
# SPDX-License-Identifier: MIT
# © 2025 Siamak Solat
"""
client_train_val_test.py – local training/validation/testing for one zone.

• Builds or resumes an LSTM, applies focal-MSE / Huber / Pinball loss  
• Splits data (2021-24 train, 2025-H1 val, 2025-H2 test)  
• Calibrates output (linear regression), writes test predictions, and
  checkpoints `<ZONE>_round<R>.keras`
"""

import sys, re, os, warnings, json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
_error_regex = re.compile(
    r"(Unable to register cuDNN factory|Unable to register cuBLAS factory|"
    r"All log messages before absl::InitializeLog)", re.IGNORECASE)
class _StderrFilter:                                   
    def __init__(self, t): self._t = t
    def write(self, m): None if _error_regex.search(m) else self._t.write(m)
    def flush(self): self._t.flush()
sys.stderr = _StderrFilter(sys.stderr)

import yaml, numpy as np, pandas as pd, tensorflow as tf
from tqdm.keras import TqdmCallback
from sklearn.linear_model import LinearRegression
import joblib, absl.logging as absl_logging, argparse
absl_logging.set_verbosity(absl_logging.ERROR)
absl_logging.set_stderrthreshold("error")
tf.get_logger().setLevel("ERROR")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="tensorflow")

def _recursive_update(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and k in dst and isinstance(dst[k], dict):
            _recursive_update(dst[k], v)
        else:
            dst[k] = v

def cfg(zone: str) -> dict:
    root = yaml.safe_load(open("all_zones.yaml"))
    base = dict(root["common"])
    _recursive_update(base, root["zones"][zone])
    base["rounds"] = root["common"]["federated"]["rounds"]
    return base


def vcol(cols): return [c for c in cols
                        if c.startswith("Number of connected")][0]
def ts(df):      return pd.to_datetime(
        df["Date"].astype(str)+" "+df["15-minute interval start time"])

def pat_ratio(zone):
    try:
        dfp = pd.read_excel("patterns.xlsx",
                            sheet_name=zone, engine="openpyxl")
        hdr = dfp[dfp.iloc[:,0].astype(str).str.contains("Five Years")].index[0]+1
        val = dfp.loc[hdr:][dfp.iloc[:,0] == "Zero/Non-zero ratio"].iloc[0,1]
        return float(str(val).replace(",","."))          
    except Exception:
        return 30.0

def windows_full(df,val_col,N,H,stride):
    counts  = df[val_col].values.astype(np.float32)
    wmean   = df["Week_Mean"].values.astype(np.float32)
    mmean   = df["Month_Mean"].values.astype(np.float32)
    dow_ohe = pd.get_dummies(df["DayName"]).reindex(
        columns=["Monday","Tuesday","Wednesday","Thursday",
                 "Friday","Saturday","Sunday"],fill_value=0).values.astype(np.float32)
    mon_ohe = pd.get_dummies(df["MonthName"]).reindex(
        columns=["January","February","March","April","May","June",
                 "July","August","September","October","November","December"],
        fill_value=0).values.astype(np.float32)
    sin_h   = df["sin_hour"].values.astype(np.float32)
    cos_h   = df["cos_hour"].values.astype(np.float32)
    X,y=[],[]
    for st in range(0,len(df)-N-H+1,stride):
        ed = st+N
        feat = np.concatenate([counts[st:ed,None],
                               wmean[st:ed,None], mmean[st:ed,None],
                               dow_ohe[st:ed], mon_ohe[st:ed],
                               sin_h[st:ed,None], cos_h[st:ed,None]],1)
        X.append(feat); y.append(counts[ed+H-1])
    return np.array(X), np.array(y)[:,None]

def focal_mse(gamma,w0,w1):
    def l(y_t,y_p):
        err=tf.square(y_t-y_p)
        cw=tf.where(tf.equal(y_t,0.0), w0, w1)
        return tf.reduce_mean(err*(1+gamma*tf.abs(y_t))*cw)
    return l
def loss_factory(cfg,w0,w1):
    n=cfg.get("loss","mse").lower()
    if n=="mse":    return focal_mse(cfg["imbalance"]["focal_gamma"],w0,w1)
    if n=="huber":  return tf.keras.losses.Huber()
    if n=="pinball":
        tau=cfg.get("pinball_tau",0.9)
        return (tf.keras.losses.PinballLoss(tau=tau)
                if hasattr(tf.keras.losses,"PinballLoss")
                else lambda y_t,y_p: tf.reduce_mean(
                        tf.maximum(tau*(y_t-y_p),(tau-1)*(y_t-y_p)) ))
    raise ValueError("Unknown loss")

def build_model(cfg,loss_fn,init=None):
    if init and tf.io.gfile.exists(init):
        mdl=tf.keras.models.load_model(init,compile=False)
    else:
        mdl=tf.keras.Sequential([tf.keras.layers.Input(
             shape=(cfg["window_length"],24))])
        for i in range(cfg["model"]["lstm_layers"]):
            mdl.add(tf.keras.layers.LSTM(cfg["model"]["lstm_units"],
                     return_sequences=(i<cfg["model"]["lstm_layers"]-1)))
        mdl.add(tf.keras.layers.Dense(cfg["model"]["dense_units"],activation="relu"))
        mdl.add(tf.keras.layers.Dense(1))
    mdl.compile(optimizer=tf.keras.optimizers.Adam(cfg["model"]["lr"]),
                loss=loss_fn,metrics=["mae"])
    return mdl

def align_patterns(pred_ser, ts_index, gt_ser,
                   iters=3, clip_min=0.5, clip_max=2.0):
    """
    Multiplicatively aligns `pred_ser` (pandas Series) to match
    hourly, weekday and monthly means of `gt_ser`.
    """
    eps = 1e-9
    aligned = pred_ser.astype(float).copy()
    for _ in range(iters):
        
        h_fac = ((gt_ser.groupby(ts_index.hour).mean()+eps) /
                 (aligned.groupby(ts_index.hour).mean()+eps)).clip(clip_min, clip_max)
        aligned *= h_fac[ts_index.hour].values
        
        d_fac = ((gt_ser.groupby(ts_index.dayofweek).mean()+eps) /
                 (aligned.groupby(ts_index.dayofweek).mean()+eps)).clip(clip_min, clip_max)
        aligned *= d_fac[ts_index.dayofweek].values
        
        m_fac = ((gt_ser.groupby(ts_index.month).mean()+eps) /
                 (aligned.groupby(ts_index.month).mean()+eps)).clip(clip_min, clip_max)
        aligned *= m_fac[ts_index.month].values
    return aligned

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--zone", required=True)
    ap.add_argument("--round",type=int,default=0)
    args=ap.parse_args()
    ZONE,RND=args.zone,args.round
    CFG=cfg(ZONE)

    ratio=pat_ratio(ZONE); w_non=np.log1p(ratio)
    print(f"\n=== {ZONE} | round {RND} ===")
    print(f"Zero/Non-zero ratio = {ratio:.2f}  → w_non = {w_non:.2f}")

    df = pd.read_excel(CFG["input_excel"], engine="openpyxl")
    val_col=vcol(df.columns)
    df["ts"]=ts(df); df.sort_values("ts", inplace=True); df.set_index("ts", inplace=True)

    pat=pd.read_excel(CFG["patterns_excel"], sheet_name=ZONE, engine="openpyxl")
    weekly_mean={r.iloc[0]:(r["Total devices"]/r["Non-zero rows"]
                  if r["Non-zero rows"] else 0.0)
                 for _,r in pat.iterrows()
                 if r.iloc[0] in ["Monday","Tuesday","Wednesday","Thursday",
                                  "Friday","Saturday","Sunday"]}
    monthly_mean={r.iloc[0]:(r["Total devices"]/r["Non-zero rows"]
                   if r["Non-zero rows"] else 0.0)
                  for _,r in pat.iterrows()
                  if r.iloc[0] in ["January","February","March","April","May","June",
                                   "July","August","September","October",
                                   "November","December"]}

    df["DayName"]=df.index.day_name(); df["MonthName"]=df.index.month_name()
    df["Hour"]=df.index.hour+df.index.minute/60
    df["Week_Mean"]=df["DayName"].map(weekly_mean).fillna(0.0)
    df["Month_Mean"]=df["MonthName"].map(monthly_mean).fillna(0.0)
    df["sin_hour"]=np.sin(2*np.pi*df["Hour"]/24)
    df["cos_hour"]=np.cos(2*np.pi*df["Hour"]/24)

    train_df=df["2021":"2024"]; val_df=df["2025-01-01":"2025-06-30"]
    test_df =df["2025-07-01":"2025-12-31"]
    N,H,S=CFG["window_length"],CFG["horizon"],CFG["stride"]
    Xtr,ytr=windows_full(train_df,val_col,N,H,S)
    Xva,yva=windows_full(val_df  ,val_col,N,H,S)
    Xte,yte=windows_full(test_df ,val_col,N,H,S)  

    loss_fn=loss_factory(CFG,1.0,w_non)
    init=f"global_round{RND-1}.keras" if RND else None
    model=build_model(CFG,loss_fn,init)

    es_cfg=CFG.get("early_stopping",{})
    es=tf.keras.callbacks.EarlyStopping(
        patience=int(es_cfg.get("patience",4)),
        min_delta=float(es_cfg.get("min_delta",0.001)),
        restore_best_weights=True)

    history=model.fit(Xtr,ytr,validation_data=(Xva,yva),
                      epochs=CFG["model"]["epochs"],
                      batch_size=CFG["model"]["batch_size"],
                      verbose=0, callbacks=[TqdmCallback(verbose=0),es])

    metrics={"zone":ZONE,"round":RND,
             "n_train":int(len(ytr)),"n_val":int(len(yva)),
             "train_loss":float(history.history["loss"][-1]),
             "val_loss":float(history.history["val_loss"][-1])}
    with open(f"{ZONE}_metrics_round{RND}.json","w") as f: json.dump(metrics,f)

    va_pred = model.predict(Xva,batch_size=512).flatten()
    lr = LinearRegression().fit(va_pred[:,None], yva.flatten())
    a,b = lr.coef_[0], lr.intercept_
    limit=1.0+np.log1p(ratio)
    if (not np.isfinite(a)) or (a<=0) or (a>limit):
        warnings.warn(f"Calibration unstable (a={a:.3g}) – revert to identity")
        a,b=1.0,0.0
    joblib.dump((a,b),f"{ZONE}_calib.pkl")
    te_raw=model.predict(Xte,batch_size=512).flatten()
    te_pred=a*te_raw+b

    idx=test_df[val_col].iloc[N+H-1::S].index[:len(te_pred)]

    if CFG.get("pattern_alignment",{}).get("enabled",True):
        pa=CFG["pattern_alignment"]

        hist_years = range(2021, 2025)        
        ref_stack  = []
        for yr in hist_years:
            ref_vals = df[df.index.year==yr][val_col].reindex(
                       idx.map(lambda t: t.replace(year=yr))).values
            ref_stack.append(ref_vals)
        gt_ref = np.nanmean(ref_stack, axis=0)  

        align = align_patterns(
            pd.Series(te_pred, index=idx),
            idx,
            pd.Series(gt_ref,   index=idx),
            iters=pa.get("iterations",3),
            clip_min=pa.get("clip_min",0.5),
            clip_max=pa.get("clip_max",2.0))
        te_pred=align.values

    def largest_remainder_rounding(arr):
        floors=np.floor(arr).astype(int)
        rem=arr-floors; k=int(np.round(arr.sum()))-floors.sum()
        if k>0: floors[np.argsort(rem)[::-1][:k]]+=1
        return floors
    te_pred_int=largest_remainder_rounding(te_pred)

    out=pd.DataFrame({"timestamp":idx,
                      "GroundTruth":yte.flatten(),
                      "Predicted":te_pred_int})

    out.to_excel(f"{ZONE}_test_predictions.xlsx", index=False)

    model.save(f"{ZONE}_round{RND}.keras")
    model.save(f"{ZONE}.keras")
    print(f"✓ saved {ZONE}_test_predictions.xlsx and model checkpoints")

if __name__=="__main__":
    gpus=tf.config.list_physical_devices("GPU")
    if gpus: tf.config.experimental.set_memory_growth(gpus[0],True)
    main()
