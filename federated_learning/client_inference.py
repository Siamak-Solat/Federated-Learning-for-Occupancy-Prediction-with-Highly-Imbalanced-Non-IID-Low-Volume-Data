# ────────────────────────────────────────────────────────────────
#  client_inference.py     
# ────────────────────────────────────────────────────────────────

import os, argparse, yaml, numpy as np, pandas as pd, tensorflow as tf, joblib
from tqdm import tqdm
from client_train_val_test_v7_for_check import (cfg, vcol, ts, pat_ratio,
                                                align_patterns)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import absl.logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)
absl_logging.set_stderrthreshold("error")

def largest_remainder_rounding(arr: np.ndarray) -> np.ndarray:
    """
    Round a non-negative real vector to integers while preserving the total
    (largest-remainder / Hamilton apportionment).
    """
    floors = np.floor(arr).astype(int)
    rema   = arr - floors
    k      = int(np.round(arr.sum())) - floors.sum()
    if k > 0:
        floors[np.argsort(rema)[::-1][:k]] += 1
    return floors

def build_feature_row(t, count, w_mean, m_mean):
    dow, mon = t.day_name(), t.month_name()
    hr = t.hour + t.minute / 60
    dow_ohe = np.zeros(7,  np.float32)
    mon_ohe = np.zeros(12, np.float32)
    dow_ohe[["Monday","Tuesday","Wednesday","Thursday",
             "Friday","Saturday","Sunday"].index(dow)] = 1
    mon_ohe[["January","February","March","April","May","June",
             "July","August","September","October","November",
             "December"].index(mon)] = 1
    return np.concatenate([[count, w_mean.get(dow,0.0), m_mean.get(mon,0.0)],
                           dow_ohe, mon_ohe,
                           [np.sin(2*np.pi*hr/24), np.cos(2*np.pi*hr/24)]],
                          dtype=np.float32)

def roll_forward(hist_df, model, N, steps, w_mean, m_mean, noise_std):
    """
    Auto-regressive walk: predict `steps` further 15-minute points, feeding each
    prediction (plus optional noise) back as the next input.
    """
    hist  = hist_df.values.astype(np.float32).tolist()
    preds = []
    for _ in tqdm(range(steps), desc="Rolling", ncols=70):
        x = np.expand_dims(hist[-N:], 0)
        p = float(model(x, training=False)[0, 0])
        preds.append(p)
        next_ts = hist_df.index[-1] + pd.Timedelta(minutes=15)

        sample = max(0.0,
                     p + (np.random.normal(0, noise_std * max(1.0, abs(p)))
                          if noise_std else 0.0))
        hist.append(build_feature_row(next_ts, sample, w_mean, m_mean))
        hist_df.loc[next_ts] = hist[-1]
    return np.array(preds, np.float32)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--zone", required=True)
    ZONE = ap.parse_args().zone
    print(f"\n=== Inference for zone «{ZONE}» ===")

    CFG       = cfg(ZONE)
    noise_std = float(CFG["autoregressive"]["feedback_noise_std"])

    df = pd.read_excel(CFG["input_excel"], engine="openpyxl")
    val_col = vcol(df.columns)
    df["ts"] = ts(df); df.sort_values("ts", inplace=True); df.set_index("ts", inplace=True)

    pat = pd.read_excel(CFG["patterns_excel"], sheet_name=ZONE, engine="openpyxl")
    w_mean = {r.iloc[0]: r["Total devices"]/r["Non-zero rows"]
              if r["Non-zero rows"] else 0.0
              for _, r in pat.iterrows()
              if r.iloc[0] in ["Monday","Tuesday","Wednesday",
                               "Thursday","Friday","Saturday","Sunday"]}
    m_mean = {r.iloc[0]: r["Total devices"]/r["Non-zero rows"]
              if r["Non-zero rows"] else 0.0
              for _, r in pat.iterrows()
              if r.iloc[0] in ["January","February","March","April","May","June",
                               "July","August","September","October",
                               "November","December"]}

    df["DayName"]   = df.index.day_name()
    df["MonthName"] = df.index.month_name()
    df["Hour"]      = df.index.hour + df.index.minute / 60
    df["Week_Mean"]  = df["DayName"].map(w_mean).fillna(0.0)
    df["Month_Mean"] = df["MonthName"].map(m_mean).fillna(0.0)
    df["sin_hour"]   = np.sin(2*np.pi*df["Hour"]/24)
    df["cos_hour"]   = np.cos(2*np.pi*df["Hour"]/24)
    feat_df = pd.DataFrame([build_feature_row(t, r[val_col], w_mean, m_mean)
                            for t, r in df.iterrows()],
                           index=df.index)

    model = tf.keras.models.load_model(f"{ZONE}.keras", compile=False)
    a, b = joblib.load(f"{ZONE}_calib.pkl")
    if not (np.isfinite(a) and a > 0):  
        a, b = 1.0, 0.0

    N      = CFG["window_length"]
    steps  = 96 * 365                          
    preds  = a * roll_forward(feat_df.iloc[-N:].copy(), model, N, steps,
                              w_mean, m_mean, noise_std) + b
    rng    = pd.date_range("2026-01-01 00:00", periods=steps, freq="15min")

    if CFG.get("pattern_alignment", {}).get("enabled", True):
        pa   = CFG["pattern_alignment"]
        ref  = np.mean([df[df.index.year == y][val_col].reindex(
                        rng.map(lambda t: t.replace(year=y))).values
                        for y in range(2021, 2026)], axis=0)
        preds = align_patterns(pd.Series(preds, index=rng), rng,
                               pd.Series(ref,   index=rng),
                               iters=pa.get("iterations", 3),
                               clip_min=pa.get("clip_min", 0.5),
                               clip_max=pa.get("clip_max", 2.0)).values

    hist_avg = {m: df[df.index.month == m][val_col].mean() for m in range(1, 13)}
    m_lbl = rng.month
    for m in range(1, 13):
        idx  = np.where(m_lbl == m)[0]
        tgt  = hist_avg[m] * len(idx)
        cur  = preds[idx].sum()
        if cur > 0:
            preds[idx] *= tgt / cur

    preds_int = largest_remainder_rounding(preds)

    gt_cols = {f"GroundTruth ({y})":
               df.reindex(rng.map(lambda t: t.replace(year=y)))[val_col].values
               for y in range(2021, 2026)}

    out = pd.DataFrame({"Date": rng.date.astype(str),
                        "15-minute interval start time": rng.strftime("%H:%M"),
                        f"{val_col} (Predicted)": preds_int,
                        **gt_cols})
    out.to_excel(f"{ZONE}_forecast_2026.xlsx", index=False)
    print("✓ saved", f"{ZONE}_forecast_2026.xlsx")

if __name__ == "__main__":
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    main()
