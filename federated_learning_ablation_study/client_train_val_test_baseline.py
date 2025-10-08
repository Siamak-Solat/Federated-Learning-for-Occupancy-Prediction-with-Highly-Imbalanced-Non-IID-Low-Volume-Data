# SPDX-License-Identifier: MIT
# © 2025 Siamak Solat
"""
client_train_val_test_baseline.py - ABLATED BASELINE BASELINE VERSION

REMOVED TECHNIQUES & REPLACEMENTS:
1. Advanced feature engineering → Only raw counts as input
2. Zero/non-zero ratio weighting → No class weighting  
3. Focal MSE / Huber / Pinball loss → Basic MSE only
4. Pattern alignment → No post-processing
5. Calibration → Direct predictions without adjustment
6. Early stopping → Fixed epochs training
7. Largest remainder rounding → Simple rounding
8. Zone-specific hyperparameters → Same config for all zones

KEPT:
- Federated learning integration (loads from previous round)
- Same model architecture (2 LSTM layers, 64 units)
- Same epochs (20) and batch size (256)
"""

import sys, os, warnings, json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import yaml, numpy as np, pandas as pd, tensorflow as tf
from tqdm.keras import TqdmCallback
import argparse
warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

def cfg(zone: str) -> dict:
    """ABLATED BASELINE: No zone-specific overrides, all use common config"""
    root = yaml.safe_load(open("all_zones_baseline.yaml"))
    base = dict(root["common"])
    base["input_excel"] = root["zones"][zone]["input_excel"]
    base["rounds"] = root["common"]["federated"]["rounds"]
    # REMOVED: No zone-specific parameter overrides
    return base

def vcol(cols): 
    return [c for c in cols if c.startswith("Number of connected")][0]

def ts(df):
    return pd.to_datetime(
        df["Date"].astype(str) + " " + df["15-minute interval start time"])

# REMOVED: pat_ratio function - no class weighting needed

def windows_basic(df, val_col, N, H, stride):
    """
    ABLATED BASELINE FEATURE EXTRACTION:
    REMOVED: Week_Mean, Month_Mean, day/month one-hot encoding, sin/cos hour
    REPLACED WITH: Only raw connection counts
    WHY: To measure impact of temporal feature engineering
    """
    counts = df[val_col].values.astype(np.float32)
    X, y = [], []
    for st in range(0, len(df) - N - H + 1, stride):
        ed = st + N
        # ABLATED BASELINE: Only use raw counts, shape (N, 1)
        feat = counts[st:ed, None]
        X.append(feat)
        y.append(counts[ed + H - 1])
    return np.array(X), np.array(y)[:, None]

# REMOVED: focal_mse function - using standard MSE
# REMOVED: loss_factory function - using standard MSE

def build_model(cfg, init_weights_path=None):
    """
    Build model with SAME architecture as original.
    Can initialize from previous round for federated learning.
    """
    if init_weights_path and tf.io.gfile.exists(init_weights_path):
        # Load model from previous federated round
        mdl = tf.keras.models.load_model(init_weights_path, compile=False)
    else:
        # Build new model with SAME architecture as original
        mdl = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(cfg["window_length"], 1)),  # Only 1 feature now
            # KEPT SAME: 2 LSTM layers with 64 units each
            tf.keras.layers.LSTM(cfg["model"]["lstm_units"], return_sequences=True),
            tf.keras.layers.LSTM(cfg["model"]["lstm_units"]),
            tf.keras.layers.Dense(cfg["model"]["dense_units"], activation="relu"),
            tf.keras.layers.Dense(1)
        ])
    
    # ABLATED BASELINE: Basic MSE loss, no focal/huber/pinball
    mdl.compile(
        optimizer=tf.keras.optimizers.Adam(cfg["model"]["lr"]),
        loss="mse",  # REMOVED: Custom loss functions
        metrics=["mae"]
    )
    return mdl

# REMOVED: align_patterns function - no pattern alignment

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", required=True)
    ap.add_argument("--round", type=int, default=0)
    args = ap.parse_args()
    ZONE, RND = args.zone, args.round
    CFG = cfg(ZONE)

    print(f"\n=== BASELINE Training: {ZONE} | round {RND} ===")
    print("ABLATED BASELINE version: basic features, MSE loss, no post-processing")
    
    # REMOVED: Pattern ratio computation
    # REMOVED: patterns.xlsx loading

    # Load data
    df = pd.read_excel(CFG["input_excel"], engine="openpyxl")
    val_col = vcol(df.columns)
    df["ts"] = ts(df)
    df.sort_values("ts", inplace=True)
    df.set_index("ts", inplace=True)

    # REMOVED: Pattern-based weekly/monthly means
    # REMOVED: DayName, MonthName, sin_hour, cos_hour features

    # Split data (SAME as original)
    train_df = df["2021":"2024"]
    val_df = df["2025-01-01":"2025-06-30"]
    test_df = df["2025-07-01":"2025-12-31"]
    
    N, H, S = CFG["window_length"], CFG["horizon"], CFG["stride"]
    
    # ABLATED BASELINE: Use basic windowing without advanced features
    Xtr, ytr = windows_basic(train_df, val_col, N, H, S)
    Xva, yva = windows_basic(val_df, val_col, N, H, S)
    Xte, yte = windows_basic(test_df, val_col, N, H, S)

    # Build model (with federated initialization if round > 0)
    init_path = None
    if RND > 0:
        # For federated learning: initialize from global model of previous round
        init_path = f"global_weights/global_round{RND-1}_baseline.keras"
    
    model = build_model(CFG, init_path)
    
    # REMOVED: Early stopping callback
    # Train for fixed epochs as specified in config (SAME as original: 20)
    history = model.fit(
        Xtr, ytr,
        validation_data=(Xva, yva),
        epochs=CFG["model"]["epochs"],  # Still 20 epochs
        batch_size=CFG["model"]["batch_size"],  # Still 256
        verbose=0,
        callbacks=[TqdmCallback(verbose=0)]
        # REMOVED: EarlyStopping callback
    )

    # Save metrics for federated aggregation
    metrics = {
        "zone": ZONE,
        "round": RND,
        "n_train": int(len(ytr)),
        "n_val": int(len(yva)),
        "train_loss": float(history.history["loss"][-1]),
        "val_loss": float(history.history["val_loss"][-1])
    }
    with open(f"{ZONE}_metrics_round{RND}_baseline.json", "w") as f:
        json.dump(metrics, f)

    # REMOVED: Calibration with linear regression
    # Use direct predictions without adjustment
    te_pred = model.predict(Xte, batch_size=512).flatten()
    
    # REMOVED: Pattern alignment post-processing
    
    # ABLATED BASELINE: Basic rounding instead of largest remainder
    # WHY: To measure impact of sophisticated rounding that preserves sums
    te_pred_int = np.round(np.maximum(0, te_pred)).astype(int)
    
    # Get timestamps for predictions
    idx = test_df[val_col].iloc[N + H - 1::S].index[:len(te_pred)]
    
    # Save predictions
    out = pd.DataFrame({
        "timestamp": idx,
        "GroundTruth": yte.flatten(),
        "Predicted": te_pred_int
    })
    
    out.to_excel(f"{ZONE}_test_predictions_baseline.xlsx", index=False)
    
    # Save model for federated aggregation
    model.save(f"{ZONE}_round{RND}_baseline.keras")
    
    # Also save as zone model for inference
    model.save(f"{ZONE}_baseline.keras")
    
    print(f"✓ Saved {ZONE}_test_predictions_baseline.xlsx and model checkpoints")

if __name__ == "__main__":
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    main()