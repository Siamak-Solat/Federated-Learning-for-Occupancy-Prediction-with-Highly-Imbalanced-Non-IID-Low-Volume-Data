# SPDX-License-Identifier: MIT
# © 2025 Siamak Solat
"""
client_inference_baseline.py - ABLATED BASELINE BASELINE VERSION

REMOVED TECHNIQUES & REPLACEMENTS:
1. Autoregressive noise injection → Deterministic predictions
2. Pattern alignment → No post-processing alignment  
3. Calibration parameters → Direct model output
4. Largest remainder rounding → Simple rounding
5. Monthly normalization → No seasonal adjustment
6. Advanced features → Only raw counts

KEPT:
- Autoregressive forecasting structure
- 96×365 steps for full year 2026
- Model trained through federated learning

WHY: To measure the cumulative impact of all inference-time improvements
"""

import os, argparse, yaml, numpy as np, pandas as pd, tensorflow as tf
from tqdm import tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import absl.logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)

def cfg(zone: str) -> dict:
    """ABLATED BASELINE: No zone-specific overrides"""
    root = yaml.safe_load(open("all_zones_baseline.yaml"))
    base = dict(root["common"])
    base["input_excel"] = root["zones"][zone]["input_excel"]
    return base

def vcol(cols):
    return [c for c in cols if c.startswith("Number of connected")][0]

def ts(df):
    return pd.to_datetime(
        df["Date"].astype(str) + " " + df["15-minute interval start time"])

# REMOVED: largest_remainder_rounding - using simple rounding
# REMOVED: build_feature_row - not needed with simple features

def roll_forward_basic(hist_counts, model, N, steps):
    """
    ABLATED BASELINE AUTOREGRESSIVE FORECASTING:
    REMOVED: Noise injection, temporal features
    REPLACED WITH: Deterministic predictions using only counts
    WHY: To measure impact of stochastic feedback and feature engineering
    """
    hist = hist_counts.tolist()
    preds = []
    
    for _ in tqdm(range(steps), desc="Rolling forecast", ncols=70):
        # ABLATED BASELINE: Only use last N count values
        x = np.array(hist[-N:], dtype=np.float32).reshape(1, N, 1)
        
        # Get prediction
        p = float(model(x, training=False)[0, 0])
        preds.append(p)
        
        # REMOVED: Noise injection (was: p + noise)
        # REMOVED: Complex feature construction
        # Simply append the prediction as next input
        hist.append(p)
    
    return np.array(preds, np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", required=True)
    ZONE = ap.parse_args().zone
    
    print(f"\n=== BASELINE Inference for zone «{ZONE}» ===")
    print("Ablated baseline version without advanced techniques")
    
    CFG = cfg(ZONE)
    
    # Load data
    df = pd.read_excel(CFG["input_excel"], engine="openpyxl")
    val_col = vcol(df.columns)
    df["ts"] = ts(df)
    df.sort_values("ts", inplace=True)
    df.set_index("ts", inplace=True)
    
    # REMOVED: Pattern loading from patterns.xlsx
    # REMOVED: Weekly and monthly means computation
    # REMOVED: Advanced feature engineering (DayName, MonthName, sin/cos hour)
    
    # Load model (trained through federated learning)
    model = tf.keras.models.load_model(f"{ZONE}_baseline.keras", compile=False)
    
    # REMOVED: Calibration parameters loading (a, b from joblib)
    # Use direct model output without linear adjustment
    
    # Prepare initial sequence (only counts)
    N = CFG["window_length"]
    steps = 96 * 365  # Full year of 15-min intervals (SAME as original)
    
    # Get last N count values for initialization
    init_counts = df[val_col].values[-N:].astype(np.float32)
    
    # ABLATED BASELINE: Basic autoregressive forecasting without noise
    # REMOVED: noise_std parameter and injection
    preds = roll_forward_basic(init_counts, model, N, steps)
    
    # REMOVED: Pattern alignment post-processing
    # No adjustment to match historical hourly/weekly/monthly patterns
    
    # REMOVED: Monthly normalization
    # No scaling to match historical monthly averages
    
    # Create timestamp range for 2026
    rng = pd.date_range("2026-01-01 00:00", periods=steps, freq="15min")
    
    # ABLATED BASELINE: Basic rounding instead of largest remainder
    # WHY: To measure impact of sophisticated rounding that preserves sums
    preds_int = np.round(np.maximum(0, preds)).astype(int)
    
    # Prepare historical data for comparison
    gt_cols = {}
    for y in range(2021, 2026):
        try:
            # Map 2026 timestamps to historical years
            hist_timestamps = rng.map(lambda t: t.replace(year=y))
            hist_vals = df.reindex(hist_timestamps)[val_col].values
            gt_cols[f"GroundTruth ({y})"] = hist_vals
        except:
            pass  # Year might not exist in data
    
    # Save output
    out = pd.DataFrame({
        "Date": rng.date.astype(str),
        "15-minute interval start time": rng.strftime("%H:%M"),
        f"{val_col} (Predicted)": preds_int,
        **gt_cols
    })
    
    out.to_excel(f"{ZONE}_forecast_2026_baseline.xlsx", index=False)
    print(f"✓ Saved {ZONE}_forecast_2026_baseline.xlsx")

if __name__ == "__main__":
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    main()