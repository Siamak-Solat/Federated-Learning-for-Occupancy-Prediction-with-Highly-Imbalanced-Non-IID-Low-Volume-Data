# ────────────────────────────────────────────────────────────────
#  synthetic_generator.py     
# ────────────────────────────────────────────────────────────────

from pathlib import Path
import pandas as pd
import numpy as np
import re
import sys

# ---------------------------- Config -------------------------------- #
MAX_ITER   = 5          # maximum κ refinement rounds
ABS_TOL    = 0.001      # 0.1 % absolute tolerance
REL_TOL    = 0.05       # 5  % relative tolerance
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
# -------------------------------------------------------------------- #

# ---------------------- Helper functions ---------------------------- #
def find_value_column(cols):
    for c in cols:
        if c.startswith("Number of connected devices in zone"):
            return c
    raise ValueError("Device-count column not found.")

def add_features(df, time_col, val_col):
    ts = pd.to_datetime(df["Date"].astype(str) + " " + df[time_col])
    df = df.copy()
    df["timestamp"]   = ts
    df["slot"]        = ts.dt.hour * 4 + ts.dt.minute // 15
    df["dow"]         = ts.dt.dayofweek
    df["month"]       = ts.dt.month
    df["is_non_zero"] = (df[val_col] > 0).astype(int)
    df["Day of Week"] = ts.dt.day_name()
    df["Month"]       = ts.dt.month_name()
    return df

def freq_dict(s):
    vc = s.value_counts(normalize=True)
    return vc.to_dict()

def build_lookups(df_feat, val_col):
    prob_triple = (
        df_feat.groupby(["slot", "dow", "month"])["is_non_zero"].mean().to_dict()
    )
    prob_slot_dow = (
        df_feat.groupby(["slot", "dow"])["is_non_zero"].mean().to_dict()
    )
    global_prob = df_feat["is_non_zero"].mean()

    value_dist_slot_dow = (
        df_feat[df_feat["is_non_zero"] == 1]
        .groupby(["slot", "dow"])[val_col]
        .apply(freq_dict)
        .to_dict()
    )
    global_value_dist = freq_dict(df_feat.loc[df_feat["is_non_zero"] == 1, val_col])

    return prob_triple, prob_slot_dow, global_prob, value_dist_slot_dow, global_value_dist

def sample_positive(slot, dow, v_dist_slot_dow, v_dist_global):
    dist = v_dist_slot_dow.get((slot, dow), v_dist_global)
    vals = np.array(list(dist.keys()))
    probs = np.array(list(dist.values()))
    return int(np.random.choice(vals, p=probs))

def generate_synthetic_rows(missing, kappa,
                            prob_triple, prob_slot_dow, gprob,
                            vdist_slot_dow, vdist_global,
                            time_col, val_col):
    rows=[]
    for ts in missing:
        slot  = ts.hour*4 + ts.minute//15
        dow   = ts.dayofweek
        mon   = ts.month

        p_raw = prob_triple.get((slot,dow,mon),
                 prob_slot_dow.get((slot,dow), gprob))
        p = max(0.0, min(1.0, p_raw * kappa))

        if np.random.random() < p:
            val = sample_positive(slot, dow, vdist_slot_dow, vdist_global)
        else:
            val = 0

        rows.append({
            "Date": ts.normalize(),
            "Day of Week": ts.day_name(),
            "Month": ts.month_name(),
            time_col: ts.strftime("%H:%M"),
            val_col: val,
            "timestamp": ts,
            "is_non_zero": int(val > 0)
        })
    return pd.DataFrame(rows)

# --------------------- Main per-zone routine ------------------------ #
def process_file(path: Path):
    print(f"\n=== Processing {path.name} ===")
    df_orig = pd.read_excel(path, engine="openpyxl")

    time_col = "15-minute interval start time"
    val_col  = find_value_column(df_orig.columns)

    # ensure original rows have 'timestamp' for proper global sort
    df_orig = df_orig.copy()
    df_orig["timestamp"] = pd.to_datetime(
        df_orig["Date"].astype(str) + " " + df_orig[time_col]
    )

    df_feat = add_features(df_orig, time_col, val_col)
    lookups = build_lookups(df_feat, val_col)

    full_range = pd.date_range("2021-01-01 00:00",
                               "2025-12-31 23:45",
                               freq="15min")
    missing = sorted(set(full_range) - set(df_feat["timestamp"]))
    print(f"Original rows: {len(df_feat):,} | Missing rows: {len(missing):,}")

    orig_ratio = df_feat["is_non_zero"].mean()
    if orig_ratio == 0:
        print("Zone has zero positive rows; no synthetic positives will be generated.")
    kappa = 1.0

    for it in range(1, MAX_ITER+1):
        df_synth = generate_synthetic_rows(
            missing, kappa,
            *lookups,
            time_col, val_col
        )
        df_all = pd.concat([df_orig, df_synth], ignore_index=True)
        new_ratio = df_all["is_non_zero"].mean()
        abs_delta = abs(new_ratio - orig_ratio)
        rel_delta = abs_delta / orig_ratio if orig_ratio > 0 else 0.0

        print(f"  Iter {it}: ratio={new_ratio:.6f} "
              f"(Δ={abs_delta:.6f}, rel={rel_delta:.2%})  κ={kappa:.4f}")

        if abs_delta <= ABS_TOL or rel_delta <= REL_TOL:
            print("  ✓ tolerance reached")
            break

        # adjust κ for next round
        if new_ratio > 0:
            kappa *= orig_ratio / new_ratio
        else:
            kappa *= 2.0  

    else:
        print("   Max iterations reached; ratio may still drift.")

    # Final formatting
    df_all["Date"] = pd.to_datetime(df_all["Date"]).dt.date
    df_all.sort_values("timestamp", inplace=True)  
    df_all = df_all[["Date", "Day of Week", "Month", time_col, val_col]]

    out_name = re.sub(r"_Columns_Added_Missing_Rows_Added\.xlsx$",
                      "_Columns_Added_Missing_Rows_Added_Synthetic_Added.xlsx",
                      path.name)
    df_all.to_excel(path.with_name(out_name),
                    index=False,
                    engine="openpyxl")
    print(f"  Written → {out_name}")

# ------------------------------ Main -------------------------------- #
if __name__ == "__main__":
    files = sorted(Path(".").glob("*_Columns_Added_Non_Existed_Rows_Added.xlsx"))
    if not files:
        print("No matching input files found.")
        sys.exit(1)

    for f in files:
        try:
            process_file(f)
        except Exception as exc:
            print(f" Error processing {f.name}: {exc}")
