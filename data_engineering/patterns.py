
# SPDX-License-Identifier: MIT
# © 2025 Siamak Solat
"""
patterns.py – build per-zone zero/Non-zero pattern tables.

- Generates weekly, monthly and yearly summaries for every
  `*_dataset.xlsx` file and writes them to *patterns.xlsx*
"""


from pathlib import Path
import pandas as pd
import numpy as np
import re

OUT_FILE = "patterns.xlsx"

def find_value_column(cols):
    for c in cols:
        if c.startswith("Number of connected devices in zone"):
            return c
    raise ValueError("Device-count column not found.")

def enrich(df, time_col, val_col):
    ts = pd.to_datetime(df["Date"].astype(str) + " " + df[time_col])
    df = df.copy()
    df["Day"]         = ts.dt.day_name()
    df["Month"]       = ts.dt.month_name()
    df["Year"]        = ts.dt.year
    df["is_non_zero"] = (df[val_col] > 0).astype(int)
    return df

def build_table(df, key, order, val_col):
    g = df.groupby(key, sort=False)
    total = g[val_col].sum()
    non0  = g["is_non_zero"].sum()
    zero  = g.size() - non0
    ratio = zero / non0.replace(0, np.nan)
    pct   = (zero / (zero + non0).replace(0, np.nan)) * 100
    return pd.DataFrame({
        "Total devices": total,
        "Non-zero rows": non0,
        "Zero/Non-zero ratio":
            ratio.map("{:.8f}".format).astype(str),
        "Percent Zero":
            pct.map("{:.2f}%".format),
    }).reindex(order)

def single_summary(df, val_col):
    total = df[val_col].sum()
    non0  = df["is_non_zero"].sum()
    zero  = len(df) - non0
    ratio = zero / non0 if non0 else np.nan
    pct   = zero / (zero + non0) * 100 if non0 else np.nan
    return pd.DataFrame({
        "Metric": ["Total devices","Non-zero rows",
                   "Zero/Non-zero ratio","Percent Zero"],
        "Value":  [total, non0,
                   "{:.8f}".format(ratio) if pd.notna(ratio) else "NaN",
                   "{:.2f}%".format(pct)   if pd.notna(pct) else "NaN"],
    })

def main():
    files = sorted(Path(".").glob(
        "*_dataset.xlsx"))
    if not files:
        print("No synthetic Excel files found.")
        return

    with pd.ExcelWriter(OUT_FILE, engine="openpyxl") as writer:
        for p in files:
            zone = re.sub(r"_Columns_Added.*", "", p.name)
            print(f"Processing {zone} …")

            df = pd.read_excel(p, engine="openpyxl")
            time_col = "15-minute interval start time"
            val_col  = find_value_column(df.columns)
            df = enrich(df, time_col, val_col)

            weekly_order  = ["Monday","Tuesday","Wednesday",
                             "Thursday","Friday","Saturday","Sunday"]
            monthly_order = ["January","February","March","April","May","June",
                             "July","August","September",
                             "October","November","December"]
            years = [2021,2022,2023,2024,2025]

            sheet = writer.book.create_sheet(zone)

            weekly = build_table(df, "Day", weekly_order, val_col)
            weekly.to_excel(writer, sheet_name=zone, startrow=0)

            monthly = build_table(df, "Month", monthly_order, val_col)
            start = len(weekly) + 3
            monthly.to_excel(writer, sheet_name=zone, startrow=start)

            row = start + len(monthly) + 3
            for yr in years:
                sheet.cell(row=row+1, column=1, value=f"Year {yr}")
                yr_tbl = single_summary(df[df["Year"] == yr], val_col)
                yr_tbl.to_excel(writer, sheet_name=zone,
                                startrow=row+2, index=False, header=True)
                row += len(yr_tbl) + 4  

            sheet.cell(row=row+1, column=1, value="Five Years (2021 - 2025)")
            global_tbl = single_summary(df, val_col)
            global_tbl.to_excel(writer, sheet_name=zone,
                                startrow=row+2, index=False, header=True)

    print(f"  Completed. Output saved as {OUT_FILE}")

if __name__ == "__main__":
    main()
