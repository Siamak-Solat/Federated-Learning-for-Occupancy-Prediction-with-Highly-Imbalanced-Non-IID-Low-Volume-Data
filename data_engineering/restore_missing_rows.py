
# SPDX-License-Identifier: MIT
# © 2025 Siamak Solat
"""
restore_missing_rows.py - fill 15-minute gaps in zone heat-map workbooks.

• Accepts `Zone_Heatmap_*_Columns_Added.xlsx` files  
• Re-creates any missing time-steps between
  2022-12-01 00:00 and 2025-04-29 23:45 and sets device count = 0  
• Outputs `<Zone>_Columns_Added_Missing_Rows_Added.xlsx`
"""

from pathlib import Path
import pandas as pd
import re
from typing import Tuple

INPUT_PATTERN = "Zone_Heatmap_*_Columns_Added.xlsx"
START_DATETIME = "2022-12-01 00:00"
END_DATETIME   = "2025-04-29 23:45"
FREQ           = "15min"                       
DATE_FORMAT    = "%Y-%m-%d %H:%M"

def detect_device_column(columns) -> str:
    """Return the column that holds the device counts for the zone."""
    for col in columns:
        if str(col).startswith("Number of connected devices in zone of"):
            return col
    raise ValueError(
        "Could not find a column starting with "
        "'Number of connected devices in zone of'"
    )


def load_original(path: Path) -> Tuple[pd.DataFrame, str]:
    """
    Load the workbook and create a helper DateTime index.

    Returns
    -------
    df          : pd.DataFrame - original rows, indexed by helper DateTime
    device_col  : str          - the name of the device-count column
    """
    df = pd.read_excel(path)

    df = df.dropna(subset=["15-minute interval start time"]).copy()

    dt_str = (
        df["Date"].astype(str).str.strip()
        + " "
        + df["15-minute interval start time"].str.strip()
    )
    df["__DateTime__"] = pd.to_datetime(dt_str, format=DATE_FORMAT)
    df = df.set_index("__DateTime__", drop=False)

    device_col = detect_device_column(df.columns)
    return df, device_col


def full_range() -> pd.DatetimeIndex:
    """Return the complete 15-minute timeline for the required period."""
    return pd.date_range(
        start=START_DATETIME,
        end=END_DATETIME,
        freq=FREQ,
        name="__DateTime__",
    )


def make_missing_rows(missing: pd.DatetimeIndex, device_col: str) -> pd.DataFrame:
    """Craft rows (zero devices) for every missing 15-minute interval."""
    if missing.empty:
        return pd.DataFrame()

    rows = pd.DataFrame(index=missing)
    rows["Date"]  = missing.normalize()
    rows["Day of Week"] = rows["Date"].dt.day_name()
    rows["Month"] = rows["Date"].dt.month_name()
    rows["15-minute interval start time"] = missing.strftime("%H:%M")
    rows[device_col] = 0  
    rows["__DateTime__"] = rows.index
    return rows


def build_output_filename(input_file: Path) -> str:
    """
    Convert 'Zone_Heatmap_<Zone>_Columns_Added.xlsx'
    to '<Zone>_Columns_Added_Missing_Rows_Added.xlsx'
    """
    m = re.match(r"Zone_Heatmap_(.+)_Columns_Added\.xlsx$", input_file.name)
    if not m:
        raise ValueError(f"Unexpected file name: {input_file.name}")
    zone = m.group(1)
    return f"{zone}_Columns_Added_Missing_Rows_Added.xlsx"


def process_file(path: Path) -> None:
    """Fill the gaps for a single zone workbook and write the result."""
    original_df, device_col = load_original(path)

    expected_index = full_range()
    missing_idx    = expected_index.difference(original_df.index)
    missing_df     = make_missing_rows(missing_idx, device_col)

    merged = pd.concat([original_df, missing_df])
    merged = (
        merged.sort_index()
        .loc[~merged.index.duplicated(keep="first")]
        .drop(columns="__DateTime__")
        .reset_index(drop=True)
    )

    merged["Date"] = pd.to_datetime(merged["Date"]).dt.date

    ordered_cols = [
        "Date",
        "Day of Week",
        "Month",
        "15-minute interval start time",
        device_col,
    ]
    merged = merged[ordered_cols]

    output_file = build_output_filename(path)
    merged.to_excel(output_file, index=False)
    print(f"✓ {path.name} → {output_file}")


def main() -> None:
    base = Path(__file__).resolve().parent
    files = sorted(base.glob(INPUT_PATTERN))

    if not files:
        print("No input files found matching:", INPUT_PATTERN)
        return

    for f in files:
        try:
            process_file(f)
        except Exception as exc:
            print(f"✗ Failed to process {f.name}: {exc}")


if __name__ == "__main__":
    main()
