# ────────────────────────────────────────────────────────────────
#  add_column.py
# ────────────────────────────────────────────────────────────────

import sys
from pathlib import Path
import pandas as pd

DEVICE_COL_PREFIX = "Number of connected devices in zone of "

def transform_file(path: Path) -> None:
    """Read, enrich, and write a single workbook."""
    df = pd.read_excel(path, engine="openpyxl")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    df["Day of Week"] = df["Date"].dt.day_name()
    df["Month"] = df["Date"].dt.month_name()

    df["Date"] = df["Date"].dt.date

    zone_col = next(
        (c for c in df.columns if c.startswith(DEVICE_COL_PREFIX)), None
    )

    base_order = [
        "Date",
        "Day of Week",
        "Month",
        "15-minute interval start time",
    ]
    if zone_col:
        base_order.append(zone_col)

    ordered_cols = base_order + [c for c in df.columns if c not in base_order]
    df = df[ordered_cols]

    new_name = path.name.replace("Original", "Columns_Added")
    df.to_excel(path.with_name(new_name), index=False)
    print(f"✓ {path.name}  →  {new_name}")


def main() -> None:
    folder = Path(__file__).resolve().parent
    originals = sorted(folder.glob("*_Original.xlsx"))

    if not originals:
        print("No *_Original.xlsx files found here.", file=sys.stderr)
        sys.exit(1)

    for p in originals:
        try:
            transform_file(p)
        except Exception as e:
            print(f"✗ Could not process {p.name}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
