"""
02_dataset_overview.py
------------------------
STAGE ONE - PART 1: Dataset Overview

Produces Table 1 - Dataset Overview, summarizing the RAW dataset's
dimensions, variable names, data types, and memory footprint.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import RAW_DATA_PATH, SHEET_NAME, TABLES_DIR, ensure_dirs


def main():
    ensure_dirs()
    df = pd.read_excel(RAW_DATA_PATH, sheet_name=SHEET_NAME)
    df = df.rename(columns=lambda c: c.strip())

    n_obs, n_vars = df.shape
    mem_bytes = df.memory_usage(deep=True).sum()
    mem_mb = mem_bytes / (1024 ** 2)

    overview_rows = [
        {"Metric": "Source file", "Value": RAW_DATA_PATH.name},
        {"Metric": "Source sheet", "Value": SHEET_NAME},
        {"Metric": "Number of observations (rows)", "Value": n_obs},
        {"Metric": "Number of variables (columns)", "Value": n_vars},
        {"Metric": "Dataset dimensions", "Value": f"{n_obs} x {n_vars}"},
        {"Metric": "Memory usage (MB, deep)", "Value": round(mem_mb, 3)},
        {"Metric": "Duplicate rows (fully identical)", "Value": int(df.duplicated().sum())},
        {"Metric": "Unique isolates (Isolate Number)", "Value": int(df["Isolate Number"].nunique())},
    ]
    overview_df = pd.DataFrame(overview_rows)

    dtypes_df = pd.DataFrame(
        {
            "Variable": df.columns,
            "Pandas dtype": [str(t) for t in df.dtypes],
            "Non-null count": df.notna().sum().values,
            "Null count": df.isna().sum().values,
        }
    )

    overview_path = TABLES_DIR / "table1_dataset_overview.csv"
    dtypes_path = TABLES_DIR / "table1b_variable_dtypes.csv"
    overview_df.to_csv(overview_path, index=False)
    dtypes_df.to_csv(dtypes_path, index=False)

    print("=== Table 1 - Dataset Overview ===")
    print(overview_df.to_string(index=False))
    print()
    print("=== Table 1b - Variable dtypes (raw, as read by pandas) ===")
    print(dtypes_df.to_string(index=False))
    print(f"\nSaved: {overview_path}")
    print(f"Saved: {dtypes_path}")


if __name__ == "__main__":
    main()
