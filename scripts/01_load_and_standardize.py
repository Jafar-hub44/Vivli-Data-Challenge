"""
01_load_and_standardize.py
---------------------------
STAGE ONE - Data loading and type standardization.

Purpose
-------
Load the raw SOAR Excel file exactly as provided and produce a single
"standardized" working copy that:
  * parses the heterogeneous 'Collection Date' field into a consistent
    representation (datetime + year, plus a flag recording how each value
    was originally encoded), because the raw column mixes real dates,
    text dates, and bare years within the same field;
  * splits each antibiotic MIC field into a numeric magnitude and a
    censoring-type flag (exact / left-censored / right-censored), because
    the raw values are text strings combining a dilution number with an
    optional operator;
  * standardizes the beta-lactamase free-text coding (Negative/NEG,
    Positive/POS) into two canonical labels.

No raw values are deleted or overwritten. The original text of every
antibiotic column and the original 'Collection Date' column are retained
alongside the derived columns, so the standardized file is strictly an
ADDITIVE, reproducible transformation of the raw file. This is the file
all subsequent Stage One scripts read from.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    RAW_DATA_PATH,
    CLEANED_DATA_PATH,
    SHEET_NAME,
    ANTIBIOTIC_VARS,
    parse_mic,
    parse_collection_date,
    standardize_betalactamase,
    ensure_dirs,
)


def main():
    ensure_dirs()

    print(f"Loading raw data from: {RAW_DATA_PATH}")
    df = pd.read_excel(RAW_DATA_PATH, sheet_name=SHEET_NAME)
    print(f"Raw shape: {df.shape[0]} rows x {df.shape[1]} columns")

    df = df.rename(columns=lambda c: c.strip())

    # --- Collection Date standardization -----------------------------------
    parsed = df["Collection Date"].apply(parse_collection_date)
    df["Collection_Date_Parsed"] = [p[0] for p in parsed]
    df["Collection_Year"] = [p[1] for p in parsed]
    df["Collection_Date_SourceType"] = [p[2] for p in parsed]

    # --- Beta-lactamase standardization -------------------------------------
    df["Betalactamase_Standardized"] = df["Betalactamase"].apply(standardize_betalactamase)

    # --- Antibiotic MIC parsing ---------------------------------------------
    for ab in ANTIBIOTIC_VARS:
        parsed_mic = df[ab].apply(parse_mic)
        df[f"{ab}_value"] = [p[0] for p in parsed_mic]
        df[f"{ab}_censor"] = [p[1] for p in parsed_mic]

    # --- Basic derived flags for later QC ------------------------------------
    df["Age_flag_extreme"] = df["Age"] > 110  # plausibility flag only, not removed

    out_path = CLEANED_DATA_PATH
    df.to_csv(out_path, index=False)
    print(f"Standardized dataset written to: {out_path}")
    print(f"Standardized shape: {df.shape[0]} rows x {df.shape[1]} columns")


if __name__ == "__main__":
    main()
