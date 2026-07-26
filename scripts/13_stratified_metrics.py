"""
13_stratified_metrics.py
----------------------------
STAGE 3 - Core stratified metric computation.

For every (Organism, Antibiotic) pair with a breakpoint in the Stage 3
reference table (12_breakpoint_reference.py), computes the full metric
panel (from Stage 2B) separately:
  (a) by Country (restricted to strata with N >= MIN_N_COUNTRY)
  (b) by Collection Year (restricted to strata with N >= MIN_N_YEAR)

using the ENTIRE cleaned SOAR dataset (not just the Stage 2B representative
subset), per the Stage 3 brief. Output feeds Parts 1-3 of the Stage 3
report (country comparisons, temporal comparisons, hidden-signal synthesis).
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

bp_module = import_module("12_breakpoint_reference".replace("-", "_"))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import ANTIBIOTIC_FULL_NAMES  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "cleaned" / "soar_stage1_standardized.csv"
TABLES_DIR = Path(__file__).resolve().parents[1] / "tables"

MIN_N_COUNTRY = 15
MIN_N_YEAR = 15


def compute_stratum_metrics(values, censors, bp):
    n = len(values)
    if n == 0:
        return None
    n_exact = int(np.sum(censors == "exact"))
    pct_censored = round(100 * (1 - n_exact / n), 2)

    sorted_v = np.sort(values)
    log2v = np.log2(values)

    def cum_pct(pct):
        idx = int(np.ceil(pct / 100 * n)) - 1
        idx = min(max(idx, 0), n - 1)
        return float(sorted_v[idx])

    mic50 = cum_pct(50)
    mic90 = cum_pct(90)
    gmean = float(2 ** np.mean(log2v))
    p25, p75 = float(np.percentile(values, 25)), float(np.percentile(values, 75))
    iqr_log2 = float(np.log2(p75) - np.log2(p25))
    skewness = float(stats.skew(log2v))
    kurtosis = float(stats.kurtosis(log2v))
    vals, counts = np.unique(values, return_counts=True)
    modal = float(vals[np.argmax(counts)])
    pct_r = round(100 * np.mean(values > bp), 2) if bp is not None else None

    return {
        "N": n, "% censored": pct_censored,
        "MIC50": mic50, "MIC90": mic90, "GeoMean": round(gmean, 4),
        "P25": p25, "P75": p75, "IQR_log2": round(iqr_log2, 3),
        "Skewness": round(skewness, 3), "Kurtosis": round(kurtosis, 3),
        "Modal_MIC": modal, "PctR": pct_r,
    }


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CLEANED_DATA_PATH)

    country_rows, year_rows = [], []

    for (organism, ab), (bp, conf, note) in bp_module.BREAKPOINTS_S.items():
        val_col, cen_col = f"{ab}_value", f"{ab}_censor"
        sub_org = df[df["Organism"] == organism]
        mask_all = sub_org[cen_col].isin(["exact", "left", "right"])
        sub_org = sub_org[mask_all]

        # --- by country ---
        for country, g in sub_org.groupby("Country"):
            values = g[val_col].to_numpy(dtype=float)
            censors = g[cen_col].to_numpy()
            if len(values) < MIN_N_COUNTRY:
                continue
            m = compute_stratum_metrics(values, censors, bp)
            if m is None:
                continue
            country_rows.append({
                "Organism": organism, "Antibiotic": ab,
                "Antibiotic (name)": ANTIBIOTIC_FULL_NAMES[ab],
                "Breakpoint confidence": conf, "Country": country, **m,
            })

        # --- by year ---
        for year, g in sub_org.groupby("Collection_Year"):
            values = g[val_col].to_numpy(dtype=float)
            censors = g[cen_col].to_numpy()
            if len(values) < MIN_N_YEAR:
                continue
            m = compute_stratum_metrics(values, censors, bp)
            if m is None:
                continue
            year_rows.append({
                "Organism": organism, "Antibiotic": ab,
                "Antibiotic (name)": ANTIBIOTIC_FULL_NAMES[ab],
                "Breakpoint confidence": conf, "Year": int(year), **m,
            })

    country_df = pd.DataFrame(country_rows)
    year_df = pd.DataFrame(year_rows)

    country_path = TABLES_DIR / "t1_stratified_metrics_by_country.csv"
    year_path = TABLES_DIR / "t2_stratified_metrics_by_year.csv"
    country_df.to_csv(country_path, index=False)
    year_df.to_csv(year_path, index=False)

    print(f"By-country strata computed: {len(country_df)} (Organism x Antibiotic x Country rows, N>={MIN_N_COUNTRY})")
    print(f"Saved: {country_path}")
    print(f"By-year strata computed: {len(year_df)} (Organism x Antibiotic x Year rows, N>={MIN_N_YEAR})")
    print(f"Saved: {year_path}")


if __name__ == "__main__":
    main()
