"""
21_sensitivity_analysis.py
------------------------------
STAGE 4 - PART 5: Sensitivity analysis

Re-runs Stage 3's country-pair mining logic (Part 1) across a grid of
alternative thresholds to test whether:
  (a) the OVERALL YIELD (number of qualifying country-pair combinations)
      is stable or highly threshold-dependent, and
  (b) the SPECIFIC HEADLINE CASE STUDIES highlighted in Stage 3 and
      statistically confirmed in Stage 4 Part 2 remain flagged under
      stricter/looser thresholds, which is the more important robustness
      question for the paper's actual conclusions.

Grid:
  - %R similarity threshold: 5, 8 (Stage 3 default), 10, 15 percentage points
  - GeoMean divergence threshold: 0.5, 1.0 (Stage 3 default), 1.5, 2.0 log2 steps
  - Minimum N per stratum: 15 (Stage 3 default), 25, 40
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

TABLES_DIR = Path(__file__).resolve().parents[1] / "tables"
STAGE3_TABLES_DIR = Path(__file__).resolve().parents[1] / "tables"

SIMILARITY_GRID = [5.0, 8.0, 10.0, 15.0]
DIVERGENCE_GRID = [0.5, 1.0, 1.5, 2.0]
MIN_N_GRID = [15, 25, 40]

# The exact headline case studies from Stage 3 / confirmed in Stage 4 Part 2
HEADLINE_CASES = [
    ("Haemophilus influenzae", "AXO", "Philippines", "Vietnam"),
    ("Streptococcus pneumoniae", "AXO", "Argentina", "Tunisia"),
    ("Streptococcus pneumoniae", "AMX", "Argentina", "Kenya"),
    ("Streptococcus pneumoniae", "AMC", "Argentina", "Kenya"),
]


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    # t1_stratified_metrics_by_country.csv already used MIN_N_COUNTRY=15 at
    # generation time in Stage 3, so it is the base table for the N>=15 and
    # N>=25/40 sub-filters below (re-filtering a superset is valid; we cannot
    # recover strata below N=15 without re-running Stage 3 extraction, which
    # is noted as a limitation of this sensitivity grid).
    base = pd.read_csv(STAGE3_TABLES_DIR / "t1_stratified_metrics_by_country.csv")

    grid_rows = []
    headline_tracking = []

    for min_n in MIN_N_GRID:
        filtered_base = base[base["N"] >= min_n]
        for sim_thresh in SIMILARITY_GRID:
            for div_thresh in DIVERGENCE_GRID:
                n_qualifying = 0
                headline_found = {case: False for case in HEADLINE_CASES}
                for (organism, ab), g in filtered_base.groupby(["Organism", "Antibiotic"]):
                    g = g[g["PctR"].notna()].reset_index(drop=True)
                    for i in range(len(g)):
                        for j in range(i + 1, len(g)):
                            r1, r2 = g.iloc[i], g.iloc[j]
                            pctr_diff = abs(r1["PctR"] - r2["PctR"])
                            gmean_gap = abs(np.log2(r1["GeoMean"]) - np.log2(r2["GeoMean"]))
                            if pctr_diff <= sim_thresh and gmean_gap >= div_thresh:
                                n_qualifying += 1
                                for case in HEADLINE_CASES:
                                    c_org, c_ab, c_a, c_b = case
                                    if organism == c_org and ab == c_ab and {r1["Country"], r2["Country"]} == {c_a, c_b}:
                                        headline_found[case] = True

                grid_rows.append({
                    "Min N": min_n, "Similarity threshold (pp)": sim_thresh,
                    "Divergence threshold (log2)": div_thresh,
                    "N qualifying country-pair combos": n_qualifying,
                    **{f"Case: {c[0].split()[0]}/{c[1]} {c[2]}-{c[3]}": headline_found[c] for c in HEADLINE_CASES},
                })

    grid_df = pd.DataFrame(grid_rows)
    out_path = TABLES_DIR / "t5_sensitivity_grid.csv"
    grid_df.to_csv(out_path, index=False)

    print(f"Sensitivity grid: {len(grid_df)} threshold combinations "
          f"({len(MIN_N_GRID)} x {len(SIMILARITY_GRID)} x {len(DIVERGENCE_GRID)})")
    print(f"Saved: {out_path}")

    print(f"\nYield range across grid: {grid_df['N qualifying country-pair combos'].min()} to "
          f"{grid_df['N qualifying country-pair combos'].max()} qualifying combinations "
          f"(Stage 3 default [MinN=15, sim=8pp, div=1.0]: "
          f"{grid_df[(grid_df['Min N']==15)&(grid_df['Similarity threshold (pp)']==8.0)&(grid_df['Divergence threshold (log2)']==1.0)]['N qualifying country-pair combos'].values[0]})")

    case_cols = [c for c in grid_df.columns if c.startswith("Case:")]
    print("\n=== Headline case persistence across all 48 threshold combinations ===")
    for col in case_cols:
        pct_persist = round(100 * grid_df[col].mean(), 1)
        print(f"{col}: flagged in {grid_df[col].sum()}/{len(grid_df)} combinations ({pct_persist}%)")

    # Stricter-than-default subset (a common robustness-check convention:
    # does the finding survive when we make the criteria MORE conservative
    # than the original analyst's choice, not just when we loosen them?)
    stricter = grid_df[(grid_df["Min N"] >= 25) & (grid_df["Similarity threshold (pp)"] <= 8.0) &
                        (grid_df["Divergence threshold (log2)"] >= 1.0)]
    print(f"\n=== Under STRICTER-than-default thresholds only (N>=25, similarity<=8pp, divergence>=1.0 log2; "
          f"{len(stricter)} combinations) ===")
    for col in case_cols:
        pct_persist = round(100 * stricter[col].mean(), 1)
        print(f"{col}: flagged in {stricter[col].sum()}/{len(stricter)} stricter combinations ({pct_persist}%)")


if __name__ == "__main__":
    main()
