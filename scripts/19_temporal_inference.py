"""
19_temporal_inference.py
----------------------------
STAGE 4 - PART 3: Formal statistical testing of Stage 3 temporal trends

Replaces Stage 3's descriptive year-by-year comparison with two formal
trend tests, applied to the three flagged Haemophilus influenzae
antibiotics (AXO, CDN, AMC) across 2016-2018 (the only organism-antibiotic
pairs with >=3 well-populated years, per Stage 3's own eligibility rule):

  1. Cochran-Armitage trend test for %resistant across ordered years -
     the standard test for a linear trend in a binomial proportion across
     ordered categories; appropriate because year is a naturally ordered
     (not merely categorical) exposure and this is the standard test used
     for exactly this situation in the epidemiological literature.
  2. Permutation test for the slope of log2(MIC) regressed on year
     (isolate-level, boundary substitution) - distribution-free trend
     test for the geometric-mean trend, avoiding the normality assumption
     of OLS significance testing on discrete, tied, boundary-substituted
     data. The observed slope's significance is assessed by permuting the
     year labels among isolates and recomputing the slope 10,000 times.

Both tests are run on the SAME isolate-level data feeding Stage 3's
descriptive year-by-year tables, so this stage tests Stage 3's findings
rather than introducing a new dataset or new hypothesis.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import ANTIBIOTIC_FULL_NAMES  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "cleaned" / "soar_stage1_standardized.csv"
TABLES_DIR = Path(__file__).resolve().parents[1] / "tables"

N_PERM = 10000
RNG_SEED = 20260724

import importlib.util
_bp_spec = importlib.util.spec_from_file_location(
    "bp_ref", Path(__file__).resolve().parent / "12_breakpoint_reference.py"
)
bp_ref = importlib.util.module_from_spec(_bp_spec)
_bp_spec.loader.exec_module(bp_ref)
BREAKPOINTS_S = bp_ref.BREAKPOINTS_S

TEMPORAL_PAIRS = [
    ("Haemophilus influenzae", "AXO"),
    ("Haemophilus influenzae", "CDN"),
    ("Haemophilus influenzae", "AMC"),
]
YEARS = [2016, 2017, 2018]


def cochran_armitage_trend_test(counts_r, counts_n, scores):
    """Cochran-Armitage trend test. counts_r = resistant counts per group,
    counts_n = total counts per group, scores = ordered numeric score per
    group (here, the calendar year itself). Returns (z_statistic, p_value)."""
    counts_r = np.asarray(counts_r, dtype=float)
    counts_n = np.asarray(counts_n, dtype=float)
    scores = np.asarray(scores, dtype=float)
    N = counts_n.sum()
    R = counts_r.sum()
    p_bar = R / N
    score_bar = np.sum(counts_n * scores) / N
    numerator = np.sum(counts_r * (scores - score_bar))
    denom = np.sqrt(p_bar * (1 - p_bar) * np.sum(counts_n * (scores - score_bar) ** 2))
    if denom == 0:
        return np.nan, np.nan
    z = numerator / denom
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p_value


def permutation_trend_test_slope(log2_values, years, n_perm=N_PERM, rng=None):
    rng = rng or np.random.default_rng(RNG_SEED)
    years = np.asarray(years, dtype=float)

    def slope(y_vals, x_vals):
        return np.polyfit(x_vals, y_vals, 1)[0]

    observed_slope = slope(log2_values, years)
    count = 0
    years_shuffled = years.copy()
    for _ in range(n_perm):
        rng.shuffle(years_shuffled)
        s = slope(log2_values, years_shuffled)
        if abs(s) >= abs(observed_slope):
            count += 1
    p_value = (count + 1) / (n_perm + 1)
    return observed_slope, p_value


def bootstrap_slope_ci(log2_values, years, n_boot=2000, rng=None, alpha=0.05):
    rng = rng or np.random.default_rng(RNG_SEED)
    n = len(log2_values)
    idx = np.arange(n)
    slopes = np.empty(n_boot)
    for i in range(n_boot):
        r = rng.choice(idx, size=n, replace=True)
        slopes[i] = np.polyfit(years[r], log2_values[r], 1)[0]
    return np.percentile(slopes, 100 * alpha / 2), np.percentile(slopes, 100 * (1 - alpha / 2))


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CLEANED_DATA_PATH)
    rng = np.random.default_rng(RNG_SEED)

    rows = []
    for organism, ab in TEMPORAL_PAIRS:
        val_col, cen_col = f"{ab}_value", f"{ab}_censor"
        bp, conf, _ = BREAKPOINTS_S[(organism, ab)]
        sub = df[(df["Organism"] == organism) & df[cen_col].isin(["exact", "left", "right"])
                 & df["Collection_Year"].isin(YEARS)]

        counts_r, counts_n = [], []
        for yr in YEARS:
            g = sub[sub["Collection_Year"] == yr]
            v = g[val_col].to_numpy(dtype=float)
            counts_r.append(int(np.sum(v > bp)))
            counts_n.append(len(v))

        ca_z, ca_p = cochran_armitage_trend_test(counts_r, counts_n, YEARS)

        all_values = sub[val_col].to_numpy(dtype=float)
        all_years = sub["Collection_Year"].to_numpy(dtype=float)
        log2_values = np.log2(all_values)
        slope_obs, perm_p = permutation_trend_test_slope(log2_values, all_years, rng=rng)
        slope_ci_lo, slope_ci_hi = bootstrap_slope_ci(log2_values, all_years, rng=rng)

        rows.append({
            "Organism": organism, "Antibiotic": ab, "Antibiotic (name)": ANTIBIOTIC_FULL_NAMES[ab],
            "Years tested": ",".join(str(y) for y in YEARS),
            "N per year": ",".join(str(n) for n in counts_n),
            "R per year (n > breakpoint)": ",".join(str(r) for r in counts_r),
            "Breakpoint (mg/L)": bp, "Breakpoint confidence": conf,
            "Cochran-Armitage z": round(ca_z, 3), "Cochran-Armitage p-value (%R trend)": round(ca_p, 5),
            "GeoMean trend slope (log2 MIC/year)": round(slope_obs, 4),
            "Slope bootstrap 95% CI": f"[{slope_ci_lo:.4f}, {slope_ci_hi:.4f}]",
            "Permutation p-value (GeoMean trend, PRIMARY)": round(perm_p, 5),
            "Fold-change per year (2^slope)": round(2 ** slope_obs, 3),
        })

    result = pd.DataFrame(rows)
    out_path = TABLES_DIR / "t3_temporal_trend_tests.csv"
    result.to_csv(out_path, index=False)
    print(f"Tested {len(result)} temporal trends ({N_PERM} permutations each)")
    print(f"Saved: {out_path}")
    with pd.option_context("display.width", 220):
        print(result[["Organism", "Antibiotic", "Cochran-Armitage p-value (%R trend)",
                       "GeoMean trend slope (log2 MIC/year)", "Slope bootstrap 95% CI",
                       "Permutation p-value (GeoMean trend, PRIMARY)", "Fold-change per year (2^slope)"]].to_string(index=False))


if __name__ == "__main__":
    main()
