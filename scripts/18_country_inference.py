"""
18_country_inference.py
---------------------------
STAGE 4 - PART 2: Formal statistical testing of Stage 3 country comparisons

For every headline country-pair finding from Stage 3, applies a battery of
methods, each justified for a specific reason:

  1. Fisher's exact test (%R difference) - exact test appropriate for
     2x2 tables, valid at any sample size (no large-sample approximation),
     which matters here because several strata have small counts of
     resistant isolates. Reported alongside a Newcombe hybrid-score 95%
     CI for the risk difference (does not assume normality of the
     difference).
  2. Permutation test on the geometric-mean difference (10,000
     relabelings) - distribution-free, makes no assumption about the
     shape of the MIC distribution, and is directly appropriate for
     discrete, heavily-tied MIC data. This is the PRIMARY test for
     distributional difference in this report.
  3. Mann-Whitney U (Wilcoxon rank-sum) test on boundary-substituted MIC
     values - a standard nonparametric confirmatory test; reported as a
     secondary check, with the explicit caveat that, like the permutation
     test, it operates on boundary-substituted (not truly interval-
     censored) values.
  4. Bootstrap CI for the difference in Turnbull (interval-censored NPMLE)
     median between the two groups - the one test in this battery that
     respects the true interval-censored nature of the raw MIC readings
     rather than relying on boundary substitution. Only computed where
     both groups' Turnbull medians are identifiable (Stage 2B established
     this can fail under heavy right-censoring).

Raw (unadjusted) p-values are saved here; multiple-comparison correction
is applied afterwards in 20_multiple_comparisons.py using ALL p-values
generated in this script and in 19_temporal_inference.py together, so
that the family of tests being corrected for is defined honestly (every
test actually run in Stage 4), not just the tests that happened to be
significant.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from scipy import stats
from statsmodels.stats.proportion import confint_proportions_2indep

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

# Every headline country-pair finding from Stage 3 (Parts 1 and 3), tested here.
COUNTRY_PAIRS = [
    ("Haemophilus influenzae", "AXO", "Philippines", "Vietnam", "Stage3 Part1 Case1"),
    ("Streptococcus pneumoniae", "AXO", "Argentina", "Tunisia", "Stage3 Part1 Case2"),
    ("Streptococcus pneumoniae", "AMX", "Argentina", "Kenya", "Stage3 Part1 Case3"),
    ("Streptococcus pneumoniae", "AMC", "Argentina", "Kenya", "Stage3 Part1 Case4"),
    ("Haemophilus influenzae", "CLA", "Chile", "Vietnam", "Stage3 Part3 hidden-signal #1"),
    ("Haemophilus influenzae", "AMC", "Tunisia", "Ukraine", "Stage3 Part3 hidden-signal #2"),
    ("Haemophilus influenzae", "POD", "Tunisia", "Ukraine", "Stage3 Part3 hidden-signal #3"),
]


def geometric_mean(values):
    return float(2 ** np.mean(np.log2(values)))


def permutation_test_geomean(a, b, n_perm=N_PERM, rng=None):
    rng = rng or np.random.default_rng(RNG_SEED)
    observed = np.log2(geometric_mean(a)) - np.log2(geometric_mean(b))
    pooled = np.concatenate([a, b])
    n_a = len(a)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        perm_a, perm_b = pooled[:n_a], pooled[n_a:]
        stat = np.log2(geometric_mean(perm_a)) - np.log2(geometric_mean(perm_b))
        if abs(stat) >= abs(observed):
            count += 1
    p_value = (count + 1) / (n_perm + 1)  # add-one correction, standard practice
    return observed, p_value


def turnbull_median(values, censors):
    values = np.asarray(values, dtype=float)
    lower, upper = values.copy(), values.copy()
    left_mask, right_mask = censors == "left", censors == "right"
    lower[left_mask], upper[left_mask] = 1e-6, values[left_mask]
    lower[right_mask], upper[right_mask] = values[right_mask], 1e6
    kmf = KaplanMeierFitter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kmf.fit_interval_censoring(lower, upper)
    md = kmf.median_survival_time_
    lo, hi = float(md["NPMLE_estimate_lower"].iloc[0]), float(md["NPMLE_estimate_upper"].iloc[0])
    if hi >= 1e5:
        return None
    return (lo * hi) ** 0.5 if lo > 0 else hi


def bootstrap_turnbull_median_diff(vals_a, cens_a, vals_b, cens_b, n_boot=200, rng=None):
    """Bootstrap CI for the difference in Turnbull medians (fewer reps than the
    percentile bootstraps above because each replicate requires an EM fit).

    PERFORMANCE NOTE (post-audit revision): reduced from 500 to 200 replicates.
    Profiling showed some bootstrap resamples produce edge-case censoring
    patterns (e.g. a resample that loses a particular dilution level) that
    make the lifelines EM algorithm converge much more slowly than on the
    original sample, making 500 replicates x 2 groups x 7 pairs impractically
    slow in aggregate. 200 replicates still gives 0.5% percentile resolution,
    which is adequate for this secondary/supplementary comparison (the
    primary confirmatory test in this script is the geometric-mean
    permutation test, N_PERM=10,000, unaffected by this change)."""
    rng = rng or np.random.default_rng(RNG_SEED)
    diffs = []
    n_a, n_b = len(vals_a), len(vals_b)
    idx_a, idx_b = np.arange(n_a), np.arange(n_b)
    for _ in range(n_boot):
        ra = rng.choice(idx_a, size=n_a, replace=True)
        rb = rng.choice(idx_b, size=n_b, replace=True)
        ma = turnbull_median(vals_a[ra], cens_a[ra])
        mb = turnbull_median(vals_b[rb], cens_b[rb])
        if ma is not None and mb is not None and ma > 0 and mb > 0:
            diffs.append(np.log2(ma) - np.log2(mb))
    if len(diffs) < n_boot * 0.5:
        return None, None, None  # too many non-identifiable replicates to trust
    diffs = np.array(diffs)
    return np.median(diffs), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CLEANED_DATA_PATH)
    rng = np.random.default_rng(RNG_SEED)

    rows = []
    for organism, ab, country_a, country_b, source in COUNTRY_PAIRS:
        val_col, cen_col = f"{ab}_value", f"{ab}_censor"
        bp, conf, _ = BREAKPOINTS_S[(organism, ab)]

        sub = df[df["Organism"] == organism]
        sub_a = sub[(sub["Country"] == country_a) & sub[cen_col].isin(["exact", "left", "right"])]
        sub_b = sub[(sub["Country"] == country_b) & sub[cen_col].isin(["exact", "left", "right"])]

        vals_a = sub_a[val_col].to_numpy(dtype=float)
        vals_b = sub_b[val_col].to_numpy(dtype=float)
        cens_a = sub_a[cen_col].to_numpy()
        cens_b = sub_b[cen_col].to_numpy()
        n_a, n_b = len(vals_a), len(vals_b)

        # --- 1. Fisher's exact test + Newcombe CI for risk difference ---
        r_a, r_b = int(np.sum(vals_a > bp)), int(np.sum(vals_b > bp))
        table = [[r_a, n_a - r_a], [r_b, n_b - r_b]]
        _, fisher_p = stats.fisher_exact(table)
        rd_lo, rd_hi = confint_proportions_2indep(r_a, n_a, r_b, n_b, method="newcomb")

        # --- 2. Permutation test on geometric mean (PRIMARY) ---
        gmean_diff_log2, perm_p = permutation_test_geomean(vals_a, vals_b, rng=rng)

        # --- 3. Mann-Whitney U (secondary confirmatory) ---
        mw_stat, mw_p = stats.mannwhitneyu(vals_a, vals_b, alternative="two-sided")

        # --- 4. Bootstrap CI for Turnbull median difference ---
        tb_diff, tb_lo, tb_hi = bootstrap_turnbull_median_diff(vals_a, cens_a, vals_b, cens_b, rng=rng)
        print(f"  ...completed {source}", flush=True)

        rows.append({
            "Source": source, "Organism": organism, "Antibiotic": ab,
            "Antibiotic (name)": ANTIBIOTIC_FULL_NAMES[ab],
            "Country A": country_a, "N_A": n_a, "R_A": r_a,
            "Country B": country_b, "N_B": n_b, "R_B": r_b,
            "Breakpoint (mg/L)": bp, "Breakpoint confidence": conf,
            "%R_A": round(100 * r_a / n_a, 2), "%R_B": round(100 * r_b / n_b, 2),
            "Risk difference (pp)": round(100 * (r_a / n_a - r_b / n_b), 2),
            "Risk difference Newcombe 95% CI": f"[{100*rd_lo:.2f}, {100*rd_hi:.2f}]",
            "Fisher exact p-value": round(fisher_p, 5),
            "GeoMean log2 difference (A-B)": round(gmean_diff_log2, 3),
            "Permutation p-value (GeoMean, PRIMARY)": round(perm_p, 5),
            "Mann-Whitney U p-value (secondary)": round(mw_p, 5),
            "Turnbull median log2 diff (bootstrap median)": round(tb_diff, 3) if tb_diff is not None else "not identifiable in >=50% of bootstrap reps",
            "Turnbull median diff 95% CI": (f"[{tb_lo:.3f}, {tb_hi:.3f}]" if tb_diff is not None else "N/A"),
        })

    result = pd.DataFrame(rows)
    out_path = TABLES_DIR / "t2_country_comparison_tests.csv"
    result.to_csv(out_path, index=False)
    print(f"Tested {len(result)} headline country-pair comparisons ({N_PERM} permutations each)")
    print(f"Saved: {out_path}")
    with pd.option_context("display.width", 260):
        print(result[["Source", "Organism", "Antibiotic", "Country A", "Country B",
                       "Risk difference (pp)", "Fisher exact p-value",
                       "GeoMean log2 difference (A-B)", "Permutation p-value (GeoMean, PRIMARY)",
                       "Mann-Whitney U p-value (secondary)"]].to_string(index=False))


if __name__ == "__main__":
    main()
