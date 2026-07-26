"""
11_metric_benchmark.py
--------------------------
STAGE 2B: Benchmarking Study of Quantitative MIC Metrics

For 8 representative Organism x Antibiotic combinations (2 per organism,
spanning the lowest and highest observed censoring level per organism -
selected from stage2a/tables/t2_censoring_by_organism_antibiotic.csv),
compute the full candidate metric panel:

  - MIC50, MIC90 (CLSI cumulative-% convention, boundary substitution)
  - Median MIC (straightforward statistical median, boundary substitution)
  - Geometric mean MIC (boundary substitution AND half-dilution substitution)
  - Turnbull NPMLE median (interval-censored, from Stage 2A machinery)
  - 25th / 75th percentile MIC, Interquartile range (IQR, log2 scale)
  - Distribution width (log2 range spanned by observed dilutions)
  - Skewness and (excess) kurtosis of log2(MIC) (boundary substitution)
  - Modal MIC (most frequently observed raw dilution reading)
  - Breakpoint exceedance (% resistant by a stated CLSI-consistent
    breakpoint, where one is available/applicable for that pair)

Robustness-to-censoring is assessed for the subset of metrics with more
than one candidate computation method (MIC50/median, geometric mean) by
comparing boundary substitution vs. half-dilution substitution vs. Turnbull
NPMLE, exactly as in Stage 2A Part 4.

This script performs NO winner selection - it only produces the comparison
tables and figures used in Part 2 (empirical benchmarking) of the Stage 2B
report.
"""

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import ANTIBIOTIC_FULL_NAMES, set_publication_style, PALETTE  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "cleaned" / "soar_stage1_standardized.csv"
TABLES_DIR = Path(__file__).resolve().parents[1] / "tables"
FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"

# 8 representative combinations, selected as the LOWEST-censored antibiotic,
# and the HIGHEST-censored antibiotic THAT STILL YIELDS AN ANALYZABLE
# (non-degenerate) DISTRIBUTION, observed for each of the 4 organisms.
#
# SELECTION RULE CORRECTION (post-audit): earlier documentation claimed these
# were simply "the lowest- and highest-censored antibiotic" per organism.
# That is true for H. influenzae and S. pneumoniae, but NOT for E. coli and
# K. pneumoniae: for both of those organisms, erythromycin (ERY) has higher
# censoring than AMP/PEN (100.0% and 98.63% vs 56.60% and 79.45%
# respectively) - see stage2a/tables/t2_censoring_by_organism_antibiotic.csv.
# ERY was not used for these two organisms because it is COMPLETELY
# DEGENERATE after boundary substitution: every single E. coli isolate (106/106)
# and every single K. pneumoniae isolate (73/73) share the identical raw
# reading ("> 16"), so the boundary-substituted distribution has exactly one
# unique value, zero variance, and undefined skewness/kurtosis (division by
# zero) - it cannot illustrate metric behavior on a genuine distribution and
# is a poor fit for a benchmarking table built to compare distributional
# shape metrics. AMP and PEN were used instead as the highest-censored
# antibiotic for which the resulting distribution is still analyzable (5 and
# 3 distinct boundary-substituted values respectively). This selection
# rationale, and the full per-organism censoring ranking it is based on, is
# regenerated and saved to tables/t0_combo_selection_rationale.csv by this
# script for independent verification.
COMBOS = [
    ("Escherichia coli", "AMC", "low"),
    ("Escherichia coli", "AMP", "high"),
    ("Klebsiella pneumoniae", "AMC", "low"),
    ("Klebsiella pneumoniae", "PEN", "high"),
    ("Haemophilus influenzae", "DIN", "low"),
    ("Haemophilus influenzae", "AXO", "high"),
    ("Streptococcus pneumoniae", "LEV", "low"),
    ("Streptococcus pneumoniae", "CLA", "high"),
]

# CLSI-consistent susceptible ("S") breakpoints (mg/L) used for the
# breakpoint-exceedance metric below. These are commonly cited, longstanding
# CLSI M100 values used here for METHODOLOGICAL DEMONSTRATION purposes only;
# several have been revised in recent CLSI editions (amoxicillin/clavulanate
# vs. H. influenzae and Enterobacterales was revised in 2022, after this
# 2019-10 dataset was collected, so the legacy/contemporaneous value is used
# here for internal consistency with the data's collection period). These
# should NOT be used for any clinical decision without verifying against the
# current CLSI M100 edition.
BREAKPOINTS_S = {
    ("Escherichia coli", "AMC"): 8.0,       # Enterobacterales, legacy CLSI (pre-2022 revision)
    ("Escherichia coli", "AMP"): 8.0,       # Enterobacterales, longstanding CLSI
    ("Klebsiella pneumoniae", "AMC"): 8.0,  # Enterobacterales, legacy CLSI (pre-2022 revision)
    ("Klebsiella pneumoniae", "PEN"): None, # No CLSI breakpoint: penicillin not indicated for Enterobacterales
    ("Haemophilus influenzae", "DIN"): 1.0, # Cefdinir vs H. influenzae, longstanding CLSI (assumes DIN = cefdinir; see Stage 1/2A code-ambiguity note)
    ("Haemophilus influenzae", "AXO"): 2.0, # Ceftriaxone vs H. influenzae, longstanding CLSI
    ("Streptococcus pneumoniae", "LEV"): 2.0,   # Levofloxacin vs S. pneumoniae, CLSI
    ("Streptococcus pneumoniae", "CLA"): 0.25,  # Clarithromycin vs S. pneumoniae, CLSI
}

SQRT2 = 2 ** 0.5
LOWER_BOUND_EPS = 1e-6
UPPER_BOUND_INF = 1e6


def boundary_substitution(values):
    return np.asarray(values, dtype=float)


def half_dilution_substitution(values, censors):
    out = np.asarray(values, dtype=float).copy()
    left_mask = censors == "left"
    right_mask = censors == "right"
    out[left_mask] = out[left_mask] / SQRT2
    out[right_mask] = out[right_mask] * SQRT2
    return out


def turnbull_median(values, censors):
    values = np.asarray(values, dtype=float)
    lower, upper = values.copy(), values.copy()
    left_mask, right_mask = censors == "left", censors == "right"
    lower[left_mask] = LOWER_BOUND_EPS
    upper[left_mask] = values[left_mask]
    lower[right_mask] = values[right_mask]
    upper[right_mask] = UPPER_BOUND_INF
    kmf = KaplanMeierFitter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kmf.fit_interval_censoring(lower, upper)
    median_df = kmf.median_survival_time_
    lo = float(median_df["NPMLE_estimate_lower"].iloc[0])
    hi = float(median_df["NPMLE_estimate_upper"].iloc[0])
    if hi >= UPPER_BOUND_INF / 10:
        # The median falls at or beyond the largest identifiable point - i.e.
        # more than half the sample is right-censored beyond the highest
        # tested dilution, so even the MEDIAN (not just the mean) is not
        # identifiable without a further parametric assumption. Returning the
        # raw numeric sentinel would be misleading, so this is flagged
        # explicitly (None) and handled downstream.
        return None
    return (lo * hi) ** 0.5 if lo > 0 else hi


def geometric_mean(values):
    return float(2 ** np.mean(np.log2(np.asarray(values, dtype=float))))


def mic_at_cumulative_pct(sorted_values, pct):
    """CLSI-style MIC50/MIC90: the smallest tested/observed concentration at
    which the cumulative percentage of isolates inhibited is >= pct."""
    n = len(sorted_values)
    idx = int(np.ceil(pct / 100 * n)) - 1
    idx = min(max(idx, 0), n - 1)
    return float(sorted_values[idx])


def compute_metrics(organism, ab, tier, df):
    sub = df[df["Organism"] == organism]
    val_col, cen_col = f"{ab}_value", f"{ab}_censor"
    mask = sub[cen_col].isin(["exact", "left", "right"])
    raw_values = sub.loc[mask, val_col].to_numpy(dtype=float)
    censors = sub.loc[mask, cen_col].to_numpy()
    n = len(raw_values)
    pct_censored = round(100 * np.mean(censors != "exact"), 2)

    bnd = boundary_substitution(raw_values)
    half = half_dilution_substitution(raw_values, censors)
    log2_bnd = np.log2(bnd)

    sorted_bnd = np.sort(bnd)
    mic50 = mic_at_cumulative_pct(sorted_bnd, 50)
    mic90 = mic_at_cumulative_pct(sorted_bnd, 90)
    median_stat = float(np.median(bnd))
    p25 = float(np.percentile(bnd, 25))
    p75 = float(np.percentile(bnd, 75))
    iqr_log2 = float(np.log2(p75) - np.log2(p25))
    width_log2 = float(np.log2(bnd.max()) - np.log2(bnd.min()))
    skewness = float(stats.skew(log2_bnd))
    kurtosis = float(stats.kurtosis(log2_bnd))  # excess kurtosis (0 = normal)

    vals, counts = np.unique(raw_values, return_counts=True)
    modal_value = float(vals[np.argmax(counts)])
    modal_pct = round(100 * counts.max() / n, 1)

    gmean_bnd = geometric_mean(bnd)
    gmean_half = geometric_mean(half)
    tb_median = turnbull_median(raw_values, censors)
    tb_median_display = round(tb_median, 4) if tb_median is not None else "Not identifiable (>50% right-censored beyond max tested dilution)"
    mean_estimable = not np.any(censors == "right")

    bp = BREAKPOINTS_S.get((organism, ab))
    if bp is not None:
        pct_exceed = round(100 * np.mean(bnd > bp), 2)
    else:
        pct_exceed = None

    return {
        "Organism": organism,
        "Antibiotic": ab,
        "Antibiotic (name)": ANTIBIOTIC_FULL_NAMES[ab],
        "Censoring tier": tier,
        "N": n,
        "% censored": pct_censored,
        "MIC50 (mg/L)": mic50,
        "MIC90 (mg/L)": mic90,
        "Median MIC - boundary (mg/L)": median_stat,
        "Median MIC - Turnbull NPMLE (mg/L)": tb_median_display,
        "Geometric mean - boundary (mg/L)": round(gmean_bnd, 4),
        "Geometric mean - half-dilution (mg/L)": round(gmean_half, 4),
        "Geometric mean well-defined (Turnbull)?": mean_estimable,
        "P25 (mg/L)": p25,
        "P75 (mg/L)": p75,
        "IQR (log2 dilution steps)": round(iqr_log2, 3),
        "Distribution width (log2 dilution steps)": round(width_log2, 3),
        "Skewness (log2 MIC)": round(skewness, 3),
        "Excess kurtosis (log2 MIC)": round(kurtosis, 3),
        "Modal MIC (mg/L)": modal_value,
        "Modal MIC (% of isolates)": modal_pct,
        "CLSI-consistent S breakpoint (mg/L)": bp,
        "% exceeding breakpoint (non-susceptible, boundary sub.)": pct_exceed,
    }


def build_combo_selection_rationale(df):
    """Regenerate, from source, the full per-organism antibiotic censoring
    ranking and an explicit note on why the highest-censoring choice for
    E. coli and K. pneumoniae is not simply the single most-censored
    antibiotic (ERY) - see COMBOS comment above. This makes the selection
    rule an independently verifiable, regenerated artifact rather than an
    unverified prose claim."""
    from utils import ANTIBIOTIC_VARS
    rows = []
    for organism in df["Organism"].unique():
        sub = df[df["Organism"] == organism]
        for ab in ANTIBIOTIC_VARS:
            cen_col = f"{ab}_censor"
            tested = sub[cen_col].isin(["exact", "left", "right"])
            n_tested = tested.sum()
            if n_tested == 0:
                continue
            pct_censored = round(100 * (sub.loc[tested, cen_col] != "exact").mean(), 2)
            n_unique_boundary = sub.loc[tested, f"{ab}_value"].nunique()
            rows.append({
                "Organism": organism, "Antibiotic": ab, "N tested": int(n_tested),
                "% censored": pct_censored,
                "Distinct boundary-substituted values": int(n_unique_boundary),
                "Degenerate (1 unique value)?": n_unique_boundary == 1,
            })
    rank = pd.DataFrame(rows)
    rank["Rank within organism (1=lowest censored)"] = rank.groupby("Organism")["% censored"].rank(method="first")
    rank = rank.sort_values(["Organism", "% censored"])
    return rank


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    set_publication_style()

    df = pd.read_csv(CLEANED_DATA_PATH)

    rationale = build_combo_selection_rationale(df)
    rationale_path = TABLES_DIR / "t0_combo_selection_rationale.csv"
    rationale.to_csv(rationale_path, index=False)
    print(f"Saved combo-selection rationale (full censoring ranking, all organism x antibiotic pairs): {rationale_path}")
    degenerate = rationale[rationale["Degenerate (1 unique value)?"]]
    print(f"Antibiotic x organism pairs excluded from 'highest-censored' selection due to complete "
          f"degeneracy (1 unique boundary-substituted value): {len(degenerate)}")
    print(degenerate[["Organism", "Antibiotic", "% censored", "N tested"]].to_string(index=False))
    print()

    rows = [compute_metrics(o, a, t, df) for o, a, t in COMBOS]
    res = pd.DataFrame(rows)
    out_path = TABLES_DIR / "t1_metric_benchmark_full.csv"
    res.to_csv(out_path, index=False)

    print("=== Stage 2B metric benchmark (8 representative Organism x Antibiotic combinations) ===")
    with pd.option_context("display.width", 260):
        print(res[["Organism", "Antibiotic", "Censoring tier", "N", "% censored",
                    "MIC50 (mg/L)", "MIC90 (mg/L)", "Median MIC - boundary (mg/L)",
                    "Median MIC - Turnbull NPMLE (mg/L)",
                    "Geometric mean - boundary (mg/L)", "Geometric mean - half-dilution (mg/L)",
                    "IQR (log2 dilution steps)", "Skewness (log2 MIC)", "Excess kurtosis (log2 MIC)",
                    "Modal MIC (mg/L)", "% exceeding breakpoint (non-susceptible, boundary sub.)"]].to_string(index=False))
    print(f"\nSaved: {out_path}")

    # --- Robustness-to-censoring table: how much do median-type metrics move? ---
    robustness_rows = []
    for r in rows:
        tb_val = r["Median MIC - Turnbull NPMLE (mg/L)"]
        if isinstance(tb_val, str):
            med_log2_range = None  # Turnbull median not identifiable - not comparable
        else:
            med_vals = [r["Median MIC - boundary (mg/L)"], tb_val]
            med_log2_range = round(float(np.log2(max(med_vals)) - np.log2(min(med_vals))), 3)
        gmean_vals = [r["Geometric mean - boundary (mg/L)"], r["Geometric mean - half-dilution (mg/L)"]]
        gmean_log2_range = float(np.log2(max(gmean_vals)) - np.log2(min(gmean_vals)))
        robustness_rows.append({
            "Organism": r["Organism"], "Antibiotic": r["Antibiotic"],
            "% censored": r["% censored"],
            "Median: log2-fold range across methods (boundary vs Turnbull)": med_log2_range,
            "Median Turnbull identifiable?": not isinstance(tb_val, str),
            "Geometric mean: log2-fold range across methods (boundary vs half-dilution)": round(gmean_log2_range, 3),
        })
    robustness = pd.DataFrame(robustness_rows).sort_values("% censored")
    robustness_path = TABLES_DIR / "t2_metric_robustness_to_censoring.csv"
    robustness.to_csv(robustness_path, index=False)
    print("\n=== Robustness to censoring: how much median/geometric-mean move across methods ===")
    print(robustness.to_string(index=False))
    print(f"Saved: {robustness_path}")

    # --- Figure 1: multi-metric panel per combination (small multiples) -----
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, (organism, ab, tier) in zip(axes.flat, COMBOS):
        r = res[(res.Organism == organism) & (res.Antibiotic == ab)].iloc[0]
        metrics = ["MIC50 (mg/L)", "MIC90 (mg/L)", "Median MIC - boundary (mg/L)",
                   "Geometric mean - boundary (mg/L)", "Modal MIC (mg/L)"]
        vals = [r[m] for m in metrics]
        ax.bar(range(len(metrics)), vals, color=PALETTE[:len(metrics)])
        ax.set_yscale("log", base=2)
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(["MIC50", "MIC90", "Median", "GeoMean", "Mode"], rotation=45, ha="right", fontsize=8)
        short_org = organism.split()[0][0] + ". " + organism.split()[1]
        ax.set_title(f"{short_org} / {ab}\n({tier} censoring, {r['% censored']}%)", fontsize=9)
    fig.suptitle("Central-Tendency Metrics Across 8 Representative Organism x Antibiotic Combinations", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_metric_panel_small_multiples.png", bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: skewness/kurtosis scatter, colored by % censored ---------
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(res["Skewness (log2 MIC)"], res["Excess kurtosis (log2 MIC)"],
                     c=res["% censored"], cmap="viridis", s=120, edgecolor="black")
    for _, r in res.iterrows():
        ax.annotate(f"{r['Organism'].split()[0][0]}.{r['Organism'].split()[1][:4]}/{r['Antibiotic']}",
                    (r["Skewness (log2 MIC)"], r["Excess kurtosis (log2 MIC)"]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.7, linestyle="--")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("% censored")
    ax.set_xlabel("Skewness (log2 MIC)")
    ax.set_ylabel("Excess kurtosis (log2 MIC)")
    ax.set_title("Distribution Shape Metrics vs. Censoring Level")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_skewness_kurtosis_scatter.png")
    plt.close(fig)

    # --- Figure 3: robustness bars (log2-fold range across methods) ---------
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(robustness))
    width = 0.35
    labels = [f"{o.split()[0][0]}.{o.split()[1][:4]}\n{a}" for o, a in zip(robustness.Organism, robustness.Antibiotic)]
    ax.bar(x - width / 2, robustness["Median: log2-fold range across methods (boundary vs Turnbull)"],
           width, label="Median (boundary vs Turnbull)", color=PALETTE[0])
    ax.bar(x + width / 2, robustness["Geometric mean: log2-fold range across methods (boundary vs half-dilution)"],
           width, label="Geometric mean (boundary vs half-dilution)", color=PALETTE[4])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Log2-fold range across methods")
    ax.set_title("Robustness of Central-Tendency Metrics to Censoring-Handling Method Choice")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_metric_robustness.png")
    plt.close(fig)

    print(f"\nSaved figures to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
