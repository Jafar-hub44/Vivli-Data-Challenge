"""
10_censoring_method_simulation.py
-------------------------------------
STAGE 2A - PART 4: Simulation and Sensitivity Assessment

Selects representative Organism x Antibiotic combinations spanning a wide
range of observed censoring levels (from <1% to >80% censored) and applies
three candidate censoring-handling methods to each, comparing the
resulting median MIC and geometric mean MIC. The purpose is solely to
assess whether the CHOICE of censoring method materially changes these
summary statistics - not to draw resistance conclusions.

Methods compared
-----------------
1. Boundary-value substitution: left-censored "<=X" -> X; right-censored
   ">X" -> X (the reported boundary value itself, unchanged).
2. Half-dilution substitution: left-censored "<=X" -> X / sqrt(2) (half a
   two-fold dilution step below X on the log2 scale); right-censored ">X"
   -> X * sqrt(2) (half a step above X).
3. Nonparametric MLE (Turnbull estimator) for interval-censored data,
   implemented via lifelines' `fit_interval_censoring`. Exact readings are
   treated as points [X, X]; left-censored as (~0, X]; right-censored as
   (X, ~infinity). The median is read directly from the fitted survival
   curve. NOTE: the mean cannot be estimated without further assumptions
   whenever the largest value in the sample is right-censored (the
   estimated survival function does not reach 0) - this is a well known
   limitation of nonparametric censored-data estimators and is reported
   explicitly for each combination rather than silently approximated.

All three methods are applied to the same underlying isolate-level MIC
observations already parsed and validated in Stage 1
(data/cleaned/soar_stage1_standardized.csv).
"""

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import ANTIBIOTIC_FULL_NAMES, set_publication_style, PALETTE  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "cleaned" / "soar_stage1_standardized.csv"
TABLES_DIR = Path(__file__).resolve().parents[1] / "tables"
FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"

# Representative Organism x Antibiotic combinations spanning the observed
# censoring range (selected from stage2a/tables/t2_censoring_by_organism_antibiotic.csv),
# restricted to combinations with n > 1,000 so results are not driven by
# small-sample noise.
COMBOS = [
    ("Streptococcus pneumoniae", "LEV", "Very low (~0%)"),
    ("Haemophilus influenzae", "AMC", "Low (~7%)"),
    ("Streptococcus pneumoniae", "AMC", "Moderate (~19%)"),
    ("Streptococcus pneumoniae", "AMP", "Moderate-high (~30%)"),
    ("Streptococcus pneumoniae", "AZM", "High (~52%)"),
    ("Haemophilus influenzae", "AXO", "Very high (~84%)"),
]

SQRT2 = 2 ** 0.5
LOWER_BOUND_EPS = 1e-6   # theoretical floor for left-censored intervals (well below smallest dilution)
UPPER_BOUND_INF = 1e6    # numeric "infinity" for right-censored intervals


def boundary_substitution(values, censors):
    return np.asarray(values, dtype=float)  # already the boundary value by construction


def half_dilution_substitution(values, censors):
    out = np.asarray(values, dtype=float).copy()
    left_mask = censors == "left"
    right_mask = censors == "right"
    out[left_mask] = out[left_mask] / SQRT2
    out[right_mask] = out[right_mask] * SQRT2
    return out


def turnbull_npmle(values, censors):
    """Fit the Turnbull NPMLE via lifelines and return (median, mean_estimable, restricted_mean_log2)."""
    values = np.asarray(values, dtype=float)
    lower = values.copy()
    upper = values.copy()
    left_mask = censors == "left"
    right_mask = censors == "right"
    lower[left_mask] = LOWER_BOUND_EPS
    upper[left_mask] = values[left_mask]
    lower[right_mask] = values[right_mask]
    upper[right_mask] = UPPER_BOUND_INF
    right_censored_flags = right_mask

    kmf = KaplanMeierFitter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kmf.fit_interval_censoring(lower, upper)

    sf = kmf.survival_function_
    median_df = kmf.median_survival_time_
    median_lower = float(median_df["NPMLE_estimate_lower"].iloc[0])
    median_upper = float(median_df["NPMLE_estimate_upper"].iloc[0])
    median_mid = (median_lower * median_upper) ** 0.5 if median_lower > 0 else median_upper  # geometric midpoint (log2 scale)
    # The mean of a right-censored (or right-censored-containing) sample is
    # only well-defined without further assumptions if no observation's true
    # value could lie beyond the largest identifiable point - i.e. there are
    # no right-censored observations at all. We check this directly from the
    # input censoring pattern (theoretically correct) rather than from the
    # fitted survival curve's tail value, which can retain a small nonzero
    # residual purely from EM numerical convergence tolerance even when no
    # right-censored observations are present.
    reaches_zero = not np.any(right_censored_flags)
    return median_mid, median_lower, median_upper, reaches_zero, sf


def geometric_mean_log2(values):
    log2vals = np.log2(np.asarray(values, dtype=float))
    return float(2 ** np.mean(log2vals))


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    set_publication_style()

    df = pd.read_csv(CLEANED_DATA_PATH)

    results = []
    turnbull_curves = {}

    for organism, ab, label in COMBOS:
        sub = df[df["Organism"] == organism]
        val_col, cen_col = f"{ab}_value", f"{ab}_censor"
        mask = sub[cen_col].isin(["exact", "left", "right"])
        values = sub.loc[mask, val_col].to_numpy(dtype=float)
        censors = sub.loc[mask, cen_col].to_numpy()
        n = len(values)
        pct_censored = round(100 * np.mean(censors != "exact"), 2)

        boundary_vals = boundary_substitution(values, censors)
        half_vals = half_dilution_substitution(values, censors)

        median_boundary = float(np.median(boundary_vals))
        gmean_boundary = geometric_mean_log2(boundary_vals)

        median_half = float(np.median(half_vals))
        gmean_half = geometric_mean_log2(half_vals)

        median_turnbull, median_turnbull_lower, median_turnbull_upper, mean_estimable, sf = turnbull_npmle(values, censors)
        turnbull_curves[(organism, ab)] = sf

        results.append({
            "Organism": organism,
            "Antibiotic": ab,
            "Antibiotic (name)": ANTIBIOTIC_FULL_NAMES[ab],
            "Censoring level": label,
            "N": n,
            "% censored": pct_censored,
            "Median MIC - Boundary substitution": median_boundary,
            "Median MIC - Half-dilution substitution": median_half,
            "Median MIC - Turnbull NPMLE (midpoint)": round(median_turnbull, 4),
            "Median MIC - Turnbull NPMLE (lower bound)": median_turnbull_lower,
            "Median MIC - Turnbull NPMLE (upper bound)": median_turnbull_upper,
            "Geometric mean MIC - Boundary substitution": round(gmean_boundary, 4),
            "Geometric mean MIC - Half-dilution substitution": round(gmean_half, 4),
            "Geometric mean well-defined (Turnbull, right tail)": mean_estimable,
            "Boundary vs Half-dilution, geometric mean ratio": round(gmean_boundary / gmean_half, 3),
            "Boundary vs Half-dilution, log2-fold difference": round(
                abs(np.log2(gmean_boundary) - np.log2(gmean_half)), 3
            ),
        })

    res_df = pd.DataFrame(results)
    out_path = TABLES_DIR / "t5_censoring_method_comparison.csv"
    res_df.to_csv(out_path, index=False)

    print("=== Censoring method comparison across representative Organism x Antibiotic combinations ===")
    with pd.option_context("display.width", 240):
        print(res_df[[
            "Organism", "Antibiotic", "Censoring level", "N", "% censored",
            "Median MIC - Boundary substitution", "Median MIC - Half-dilution substitution",
            "Median MIC - Turnbull NPMLE (midpoint)",
            "Geometric mean MIC - Boundary substitution", "Geometric mean MIC - Half-dilution substitution",
            "Boundary vs Half-dilution, log2-fold difference",
        ]].to_string(index=False))
    print(f"\nSaved: {out_path}")

    # --- Figure: geometric mean MIC by method, per combination --------------
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(res_df))
    width = 0.35
    ax.bar(x - width / 2, res_df["Geometric mean MIC - Boundary substitution"], width,
           label="Boundary substitution", color=PALETTE[0])
    ax.bar(x + width / 2, res_df["Geometric mean MIC - Half-dilution substitution"], width,
           label="Half-dilution substitution", color=PALETTE[4])
    ax.set_yscale("log", base=2)
    labels = [f"{o.split()[0][0]}. {o.split()[1]}\n{a}\n({c.split()[0]})"
              for o, a, c in zip(res_df["Organism"], res_df["Antibiotic"], res_df["Censoring level"])]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Geometric mean MIC (mg/L, log2 scale)")
    ax.set_title("Sensitivity of Geometric Mean MIC to Censoring-Handling Method\n(Stage 2A simulation, representative Organism x Antibiotic combinations)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_method_sensitivity_geomean.png")
    plt.close(fig)

    # --- Figure: Turnbull NPMLE survival curves for the two most-censored combos ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    high_combos = [("Streptococcus pneumoniae", "AZM"), ("Haemophilus influenzae", "AXO")]
    for ax, (organism, ab) in zip(axes, high_combos):
        sf = turnbull_curves[(organism, ab)]
        ax.step(sf.index, sf["NPMLE_estimate_lower"], where="post", color=PALETTE[1], label="Lower NPMLE bound")
        ax.step(sf.index, sf["NPMLE_estimate_upper"], where="post", color=PALETTE[4], label="Upper NPMLE bound")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("MIC (mg/L, log2 scale)")
        ax.set_ylabel("Survival probability (P[true MIC > x])")
        ax.set_title(f"{organism}\n{ab} ({ANTIBIOTIC_FULL_NAMES[ab]})")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Turnbull NPMLE Survival Curves - Heavily Censored Combinations", y=1.03)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig6_turnbull_survival_curves.png", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figures to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
