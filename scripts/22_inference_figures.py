"""
22_inference_figures.py
---------------------------
STAGE 4 - Publication-quality figures with confidence intervals.

Produces:
  fig1_forest_geomean_country.png - forest plot of geometric-mean MIC with
    bootstrap 95% CIs for every headline country stratum, grouped by
    comparison pair, making non-overlap (or overlap) visually inspectable.
  fig2_forest_temporal_trend.png - forest plot of the geometric-mean trend
    slope (log2 MIC/year) with bootstrap 95% CIs for the three temporal
    comparisons, with a reference line at slope=0 (no trend).
  fig3_pvalue_comparison.png - dot plot comparing raw vs. Bonferroni/Holm/
    BH-FDR-adjusted p-values for every test, on a -log10 scale, with the
    alpha=0.05 threshold marked.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import set_publication_style, PALETTE  # noqa: E402

TABLES_DIR = Path(__file__).resolve().parents[1] / "tables"
FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"


def fig_forest_country():
    ci = pd.read_csv(TABLES_DIR / "t1_confidence_intervals.csv")
    ci_country = ci[ci["Stratum type"] == "country"].copy()
    ci_country["label"] = ci_country["Organism"].str.split().str[0].str[0] + ". " + \
        ci_country["Organism"].str.split().str[1] + " / " + ci_country["Antibiotic"] + " — " + ci_country["Stratum"]
    ci_country = ci_country.sort_values(["Antibiotic", "Organism"])

    fig, ax = plt.subplots(figsize=(9, 8))
    y = np.arange(len(ci_country))
    x = ci_country["GeoMean (point estimate, mg/L)"]
    lo = ci_country["GeoMean bootstrap 95% CI low"]
    hi = ci_country["GeoMean bootstrap 95% CI high"]
    err_lo = x - lo
    err_hi = hi - x
    ax.errorbar(x, y, xerr=[err_lo, err_hi], fmt="o", color=PALETTE[0], ecolor=PALETTE[1],
                elinewidth=2, capsize=3, markersize=6)
    ax.set_yticks(y)
    ax.set_yticklabels(ci_country["label"], fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Geometric mean MIC (mg/L, log2 scale) with bootstrap 95% CI")
    ax.set_title("Forest Plot: Geometric Mean MIC by Country Stratum\n(Stage 3 headline comparisons, Stage 4 bootstrap 95% CIs)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_forest_geomean_country.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig1_forest_geomean_country.png")


def fig_forest_temporal():
    t = pd.read_csv(TABLES_DIR / "t3_temporal_trend_tests.csv")
    t["ci_lo"] = t["Slope bootstrap 95% CI"].str.extract(r"\[(-?[\d.]+),")[0].astype(float)
    t["ci_hi"] = t["Slope bootstrap 95% CI"].str.extract(r", (-?[\d.]+)\]")[0].astype(float)

    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(t))
    x = t["GeoMean trend slope (log2 MIC/year)"]
    err_lo = x - t["ci_lo"]
    err_hi = t["ci_hi"] - x
    ax.errorbar(x, y, xerr=[err_lo, err_hi], fmt="s", color=PALETTE[4], ecolor=PALETTE[1],
                elinewidth=2, capsize=4, markersize=8)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"H. influenzae / {a}" for a in t["Antibiotic"]])
    ax.set_xlabel("GeoMean MIC trend slope (log2 dilution steps / year), bootstrap 95% CI")
    ax.set_title("Forest Plot: Temporal Trend Slope, 2016-2018\n(dashed line = no trend)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_forest_temporal_trend.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig2_forest_temporal_trend.png")


def fig_pvalue_comparison():
    mc = pd.read_csv(TABLES_DIR / "t4_multiple_comparisons.csv")
    mc = mc.sort_values("p_value").reset_index(drop=True)
    y = np.arange(len(mc))

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.scatter(-np.log10(mc["p_value"].clip(lower=1e-6)), y, label="Raw p-value", color=PALETTE[0], s=40, zorder=3)
    ax.scatter(-np.log10(mc["Bonferroni adj. p"].clip(lower=1e-6)), y, label="Bonferroni adj.", color=PALETTE[4], s=25, marker="x")
    ax.scatter(-np.log10(mc["BH-FDR adj. p (q-value)"].clip(lower=1e-6)), y, label="BH-FDR adj. (q-value)", color=PALETTE[2], s=25, marker="^")
    ax.axvline(-np.log10(0.05), color="black", linestyle="--", linewidth=1, label="alpha = 0.05")
    labels = [f"{row['Antibiotic']} {row['Comparison']} [{row['Test']}]" for _, row in mc.iterrows()]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlabel("-log10(p-value)")
    ax.set_title("Raw vs. Corrected p-values Across All 26 Stage 4 Tests")
    ax.legend(loc="lower right", fontsize=8)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_pvalue_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig3_pvalue_comparison.png")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    set_publication_style()
    fig_forest_country()
    fig_forest_temporal()
    fig_pvalue_comparison()


if __name__ == "__main__":
    main()
