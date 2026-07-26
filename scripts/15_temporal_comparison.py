"""
15_temporal_comparison.py
-----------------------------
STAGE 3 - PART 2: Temporal comparisons (2015-2018)

For every (Organism, Antibiotic) pair with >= 2 qualifying years
(N >= 15/year, from 13_stratified_metrics.py), evaluates whether the MIC
distribution changes across 2015-2018 even when breakpoint-based %resistant
stays stable, and flags evidence of MIC creep/drift or distributional shift.

"Stable resistance, shifting distribution" is operationalized as:
  - %R range across available years <= PCTR_STABLE_THRESHOLD_PP percentage points
  - AND geometric-mean log2 range across years >= GMEAN_SHIFT_THRESHOLD_LOG2

Produces a ranked summary table and year-by-year distribution figures for
the top-ranked case studies (only Organism x Antibiotic pairs with >=3
qualifying years are eligible for "trend" figures, to show a real trajectory
rather than a single before/after comparison).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import set_publication_style, PALETTE  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "cleaned" / "soar_stage1_standardized.csv"
TABLES_DIR = Path(__file__).resolve().parents[1] / "tables"
FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"

PCTR_STABLE_THRESHOLD_PP = 8.0
GMEAN_SHIFT_THRESHOLD_LOG2 = 0.5
MIN_YEARS = 2


def mine_temporal_trends(year_df):
    rows = []
    for (organism, ab), g in year_df.groupby(["Organism", "Antibiotic"]):
        g = g.sort_values("Year")
        if len(g) < MIN_YEARS or g["PctR"].isna().any():
            continue
        pctr_range = g["PctR"].max() - g["PctR"].min()
        gmean_log2 = np.log2(g["GeoMean"])
        gmean_range = gmean_log2.max() - gmean_log2.min()
        gmean_direction = "increasing" if gmean_log2.iloc[-1] > gmean_log2.iloc[0] else "decreasing"
        mic50_constant = g["MIC50"].nunique() == 1
        mic90_constant = g["MIC90"].nunique() == 1
        skew_range = g["Skewness"].max() - g["Skewness"].min()
        iqr_range = g["IQR_log2"].max() - g["IQR_log2"].min()

        rows.append({
            "Organism": organism, "Antibiotic": ab,
            "Antibiotic (name)": g["Antibiotic (name)"].iloc[0],
            "Years": ",".join(str(y) for y in g["Year"]),
            "N per year": ",".join(str(n) for n in g["N"]),
            "%R range (pp)": round(pctr_range, 2),
            "%R by year": ",".join(f"{v:.1f}" for v in g["PctR"]),
            "GeoMean log2 range": round(gmean_range, 3),
            "GeoMean direction (first->last yr)": gmean_direction,
            "GeoMean by year (mg/L)": ",".join(f"{v:.4f}" for v in g["GeoMean"]),
            "MIC50 constant across years?": mic50_constant,
            "MIC50 by year": ",".join(str(v) for v in g["MIC50"]),
            "MIC90 constant across years?": mic90_constant,
            "MIC90 by year": ",".join(str(v) for v in g["MIC90"]),
            "Skewness range": round(skew_range, 3),
            "IQR_log2 range": round(iqr_range, 3),
            "Stable %R + shifting GeoMean?": (pctr_range <= PCTR_STABLE_THRESHOLD_PP
                                               and gmean_range >= GMEAN_SHIFT_THRESHOLD_LOG2),
            "N years": len(g),
        })
    return pd.DataFrame(rows)


def plot_temporal_case(df_raw, organism, ab, ab_name, years, bp):
    sub = df_raw[df_raw["Organism"] == organism]
    val_col, cen_col = f"{ab}_value", f"{ab}_censor"
    sub = sub[sub[cen_col].isin(["exact", "left", "right"])]
    sub = sub[sub["Collection_Year"].isin(years)]

    all_vals = sorted(sub[val_col].unique())
    fig, axes = plt.subplots(1, len(years), figsize=(4.2 * len(years), 4.5), sharey=True)
    if len(years) == 1:
        axes = [axes]
    for ax, yr in zip(axes, years):
        vals = sub[sub["Collection_Year"] == yr][val_col]
        counts = vals.value_counts().reindex(all_vals, fill_value=0)
        pct = 100 * counts / counts.sum()
        colors = [PALETTE[4] if (bp is not None and v > bp) else PALETTE[0] for v in all_vals]
        ax.bar(range(len(all_vals)), pct.values, color=colors)
        ax.set_xticks(range(len(all_vals)))
        ax.set_xticklabels([str(v) for v in all_vals], rotation=90, fontsize=7)
        if bp is not None:
            ax.axvline(np.searchsorted(all_vals, bp) - 0.5, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"{int(yr)} (n={int(counts.sum())})")
        ax.set_xlabel("MIC (mg/L)")
    axes[0].set_ylabel("% of isolates")
    fig.suptitle(f"{organism} / {ab_name} ({ab})\nMIC distribution by year — stable %R, shifting distribution", y=1.03)
    fig.tight_layout()
    fname = FIGURES_DIR / f"fig2_temporal_case_{organism.split()[0]}_{ab}.png"
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    return fname


def main():
    set_publication_style()
    year_df = pd.read_csv(TABLES_DIR / "t2_stratified_metrics_by_year.csv")
    raw_df = pd.read_csv(CLEANED_DATA_PATH)

    trends = mine_temporal_trends(year_df)
    trends = trends.sort_values("GeoMean log2 range", ascending=False).reset_index(drop=True)
    out_path = TABLES_DIR / "t4_temporal_trends.csv"
    trends.to_csv(out_path, index=False)
    print(f"Computed temporal trends for {len(trends)} Organism x Antibiotic pairs (>= {MIN_YEARS} qualifying years)")
    print(f"Saved: {out_path}")

    flagged = trends[trends["Stable %R + shifting GeoMean?"]].sort_values("N years", ascending=False)
    print(f"\n=== Flagged: stable %R (<= {PCTR_STABLE_THRESHOLD_PP}pp range) but shifting GeoMean "
          f"(>= {GMEAN_SHIFT_THRESHOLD_LOG2} log2 steps) ===")
    with pd.option_context("display.width", 200):
        print(flagged[["Organism", "Antibiotic", "N years", "%R range (pp)", "GeoMean log2 range",
                        "GeoMean direction (first->last yr)", "MIC50 constant across years?",
                        "MIC90 constant across years?"]].to_string(index=False))

    bp_module_path = Path(__file__).resolve().parent / "12_breakpoint_reference.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("bp_ref", bp_module_path)
    bp_ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bp_ref)

    top_trend_cases = flagged[flagged["N years"] >= 3].head(3)
    for _, row in top_trend_cases.iterrows():
        years = [int(y) for y in row["Years"].split(",")]
        bp, _, _ = bp_ref.BREAKPOINTS_S[(row["Organism"], row["Antibiotic"])]
        fpath = plot_temporal_case(raw_df, row["Organism"], row["Antibiotic"], row["Antibiotic (name)"], years, bp)
        print(f"Saved temporal case figure: {fpath}")


if __name__ == "__main__":
    main()
