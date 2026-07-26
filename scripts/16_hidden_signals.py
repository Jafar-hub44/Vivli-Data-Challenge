"""
16_hidden_signals.py
------------------------
STAGE 3 - PART 3: Hidden surveillance signals

Mines both the country-comparison table (t3) and the temporal-trend table
(t4) for the strictest version of the "hidden signal" pattern requested:
cases where MIC50, MIC90, AND breakpoint %resistant are all effectively
unchanged/equal between two strata, yet other quantitative metrics
(geometric mean, skewness, IQR) show a material difference. These are
compiled into a ranked case-study table and visualized for the strongest
examples.
"""

import sys
from pathlib import Path

import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import set_publication_style, PALETTE  # noqa: E402

TABLES_DIR = Path(__file__).resolve().parents[1] / "tables"
FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "cleaned" / "soar_stage1_standardized.csv"

PCTR_CLOSE_THRESHOLD_PP = 5.0
GMEAN_NOTABLE_LOG2 = 0.35   # a quarter-to-third of a dilution step: small but consistent


def mine_country_hidden_signals():
    df = pd.read_csv(TABLES_DIR / "t3_country_pairs_similar_pctR_divergent_distribution.csv")
    frozen = df[
        (df["MIC50_A"] == df["MIC50_B"]) &
        (df["MIC90_A"] == df["MIC90_B"]) &
        (df["%R difference (pp)"] <= PCTR_CLOSE_THRESHOLD_PP) &
        (df["GeoMean gap (log2 steps)"] >= GMEAN_NOTABLE_LOG2)
    ].copy()
    frozen["Source"] = "country"
    frozen = frozen.sort_values("GeoMean gap (log2 steps)", ascending=False)
    return frozen


def mine_temporal_hidden_signals():
    df = pd.read_csv(TABLES_DIR / "t4_temporal_trends.csv")
    frozen = df[
        (df["MIC50 constant across years?"]) &
        (df["MIC90 constant across years?"]) &
        (df["%R range (pp)"] <= PCTR_CLOSE_THRESHOLD_PP) &
        (df["GeoMean log2 range"] >= GMEAN_NOTABLE_LOG2)
    ].copy()
    frozen["Source"] = "temporal"
    frozen = frozen.sort_values("GeoMean log2 range", ascending=False)
    return frozen


def plot_hidden_signal_case(df_raw, organism, ab, ab_name, country_a, country_b, bp, mic50, mic90, rank):
    sub = df_raw[(df_raw["Organism"] == organism) & (df_raw["Country"].isin([country_a, country_b]))]
    val_col, cen_col = f"{ab}_value", f"{ab}_censor"
    sub = sub[sub[cen_col].isin(["exact", "left", "right"])]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    all_vals = sorted(sub[val_col].unique())
    for ax, country in zip(axes, [country_a, country_b]):
        vals = sub[sub["Country"] == country][val_col]
        counts = vals.value_counts().reindex(all_vals, fill_value=0)
        pct = 100 * counts / counts.sum()
        colors = [PALETTE[4] if (bp is not None and v > bp) else PALETTE[0] for v in all_vals]
        ax.bar(range(len(all_vals)), pct.values, color=colors)
        ax.set_xticks(range(len(all_vals)))
        ax.set_xticklabels([str(v) for v in all_vals], rotation=90, fontsize=7)
        if bp is not None:
            ax.axvline(np.searchsorted(all_vals, bp) - 0.5, color="black", linestyle="--", linewidth=1)
        ax.set_ylabel("% of isolates")
        ax.set_xlabel("MIC (mg/L)")
        ax.set_title(f"{country} (n={int(counts.sum())})")
    fig.suptitle(f"{organism} / {ab_name} ({ab}) — HIDDEN SIGNAL CASE STUDY\n"
                 f"MIC50={mic50}, MIC90={mic90} identical in both countries; %R nearly identical; "
                 f"geometric mean differs materially", y=1.05)
    fig.tight_layout()
    fname = FIGURES_DIR / f"fig3_hidden_signal_case_{rank}_{organism.split()[0]}_{ab}.png"
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    return fname


def main():
    set_publication_style()
    country_hidden = mine_country_hidden_signals()
    temporal_hidden = mine_temporal_hidden_signals()

    country_out = TABLES_DIR / "t5_hidden_signals_country.csv"
    temporal_out = TABLES_DIR / "t6_hidden_signals_temporal.csv"
    country_hidden.to_csv(country_out, index=False)
    temporal_hidden.to_csv(temporal_out, index=False)

    print(f"Country-domain hidden signals (MIC50=MIC90 identical, %R within {PCTR_CLOSE_THRESHOLD_PP}pp, "
          f"GeoMean gap >= {GMEAN_NOTABLE_LOG2} log2): {len(country_hidden)} found")
    print(f"Saved: {country_out}")
    with pd.option_context("display.width", 220):
        print(country_hidden[["Organism", "Antibiotic", "Country A", "%R_A", "MIC50_A", "MIC90_A", "GeoMean_A",
                               "Country B", "%R_B", "GeoMean_B", "GeoMean gap (log2 steps)", "Min N"]].head(10).to_string(index=False))

    print(f"\nTemporal-domain hidden signals: {len(temporal_hidden)} found")
    print(f"Saved: {temporal_out}")
    if len(temporal_hidden):
        with pd.option_context("display.width", 200):
            print(temporal_hidden[["Organism", "Antibiotic", "N years", "%R range (pp)",
                                    "GeoMean log2 range"]].to_string(index=False))

    # Combined case-study summary for the report
    top_cases = country_hidden.head(3)
    summary_rows = []
    for _, r in top_cases.iterrows():
        summary_rows.append({
            "Case study": f"{r['Organism']} / {r['Antibiotic (name)']} ({r['Antibiotic']}): {r['Country A']} vs {r['Country B']}",
            "MIC50 (both)": r["MIC50_A"], "MIC90 (both)": r["MIC90_A"],
            "%R (A / B)": f"{r['%R_A']:.1f}% / {r['%R_B']:.1f}%",
            "GeoMean (A / B, mg/L)": f"{r['GeoMean_A']:.4f} / {r['GeoMean_B']:.4f}",
            "GeoMean gap (log2 steps)": r["GeoMean gap (log2 steps)"],
            "Min N": r["Min N"],
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = TABLES_DIR / "t7_top_case_studies_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved top case-study summary: {summary_path}")
    print(summary_df.to_string(index=False))

    # --- visualize the top 3 hidden-signal case studies ---
    raw_df = pd.read_csv(CLEANED_DATA_PATH)
    bp_module_path = Path(__file__).resolve().parent / "12_breakpoint_reference.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("bp_ref", bp_module_path)
    bp_ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bp_ref)

    for rank, (_, r) in enumerate(top_cases.iterrows(), start=1):
        bp, _, _ = bp_ref.BREAKPOINTS_S[(r["Organism"], r["Antibiotic"])]
        fpath = plot_hidden_signal_case(raw_df, r["Organism"], r["Antibiotic"], r["Antibiotic (name)"],
                                         r["Country A"], r["Country B"], bp, r["MIC50_A"], r["MIC90_A"], rank)
        print(f"Saved hidden-signal case figure: {fpath}")


if __name__ == "__main__":
    main()
