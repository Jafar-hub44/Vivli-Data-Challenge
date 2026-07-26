"""
14_country_comparison.py
----------------------------
STAGE 3 - PART 1: Country comparisons

Systematically scans every (Organism, Antibiotic) pair (with N>=15 per
country, from 13_stratified_metrics.py) for pairs of countries where:
  - breakpoint-based %resistant is SIMILAR (|delta| <= SIMILARITY_THRESHOLD_PP
    percentage points), i.e. conventional surveillance would call them
    "about the same"
  - AND the MIC distribution differs materially, operationalized as a
    geometric-mean gap >= DIVERGENCE_THRESHOLD_LOG2 log2 dilution steps.

CRITERION CORRECTION (post-audit): an earlier version of this script also
flagged a pair if its modal MIC merely differed at all ("OR different modal
MIC values"), even without a material geometric-mean gap. That OR-clause is
almost trivially satisfied by any two countries (modal MIC differing by even
one dilution step, including differences driven by censoring/substitution
artifacts documented in Stage 2B, was enough) and produced a much larger,
less meaningful "qualifying" count (363 at Min N>=25) that did not match
what the geometric-mean-focused downstream analysis (Stage 4) actually
tested or intended. The modal-MIC OR-clause has been removed so this
script's criterion now exactly matches the one used throughout Stage 4
(confirmatory testing, sensitivity grid): geometric-mean divergence is
REQUIRED, not merely sufficient-as-an-alternative-to-modal-difference. This
is the single authoritative mining criterion used everywhere in this
project from this point forward.

Produces a ranked table of all qualifying pairs and publication-quality
comparison figures (grouped bar chart of full MIC distribution) for the
top-ranked case studies.
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

SIMILARITY_THRESHOLD_PP = 8.0      # %R considered "similar" if within this many percentage points
DIVERGENCE_THRESHOLD_LOG2 = 1.0    # geometric-mean gap (log2 dilution steps) considered "materially different"
MIN_N_FOR_CASE_STUDY = 25          # only surface case studies with reasonably powered strata


def mine_country_pairs(country_df):
    rows = []
    for (organism, ab), g in country_df.groupby(["Organism", "Antibiotic"]):
        g = g[g["PctR"].notna()].reset_index(drop=True)
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                r1, r2 = g.iloc[i], g.iloc[j]
                pctr_diff = abs(r1["PctR"] - r2["PctR"])
                gmean_gap = abs(np.log2(r1["GeoMean"]) - np.log2(r2["GeoMean"]))
                modal_diff = r1["Modal_MIC"] != r2["Modal_MIC"]
                if pctr_diff <= SIMILARITY_THRESHOLD_PP and gmean_gap >= DIVERGENCE_THRESHOLD_LOG2:
                    rows.append({
                        "Organism": organism, "Antibiotic": ab,
                        "Antibiotic (name)": r1["Antibiotic (name)"],
                        "Breakpoint confidence": r1["Breakpoint confidence"],
                        "Country A": r1["Country"], "N_A": r1["N"], "%R_A": r1["PctR"],
                        "MIC50_A": r1["MIC50"], "MIC90_A": r1["MIC90"], "GeoMean_A": r1["GeoMean"],
                        "Modal_A": r1["Modal_MIC"], "Skew_A": r1["Skewness"],
                        "Country B": r2["Country"], "N_B": r2["N"], "%R_B": r2["PctR"],
                        "MIC50_B": r2["MIC50"], "MIC90_B": r2["MIC90"], "GeoMean_B": r2["GeoMean"],
                        "Modal_B": r2["Modal_MIC"], "Skew_B": r2["Skewness"],
                        "%R difference (pp)": round(pctr_diff, 2),
                        "GeoMean gap (log2 steps)": round(gmean_gap, 3),
                        "Modal MIC differs?": modal_diff,
                        "Min N": min(r1["N"], r2["N"]),
                    })
    return pd.DataFrame(rows)


def plot_country_case_study(df_raw, organism, ab, ab_name, country_a, country_b, rank, bp):
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
        ax.set_ylabel("% of isolates")
        ax.set_xlabel("MIC (mg/L)")
        if bp is not None:
            ax.axvline(np.searchsorted(all_vals, bp) - 0.5, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"{country} (n={counts.sum()})")
    fig.suptitle(f"{organism} / {ab_name} ({ab})\nMIC distribution: {country_a} vs {country_b} "
                 f"(similar %R, divergent distribution)", y=1.03)
    fig.tight_layout()
    fname = FIGURES_DIR / f"fig1_country_case_{rank}_{organism.split()[0]}_{ab}.png"
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    return fname


def main():
    set_publication_style()
    country_df = pd.read_csv(TABLES_DIR / "t1_stratified_metrics_by_country.csv")
    raw_df = pd.read_csv(CLEANED_DATA_PATH)

    pairs = mine_country_pairs(country_df)
    pairs = pairs[pairs["Min N"] >= MIN_N_FOR_CASE_STUDY].copy()
    # Deterministic ordering: primary key is GeoMean gap (descending). Exact ties in
    # GeoMean gap are broken by (1) larger combined sample size N_A+N_B (more
    # statistical power = a better representative exemplar), then (2) alphabetical
    # Country A/B, so re-running this script always yields the same case-study
    # selection regardless of incidental groupby/row order.
    pairs["_combined_N"] = pairs["N_A"] + pairs["N_B"]
    pairs = pairs.sort_values(
        ["GeoMean gap (log2 steps)", "_combined_N", "Country A", "Country B"],
        ascending=[False, False, True, True],
    ).drop(columns="_combined_N").reset_index(drop=True)

    out_path = TABLES_DIR / "t3_country_pairs_similar_pctR_divergent_distribution.csv"
    pairs.to_csv(out_path, index=False)
    print(f"Found {len(pairs)} qualifying country-pair x antibiotic combinations (Min N >= {MIN_N_FOR_CASE_STUDY})")
    print(f"Saved: {out_path}")

    top = pairs.drop_duplicates(subset=["Organism", "Antibiotic"]).head(4)
    print("\n=== Top case studies (one per Organism x Antibiotic, ranked by GeoMean gap) ===")
    with pd.option_context("display.width", 220):
        print(top[["Organism", "Antibiotic", "Country A", "%R_A", "GeoMean_A", "Country B", "%R_B", "GeoMean_B",
                    "%R difference (pp)", "GeoMean gap (log2 steps)"]].to_string(index=False))

    bp_module_path = Path(__file__).resolve().parent / "12_breakpoint_reference.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("bp_ref", bp_module_path)
    bp_ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bp_ref)

    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        bp, _, _ = bp_ref.BREAKPOINTS_S[(row["Organism"], row["Antibiotic"])]
        fpath = plot_country_case_study(raw_df, row["Organism"], row["Antibiotic"], row["Antibiotic (name)"],
                                         row["Country A"], row["Country B"], rank, bp)
        print(f"Saved case study figure: {fpath}")


if __name__ == "__main__":
    main()
