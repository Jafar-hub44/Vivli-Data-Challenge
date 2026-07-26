"""
09_censoring_characterization.py
------------------------------------
STAGE 2A - PART 1: Characterize MIC Censoring

Builds on the Stage 1 standardized dataset (data/cleaned/soar_stage1_standardized.csv)
and the {antibiotic}_value / {antibiotic}_censor columns produced there.

Produces:
  - Per-antibiotic censoring table (exact / left / right counts, % censored,
    lowest & highest observed dilution)
  - Censoring summary by Organism x Antibiotic
  - Censoring summary by Country (aggregated across antibiotics)
  - Censoring summary by Collection Year (aggregated across antibiotics)
  - Supporting figures

No MIC50/MIC90 or other surveillance metric is calculated here - this is
purely a characterization of the censoring pattern itself.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (  # noqa: E402
    ANTIBIOTIC_VARS,
    ANTIBIOTIC_FULL_NAMES,
    set_publication_style,
    PALETTE,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "cleaned" / "soar_stage1_standardized.csv"
TABLES_DIR = Path(__file__).resolve().parents[1] / "tables"
FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"


def per_antibiotic_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ab in ANTIBIOTIC_VARS:
        val_col, cen_col = f"{ab}_value", f"{ab}_censor"
        n_exact = (df[cen_col] == "exact").sum()
        n_left = (df[cen_col] == "left").sum()
        n_right = (df[cen_col] == "right").sum()
        n_missing = (df[cen_col] == "missing").sum()
        n_tested = n_exact + n_left + n_right
        tested_vals = df.loc[df[cen_col].isin(["exact", "left", "right"]), val_col]
        rows.append({
            "Antibiotic (code)": ab,
            "Antibiotic (name)": ANTIBIOTIC_FULL_NAMES[ab],
            "N tested": int(n_tested),
            "N missing": int(n_missing),
            "N exact": int(n_exact),
            "N left-censored (<=)": int(n_left),
            "N right-censored (>)": int(n_right),
            "% censored (of tested)": round(100 * (n_left + n_right) / n_tested, 2) if n_tested else 0.0,
            "% left-censored (of tested)": round(100 * n_left / n_tested, 2) if n_tested else 0.0,
            "% right-censored (of tested)": round(100 * n_right / n_tested, 2) if n_tested else 0.0,
            "Lowest observed dilution (mg/L)": tested_vals.min() if len(tested_vals) else None,
            "Highest observed dilution (mg/L)": tested_vals.max() if len(tested_vals) else None,
        })
    return pd.DataFrame(rows).sort_values("% censored (of tested)", ascending=False)


def by_organism_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for org in sorted(df["Organism"].unique()):
        sub = df[df["Organism"] == org]
        for ab in ANTIBIOTIC_VARS:
            cen_col = f"{ab}_censor"
            n_exact = (sub[cen_col] == "exact").sum()
            n_left = (sub[cen_col] == "left").sum()
            n_right = (sub[cen_col] == "right").sum()
            n_tested = n_exact + n_left + n_right
            if n_tested == 0:
                continue  # antibiotic not part of this organism's panel (documented in Stage 1)
            rows.append({
                "Organism": org,
                "Antibiotic": ab,
                "N tested": int(n_tested),
                "N exact": int(n_exact),
                "N left-censored": int(n_left),
                "N right-censored": int(n_right),
                "% censored": round(100 * (n_left + n_right) / n_tested, 2),
            })
    return pd.DataFrame(rows)


def by_country_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in sorted(df["Country"].unique()):
        sub = df[df["Country"] == c]
        n_exact = n_left = n_right = n_tested = 0
        for ab in ANTIBIOTIC_VARS:
            cen_col = f"{ab}_censor"
            n_exact += (sub[cen_col] == "exact").sum()
            n_left += (sub[cen_col] == "left").sum()
            n_right += (sub[cen_col] == "right").sum()
        n_tested = n_exact + n_left + n_right
        rows.append({
            "Country": c,
            "N isolates": len(sub),
            "N MIC observations tested": int(n_tested),
            "% censored (all antibiotics pooled)": round(100 * (n_left + n_right) / n_tested, 2) if n_tested else 0.0,
        })
    return pd.DataFrame(rows).sort_values("% censored (all antibiotics pooled)", ascending=False)


def by_year_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for y in sorted(df["Collection_Year"].dropna().unique()):
        sub = df[df["Collection_Year"] == y]
        n_exact = n_left = n_right = 0
        for ab in ANTIBIOTIC_VARS:
            cen_col = f"{ab}_censor"
            n_exact += (sub[cen_col] == "exact").sum()
            n_left += (sub[cen_col] == "left").sum()
            n_right += (sub[cen_col] == "right").sum()
        n_tested = n_exact + n_left + n_right
        rows.append({
            "Collection Year": int(y),
            "N isolates": len(sub),
            "N MIC observations tested": int(n_tested),
            "% censored (all antibiotics pooled)": round(100 * (n_left + n_right) / n_tested, 2) if n_tested else 0.0,
        })
    return pd.DataFrame(rows)


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    set_publication_style()

    df = pd.read_csv(CLEANED_DATA_PATH)

    # --- Per-antibiotic ------------------------------------------------
    ab_table = per_antibiotic_table(df)
    ab_table.to_csv(TABLES_DIR / "t1_censoring_by_antibiotic.csv", index=False)
    print("=== Censoring by antibiotic ===")
    with pd.option_context("display.width", 220):
        print(ab_table.to_string(index=False))

    # --- By organism -----------------------------------------------------
    org_table = by_organism_table(df)
    org_table.to_csv(TABLES_DIR / "t2_censoring_by_organism_antibiotic.csv", index=False)
    print("\n=== Censoring by organism x antibiotic (head) ===")
    print(org_table.head(10).to_string(index=False))

    # --- By country --------------------------------------------------------
    country_table = by_country_table(df)
    country_table.to_csv(TABLES_DIR / "t3_censoring_by_country.csv", index=False)
    print("\n=== Censoring by country (pooled across antibiotics) ===")
    print(country_table.to_string(index=False))

    # --- By year ----------------------------------------------------------
    year_table = by_year_table(df)
    year_table.to_csv(TABLES_DIR / "t4_censoring_by_year.csv", index=False)
    print("\n=== Censoring by collection year (pooled across antibiotics) ===")
    print(year_table.to_string(index=False))

    # --- Figure: % censored by antibiotic, split left/right ----------------
    fig, ax = plt.subplots(figsize=(11, 6))
    order = ab_table.sort_values("% censored (of tested)", ascending=False)
    x = range(len(order))
    ax.bar(x, order["% left-censored (of tested)"], label="Left-censored (<=)", color=PALETTE[1])
    ax.bar(x, order["% right-censored (of tested)"], bottom=order["% left-censored (of tested)"],
           label="Right-censored (>)", color=PALETTE[4])
    ax.set_xticks(list(x))
    ax.set_xticklabels(order["Antibiotic (code)"], rotation=45, ha="right")
    ax.set_ylabel("Percentage of tested isolates (%)")
    ax.set_title("MIC Censoring by Antibiotic\nGSK SOAR 2019-10 (Stage 2A characterization)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_censoring_by_antibiotic.png")
    plt.close(fig)

    # --- Figure: % censored by organism (heatmap, organism x antibiotic) ---
    organisms = sorted(df["Organism"].unique())
    pivot = org_table.pivot(index="Organism", columns="Antibiotic", values="% censored").reindex(
        index=organisms, columns=ANTIBIOTIC_VARS
    )
    fig, ax = plt.subplots(figsize=(11, 4.5))
    im = ax.imshow(pivot.values.astype(float), cmap="Reds", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(ANTIBIOTIC_VARS)))
    ax.set_xticklabels(ANTIBIOTIC_VARS, rotation=45, ha="right")
    ax.set_yticks(range(len(organisms)))
    ax.set_yticklabels(organisms)
    for i in range(len(organisms)):
        for j in range(len(ANTIBIOTIC_VARS)):
            v = pivot.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                         color="white" if v > 60 else "black", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("% censored")
    ax.set_title("MIC Censoring by Organism x Antibiotic\n(blank cell = antibiotic not in that organism's panel)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_censoring_by_organism_heatmap.png")
    plt.close(fig)

    # --- Figure: % censored by country --------------------------------
    fig, ax = plt.subplots(figsize=(10, 7))
    cs = country_table.sort_values("% censored (all antibiotics pooled)")
    ax.barh(cs["Country"], cs["% censored (all antibiotics pooled)"], color=PALETTE[2])
    ax.set_xlabel("% censored (all antibiotics pooled)")
    ax.set_title("MIC Censoring by Country (pooled across all antibiotics)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_censoring_by_country.png")
    plt.close(fig)

    # --- Figure: % censored by year --------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(year_table["Collection Year"].astype(str), year_table["% censored (all antibiotics pooled)"], color=PALETTE[3])
    ax.set_xlabel("Collection Year")
    ax.set_ylabel("% censored (all antibiotics pooled)")
    ax.set_title("MIC Censoring by Collection Year")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_censoring_by_year.png")
    plt.close(fig)

    print(f"\nSaved tables to: {TABLES_DIR}")
    print(f"Saved figures to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
