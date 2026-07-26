"""
07_antibiotic_coverage.py
----------------------------
STAGE ONE - PART 6: Antibiotic Testing Coverage

Determines which antibiotics were tested, overall testing frequency,
and coverage broken down by organism and by country. Produces a
publication-quality heatmap of organism x antibiotic testing coverage.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import CLEANED_DATA_PATH, TABLES_DIR, FIGURES_DIR, ANTIBIOTIC_VARS, set_publication_style, ensure_dirs


def main():
    ensure_dirs()
    set_publication_style()
    df = pd.read_csv(CLEANED_DATA_PATH)
    n = len(df)

    # --- Overall testing frequency ------------------------------------
    overall = []
    for ab in ANTIBIOTIC_VARS:
        n_tested = (df[f"{ab}_censor"] != "missing").sum()
        overall.append({"Antibiotic": ab, "Number tested": int(n_tested),
                         "Percentage tested (%)": round(100 * n_tested / n, 2)})
    overall_df = pd.DataFrame(overall).sort_values("Percentage tested (%)", ascending=False)
    overall_df.to_csv(TABLES_DIR / "table6a_overall_testing_frequency.csv", index=False)
    print("=== Overall antibiotic testing frequency ===")
    print(overall_df.to_string(index=False))

    # --- Coverage by organism (% of that organism's isolates tested) ------
    organisms = sorted(df["Organism"].unique())
    org_matrix = pd.DataFrame(index=organisms, columns=ANTIBIOTIC_VARS, dtype=float)
    for org in organisms:
        sub = df[df["Organism"] == org]
        for ab in ANTIBIOTIC_VARS:
            org_matrix.loc[org, ab] = round(100 * (sub[f"{ab}_censor"] != "missing").mean(), 1)
    org_matrix.to_csv(TABLES_DIR / "table6b_coverage_by_organism.csv")
    print("\n=== Coverage by organism (%) ===")
    print(org_matrix.to_string())

    # --- Coverage by country -----------------------------------------------
    countries = sorted(df["Country"].unique())
    country_matrix = pd.DataFrame(index=countries, columns=ANTIBIOTIC_VARS, dtype=float)
    for c in countries:
        sub = df[df["Country"] == c]
        for ab in ANTIBIOTIC_VARS:
            country_matrix.loc[c, ab] = round(100 * (sub[f"{ab}_censor"] != "missing").mean(), 1)
    country_matrix.to_csv(TABLES_DIR / "table6c_coverage_by_country.csv")
    print("\n=== Coverage by country (%) ===")
    print(country_matrix.to_string())

    # --- Heatmap: organism x antibiotic coverage ----------------------------
    fig, ax = plt.subplots(figsize=(11, 4.5))
    data = org_matrix.values.astype(float)
    im = ax.imshow(data, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(ANTIBIOTIC_VARS)))
    ax.set_xticklabels(ANTIBIOTIC_VARS, rotation=45, ha="right")
    ax.set_yticks(range(len(organisms)))
    ax.set_yticklabels(organisms)
    for i in range(len(organisms)):
        for j in range(len(ANTIBIOTIC_VARS)):
            ax.text(j, i, f"{data[i, j]:.0f}", ha="center", va="center",
                     color="white" if data[i, j] > 60 else "black", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Percent of isolates tested (%)")
    ax.set_title("Antibiotic Testing Coverage by Organism\nGSK SOAR 2019-10 raw data extract (descriptive only)")
    fig.tight_layout()
    fig_path = FIGURES_DIR / "fig_antibiotic_coverage_heatmap.png"
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"\nSaved: {fig_path}")

    # --- Heatmap: country x antibiotic coverage --------------------------
    fig, ax = plt.subplots(figsize=(11, 7))
    data_c = country_matrix.values.astype(float)
    im = ax.imshow(data_c, cmap="YlOrRd", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(ANTIBIOTIC_VARS)))
    ax.set_xticklabels(ANTIBIOTIC_VARS, rotation=45, ha="right")
    ax.set_yticks(range(len(countries)))
    ax.set_yticklabels(countries, fontsize=8)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Percent of isolates tested (%)")
    ax.set_title("Antibiotic Testing Coverage by Country\nGSK SOAR 2019-10 raw data extract (descriptive only)")
    fig.tight_layout()
    fig_path2 = FIGURES_DIR / "fig_antibiotic_coverage_by_country_heatmap.png"
    fig.savefig(fig_path2)
    plt.close(fig)
    print(f"Saved: {fig_path2}")


if __name__ == "__main__":
    main()
