"""
06_descriptive_characteristics.py
------------------------------------
STAGE ONE - PART 5: Descriptive Dataset Characteristics

Purely descriptive frequency/percentage summaries and bar charts /
histogram for: Organism, Country, Collection year, Age, Sex, Body site,
Beta-lactamase status. No biological interpretation.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import CLEANED_DATA_PATH, TABLES_DIR, FIGURES_DIR, set_publication_style, PALETTE, ensure_dirs


def freq_table(df, col, dropna=False):
    counts = df[col].value_counts(dropna=dropna)
    pct = (counts / len(df) * 100).round(2)
    out = pd.DataFrame({col: counts.index, "Frequency": counts.values, "Percentage": pct.values})
    return out


def bar_chart(labels, values, title, xlabel, ylabel, fname, horizontal=False, color=None, rotate=45):
    fig, ax = plt.subplots(figsize=(10, 6))
    color = color or PALETTE[0]
    if horizontal:
        ax.barh(labels, values, color=color)
        ax.invert_yaxis()
        ax.set_xlabel(ylabel)
        ax.set_ylabel(xlabel)
    else:
        ax.bar(labels, values, color=color)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.setp(ax.get_xticklabels(), rotation=rotate, ha="right")
    ax.set_title(title)
    fig.tight_layout()
    path = FIGURES_DIR / fname
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    ensure_dirs()
    set_publication_style()
    df = pd.read_csv(CLEANED_DATA_PATH)

    # --- Organisms -----------------------------------------------------
    t = freq_table(df, "Organism")
    t.to_csv(TABLES_DIR / "table5a_organism_frequency.csv", index=False)
    print("=== Organism frequency ===")
    print(t.to_string(index=False))
    bar_chart(t["Organism"], t["Frequency"], "Organism Distribution\n(GSK SOAR 2019-10 raw data extract, descriptive only)",
              "Organism", "Number of isolates", "fig_organism_distribution.png", horizontal=True, color=PALETTE[0])

    # --- Countries -------------------------------------------------------
    t = freq_table(df, "Country").sort_values("Frequency", ascending=False)
    t.to_csv(TABLES_DIR / "table5b_country_frequency.csv", index=False)
    print("\n=== Country frequency ===")
    print(t.to_string(index=False))
    bar_chart(t["Country"], t["Frequency"], "Country Distribution\n(GSK SOAR 2019-10 raw data extract, descriptive only)",
              "Country", "Number of isolates", "fig_country_distribution.png", horizontal=True, color=PALETTE[2])

    # --- Collection years --------------------------------------------------
    t = freq_table(df, "Collection_Year").sort_values("Collection_Year")
    t = t.rename(columns={"Collection_Year": "Collection Year"})
    t.to_csv(TABLES_DIR / "table5c_collection_year_frequency.csv", index=False)
    print("\n=== Collection year frequency ===")
    print(t.to_string(index=False))
    bar_chart(t["Collection Year"].astype(int).astype(str), t["Frequency"],
              "Collection Year Distribution\n(GSK SOAR 2019-10 raw data extract, descriptive only)",
              "Collection Year", "Number of isolates", "fig_collection_year_distribution.png",
              color=PALETTE[3], rotate=0)

    # --- Age ------------------------------------------------------------
    age_summary = df["Age"].describe().to_frame(name="Age (years)")
    age_summary.to_csv(TABLES_DIR / "table5d_age_summary_statistics.csv")
    print("\n=== Age summary statistics ===")
    print(age_summary.to_string())

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(df["Age"], bins=30, color=PALETTE[5], edgecolor="white")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Number of isolates")
    ax.set_title("Age Distribution\n(GSK SOAR 2019-10 raw data extract, descriptive only)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_age_distribution.png")
    plt.close(fig)
    print(f"Saved: {FIGURES_DIR / 'fig_age_distribution.png'}")

    # --- Sex --------------------------------------------------------------
    t = freq_table(df, "Gender")
    t.to_csv(TABLES_DIR / "table5e_sex_frequency.csv", index=False)
    print("\n=== Sex frequency ===")
    print(t.to_string(index=False))
    bar_chart(t["Gender"], t["Frequency"], "Sex Distribution\n(GSK SOAR 2019-10 raw data extract, descriptive only)",
              "Sex", "Number of isolates", "fig_sex_distribution.png", color=PALETTE[6], rotate=0)

    # --- Body site ----------------------------------------------------
    t = freq_table(df, "BodyLocation").sort_values("Frequency", ascending=False)
    t.to_csv(TABLES_DIR / "table5f_bodysite_frequency.csv", index=False)
    print("\n=== Body site frequency ===")
    print(t.to_string(index=False))
    bar_chart(t["BodyLocation"], t["Frequency"], "Body Site (Specimen Source) Distribution\n(GSK SOAR 2019-10 raw data extract, descriptive only)",
              "Body site", "Number of isolates", "fig_bodysite_distribution.png", horizontal=True, color=PALETTE[1])

    # --- Beta-lactamase -----------------------------------------------
    t = freq_table(df, "Betalactamase_Standardized", dropna=False)
    t = t.rename(columns={"Betalactamase_Standardized": "Betalactamase"})
    t["Betalactamase"] = t["Betalactamase"].fillna("Missing / Not tested")
    t.to_csv(TABLES_DIR / "table5g_betalactamase_frequency.csv", index=False)
    print("\n=== Beta-lactamase frequency (standardized coding) ===")
    print(t.to_string(index=False))
    bar_chart(t["Betalactamase"], t["Frequency"], "Beta-lactamase Status Distribution\n(GSK SOAR 2019-10 raw data extract, descriptive only)",
              "Beta-lactamase status", "Number of isolates", "fig_betalactamase_distribution.png",
              color=PALETTE[4], rotate=0)


if __name__ == "__main__":
    main()
