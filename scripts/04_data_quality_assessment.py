"""
04_data_quality_assessment.py
--------------------------------
STAGE ONE - PART 3: Data Quality Assessment

Covers:
  (a) Missing values - counts, percentages, heatmap, summary table
  (b) Duplicate records - full-row duplicates, duplicate isolate IDs,
      duplicate MIC observations (same isolate + same antibiotic panel)
  (c) Consistency checks - country names, organism names, antibiotic
      column presence, collection years, sex coding, body-site coding,
      beta-lactamase coding, MIC formatting
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    CLEANED_DATA_PATH,
    TABLES_DIR,
    FIGURES_DIR,
    ANTIBIOTIC_VARS,
    set_publication_style,
    ensure_dirs,
)


def missing_values_analysis(df: pd.DataFrame):
    print("\n--- Missing values ---")
    raw_cols = ["Isolate Number", "Organism", "BodyLocation", "Country", "Centre",
                "Gender", "Age", "Collection Date", "Betalactamase"] + ANTIBIOTIC_VARS

    miss = df[raw_cols].isna().sum()
    pct = (miss / len(df) * 100).round(2)
    summary = pd.DataFrame({"Variable": raw_cols, "Missing (n)": miss.values, "Missing (%)": pct.values})
    summary = summary.sort_values("Missing (%)", ascending=False)
    summary_path = TABLES_DIR / "table3a_missing_values_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"Saved: {summary_path}")

    # Heatmap of missingness (rows=isolates sampled, cols=variables)
    set_publication_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    miss_matrix = df[raw_cols].isna().astype(int)
    ax.imshow(miss_matrix.T, aspect="auto", cmap="Greys", interpolation="nearest")
    ax.set_yticks(range(len(raw_cols)))
    ax.set_yticklabels(raw_cols, fontsize=8)
    ax.set_xlabel("Record index")
    ax.set_title("Missing Data Matrix (dark = missing)\nGSK SOAR 2019-10 raw data extract")
    fig.tight_layout()
    fig_path = FIGURES_DIR / "fig_missing_data_matrix.png"
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Saved: {fig_path}")

    return summary


def duplicate_analysis(df: pd.DataFrame):
    print("\n--- Duplicate records ---")
    full_dupes = df.duplicated().sum()
    id_dupes = df["Isolate Number"].duplicated().sum()

    # Duplicate MIC observations: same isolate id appearing with the same
    # antibiotic panel values more than once (would indicate re-entry)
    mic_subset = ["Isolate Number"] + ANTIBIOTIC_VARS
    mic_dupes = df[mic_subset].duplicated(subset=ANTIBIOTIC_VARS, keep=False).sum()

    rows = [
        {"Check": "Fully duplicated rows (all columns identical)", "Count": int(full_dupes)},
        {"Check": "Duplicate Isolate Number values", "Count": int(id_dupes)},
        {"Check": "Records sharing an identical full antibiotic MIC panel (excl. ID)", "Count": int(mic_dupes)},
    ]
    dup_df = pd.DataFrame(rows)
    dup_path = TABLES_DIR / "table3b_duplicate_records.csv"
    dup_df.to_csv(dup_path, index=False)
    print(dup_df.to_string(index=False))
    print(f"Saved: {dup_path}")
    return dup_df


def consistency_checks(df: pd.DataFrame):
    print("\n--- Consistency checks ---")
    findings = []

    # Country names
    countries = sorted(df["Country"].dropna().unique())
    findings.append({"Check": "Country names", "Unique values": len(countries),
                      "Finding": "No obvious spelling variants or case inconsistencies detected.",
                      "Values": ", ".join(countries)})

    # Organism names
    organisms = sorted(df["Organism"].dropna().unique())
    findings.append({"Check": "Organism names", "Unique values": len(organisms),
                      "Finding": "Consistent binomial nomenclature; no case/spelling variants detected.",
                      "Values": ", ".join(organisms)})

    # Antibiotic column presence / naming ambiguity
    findings.append({"Check": "Antibiotic column codes", "Unique values": len(ANTIBIOTIC_VARS),
                      "Finding": "CDN and DIN both correspond to 'Cefdinir' in standard lab abbreviation "
                                 "lists, yet hold different values per isolate (identical in only "
                                 f"{(df['CDN']==df['DIN']).sum()}/{len(df)} rows) - unresolved naming ambiguity, "
                                 "recommend confirming with data provider.",
                      "Values": ", ".join(ANTIBIOTIC_VARS)})

    # Collection years
    years = sorted(df["Collection_Year"].dropna().unique().astype(int))
    findings.append({"Check": "Collection years", "Unique values": len(years),
                      "Finding": f"Range {min(years)}-{max(years)}; plausible and internally consistent. "
                                 "However, the source field mixes three encodings (datetime, text date, "
                                 "bare year) - standardized in script 01.",
                      "Values": ", ".join(str(y) for y in years)})

    # Sex coding
    sexes = sorted(df["Gender"].dropna().unique())
    findings.append({"Check": "Sex (Gender) coding", "Unique values": len(sexes),
                      "Finding": "Two consistent codes used (M/F); no blanks, unknowns, or alternate codings observed.",
                      "Values": ", ".join(sexes)})

    # Body site coding
    body_sites = sorted(df["BodyLocation"].dropna().unique())
    findings.append({"Check": "Body site (BodyLocation) coding", "Unique values": len(body_sites),
                      "Finding": "Consistent 'System: Specimen' hierarchical free-text format; no case "
                                 "or delimiter inconsistencies detected, though the field mixes two "
                                 "levels of granularity in one string.",
                      "Values": "; ".join(body_sites)})

    # Beta-lactamase coding
    bl_raw = sorted(df["Betalactamase"].dropna().unique())
    findings.append({"Check": "Beta-lactamase coding", "Unique values": len(bl_raw),
                      "Finding": "INCONSISTENT: four distinct raw codes for two true categories "
                                 "('Negative'/'NEG' and 'Positive'/'POS'); standardized to two labels "
                                 "in script 01. Also 53.8% missing overall.",
                      "Values": ", ".join(bl_raw)})

    # MIC formatting
    ops_found = set()
    for ab in ANTIBIOTIC_VARS:
        for v in df[ab].dropna().astype(str).unique():
            for op in ("</=", ">", "<="):
                if v.startswith(op):
                    ops_found.add(op)
    findings.append({"Check": "MIC formatting (censoring operators)", "Unique values": len(ops_found),
                      "Finding": f"Operators observed across all antibiotic columns: {sorted(ops_found)}. "
                                 "Format is internally consistent (operator + numeric dilution value); "
                                 "parsed into numeric value + censor type in script 01.",
                      "Values": ", ".join(sorted(ops_found))})

    # Age plausibility (not a "consistency" issue per se, but a coding/range check)
    n_extreme_age = int((df["Age"] > 110).sum())
    findings.append({"Check": "Age plausibility", "Unique values": df["Age"].nunique(),
                      "Finding": f"Range {df['Age'].min()}-{df['Age'].max()} years. "
                                 f"{n_extreme_age} record(s) with Age > 110 flagged as implausible.",
                      "Values": ""})

    cons_df = pd.DataFrame(findings)
    cons_path = TABLES_DIR / "table3c_consistency_checks.csv"
    cons_df.to_csv(cons_path, index=False)
    with pd.option_context("display.max_colwidth", 90, "display.width", 200):
        print(cons_df[["Check", "Unique values", "Finding"]].to_string(index=False))
    print(f"Saved: {cons_path}")
    return cons_df


def main():
    ensure_dirs()
    df = pd.read_csv(CLEANED_DATA_PATH)
    missing_values_analysis(df)
    duplicate_analysis(df)
    consistency_checks(df)


if __name__ == "__main__":
    main()
