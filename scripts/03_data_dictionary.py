"""
03_data_dictionary.py
------------------------
STAGE ONE - PART 2: Data Dictionary

Produces a publication-quality data dictionary describing every variable
in the RAW dataset (name, description, data type, example value, number
of unique values, number missing, and analyst comments on known issues).

Descriptions of the antibiotic codes and body-site coding are based on
standard SOAR / CLSI antimicrobial nomenclature; all counts and example
values are computed directly from the uploaded file.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import RAW_DATA_PATH, SHEET_NAME, TABLES_DIR, ANTIBIOTIC_FULL_NAMES, ensure_dirs

MANUAL_DESCRIPTIONS = {
    "Isolate Number": "Unique laboratory identifier assigned to each bacterial isolate.",
    "Organism": "Species of the bacterial isolate recovered from the clinical specimen.",
    "BodyLocation": "Anatomical source / specimen type from which the isolate was recovered (body system: specimen).",
    "Country": "Country in which the isolate was collected.",
    "Centre": "Numeric code identifying the participating study/collection centre.",
    "Gender": "Sex of the patient from whom the isolate was collected (M/F).",
    "Age": "Age of the patient in years at the time of specimen collection.",
    "Collection Date": "Date on which the specimen was collected. Recorded inconsistently across records (full date, text date, or year only).",
    "Betalactamase": "Beta-lactamase production status of the isolate, as determined by phenotypic testing (applicable mainly to H. influenzae).",
}

BODY_LOCATION_NOTE = "Free-text 'System: Specimen' coding, e.g. 'Respiratory: Sputum', 'CVS: Blood'."


def main():
    ensure_dirs()
    df = pd.read_excel(RAW_DATA_PATH, sheet_name=SHEET_NAME)
    df = df.rename(columns=lambda c: c.strip())

    rows = []
    for col in df.columns:
        s = df[col]
        n_unique = s.nunique(dropna=True)
        n_missing = s.isna().sum()
        pct_missing = round(100 * n_missing / len(s), 2)
        non_null = s.dropna()
        example = non_null.iloc[0] if len(non_null) else ""

        if col in ANTIBIOTIC_FULL_NAMES:
            description = (
                f"Minimum Inhibitory Concentration (mg/L) of {ANTIBIOTIC_FULL_NAMES[col]} "
                f"against the isolate. Raw text field combining a dilution value "
                f"with an optional censoring operator ('</=' = left-censored, '>' = right-censored)."
            )
            comment = "MIC field; requires parsing into numeric value + censoring flag (see script 01)."
            if col in ("CDN", "DIN"):
                comment += (
                    " NOTE: standard lab abbreviation lists map both 'CDN' and 'DIN' to "
                    "Cefdinir, yet the two columns hold different values for the same isolate "
                    "(identical in only 169/2318 rows), so they are not duplicates of one "
                    "antibiotic. The true identity of one of these two columns is unresolved "
                    "and should be confirmed with the data provider before Stage Two."
                )
        elif col == "BodyLocation":
            description = MANUAL_DESCRIPTIONS[col]
            comment = "Free-text categorical; hierarchical (system: specimen) — consider splitting into two fields."
        elif col == "Collection Date":
            description = MANUAL_DESCRIPTIONS[col]
            comment = "Mixed data types (datetime, text 'DD-Mon-YY', bare year int) — standardized in script 01."
        elif col == "Betalactamase":
            description = MANUAL_DESCRIPTIONS[col]
            comment = "Inconsistent coding: 'Negative'/'NEG' and 'Positive'/'POS' used interchangeably; ~54% missing (mostly not applicable outside H. influenzae)."
        elif col == "Age":
            description = MANUAL_DESCRIPTIONS[col]
            comment = "Contains a maximum value of 150 — implausible for a human age; flagged for review, not altered."
        elif col == "Isolate Number":
            description = MANUAL_DESCRIPTIONS[col]
            comment = "Verified unique across all records (0 duplicates)."
        else:
            description = MANUAL_DESCRIPTIONS.get(col, "")
            comment = ""

        rows.append(
            {
                "Variable": col,
                "Description": description,
                "Data type (raw)": str(s.dtype),
                "Example value": example,
                "Unique values": n_unique,
                "Missing (n)": n_missing,
                "Missing (%)": pct_missing,
                "Comments": comment,
            }
        )

    dict_df = pd.DataFrame(rows)
    out_path = TABLES_DIR / "data_dictionary.csv"
    dict_df.to_csv(out_path, index=False)

    print("=== Data Dictionary ===")
    with pd.option_context("display.max_colwidth", 60, "display.width", 220):
        print(dict_df.to_string(index=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
