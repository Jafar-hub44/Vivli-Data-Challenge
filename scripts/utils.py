"""
utils.py
--------
Shared constants and helper functions used across all Stage One scripts
for the MIC-Sentinel / Vivli AMR Data Challenge project.

Author: MIC-Sentinel Stage One pipeline
Dataset: GSK SOAR (Survey of Antibiotic Resistance) 2019-10 raw data extract
"""

import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Project paths (relative to the MIC_Sentinel project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "GSK_SOAR_201910_raw_data.xlsx"
CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "cleaned" / "soar_stage1_standardized.csv"
FIGURES_DIR = PROJECT_ROOT / "figures"
TABLES_DIR = PROJECT_ROOT / "tables"
REPORTS_DIR = PROJECT_ROOT / "reports"

SHEET_NAME = "3550 valid MIC data (2)"

# ---------------------------------------------------------------------------
# Column groupings
# ---------------------------------------------------------------------------
ID_VARS = ["Isolate Number"]

DEMOGRAPHIC_VARS = ["Country", "Centre", "Gender", "Age"]

CLINICAL_VARS = ["Organism", "BodyLocation", "Betalactamase"]

DATE_VARS = ["Collection Date"]

# Antibiotic abbreviation -> full name (per SOAR / GSK documentation and
# standard antimicrobial nomenclature). Used for readable labels in
# tables and figures; the original short codes are preserved as the
# authoritative column names throughout the analysis.
ANTIBIOTIC_FULL_NAMES = {
    "AMC": "Amoxicillin/Clavulanic acid",
    "AMP": "Ampicillin",
    "AMX": "Amoxicillin",
    "AXO": "Ceftriaxone",
    "AZM": "Azithromycin",
    "CDN": "Cefdinir",
    "CEC": "Cefaclor",
    "CLA": "Clarithromycin",
    "CXM": "Cefuroxime",
    "DIN": "Cefdinir (alternate code - see note)",
    "ERY": "Erythromycin",
    "FIX": "Cefixime",
    "LEV": "Levofloxacin",
    "MXF": "Moxifloxacin",
    "PEN": "Penicillin",
    "POD": "Cefpodoxime",
    "SXT": "Trimethoprim/Sulfamethoxazole",
}

# NOTE ON CODE AMBIGUITY: Standard CLSI/laboratory abbreviation lists map BOTH
# "CDN" and "DIN" to Cefdinir (e.g. Cdn, CDR, DIN, CD, CFD are all documented
# cefdinir abbreviations). This dataset contains both CDN and DIN as separate
# columns with different values per row, so they cannot both simply be
# "Cefdinir" duplicated. This is flagged as an unresolved naming ambiguity in
# the Stage One QC report rather than guessed at - the true identity of one of
# these two columns should be confirmed with the data provider before Stage Two.

ANTIBIOTIC_VARS = list(ANTIBIOTIC_FULL_NAMES.keys())

ALL_EXPECTED_VARS = ID_VARS + CLINICAL_VARS + DEMOGRAPHIC_VARS + DATE_VARS + ANTIBIOTIC_VARS

# ---------------------------------------------------------------------------
# MIC parsing
# ---------------------------------------------------------------------------
# The raw MIC fields are text strings that combine a numeric dilution value
# with an optional censoring operator:
#   "</="  -> left-censored (true value <= reported value; at/below lowest
#             dilution tested)
#   ">"    -> right-censored (true value > reported value; at/above highest
#             dilution tested)
#   none   -> exact (uncensored) MIC reading
MIC_PATTERN = re.compile(r"^\s*(</=|<=|>|<)?\s*([0-9]*\.?[0-9]+)\s*$")


def parse_mic(value):
    """
    Parse a raw MIC string into its numeric magnitude and censoring type.

    Returns a tuple: (numeric_value: float or None, censor_type: str)
    censor_type is one of {"exact", "left", "right", "missing", "unparsed"}.
    """
    if pd.isna(value):
        return (None, "missing")
    s = str(value).strip()
    if s == "":
        return (None, "missing")
    m = MIC_PATTERN.match(s)
    if not m:
        return (None, "unparsed")
    operator, number = m.group(1), m.group(2)
    numeric = float(number)
    if operator in ("</=", "<="):
        return (numeric, "left")
    if operator in (">",):
        return (numeric, "right")
    if operator in ("<",):
        return (numeric, "left")
    return (numeric, "exact")


def parse_collection_date(value):
    """
    Normalize the heterogeneous 'Collection Date' field, which mixes:
      - native datetime objects (Excel date-formatted cells)
      - strings such as '12-Nov-16' (Excel text-formatted cells)
      - bare 4-digit years stored as integers (e.g. 2016)

    Returns a tuple (parsed_timestamp: pd.Timestamp or NaT, year: Int or NA,
    source_type: str) so that both the fully-resolved date (where available)
    and the coarser year (always available when parseable) can be retained
    without fabricating a day/month for records that only ever recorded a year.
    """
    if pd.isna(value):
        return (pd.NaT, pd.NA, "missing")

    # Case 1: already a real date/datetime object
    if isinstance(value, (pd.Timestamp,)) or hasattr(value, "year") and hasattr(value, "month") and not isinstance(value, int):
        ts = pd.Timestamp(value)
        return (ts, ts.year, "datetime")

    # Case 2: bare integer year (e.g. 2016), NOT an Excel serial date
    if isinstance(value, int):
        if 1900 < value < 2100:
            return (pd.NaT, value, "year_only")
        return (pd.NaT, pd.NA, "unparsed")

    # Case 3: string date, e.g. '12-Nov-16'
    s = str(value).strip()
    try:
        ts = pd.to_datetime(s, format="%d-%b-%y", errors="raise")
        return (ts, ts.year, "string_date")
    except (ValueError, TypeError):
        pass

    # Case 3b: month-year only, e.g. 'Mar-17' (no day recorded). Handled
    # explicitly because pandas' generic parser misreads a bare 'Mon-YY'
    # string as day=YY with year defaulting to 1 (e.g. 'Mar-17' -> year 1
    # rather than 2017), which would silently corrupt the year.
    m = re.fullmatch(r"([A-Za-z]{3})-(\d{2})", s)
    if m:
        try:
            ts = pd.to_datetime(f"01-{m.group(1)}-{m.group(2)}", format="%d-%b-%y", errors="raise")
            return (pd.NaT, ts.year, "month_year_only")
        except (ValueError, TypeError):
            pass

    try:
        ts = pd.to_datetime(s, errors="raise")
        return (ts, ts.year, "string_date_fallback")
    except (ValueError, TypeError):
        pass
    # Bare year as string
    if re.fullmatch(r"(19|20)\d{2}", s):
        return (pd.NaT, int(s), "year_only_string")
    return (pd.NaT, pd.NA, "unparsed")


def standardize_betalactamase(value):
    """Collapse inconsistent beta-lactamase coding into a canonical label."""
    if pd.isna(value):
        return pd.NA
    s = str(value).strip().upper()
    if s in ("NEG", "NEGATIVE"):
        return "Negative"
    if s in ("POS", "POSITIVE"):
        return "Positive"
    return str(value).strip()


# ---------------------------------------------------------------------------
# Plotting style (publication quality, consistent across all figures)
# ---------------------------------------------------------------------------
def set_publication_style():
    matplotlib.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.edgecolor": "#333333",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


PALETTE = [
    "#2E5A88", "#4C8C9B", "#84A98C", "#C9A227",
    "#B5546A", "#6D597A", "#8B8C89", "#3D5A80",
]


def ensure_dirs():
    for d in (FIGURES_DIR, TABLES_DIR, REPORTS_DIR, CLEANED_DATA_PATH.parent):
        d.mkdir(parents=True, exist_ok=True)
