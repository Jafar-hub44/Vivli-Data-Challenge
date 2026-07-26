"""
12_breakpoint_reference.py
------------------------------
STAGE 3 - Shared breakpoint reference table.

Defines a curated set of CLSI-consistent susceptible ("S") breakpoints
(mg/L) for organism-antibiotic pairs with reasonably high confidence,
used consistently across all Stage 3 scripts for the "% resistant"
(breakpoint-exceedance) calculations that anchor the comparison against
distributional MIC metrics.

IMPORTANT CAVEATS (documented once here, applicable everywhere this table
is used):
  - These are commonly cited, longstanding CLSI M100 values used for
    METHODOLOGICAL DEMONSTRATION in this research project. They are NOT
    independently re-verified against the current CLSI M100 edition for
    every pair, and must not be used for any clinical, regulatory, or
    publication purpose without checking the current CLSI M100 edition.
  - Amoxicillin/clavulanate breakpoints (H. influenzae and Enterobacterales)
    were revised by CLSI in 2022; the legacy/contemporaneous value is used
    here because the SOAR 2019-10 dataset was collected before that
    revision, so it is the internally consistent choice for this dataset's
    collection period - NOT necessarily the currently valid clinical value.
  - Only pairs with reasonably high recall confidence are included. Pairs
    omitted (value None) are treated as "no breakpoint applied in this
    pipeline" rather than guessed at.
  - Trimethoprim/sulfamethoxazole (SXT) breakpoints are expressed against
    the trimethoprim component only, per standard CLSI convention, and
    differ by organism group (Streptococcus/Haemophilus vs Enterobacterales).
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import ANTIBIOTIC_FULL_NAMES  # noqa: E402

TABLES_DIR = Path(__file__).resolve().parents[1] / "tables"

# (Organism, Antibiotic code): (S breakpoint mg/L, confidence, note)
BREAKPOINTS_S = {
    # --- Streptococcus pneumoniae (CLSI, nonmeningitis oral/IV as applicable) ---
    ("Streptococcus pneumoniae", "PEN"): (0.06, "high", "Oral penicillin, nonmeningitis"),
    ("Streptococcus pneumoniae", "AMX"): (2.0, "moderate", "Oral amoxicillin"),
    ("Streptococcus pneumoniae", "AMC"): (2.0, "high", "Amoxicillin/clavulanate, amoxicillin component"),
    ("Streptococcus pneumoniae", "AXO"): (1.0, "high", "Ceftriaxone, nonmeningitis"),
    ("Streptococcus pneumoniae", "POD"): (0.5, "high", "Cefpodoxime"),
    ("Streptococcus pneumoniae", "CXM"): (0.5, "moderate", "Cefuroxime, oral"),
    ("Streptococcus pneumoniae", "ERY"): (0.25, "high", "Erythromycin"),
    ("Streptococcus pneumoniae", "CLA"): (0.25, "high", "Clarithromycin"),
    ("Streptococcus pneumoniae", "AZM"): (0.5, "high", "Azithromycin"),
    ("Streptococcus pneumoniae", "SXT"): (0.5, "moderate", "Trimethoprim/sulfamethoxazole, trimethoprim component"),
    ("Streptococcus pneumoniae", "LEV"): (2.0, "high", "Levofloxacin"),
    ("Streptococcus pneumoniae", "MXF"): (1.0, "high", "Moxifloxacin"),
    ("Streptococcus pneumoniae", "CDN"): (0.5, "moderate", "Cefdinir"),
    ("Streptococcus pneumoniae", "DIN"): (0.5, "low", "Assumes DIN = cefdinir (unresolved code ambiguity, see Stage 1/2A)"),
    # --- Haemophilus influenzae (CLSI) ---
    ("Haemophilus influenzae", "AMP"): (1.0, "high", "Ampicillin"),
    ("Haemophilus influenzae", "AMC"): (4.0, "moderate", "Amoxicillin/clavulanate, legacy pre-2022 breakpoint, amoxicillin component"),
    ("Haemophilus influenzae", "AXO"): (2.0, "high", "Ceftriaxone"),
    ("Haemophilus influenzae", "CXM"): (4.0, "moderate", "Cefuroxime, oral"),
    ("Haemophilus influenzae", "FIX"): (1.0, "high", "Cefixime"),
    ("Haemophilus influenzae", "POD"): (2.0, "moderate", "Cefpodoxime"),
    ("Haemophilus influenzae", "AZM"): (4.0, "high", "Azithromycin"),
    ("Haemophilus influenzae", "CLA"): (8.0, "moderate", "Clarithromycin"),
    ("Haemophilus influenzae", "SXT"): (0.5, "moderate", "Trimethoprim/sulfamethoxazole, trimethoprim component"),
    ("Haemophilus influenzae", "LEV"): (2.0, "high", "Levofloxacin"),
    ("Haemophilus influenzae", "MXF"): (2.0, "low", "Moxifloxacin, lower-confidence recall"),
    ("Haemophilus influenzae", "CDN"): (1.0, "moderate", "Cefdinir"),
    ("Haemophilus influenzae", "DIN"): (1.0, "low", "Assumes DIN = cefdinir (unresolved code ambiguity, see Stage 1/2A)"),
    # --- Enterobacterales (E. coli, K. pneumoniae), legacy pre-2022 where relevant ---
    ("Escherichia coli", "AMP"): (8.0, "high", "Ampicillin, Enterobacterales"),
    ("Escherichia coli", "AMC"): (8.0, "moderate", "Amoxicillin/clavulanate, legacy pre-2022, amoxicillin component"),
    ("Escherichia coli", "AXO"): (1.0, "high", "Ceftriaxone, Enterobacterales"),
    ("Escherichia coli", "FIX"): (1.0, "moderate", "Cefixime, Enterobacterales"),
    ("Escherichia coli", "SXT"): (2.0, "high", "Trimethoprim/sulfamethoxazole, trimethoprim component, Enterobacterales"),
    ("Escherichia coli", "LEV"): (2.0, "high", "Levofloxacin, Enterobacterales"),
    ("Klebsiella pneumoniae", "AMC"): (8.0, "moderate", "Amoxicillin/clavulanate, legacy pre-2022, amoxicillin component"),
    ("Klebsiella pneumoniae", "AXO"): (1.0, "high", "Ceftriaxone, Enterobacterales"),
    ("Klebsiella pneumoniae", "FIX"): (1.0, "moderate", "Cefixime, Enterobacterales"),
    ("Klebsiella pneumoniae", "SXT"): (2.0, "high", "Trimethoprim/sulfamethoxazole, trimethoprim component, Enterobacterales"),
    ("Klebsiella pneumoniae", "LEV"): (2.0, "high", "Levofloxacin, Enterobacterales"),
    ("Klebsiella pneumoniae", "PEN"): (None, "n/a", "No CLSI breakpoint: penicillin not clinically indicated for Enterobacterales"),
}


def get_breakpoint_table():
    rows = []
    for (org, ab), (bp, conf, note) in BREAKPOINTS_S.items():
        rows.append({
            "Organism": org, "Antibiotic": ab,
            "Antibiotic (name)": ANTIBIOTIC_FULL_NAMES.get(ab, ab),
            "S breakpoint (mg/L)": bp, "Confidence": conf, "Note": note,
        })
    return pd.DataFrame(rows)


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    t = get_breakpoint_table()
    out = TABLES_DIR / "t0_breakpoint_reference.csv"
    t.to_csv(out, index=False)
    print(f"Saved breakpoint reference table ({len(t)} pairs): {out}")
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
