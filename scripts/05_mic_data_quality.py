"""
05_mic_data_quality.py
-------------------------
STAGE ONE - PART 4: MIC Data Quality

For every antibiotic, compute (from the standardized numeric MIC values):
  - Number tested (non-missing)
  - Number missing
  - Number exact MICs
  - Number left-censored (</=)
  - Number right-censored (>)
  - Minimum MIC, Maximum MIC (of reported dilution values, censoring noted)
  - Number of distinct dilution steps observed
  - Percentage censored

Explicitly does NOT calculate MIC50, MIC90, or any other surveillance
metric - those belong to a later stage.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    CLEANED_DATA_PATH,
    TABLES_DIR,
    FIGURES_DIR,
    ANTIBIOTIC_VARS,
    ANTIBIOTIC_FULL_NAMES,
    set_publication_style,
    PALETTE,
    ensure_dirs,
)


def main():
    ensure_dirs()
    df = pd.read_csv(CLEANED_DATA_PATH)
    n = len(df)

    rows = []
    for ab in ANTIBIOTIC_VARS:
        val_col = f"{ab}_value"
        cen_col = f"{ab}_censor"
        n_missing = (df[cen_col] == "missing").sum()
        n_tested = n - n_missing
        n_exact = (df[cen_col] == "exact").sum()
        n_left = (df[cen_col] == "left").sum()
        n_right = (df[cen_col] == "right").sum()
        n_unparsed = (df[cen_col] == "unparsed").sum()

        tested_vals = df.loc[df[cen_col].isin(["exact", "left", "right"]), val_col]
        min_mic = tested_vals.min() if len(tested_vals) else None
        max_mic = tested_vals.max() if len(tested_vals) else None
        n_dilution_steps = tested_vals.nunique()

        pct_censored = round(100 * (n_left + n_right) / n_tested, 2) if n_tested else 0.0

        rows.append(
            {
                "Antibiotic (code)": ab,
                "Antibiotic (name)": ANTIBIOTIC_FULL_NAMES[ab],
                "Number tested": int(n_tested),
                "Number missing": int(n_missing),
                "Number exact MICs": int(n_exact),
                "Number left-censored (</=)": int(n_left),
                "Number right-censored (>)": int(n_right),
                "Number unparsed values": int(n_unparsed),
                "Minimum MIC (mg/L)": min_mic,
                "Maximum MIC (mg/L)": max_mic,
                "Distinct dilution values observed": int(n_dilution_steps),
                "Percent censored (%)": pct_censored,
            }
        )

    mic_qual = pd.DataFrame(rows).sort_values("Antibiotic (code)")
    out_path = TABLES_DIR / "table4_mic_quality_summary.csv"
    mic_qual.to_csv(out_path, index=False)

    with pd.option_context("display.width", 220):
        print("=== MIC Quality Summary Table ===")
        print(mic_qual.to_string(index=False))
    print(f"\nSaved: {out_path}")

    # --- Figure: censoring composition per antibiotic ----------------------
    set_publication_style()
    fig, ax = plt.subplots(figsize=(11, 6))
    order = mic_qual.sort_values("Percent censored (%)", ascending=False)["Antibiotic (code)"]
    exact_pct = []
    left_pct = []
    right_pct = []
    for ab in order:
        r = mic_qual[mic_qual["Antibiotic (code)"] == ab].iloc[0]
        tested = r["Number tested"]
        exact_pct.append(100 * r["Number exact MICs"] / tested if tested else 0)
        left_pct.append(100 * r["Number left-censored (</=)"] / tested if tested else 0)
        right_pct.append(100 * r["Number right-censored (>)"] / tested if tested else 0)

    x = range(len(order))
    ax.bar(x, exact_pct, label="Exact", color=PALETTE[0])
    ax.bar(x, left_pct, bottom=exact_pct, label="Left-censored (</=)", color=PALETTE[1])
    bottom2 = [e + l for e, l in zip(exact_pct, left_pct)]
    ax.bar(x, right_pct, bottom=bottom2, label="Right-censored (>)", color=PALETTE[4])
    ax.set_xticks(list(x))
    ax.set_xticklabels(order, rotation=45, ha="right")
    ax.set_ylabel("Percentage of tested isolates (%)")
    ax.set_title("MIC Censoring Composition by Antibiotic\nGSK SOAR 2019-10 raw data extract (descriptive only)")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig_path = FIGURES_DIR / "fig_mic_censoring_summary.png"
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
