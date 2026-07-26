"""
20_multiple_comparisons.py
------------------------------
STAGE 4 - PART 4: Multiple comparisons correction

Pools EVERY p-value actually generated in Stage 4 (7 country-pair
comparisons x 3 tests each that produce a p-value [Fisher exact,
permutation-GeoMean, Mann-Whitney] = 21 p-values, plus 3 temporal
Cochran-Armitage + 3 temporal permutation-trend p-values = 6 more;
27 p-values total) and applies three correction procedures:

  - Bonferroni: controls family-wise error rate (FWER) via a fixed
    alpha/m threshold. Most conservative; guards against ANY false
    positive among all tests.
  - Holm: also controls FWER, but sequentially, and is uniformly more
    powerful than Bonferroni while retaining the same FWER guarantee.
  - Benjamini-Hochberg (FDR): controls the expected PROPORTION of false
    discoveries among rejected hypotheses, not the probability of any
    false positive at all. Substantially more powerful than either FWER
    method when many tests are run, at the cost of tolerating some
    false positives among the discoveries.

The three are compared directly (how many findings remain "significant"
under each), and a recommendation is made for which is most appropriate
given this project's exploratory-surveillance context.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

TABLES_DIR = Path(__file__).resolve().parents[1] / "tables"


def bonferroni(p_values, alpha=0.05):
    m = len(p_values)
    return np.array(p_values) < (alpha / m), np.minimum(np.array(p_values) * m, 1.0)


def holm(p_values, alpha=0.05):
    p = np.asarray(p_values)
    m = len(p)
    order = np.argsort(p)
    sorted_p = p[order]
    adj = np.empty(m)
    running_max = 0.0
    for i in range(m):
        val = min((m - i) * sorted_p[i], 1.0)
        running_max = max(running_max, val)
        adj[i] = running_max
    adj_p = np.empty(m)
    adj_p[order] = adj
    reject = adj_p < alpha
    return reject, adj_p


def benjamini_hochberg(p_values, alpha=0.05):
    p = np.asarray(p_values)
    m = len(p)
    order = np.argsort(p)
    sorted_p = p[order]
    adj = np.empty(m)
    running_min = 1.0
    for i in range(m - 1, -1, -1):
        val = min(sorted_p[i] * m / (i + 1), 1.0)
        running_min = min(running_min, val)
        adj[i] = running_min
    adj_p = np.empty(m)
    adj_p[order] = adj
    reject = adj_p < alpha
    return reject, adj_p


def main():
    country = pd.read_csv(TABLES_DIR / "t2_country_comparison_tests.csv")
    temporal = pd.read_csv(TABLES_DIR / "t3_temporal_trend_tests.csv")

    tests = []
    for _, r in country.iterrows():
        base = {"Domain": "country", "Source": r["Source"], "Organism": r["Organism"], "Antibiotic": r["Antibiotic"],
                "Comparison": f"{r['Country A']} vs {r['Country B']}"}
        tests.append({**base, "Test": "Fisher exact (%R)", "p_value": r["Fisher exact p-value"]})
        tests.append({**base, "Test": "Permutation (GeoMean, PRIMARY)", "p_value": r["Permutation p-value (GeoMean, PRIMARY)"]})
        tests.append({**base, "Test": "Mann-Whitney U (secondary)", "p_value": r["Mann-Whitney U p-value (secondary)"]})
    for _, r in temporal.iterrows():
        base = {"Domain": "temporal", "Source": "Stage3 Part2", "Organism": r["Organism"], "Antibiotic": r["Antibiotic"],
                "Comparison": "2016-2018 trend"}
        if not pd.isna(r["Cochran-Armitage p-value (%R trend)"]):
            tests.append({**base, "Test": "Cochran-Armitage (%R trend)", "p_value": r["Cochran-Armitage p-value (%R trend)"]})
        tests.append({**base, "Test": "Permutation (GeoMean trend, PRIMARY)", "p_value": r["Permutation p-value (GeoMean trend, PRIMARY)"]})

    result = pd.DataFrame(tests)
    p_values = result["p_value"].to_numpy()

    bonf_reject, bonf_adj = bonferroni(p_values)
    holm_reject, holm_adj = holm(p_values)
    bh_reject, bh_adj = benjamini_hochberg(p_values)

    result["Bonferroni adj. p"] = np.round(bonf_adj, 5)
    result["Bonferroni significant (a=0.05)"] = bonf_reject
    result["Holm adj. p"] = np.round(holm_adj, 5)
    result["Holm significant (a=0.05)"] = holm_reject
    result["BH-FDR adj. p (q-value)"] = np.round(bh_adj, 5)
    result["BH-FDR significant (a=0.05)"] = bh_reject

    out_path = TABLES_DIR / "t4_multiple_comparisons.csv"
    result.to_csv(out_path, index=False)

    print(f"Total tests pooled for multiple-comparison correction: {len(result)}")
    print(f"Raw (uncorrected) significant at a=0.05: {(p_values < 0.05).sum()} / {len(result)}")
    print(f"Bonferroni significant: {bonf_reject.sum()} / {len(result)} (alpha/m = {0.05/len(result):.5f})")
    print(f"Holm significant: {holm_reject.sum()} / {len(result)}")
    print(f"BH-FDR significant: {bh_reject.sum()} / {len(result)}")
    print(f"\nSaved: {out_path}")

    with pd.option_context("display.width", 240):
        print("\n=== Full comparison ===")
        print(result[["Domain", "Organism", "Antibiotic", "Comparison", "Test", "p_value",
                       "Bonferroni significant (a=0.05)", "Holm significant (a=0.05)",
                       "BH-FDR significant (a=0.05)"]].to_string(index=False))


if __name__ == "__main__":
    main()
