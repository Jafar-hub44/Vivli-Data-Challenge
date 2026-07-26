"""
17_confidence_intervals.py
------------------------------
STAGE 4 - PART 1: Confidence intervals for headline Stage 3 metrics

For every (Organism, Antibiotic, Stratum) combination that appeared in a
Stage 3 headline finding (country case studies, hidden-signal case studies,
and temporal trend strata), computes appropriate confidence intervals:

  - % resistant (breakpoint exceedance): Wilson score interval (primary)
    and Clopper-Pearson exact interval (conservative alternative), both
    standard, well-justified binomial-proportion CIs.
  - Geometric mean MIC: nonparametric percentile bootstrap (2,000
    resamples of isolates WITH replacement, boundary substitution applied
    identically inside every resample) - appropriate because MIC data are
    discrete, ties are common, and no parametric distributional shape is
    assumed.
  - MIC50 / MIC90: nonparametric percentile bootstrap (same resampling
    scheme), because these CLSI-convention order statistics have no simple
    closed-form CI once ties and censoring are present.
  - Modal MIC: NOT a metric with a standard CI. Instead, "mode stability"
    is reported - the percentage of bootstrap resamples in which the same
    value remains the most frequent - as an honest, appropriately-hedged
    substitute for a conventional interval.

All computations reuse the exact same boundary-substitution and MIC50/90
(CLSI cumulative-%) conventions used throughout Stages 2B-3, so that Stage 4
is testing Stage 3's actual point estimates, not new metric definitions.
"""

import sys
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import ANTIBIOTIC_FULL_NAMES  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "cleaned" / "soar_stage1_standardized.csv"
TABLES_DIR = Path(__file__).resolve().parents[1] / "tables"

N_BOOT = 2000
RNG_SEED = 20260724  # fixed seed for full reproducibility of bootstrap CIs


def deterministic_seed(*parts, base=RNG_SEED, modulus=10_000):
    """Derive a reproducible per-stratum seed offset from string parts.

    REPRODUCIBILITY FIX (post-audit): the previous implementation used
    Python's built-in hash() on a tuple of strings. hash() on str/tuple
    objects is randomized per-process (via PYTHONHASHSEED) unless that
    environment variable is explicitly fixed, so the previous code produced
    a DIFFERENT per-stratum seed - and therefore different bootstrap CI
    bounds and mode-stability values - on every fresh Python invocation,
    contradicting this project's reproducibility claims. This was confirmed
    directly: re-running the script twice produced different values in
    13 of 23 rows of the output table.

    hashlib.md5 (or any hashlib digest) is stable across processes and
    Python versions for a given input, unlike the built-in hash(), so it is
    used here instead. This is NOT a cryptographic use case - md5 is used
    purely as a fast, deterministic, well-distributed integer generator.
    """
    key = "|".join(str(p) for p in parts).encode("utf-8")
    digest_int = int(hashlib.md5(key).hexdigest(), 16)
    return base + (digest_int % modulus)

# Breakpoint reference (re-import from Stage 3 without duplicating the dict)
import importlib.util
_bp_spec = importlib.util.spec_from_file_location(
    "bp_ref", Path(__file__).resolve().parent / "12_breakpoint_reference.py"
)
bp_ref = importlib.util.module_from_spec(_bp_spec)
_bp_spec.loader.exec_module(bp_ref)
BREAKPOINTS_S = bp_ref.BREAKPOINTS_S

# --- Headline strata carried forward from Stage 3 (country domain) ---------
COUNTRY_STRATA = [
    ("Haemophilus influenzae", "AXO", "Philippines"),
    ("Haemophilus influenzae", "AXO", "Vietnam"),
    ("Streptococcus pneumoniae", "AXO", "Argentina"),
    ("Streptococcus pneumoniae", "AXO", "Tunisia"),
    ("Streptococcus pneumoniae", "AMX", "Argentina"),
    ("Streptococcus pneumoniae", "AMX", "Kenya"),
    ("Streptococcus pneumoniae", "AMC", "Argentina"),
    ("Streptococcus pneumoniae", "AMC", "Kenya"),
    ("Haemophilus influenzae", "CLA", "Chile"),
    ("Haemophilus influenzae", "CLA", "Vietnam"),
    ("Haemophilus influenzae", "AMC", "Tunisia"),
    ("Haemophilus influenzae", "AMC", "Ukraine"),
    ("Haemophilus influenzae", "POD", "Tunisia"),
    ("Haemophilus influenzae", "POD", "Ukraine"),
]

# --- Headline strata carried forward from Stage 3 (temporal domain) --------
TEMPORAL_STRATA = [
    (org, ab, year)
    for org, ab in [("Haemophilus influenzae", "AXO"),
                     ("Haemophilus influenzae", "CDN"),
                     ("Haemophilus influenzae", "AMC")]
    for year in (2016, 2017, 2018)
]


def mic_at_cumulative_pct(sorted_values, pct):
    n = len(sorted_values)
    idx = int(np.ceil(pct / 100 * n)) - 1
    idx = min(max(idx, 0), n - 1)
    return float(sorted_values[idx])


def geometric_mean(values):
    return float(2 ** np.mean(np.log2(values)))


def bootstrap_ci(values, statistic_fn, n_boot=N_BOOT, rng=None, alpha=0.05):
    rng = rng or np.random.default_rng(RNG_SEED)
    n = len(values)
    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_stats[i] = statistic_fn(sample)
    lo = np.percentile(boot_stats, 100 * alpha / 2)
    hi = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return lo, hi, boot_stats


def mode_stability(values, rng, n_boot=N_BOOT):
    n = len(values)
    original_mode = stats.mode(values, keepdims=False).mode
    matches = 0
    for _ in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        vals, counts = np.unique(sample, return_counts=True)
        boot_mode = vals[np.argmax(counts)]
        if boot_mode == original_mode:
            matches += 1
    return float(original_mode), round(100 * matches / n_boot, 1)


def compute_ci_row(df, organism, ab, country=None, year=None):
    val_col, cen_col = f"{ab}_value", f"{ab}_censor"
    sub = df[df["Organism"] == organism]
    if country is not None:
        sub = sub[sub["Country"] == country]
        stratum_label = country
        stratum_type = "country"
    else:
        sub = sub[sub["Collection_Year"] == year]
        stratum_label = str(int(year))
        stratum_type = "year"
    sub = sub[sub[cen_col].isin(["exact", "left", "right"])]
    values = sub[val_col].to_numpy(dtype=float)  # boundary substitution
    n = len(values)
    if n == 0:
        return None

    bp, conf, _ = BREAKPOINTS_S.get((organism, ab), (None, "n/a", ""))
    n_r = int(np.sum(values > bp)) if bp is not None else None

    rng = np.random.default_rng(deterministic_seed(organism, ab, stratum_label))

    row = {
        "Organism": organism, "Antibiotic": ab,
        "Antibiotic (name)": ANTIBIOTIC_FULL_NAMES[ab],
        "Stratum type": stratum_type, "Stratum": stratum_label, "N": n,
    }

    # --- %R: Wilson + Clopper-Pearson ---
    if bp is not None:
        pct_r = round(100 * n_r / n, 2)
        wilson_lo, wilson_hi = proportion_confint(n_r, n, alpha=0.05, method="wilson")
        cp_lo, cp_hi = proportion_confint(n_r, n, alpha=0.05, method="beta")
        wilson_lo, wilson_hi = max(0.0, wilson_lo), min(1.0, wilson_hi)
        cp_lo, cp_hi = max(0.0, cp_lo), min(1.0, cp_hi)
        row.update({
            "Breakpoint (mg/L)": bp, "Breakpoint confidence": conf,
            "%R (point estimate)": pct_r, "n_R": n_r,
            "%R Wilson 95% CI low": round(100 * wilson_lo, 2),
            "%R Wilson 95% CI high": round(100 * wilson_hi, 2),
            "%R Clopper-Pearson 95% CI low": round(100 * cp_lo, 2),
            "%R Clopper-Pearson 95% CI high": round(100 * cp_hi, 2),
        })
    else:
        row.update({"Breakpoint (mg/L)": None, "%R (point estimate)": None})

    # --- Geometric mean: bootstrap percentile CI ---
    gmean_point = geometric_mean(values)
    lo, hi, _ = bootstrap_ci(values, geometric_mean, rng=rng)
    row.update({
        "GeoMean (point estimate, mg/L)": round(gmean_point, 4),
        "GeoMean bootstrap 95% CI low": round(lo, 4),
        "GeoMean bootstrap 95% CI high": round(hi, 4),
    })

    # --- MIC50 / MIC90: bootstrap percentile CI ---
    mic50_point = mic_at_cumulative_pct(np.sort(values), 50)
    mic90_point = mic_at_cumulative_pct(np.sort(values), 90)
    lo50, hi50, _ = bootstrap_ci(values, lambda s: mic_at_cumulative_pct(np.sort(s), 50), rng=rng)
    lo90, hi90, _ = bootstrap_ci(values, lambda s: mic_at_cumulative_pct(np.sort(s), 90), rng=rng)
    row.update({
        "MIC50 (point estimate)": mic50_point,
        "MIC50 bootstrap 95% CI": f"[{lo50:.4g}, {hi50:.4g}]",
        "MIC90 (point estimate)": mic90_point,
        "MIC90 bootstrap 95% CI": f"[{lo90:.4g}, {hi90:.4g}]",
    })

    # --- Modal MIC: bootstrap mode-stability ---
    modal_val, stability_pct = mode_stability(values, rng)
    row.update({"Modal MIC (point estimate)": modal_val,
                "Modal MIC bootstrap stability (%)": stability_pct})

    return row


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CLEANED_DATA_PATH)

    rows = []
    for organism, ab, country in COUNTRY_STRATA:
        r = compute_ci_row(df, organism, ab, country=country)
        if r:
            rows.append(r)
    for organism, ab, year in TEMPORAL_STRATA:
        r = compute_ci_row(df, organism, ab, year=year)
        if r:
            rows.append(r)

    result = pd.DataFrame(rows)
    out_path = TABLES_DIR / "t1_confidence_intervals.csv"
    result.to_csv(out_path, index=False)
    print(f"Computed CIs for {len(result)} strata ({N_BOOT} bootstrap replicates each)")
    print(f"Saved: {out_path}")
    with pd.option_context("display.width", 260):
        print(result[["Organism", "Antibiotic", "Stratum", "N", "%R (point estimate)",
                       "%R Wilson 95% CI low", "%R Wilson 95% CI high",
                       "GeoMean (point estimate, mg/L)", "GeoMean bootstrap 95% CI low",
                       "GeoMean bootstrap 95% CI high", "Modal MIC bootstrap stability (%)"]].to_string(index=False))


if __name__ == "__main__":
    main()
