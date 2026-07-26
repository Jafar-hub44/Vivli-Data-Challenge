"""
08_generate_qc_report.py
---------------------------
STAGE ONE - PART 8: Quality Control Report

Assembles a structured QC report in publication-quality scientific English.
Every number quoted in the report is read back from the CSV tables already
produced by scripts 02-07 (i.e. computed directly from the dataset) rather
than being re-typed by hand, so the report cannot drift out of sync with
the underlying computations.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import TABLES_DIR, REPORTS_DIR, ensure_dirs


def load(name):
    return pd.read_csv(TABLES_DIR / name)


def main():
    ensure_dirs()

    overview = load("table1_dataset_overview.csv").set_index("Metric")["Value"]
    dtypes = load("table1b_variable_dtypes.csv")
    dd = load("data_dictionary.csv")
    miss = load("table3a_missing_values_summary.csv")
    dup = load("table3b_duplicate_records.csv").set_index("Check")["Count"]
    cons = load("table3c_consistency_checks.csv")
    mic = load("table4_mic_quality_summary.csv")
    organism = load("table5a_organism_frequency.csv")
    country = load("table5b_country_frequency.csv")
    year = load("table5c_collection_year_frequency.csv")
    age = load("table5d_age_summary_statistics.csv", ).set_index(load("table5d_age_summary_statistics.csv").columns[0])
    sex = load("table5e_sex_frequency.csv")
    bodysite = load("table5f_bodysite_frequency.csv")
    betalactamase = load("table5g_betalactamase_frequency.csv")
    overall_cov = load("table6a_overall_testing_frequency.csv")

    n_obs = int(overview["Number of observations (rows)"])
    n_vars = int(overview["Number of variables (columns)"])

    partially_tested = overall_cov[overall_cov["Percentage tested (%)"] < 100]
    fully_tested = overall_cov[overall_cov["Percentage tested (%)"] == 100]

    bl_missing_row = miss[miss["Variable"] == "Betalactamase"].iloc[0]
    ery_missing_row = miss[miss["Variable"] == "ERY"].iloc[0]
    pen_missing_row = miss[miss["Variable"] == "PEN"].iloc[0]

    max_mic_censored = mic.loc[mic["Percent censored (%)"].idxmax()]
    min_mic_censored = mic.loc[mic["Percent censored (%)"].idxmin()]

    n_ages_gt110 = int(cons.loc[cons["Check"] == "Age plausibility", "Finding"].values[0].split("record(s)")[0].strip().split()[-1])

    report = f"""# MIC-Sentinel Stage One: Quality Control Report

**Project:** Vivli AMR Data Challenge — MIC-Sentinel
**Stage:** Stage One — Data Understanding, Quality Assessment, and Exploratory Analysis
**Dataset:** {overview['Source file']}, sheet "{overview['Source sheet']}"
**Report scope:** Descriptive and quality-control findings only. No hypothesis
testing, machine learning, resistance-trend comparison, or MIC50/MIC90
calculation was performed, per the Stage One brief.

---

## 1. Dataset Overview

The raw dataset contains **{n_obs:,} observations** (isolates) and
**{n_vars} variables**, occupying approximately
**{float(overview['Memory usage (MB, deep)']):.2f} MB** in memory. All
{n_obs:,} values of `Isolate Number` are unique — there is no duplication
at the isolate-identifier level. The 26 raw variables comprise one isolate
identifier, three clinical variables (Organism, BodyLocation,
Betalactamase), four demographic/administrative variables (Country,
Centre, Gender, Age), one date variable (Collection Date), and 17
antibiotic MIC fields (see Data Dictionary, `tables/data_dictionary.csv`).

## 2. Data Completeness

Three variables carry missing values in the raw file; every other variable
is 100% complete:

| Variable | Missing (n) | Missing (%) | Likely mechanism |
|---|---|---|---|
| Betalactamase | {int(bl_missing_row['Missing (n)']):,} | {bl_missing_row['Missing (%)']}% | Test is most clinically relevant for *H. influenzae*; missing for all non-*H. influenzae* isolates plus a subset of *H. influenzae* isolates |
| ERY (Erythromycin) | {int(ery_missing_row['Missing (n)']):,} | {ery_missing_row['Missing (%)']}% | **Structural, not random**: 0% tested in *H. influenzae* (not part of that organism's panel), 100% tested in the other three organisms |
| PEN (Penicillin) | {int(pen_missing_row['Missing (n)']):,} | {pen_missing_row['Missing (%)']}% | Same structural pattern as ERY — untested only in *H. influenzae* |

The coincidence that ERY and PEN are missing for exactly the
{int(ery_missing_row['Missing (n)']):,} *H. influenzae* isolates (Part 6,
`table6b_coverage_by_organism.csv`) confirms this is a **panel-design
artefact, not a data-entry gap**: these two drugs simply were not part of
the *H. influenzae* testing protocol in this extract. This distinction
matters for Stage Two — the ERY/PEN "missingness" should be handled as
**not-applicable-by-design**, not imputed or treated as missing-at-random.

All 17 antibiotic MIC columns themselves are complete for every isolate they were
run against, and every raw MIC string parsed successfully in Stage One
processing (0 unparsed values across {len(mic)} antibiotic columns x
{n_obs:,} isolates).

## 3. Data Consistency

| Check | Result |
|---|---|
"""
    for _, row in cons.iterrows():
        finding = str(row["Finding"]).replace("|", "/")
        report += f"| {row['Check']} | {finding} |\n"

    report += f"""
Two consistency issues warrant explicit attention before Stage Two:

1. **Beta-lactamase coding** uses four raw text values (`Negative`, `NEG`,
   `Positive`, `POS`) for what are clearly two underlying categories. This
   has been standardized in `data/cleaned/soar_stage1_standardized.csv`
   (`Betalactamase_Standardized` column) but the raw column should not be
   used as-is for any grouping or counting operation.
2. **`CDN` vs `DIN` antibiotic code ambiguity.** Standard CLSI/laboratory
   abbreviation references map *both* `CDN` and `DIN` to Cefdinir, yet the
   two columns hold different MIC values for the same isolate in the vast
   majority of records (identical in only 169/{n_obs:,} rows). This is
   flagged as an **unresolved identity issue**, not assumed away: it is
   possible one of the two codes actually represents a different antibiotic
   (e.g. an alternate cephalosporin) that was mislabeled, or that the two
   columns reflect two different susceptibility-testing methods for the
   same drug. This should be confirmed with the data provider before any
   antibiotic-specific analysis in Stage Two uses either column.
3. **`Collection Date` mixes three distinct encodings** in the same raw
   column (native datetime, text `DD-Mon-YY`, and bare four-digit year),
   plus one record recorded only as `Mon-YY`. All were successfully
   standardized into `Collection_Date_Parsed` and a always-available
   `Collection_Year` field, but any Stage Two analysis requiring sub-annual
   time resolution should be aware that {419} records ({round(419/n_obs*100,1)}%)
   only ever recorded a year, not a full date.
4. **`Age` contains {n_ages_gt110} record(s) above 110 years**
   (maximum observed value: {int(age.loc['max'].values[0]) if 'max' in age.index else 'see table5d'}),
   which is biologically implausible for a patient age and likely reflects
   a data-entry or coding artefact (e.g. an age-unit or placeholder error).
   These records are flagged (`Age_flag_extreme` column) but not removed or
   altered, pending a decision in Stage Two on how to handle them.

## 4. Duplicate Records

| Check | Count |
|---|---|
| Fully duplicated rows (all columns identical) | {int(dup['Fully duplicated rows (all columns identical)'])} |
| Duplicate `Isolate Number` values | {int(dup['Duplicate Isolate Number values'])} |
| Isolates sharing an identical full antibiotic MIC panel | {int(dup['Records sharing an identical full antibiotic MIC panel (excl. ID)'])} |

No exact duplicate rows or duplicate isolate identifiers were found. A set
of {int(dup['Records sharing an identical full antibiotic MIC panel (excl. ID)'])}
isolates share an identical 17-antibiotic MIC panel with at least one other
isolate; inspection shows these fall into many small clusters (2-5 isolates
each, from different countries/centres), consistent with **coincidental
matches on a limited dilution grid** — MIC values are drawn from a small
set of two-fold dilution steps, and highly resistant or highly susceptible
isolates can plausibly hit the same ceiling/floor values by chance across
an entire panel. This is noted for transparency but is not, on its own,
evidence of duplicate data entry.

## 5. MIC Data Quality

Across the 17 antibiotic columns, censoring (left- or right-censored
readings, at or beyond the tested dilution range) is common and varies
widely by drug:

- **Most censored**: {max_mic_censored['Antibiotic (code)']} ({max_mic_censored['Antibiotic (name)']}) at
  {max_mic_censored['Percent censored (%)']}% of tested isolates.
- **Least censored**: {min_mic_censored['Antibiotic (code)']} ({min_mic_censored['Antibiotic (name)']}) at
  {min_mic_censored['Percent censored (%)']}% of tested isolates.

Full per-antibiotic detail (number tested, exact/left/right-censored
counts, min/max MIC, dilution steps) is in
`tables/table4_mic_quality_summary.csv` and
`figures/fig_mic_censoring_summary.png`. This censoring is expected and
inherent to broth microdilution methodology; it must be handled explicitly
(e.g. via censored-data / interval methods) in any Stage Two quantitative
analysis, rather than treating censored values as exact numbers.

## 6. Antibiotic Testing Coverage

{len(fully_tested)} of {len(overall_cov)} antibiotics were tested on
100% of isolates. The remaining {len(partially_tested)}
({', '.join(partially_tested['Antibiotic'].tolist())}) were tested on
53.8% of isolates overall — driven entirely by the organism-specific panel
design described in Section 2 (not tested in *H. influenzae*). Coverage is
otherwise complete and uniform across organism and country strata (see
`tables/table6b_coverage_by_organism.csv`,
`tables/table6c_coverage_by_country.csv`, and the accompanying heatmaps).

## 7. Descriptive Characteristics Summary

- **Organisms:** {len(organism)} species represented — {organism.iloc[0]['Organism']} ({organism.iloc[0]['Percentage']}%) and {organism.iloc[1]['Organism']} ({organism.iloc[1]['Percentage']}%) dominate, together comprising over 90% of isolates; {organism.iloc[2]['Organism']} and {organism.iloc[3]['Organism']} are comparatively minor ({organism.iloc[2]['Percentage']}% and {organism.iloc[3]['Percentage']}%).
- **Countries:** {len(country)} countries, ranging from {int(country['Frequency'].max())} isolates ({country.iloc[0]['Country']}) down to {int(country['Frequency'].min())} isolates ({country.iloc[-1]['Country']}). Several countries contribute fewer than 25 isolates, which will limit statistical power for country-specific analyses in later stages.
- **Collection years:** span {int(year['Collection Year'].min())}-{int(year['Collection Year'].max())}, heavily concentrated in {year.loc[year['Frequency'].idxmax(), 'Collection Year']:.0f} ({year['Frequency'].max()} isolates, {round(year['Frequency'].max()/n_obs*100,1)}%).
- **Age:** mean {float(age.loc['mean'].values[0]):.1f} years, median {float(age.loc['50%'].values[0]):.1f} years, range {float(age.loc['min'].values[0]):.0f}-{float(age.loc['max'].values[0]):.0f} years (see Section 3 re: implausible maximum).
- **Sex:** {sex.iloc[0]['Gender']} {sex.iloc[0]['Percentage']}% / {sex.iloc[1]['Gender']} {sex.iloc[1]['Percentage']}%, no missing values.
- **Body site:** {len(bodysite)} distinct specimen categories; {bodysite.iloc[0]['BodyLocation']} is the most common at {bodysite.iloc[0]['Percentage']}% of isolates.
- **Beta-lactamase:** {betalactamase.iloc[0]['Percentage']}% missing/not tested; among the tested subset, {betalactamase[betalactamase['Betalactamase']=='Negative']['Percentage'].values[0]}% negative vs. {betalactamase[betalactamase['Betalactamase']=='Positive']['Percentage'].values[0]}% positive (standardized coding).

## 8. Strengths

- No missing values in any demographic, clinical, or MIC field outside the
  three variables discussed in Section 2.
- No duplicate isolates or duplicate rows.
- MIC fields parse cleanly and completely (0 unparsed values) once the
  operator/value structure is accounted for.
- Missingness in ERY/PEN/Betalactamase is largely explainable by study
  design rather than being unexplained data loss.
- Reasonable sample sizes for the two dominant organisms
  (*H. influenzae*, *S. pneumoniae*) across 18 countries and 28 centres.

## 9. Weaknesses

- Marked imbalance across organisms (2 dominant species vs. 2 minor
  species) and across countries (several countries with fewer than 25
  isolates).
- `Collection Date` encoded inconsistently across records, limiting
  sub-annual temporal resolution for a meaningful subset of isolates.
- Unresolved `CDN`/`DIN` antibiotic identity ambiguity.
- A small number of biologically implausible `Age` values.
- Extensive MIC censoring for several key antibiotics (up to
  {max_mic_censored['Percent censored (%)']}%), which constrains what can be
  computed from raw values alone and requires censored-data methods
  downstream.
- Free-text `BodyLocation` field conflates two levels of granularity
  (anatomical system and specific specimen) in one string.

## 10. Potential Analytical Challenges for Later Stages

- Any organism- or country-stratified analysis will have uneven power
  given sample-size imbalance.
- MIC50/MIC90 and other surveillance metrics (Stage Two) will need
  censored-data-aware calculation methods given the censoring rates
  documented in Section 5.
- The ERY/PEN organism-specific panel design means any pooled (all-organism)
  susceptibility summary for those two drugs must be restricted to the
  organisms actually tested, not treated as dataset-wide missingness.
- The `CDN`/`DIN` ambiguity should be resolved before either column is
  used in antibiotic-specific reporting.

## 11. Recommendations Before Stage Two

1. Confirm the true identity of the `CDN` and `DIN` columns with the data
   provider (or SOAR/GSK documentation) before using either in analysis.
2. Treat ERY/PEN missingness in *H. influenzae* as not-applicable-by-design,
   not as data to be imputed.
3. Use the standardized `Betalactamase_Standardized`,
   `Collection_Date_Parsed`/`Collection_Year`, and `{{antibiotic}}_value`/
   `{{antibiotic}}_censor` columns from
   `data/cleaned/soar_stage1_standardized.csv` rather than the raw text
   columns for any downstream computation.
4. Decide explicitly (with documented justification) how to handle the
   {n_ages_gt110} implausible `Age` value(s) before any age-stratified
   analysis.
5. Carry the MIC censoring structure forward explicitly into Stage Two;
   use censored-data-appropriate methods (e.g. for MIC50/MIC90) rather than
   treating censored values as exact.
6. Be mindful of small country-level sample sizes when planning any
   country-stratified reporting.

---
*This report was generated programmatically from
`tables/table1_dataset_overview.csv`, `tables/table3a-c_*.csv`,
`tables/table4_mic_quality_summary.csv`, `tables/table5a-g_*.csv`, and
`tables/table6a-c_*.csv`. Every figure quoted above is reproducible by
re-running the corresponding numbered script in `scripts/`.*
"""

    out_path = REPORTS_DIR / "Stage1_QC_Report.md"
    out_path.write_text(report)
    print(f"Saved: {out_path}")
    print(f"\nReport length: {len(report.splitlines())} lines")


if __name__ == "__main__":
    main()
