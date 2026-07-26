# Data Access Note

This repository contains the fully de-identified, **derived** analysis dataset
(`data/cleaned/soar_stage1_standardized.csv`) and every table/figure produced from it. It
does **not** contain the original raw vendor file (`GSK_SOAR_201910_raw_data.xlsx`).

**Why:** the raw SOAR dataset (2019-10 extract) was contributed by GSK to the Vivli
platform and accessed under the data use agreement governing the Vivli AMR Surveillance
Data Challenge. Redistribution of that raw file outside the Vivli platform is governed by
that agreement, not by this project.

**If you want the raw file:** request access directly through Vivli (https://vivli.org).

**If you want to verify or reproduce this analysis:** `data/cleaned/soar_stage1_standardized.csv`
retains every original isolate-level field alongside the additive standardization columns
described in the manuscript's Methods section, and is sufficient to independently re-run
every script numbered 09 and above in `scripts/` (scripts 01-08 perform the raw-to-cleaned
transformation itself and require the raw file — see `README.md`).
