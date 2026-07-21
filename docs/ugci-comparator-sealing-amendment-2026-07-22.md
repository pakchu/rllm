# UGCI-288 comparator sealing amendment

## Purpose

The UGCI-288 preregistration fixed novelty comparisons against AMTR-48,
SQFD-6, SDDR-12, and UCBR-12. One bound SQFD clock physically extends beyond
the 2023 selection boundary even though the comparison interval itself ends at
`2024-01-01T00:00:00Z`.

Before opening UGCI source incidence, this amendment freezes one timestamp-only
bundle containing only the preregistered candidate/control rows inside each
fixed half-open comparison interval:

- clock: `results/ugci_prior_comparator_views_pre2024_2026-07-22.csv.gz`
- manifest: `results/ugci_prior_comparator_views_pre2024_manifest_2026-07-22.json`
- builder: `training/freeze_ugci_prior_comparator_views.py`

The support evaluator must consume that sealed bundle and must not reopen the
original comparator files.

## What does not change

- UGCI signal definition, direction, latency, holding period, and leverage
- train/selection dates
- support thresholds
- comparator identities, controls, entry timestamps, or comparison intervals
- failure action

The sanitizer does not read UGCI source rows, BTC market data, funding, labels,
returns, PnL, CAGR, or MDD. It reads the already frozen comparator clock files
only to create the interval-contained physical view. No post-2023 comparator row
is retained in the output.
