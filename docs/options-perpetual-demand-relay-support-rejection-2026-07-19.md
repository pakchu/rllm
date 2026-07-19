# OPDR-24 source-only support rejection — 2026-07-19

## Verdict

`OPDR-24` was rejected before any BTCUSDT execution price, return, funding cash
flow, or strategy PnL was opened. The singleton policy passed every frozen
novelty check, but failed four preregistered support checks:

1. Test events were concentrated in one month: **15 of 56** in February 2024,
   or **26.79%**, above the frozen **20%** maximum.
2. Eval produced **25 events**, below the frozen minimum of **40**.
3. Eval events were concentrated in September 2025: **20 of 25**, or **80.00%**,
   above the frozen **20%** maximum.
4. Final events were concentrated in June 2026: **20 of 29**, or **68.97%**,
   above the frozen **30%** maximum.

No threshold, source-validity rule, direction, holding period, support minimum,
or concentration limit was changed after these source-only observations. All
train/test/eval/final returns remain sealed; no strict evaluator will be opened
for this candidate.

## Outcome-blind support summary

| Split | Events | Long | Short | Largest month share | Gate |
|---|---:|---:|---:|---:|---|
| Train, 2023-H2 | 35 | 15 | 20 | 31.43% | pass |
| Test, 2024 | 56 | 27 | 29 | **26.79%** | **fail** |
| Eval, 2025 | **25** | 12 | 13 | **80.00%** | **fail** |
| Final, 2026-H1 | 29 | 12 | 17 | **68.97%** | **fail** |

Train subperiod support passed independently: Q3 had 18 events and Q4 had 17,
against frozen minima of 6 and 8. Every split also passed the 25% minimum
long/short share. The rejection therefore comes from temporal instability, not
directional imbalance.

The concentration is consistent with the deliberately strict source-validity
contract: each rolling 720-hour threshold needs at least 672 joint-valid hours.
Only 12,864 of 26,567 hourly anchors became threshold-ready after BVOL archive
gaps were retained as invalid rather than imputed. Relaxing that rule after
observing the support pattern would be an undeclared candidate repair.

## Novelty audit

| Comparator | Common window | Exact Jaccard | Near window | OPDR near share | Gate |
|---|---|---:|---:|---:|---|
| Legacy DVOL price-follow | 2023-H2 | 3.23% | 1h | 14.29% | pass |
| PSR-30/6 | 2023-H2–2026-H1 | 0.00% | 6h | 26.90% | pass |
| PCBR-12 | 2023-H2–2026-H1 | 0.30% | 6h | 8.28% | pass |
| CMSR-36 | 2023-H2 | 0.00% | 6h | 28.57% | pass |

Every denominator is restricted to the comparator's explicit common coverage.
Events after 2023 were not allowed to dilute the legacy DVOL or CMSR containment
tests.

## Reproducibility and sealing evidence

- Premium-only source rows loaded: `3,417,120`
- Derived premium hours: `26,568` (`26,541` valid)
- Joint feature hours: `26,567` (`23,743` valid)
- BTC execution rows loaded: `0`
- Funding rows loaded: `0`
- Primary clocks: `145`
- Primary clock SHA-256:
  `ceb79b206c3e1f6bf78b02cd2ace9a94f875ce930a704cc6e7a5a8b255021b99`
- Support report SHA-256:
  `d8a82c072c45a2e965b8e4d05383aa3cb7f39d92728aef54ccd51ad54a02b9f3`
- Support manifest hash:
  `52eae4c00d672af469ef5d09a94e9ab2a92f6ab8128bab6e948225f960654840`

Two consecutive builds produced byte-identical report, primary clock, and all
six control-clock files. Peak RSS was approximately 1.47 GiB. `OPDR-24` cannot
be repaired using these observations; the next attempt must be a separately
preregistered mechanism.
