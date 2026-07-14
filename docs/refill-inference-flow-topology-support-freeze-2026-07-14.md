# RIFT-96 support decision and clock freeze — 2026-07-14

## Decision

RIFT-96 passed its preregistered outcome-blind support/novelty gate at setup
score quantile `0.925`. No execution return, future OHLC path, win rate, CAGR,
or MDD was opened.

The chosen percentile is the highest frozen grid member passing every count
floor. Quantile `0.95` failed because 2023 H1 had 29 events versus the fixed
minimum of 30; no threshold was relaxed.

## Selected support

- raw confirmed sequences: **527**
- 8-hour non-overlapping events: **460**
- 2020 / 2021 / 2022 / 2023: **103 / 153 / 119 / 85**
- 2023 H1 / H2: **42 / 43**
- actions: **460 long, 0 short**, fixed by design

Selected raw-clock Jaccards:

- same-bar static: `0.0058`
- no path quality: `0.1917`
- no derivatives crowd: `0.0829`
- centroid-free momentum: `0.6512`
- Spot-only: `0.0764`
- stale setup 1h / 24h: `0.0228 / 0.0259`
- one-bar delay: `0.0038`
- simple two-bar momentum: `0.0057`
- CSPR primary: `0.0000`

Thus the clock is neither a CSPR repair nor a relabeled static/two-bar momentum
clock at the preregistered overlap limits. This is only a structural result;
it makes no profitability claim.

## Frozen artifacts

- preregistration commit:
  `b9af831b6117f0eb5843fc3ee71d4ddff51c3a39`
- support JSON SHA-256:
  `d026ceb2af9d586c5327659cb98d69a4d19ec3bfb8e2357891ac20c8e53a31e6`
- selected event clock SHA-256:
  `83becb88da83ce55e235f10c2e91ed3a2ad478c6aea2e9298d37c43b36cfff00`
- clock manifest SHA-256:
  `47460778b45fe861c787e49a6f374c54e8155594350d1d30d54b4e54b0d63789`

The clock contains only signal/entry/exit positions and timestamps, fixed long
side, branch, and 96-bar hold. It contains no price or return column. The first
signal is `2020-01-12 08:10:00`; the last is `2023-12-28 09:35:00`.

## Next gate

Implement, test, independently review, commit, and hash-freeze the evaluator
and every control action/clock before opening returns. Only 2020–2022 train and
2023 selection/H1/H2 may then open. Calendar 2024 onward remains sealed unless
the unchanged pre-2024 gate passes.
