# BAFR-24F outcome-blind support freeze — 2026-07-20

## Decision

**PASS the frozen source-quality and temporal-support gate.** BAFR-24F may
advance only to the separately frozen timestamp/side novelty gate. No market
OHLC value, funding cash flow, post-entry return, PnL, absolute return, CAGR,
strict MDD, hit rate, or outcome-derived repair was read in this stage.

The later novelty implementation must consume pre-frozen comparator clocks.
It must not reconstruct prior clocks by loading market OHLC inside the BAFR
admission process.

## Verified source build

- source range: `2020-01-01 00:00:00` through `2023-12-31 23:55:00` UTC;
- completed five-minute feature rows: **420,732**;
- official market timestamp-grid rows: **420,768**;
- combined feature SHA256:
  `e46dc9a4f5e4d4a93bc260d40c0a599ccd0e609d5cb8ebf438c716f7272f7275`;
- source manifest SHA256:
  `9fa1025c90fb8ad1729f2278236a73e94b0d20bcf9b79178610306cf3b85a28b`;
- timestamp-only market SHA256:
  `e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d`;
- full build elapsed time: **20m 02.82s** with four workers;
- maximum resident set size: **4,477,204 KiB**;
- build exit status: **0**.

All schema identities, archive checksum bindings, monthly artifact hashes,
within-month state chains, prior-day warmup chains, aggregate-ID terminal
states, row counts, and pre-2024 cutoff checks passed.

## Source quarantine

Five UTC source-gap/reset days were detected:

- `2020-04-15`;
- `2021-02-09`;
- `2021-02-24`;
- `2021-05-19`; and
- `2022-09-06`.

Missing/reset rows and their following 24 bars produced **1,682** quarantined
market rows, or **0.39975%** of the full grid. This passes the frozen maximum of
2%. No missing value was filled.

## Frozen support clock

The single preregistered q90/prior-8,640-clean-observation/24-bar policy yielded:

| Check | Observed | Frozen floor/limit | Result |
|---|---:|---:|---|
| non-overlapping total | 11,248 | >= 250 | pass |
| 2020 | 2,576 | >= 40 | pass |
| 2021 | 2,759 | >= 40 | pass |
| 2022 | 2,886 | >= 40 | pass |
| 2023 | 3,027 | >= 40 | pass |
| 2023 H1 | 1,467 | >= 20 | pass |
| 2023 H2 | 1,560 | >= 20 | pass |
| long share | 47.11% | 25–75% | pass |
| short share | 52.89% | 25–75% | pass |
| largest month share | 2.48% | <= 20% | pass |

The clock is intentionally dense; density is support evidence, not alpha
evidence. Its economic usefulness remains completely unknown.

## Frozen artifacts

- support result:
  `results/binance_aggressor_frustration_support_2026-07-20.json`
  (SHA256 `cf6edad6a4eb46c6630dbb5008c88da1ddd39f9ac5c1606785be02f2b323fb62`);
- timestamp/side-bearing execution clock:
  `results/binance_aggressor_frustration_clock_2026-07-20.csv`
  (SHA256 `f3b816a76decce31136ed23d22f043eb8e80ef1b8697b869241b060062f01747`).

The next permitted action is the outcome-blind novelty gate against CBFR-72,
MFIC, NETF, WFRS, and terminal absorption. Failure against any frozen
comparison retires BAFR without opening returns.
