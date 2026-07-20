# WCTR-288 source-support rejection — 2026-07-20

## Verdict

**WCTR-288 is rejected before BTC outcomes.** The frozen primary clock passed
every count, direction, half-year, quarter, and source-continuity requirement,
but failed the preregistered forward calendar-dispersion cap:

```text
observed forward maximum month share = 10 / 34 = 29.4118%
frozen maximum                         = 28.0000%
```

The excess is 1.4118 percentage points. The threshold, support floor, rank
window, side, latency, hold, and calendar remain unchanged. No repair or
performance lookup is permitted under `WCTR-288`.

## Frozen support result

The outcome-blind builder committed at `6f44214` opened only the exact
hash-bound mined-block size/weight artifact and produced:

| Artifact | Rows | SHA-256 |
| --- | ---: | --- |
| Primary source clock | 230 | `7a6b56a3024d0d087322fad7b3229276c539b93374691cd2812af0630dc752b1` |
| Control source clocks | 2,716 | `ca5ef092d30ed9135429d8d2b830546681e289f0798ab50ef60c85ed5fd9a1f7` |
| Support manifest | — | `35e3c4623be670d690b914f96e6a40d8e314e3d14554d3024d1f201fcf8ffb30` |

The support manifest canonical hash is
`b07e6cab43ba4bcaeb3b76ded9af35e9699e5b41a346e1df8e0563337512978d`.
It binds preregistration policy hash
`510cedafde2775d65e3bc77eaefeccb9d526b9d738e503aa7c6c0e277974ddeb`.

## Primary incidence

| Window | Total | Long | Short | Max month share | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Train: 2022-11 through 2023 | 81 | 39 | 42 | 17.2839% | pass |
| Test: 2024 | 70 | 34 | 36 | 14.2857% | pass |
| Eval: 2025 | 45 | 22 | 23 | 20.0000% | pass |
| Forward: 2026 through 2026-07-20 | 34 | 18 | 16 | **29.4118%** | **fail** |

Additional dispersion counts all passed:

- train 2022 Nov-Dec: 14; 2023H1: 38; 2023H2: 29;
- test 2024 quarters: 19, 14, 13, 24;
- eval 2025 quarters: 14, 10, 8, 13; and
- forward 2026H1: 34.

The forward monthly counts were January 10, February 3, March 5, April 10,
and May 6. No June event survived the frozen magnitude, fullness, sign,
split-containment, and chronological non-overlap rules. January and April each
therefore exceeded the maximum permitted concentration.

## Source and leakage boundary

- Frozen source rows: 2,921 continuous 12-hour buckets.
- Base-valid feature rows: 2,907.
- Strict-prior rank-ready rows: 2,787.
- Market rows loaded: 0.
- Funding rows loaded: 0.
- Premium/OI rows loaded: 0.
- Return/PnL fields opened: 0.
- Performance values opened: false.

The support implementation reads the normalized compressed source exactly
once, verifies the exact bytes, and parses that same in-memory byte sequence.
It does not open the raw source artifact during the support build. Published
artifacts are immutable hard links with same-inode rollback protection.

## Controls and stop

All preregistered control incidences were generated source-only. `impulse_only`
met its own source-support floors; this is not a performance result and cannot
rescue or replace WCTR-288. The primary support failure stops the sequence
before any control or primary BTC return is opened.

WCTR-288 must not proceed to evaluator construction. A future witness-based
idea requires a new mechanism rationale, candidate ID, and preregistration; it
may not merely relax the failed month-share cap or retune this singleton.
