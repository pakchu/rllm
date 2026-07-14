# CATCH-12 support freeze — 2026-07-14

## Decision

**PASS support/novelty; outcomes remain unopened.** No future return, held
high/low path, win rate, CAGR, or MDD was read during selection.

- preregistration commit:
  `5d1270bf0d3c453ecfc3a5e02a193db8db59f1e5`
- frozen source SHA-256:
  `00ab6a55fc7bfeb3012584db5bc97a7d7b98dd995491acfd3f865c6bd41f92cc`
- support artifact:
  `results/cash_auction_transfer_catchup_handoff_support_2026-07-14.json`
- support SHA-256:
  `454c54cb234a34b51fca12a810332039d7b21e4395f47f4e6b8ad6375370be02`
- selected strictly-prior score quantile: **0.975**
- fixed hold: **12 five-minute bars (one hour)**

Lower quantiles all passed count support but failed at least one frozen novelty
limit. The highest tested quantile passed every count and novelty rule, so no
post-result threshold repair occurred.

## Selected support

| Period | Non-overlapping events |
|---|---:|
| 2020 | 1,108 |
| 2021 | 1,075 |
| 2022 | 919 |
| 2023 | 855 |
| **Total** | **3,957** |

- 2023 H1/H2: **415 / 440**
- 2023 Q1/Q2/Q3/Q4: **181 / 234 / 211 / 229**
- long/short share: **44.02% / 55.98%**
- long/short by year:
  - 2020: `490 / 618`
  - 2021: `471 / 604`
  - 2022: `408 / 511`
  - 2023: `373 / 482`
- all 48 months have at least 20 scheduled events
- raw candidate count before fixed-hold reservation: **4,503**

## Novelty evidence

All limits passed on both raw incidence and each control's own fixed-hold
schedule. Values below are `Jaccard / primary containment`.

| Control | Raw | Scheduled |
|---|---:|---:|
| reverse-time | `0.1213 / 0.2163` | `0.1081 / 0.1941` |
| venue-swap | `0 / 0` | `0 / 0` |
| simultaneous-only | `0.0203 / 0.0422` | `0.0171 / 0.0344` |
| aggregate-only | `0.0548 / 0.1097` | `0.0460 / 0.0874` |
| basis-only | `0.0125 / 0.0333` | `0.0109 / 0.0260` |
| flow/return-asymmetry-only | `0.0697 / 0.1244` | `0.0605 / 0.1079` |
| no activity ordering | `0.2132 / 0.3511` | `0.1906 / 0.3161` |
| stale 1h | `0.0070 / 0.0138` | `0.0069 / 0.0136` |
| stale 24h | `0.0056 / 0.0111` | `0.0045 / 0.0088` |
| one-bar delay | `0.0072 / 0.0142` | `0 / 0` |

Residual-basis ablation retention was **0.5348 raw** and **0.5867
scheduled**, below the frozen `0.65` maximum.

Scheduled overlap with prior candidate clocks was also below `0.01`:

- CSPR: Jaccard `0.00313`, containment `0.00379`, 15 shared events;
- RIFT: Jaccard `0.00181`, containment `0.00202`, 8 shared events;
- CSPR ∪ RIFT: Jaccard `0.00439`, containment `0.00581`, 23 shared events.

## Interpretation and boundary

This is strong clock support, not alpha evidence. The result only shows that a
high-strength, bidirectional, cash-led ordering event exists throughout all
pre-2024 eras and is not merely a stale, simultaneous, aggregate, basis-only,
reverse-time, CSPR, or RIFT relabeling under the frozen overlap rules.

The next step may serialize the selected non-overlapping clock and its hash.
Only after that clock and a separate pre-outcome evaluator implementation are
committed may train/2023 returns be opened. 2024+ remains sealed.
