# CLV v1 frozen selection result — 2026-07-14

## Decision

**CLV v1 is rejected.** The evaluator was committed in `0901908` and hardened
in `4f7c652` before any CLV return was opened. It then replayed the frozen
support clock byte-for-byte and opened only calendar-2023 train/select windows.

- result:
  `results/cross_collateral_liquidity_vacuum_selection_2026-07-14.json`
- result SHA256:
  `a18e56b41045c7b83e2c9a5597fcbdea7380e1d31836cf3353c481b160ec6409`
- evaluator SHA256 recorded at execution:
  `5c7733e6dee9ef40c06767fb8802d1f14957cfeb1d0b37a0de37bd01f3ade8fc`
- still sealed: full 2024, full 2025, and 2026 YTD

No threshold, feature, direction, holding period, stop, or regime was changed
after the outcomes opened. No Gemma or RL policy was attached.

## Frozen CLV results

All returns include 0.5x leverage, 5 bp fee plus 1 bp slippage per notional
side, next-five-minute-open entry, scheduled-open exit after 12 bars, full split
clock CAGR, and favorable-first/adverse-second held-path strict MDD.

| window | absolute return | CAGR | strict MDD | CAGR/MDD | trades | weekly p |
|---|---:|---:|---:|---:|---:|---:|
| train 2023 H1 | -3.6416% | -7.2125% | 10.9753% | -0.6572 | 220 | 0.74151 |
| select 2023 H2 | -21.7587% | -38.5581% | 23.2907% | -1.6555 | 301 | 0.99999 |
| 2023 Q1 | +4.7091% | +20.5323% | 3.5049% | 5.8583 | 91 | 0.19418 |
| 2023 Q2 | -7.9752% | -28.3652% | 9.6495% | -2.9396 | 129 | 0.99927 |
| 2023 Q3 | -11.6599% | -38.8720% | 12.5511% | -3.0971 | 148 | 1.00000 |
| 2023 Q4 | -11.4317% | -38.2427% | 14.6441% | -2.6115 | 153 | 0.99545 |

CLV fails both half-year absolute-return and CAGR/MDD gates, H2 exceeds the
15% strict-MDD ceiling, both half-year weekly tests fail, and Q2 through Q4 are
negative. Q1's positive ratio is not statistically significant under the
frozen weekly-cluster test and does not generalize to the next three quarters.

## Frozen control diagnosis

The opportunity clock was reserved before every control changed its action.

| policy | 2023 H1 abs | H1 CAGR/MDD | 2023 H2 abs | H2 CAGR/MDD |
|---|---:|---:|---:|---:|
| CLV follow vacuum | -3.6416% | -0.6572 | -21.7587% | -1.6555 |
| exact reverse | -20.5720% | -1.7597 | -11.1858% | -1.5504 |
| always long | -2.4446% | -0.7031 | -19.2469% | -1.7721 |
| always short | -21.5471% | -1.7280 | -13.9489% | -1.4658 |
| permuted sign | -16.2148% | -1.6269 | -18.9930% | -1.7074 |

Both CLV and its exact reversal lose in both halves. This rules out promoting
the reverse control as though it were an independent out-of-sample discovery.
Directional controls also lose. The failure is therefore not a stable sign
mistake; the one-hour cross-collateral depth response does not deliver enough
net expectancy under the frozen execution contract.

Net mean CLV trade return is -1.61 bp of account equity in H1 and -8.10 bp in
H2. The round-trip account cost is approximately 6 bp before compounding. H1
contains a small pre-cost tendency, but it reverses in H2 and is far below the
required statistical and risk-adjusted edge. This is not a monetizable alpha.

## Qualification failures

The frozen evaluator reported:

1. H1 and H2 non-positive absolute return;
2. H1 and H2 CAGR/strict-MDD below 3;
3. H2 strict MDD above 15%;
4. H1 and H2 weekly-cluster p-values not below 0.10;
5. Q2, Q3, and Q4 non-positive absolute return.

Trade-count floors pass, so insufficient sample incidence is not the cause.
The result is a direct economic rejection, not a support rejection.

## Consequence

CLV v1 is closed. Its 2023 candidate clock, controls, and result remain useful
negative evidence, but none may be repaired under the CLV v1 name. The sealed
2024+ book-depth data will not be downloaded merely to rescue this mechanism.

The next experiment must change the economic object rather than optimize this
gate: in particular, it should test a lower-turnover *persistent liquidity
state transition* or cross-contract disagreement duration, with a holding
horizon justified before outcomes, instead of reacting to every one-hour
vacuum impulse.
