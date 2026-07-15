# Absorbed Basis-Recoil Pullback Alpha — Pre-OOS Selection

## Verdict

**REJECTED_PRE_OOS.** The complete 12-rule family failed the frozen pre-2024
selection contract. No 2024+ observation was loaded or evaluated.

## Hypothesis

The experiment tested a two-stage interaction instead of a weighted feature
blend:

1. an audited confirmed pullback-squeeze setup becomes the anchor;
2. one, three, or six hours later, perp premium must have recovered from the
   anchor while 1-hour taker flow remains seller-dominant;
3. price must no longer be in the weakest train-fitted 40% of 1-hour returns;
4. 4-hour dollar-flow participation must exceed its train q50 or q70; and
5. entry occurs at the next five-minute open.

This was intended to identify passive absorption: sellers remain aggressive,
but price and basis begin to recover under meaningful participation.

## Frozen protocol

- Physical source cutoff: `2024-01-01`
- Threshold fit: `2020-07-01` through `2022-12-31`
- Selection: calendar 2023, with H1 and H2 both required
- Family: `3 lag windows × 2 sell quantiles × 2 participation quantiles = 12`
- Position: long, 0.5× leverage, 48-hour maximum hold, 10% take-profit, no stop
- Cost: 6 bp per notional side
- Funding: realized funding cash flows
- Execution: completed hourly signal, next five-minute open
- Strict MDD: global/pre-entry high-water mark plus the position-wide favorable
  envelope followed by the adverse envelope
- Promotion target: train, 2023, and combined pre-2024 CAGR/strict-MDD at least
  3.0; strict MDD at most 15%; positive subperiods and minimum trade counts

## Best rule

`abr_L3_sell0.4_part0.5`

| Period | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| Train (2020H2–2022) | 13.19% | 5.08% | 8.83% | 0.575 | 21 |
| Selection (2023) | 2.61% | 2.61% | 5.70% | 0.458 | 4 |
| Combined pre-2024 | 16.14% | 4.37% | 8.83% | 0.494 | 25 |

The best member missed both the return/risk target and every minimum trade
count. Increasing the trigger window raised activity but did not improve
return/risk enough. The interaction therefore does not justify OOS exposure.

## Interpretation

Premium recoupling plus seller imbalance is not sufficient after this already
sparse setup. It mostly removes valid pullback entries rather than isolating a
better payoff state. The next family should move the event anchor itself: use
the **duration and first release of a deeply negative funding state**, then
require price/flow confirmation at the transition. That tests lifecycle
information unavailable to a same-row conjunction.

## Reproduction

```bash
PYTHONPATH="$PWD" python training/search_absorbed_basis_recoil_pullback_alpha.py \
  --input-csv data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz \
  --funding-csv data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz \
  --premium-csv data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz
```

Machine-readable evidence:
`results/absorbed_basis_recoil_pullback_alpha_selection_2026-07-15.json`.
