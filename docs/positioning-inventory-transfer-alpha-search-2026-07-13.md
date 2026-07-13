# Positioning Inventory-Transfer Alpha Search — 2026-07-13

## Thesis

The strongest recent positioning result was not the disagreement extreme but
the first zero-cross after it remained alive for 36 hours. This fixed follow-up
asked whether open interest distinguishes two economically different exits:

- OI preserved or increased from episode start: inventory changed hands, so
  follow the disagreement-resolution direction;
- OI declined: inventory was destroyed by deleveraging, so fade the resolution.

No nearby z-score, age, OI threshold or direction was tuned. The only two
policies were fixed 6-hour and 18-hour holds.

## Causal protocol

- Market and Binance USD-M metric sources physically stop before `2024-01-01`.
- Every positioning/OI observation is delayed by one complete 5-minute bar.
- Disagreement z-scores use history through `t-1` only.
- Episode-start OI is frozen when `|z|>=1.5`; current OI is read only on the
  completed zero-cross bar after at least 36 hours.
- Missing metrics reset state; all of 2022 remains quarantined because of known
  top-trader coverage gaps.
- Entry is next 5-minute open, leverage is `0.5x`, and cost is `6bp/side`.
- Strict MDD uses favorable-first/adverse-second OHLC high-water.
- `2024+` OOS was not opened.

## Results

The 66 events split into 36 OI-preserved and 30 OI-contracted resolutions.

| Hold / period | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| 6h fit | +2.43% | +2.00% | 6.72% | 0.30 | 40 |
| 6h 2023 | -7.04% | -7.04% | 7.93% | -0.89 | 24 |
| 18h fit | +7.87% | +6.44% | 10.36% | 0.62 | 40 |
| 18h 2023 | -10.07% | -10.08% | 17.97% | -0.56 | 24 |

The 18-hour control results make the failure unambiguous:

- ignore OI and retain the parent lifecycle direction: fit `+30.96%`, CAGR
  `24.90%`, strict MDD `6.76%`, ratio `3.68`; 2023 `+7.04%`, CAGR `7.04%`,
  strict MDD `6.52%`, ratio `1.08`;
- invert the OI routing: fit `-12.50%`, ratio `-0.61`; 2023 `+7.47%`, ratio
  `1.17`;
- exact direction flip: identical to inverted routing because the rule has two
  exhaustive branches.

## Decision

**Rejected.** OI conservation did not refine the lifecycle edge; it destroyed
the positive parent in 2023, while the opposite routing changed which regime
won. This is regime-dependent branch selection, not a stable inventory-transfer
law. Zero of two policies passed pre-2024 admission, so OOS remained sealed.

Record only the exact static OI-sign routing as gamma failure provenance. Raw
delayed OI and the continuous lifecycle state are not universally rejected.

Artifacts:

- `training/search_positioning_inventory_transfer_alpha.py`
- `tests/test_search_positioning_inventory_transfer_alpha.py`
- `results/positioning_inventory_transfer_alpha_scan_2026-07-13.json`
