# Pullback-squeeze profit-lock selection — 2026-07-15

## Status

**Pre-OOS candidate frozen; 2024 onward remains unopened by this experiment.**

The corrected live-parity pullback-squeeze failed as a 48-hour time-exit
strategy because its strict drawdown can be dominated by a profitable trade's
position-wide favorable-to-adverse envelope. This experiment changes only the
exit constitution: it permits a causal take-profit to lock the favorable move,
but deliberately adds no hard stop. Entry features and thresholds are
unchanged.

The frozen manifest hash is:

```text
ca64871a079b5e7fe558417d6870bed63e0e39a9392137bb9392a9f8825c6325
```

## Frozen candidate

```text
LONG confirmed pullback-squeeze
entry: next 5-minute open
exit: first +10% underlying take-profit or 48-hour time exit
hard stop: none
leverage: 0.5x
```

At 0.5x, the +10% underlying take-profit is approximately a +5% gross account
move before exit costs and funding. One position is held at a time.

## Selection protocol

- Every source was physically truncated before `2024-01-01`.
- Signal row: top-of-hour `:00`.
- Market-derived inputs: shifted one completed 5-minute bar.
- Boundary funding and premium: current as-of values only; premium freshness is
  at most ten minutes.
- Entry: next 5-minute open (`:05`).
- Cost: 5 bp fee + 1 bp slippage per notional per side.
- Realized funding is compounded while the position is open.
- CAGR uses the full wall-clock window, including idle periods.
- Strict MDD includes the global/pre-entry high-water mark and the complete
  position's favorable envelope before its adverse envelope.
- A trade crossing a split boundary is purged.
- Search family: 4 fixed holds (`24h`, `36h`, `48h`, `60h`) × 11 profit-lock
  levels/time-only control = 44 pre-registered execution candidates.
- No stop-loss pair was searched in this family; the previous 28-candidate
  audit had already rejected all fixed TP+SL pairs it tested.

Admission required positive returns in 2020H2, 2021, 2022, and both 2023 halves;
at least 60 train, 12 selection, and 5 trades per 2023 half; train and 2023
ratios at least 3; combined pre-2024 ratio at least 2.5; and strict MDD no more
than 15%.

## Frozen pre-2024 statistics

| Window | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| Train 2020H2–2022 | +99.94% | 31.90% | 10.38% | **3.07** | 77 |
| 2020H2 | +16.91% | 36.36% | 6.37% | **5.71** | 13 |
| 2021 | +35.40% | 35.43% | 10.10% | **3.51** | 25 |
| 2022 | +23.63% | 23.65% | 10.38% | 2.28 | 38 |
| Select 2023 | +16.51% | 16.52% | 4.86% | **3.40** | 17 |
| 2023H1 | +11.16% | 23.81% | 4.86% | **4.90** | 9 |
| 2023H2 | +4.81% | 9.77% | 3.08% | **3.17** | 8 |
| Combined pre-2024 | +132.94% | 27.31% | 10.38% | **2.63** | 94 |

The annual 2022 ratio is below 3, so the result is not uniformly strong. The
admission rule intentionally uses positive annual return plus aggregate train
and selection ratio thresholds rather than demanding ratio 3 in every year.

## Neighborhood and stress checks

Only the 10% lock cleared the full pre-OOS gate. Nearby 48-hour variants were:

| Exit | Train ratio | 2023 ratio | Pre-2024 ratio | Pre-2024 trades |
|---|---:|---:|---:|---:|
| +8% TP | 2.66 | 3.36 | 2.31 | 95 |
| **+10% TP** | **3.07** | **3.40** | **2.63** | **94** |
| +12% TP | 3.21 | 2.61 | 2.76 | 93 |
| +15% TP | 3.25 | 2.57 | 2.84 | 93 |
| +20% TP | 3.01 | 2.31 | 2.69 | 93 |
| Time only | 2.80 | 2.29 | 2.48 | 93 |

The surface is directionally smooth, but the exact 10% bridge is selection-
sensitive and must be treated as one selected hypothesis, not an independent
discovery.

At 10 bp per side, train remains +93.87% with ratio 2.83, 2023 remains +15.72%
with ratio 3.23, and combined pre-2024 remains +124.35% with ratio 2.43. With
entry delayed to the third 5-minute open, train remains +97.11% with ratio
2.83, 2023 +15.77% with ratio 3.24, and combined pre-2024 +128.20% with ratio
2.41. These stresses remain profitable but fall below the primary ratio gate.

## Interpretation and risk

This is a plausible execution alpha, not yet an accepted live alpha. The
mechanism is coherent: pullback-squeeze entries often make a large favorable
move and then mean-revert before the 48-hour close; a profit lock changes that
path constitution without adding a future-dependent signal. However:

1. exactly one of 44 exit candidates passed;
2. nearby cost and latency stresses miss the aggregate ratio threshold;
3. no hard stop means every losing trade can remain open for 48 hours;
4. 2023 contains only 17 trades; and
5. the broader research programme has high multiplicity even though this
   experiment's 2024+ replay is still physically sealed.

The candidate must therefore be committed and pushed before the manifest is
used to open 2024–2026. No parameter may change after that commit.

## Artifacts

- `training/search_pullback_squeeze_profit_lock_alpha.py`
- `tests/test_search_pullback_squeeze_profit_lock_alpha.py`
- `results/pullback_squeeze_profit_lock_selection_2026-07-15.json`
- `results/pullback_squeeze_profit_lock_manifest_2026-07-15.json`

Reproduce the sealed selection:

```bash
PYTHONPATH=. /home/pakchu/rllm/.venv/bin/python -m \
  training.search_pullback_squeeze_profit_lock_alpha \
  --input-csv /home/pakchu/rllm/data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz \
  --funding-csv /home/pakchu/rllm/data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz \
  --premium-csv /home/pakchu/rllm/data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz \
  --output results/pullback_squeeze_profit_lock_selection_2026-07-15.json \
  --manifest-output results/pullback_squeeze_profit_lock_manifest_2026-07-15.json
```
