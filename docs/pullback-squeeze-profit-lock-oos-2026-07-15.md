# Pullback-squeeze profit-lock frozen OOS — 2026-07-15

## Verdict

**Rejected.** After commit `3c1ed23` pinned manifest
`ca64871a079b5e7fe558417d6870bed63e0e39a9392137bb9392a9f8825c6325`,
the single frozen 48-hour/+10%-TP/no-stop candidate was replayed once on
2024–2026. No OOS parameter was searched or changed.

The candidate passes 2024 and the partial 2026 holdout, but it fails 2025 and
the combined OOS target. It is not promoted as a standalone alpha.

## Frozen OOS statistics

All results use 0.5x leverage, 6 bp per notional per side, realized funding,
full wall-clock CAGR, split-contained exits, and the current strict-MDD
definition.

| Window | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| Test 2024 | +20.18% | 20.13% | 5.29% | **3.80** | 17 |
| Eval 2025 | +7.20% | 7.20% | 5.39% | **1.34** | 11 |
| Holdout 2026 through June 1 | +11.35% | 29.48% | 4.99% | **5.91** | 19 |
| OOS 2024–2026 | +43.45% | 16.10% | 5.72% | **2.82** | 47 |

At 10 bp per side, the combined OOS result falls to +40.78% absolute return,
CAGR 15.20%, strict MDD 5.94%, and ratio 2.56. The 2025 ratio is 1.22.

## Why it failed

The take-profit repaired the pre-2024 strict path constitution but did not
create a stable new edge in later data. In fact, the frozen 2024–2026 aggregate
is effectively the same weak regime profile as the corrected time-exit parent:
strong in 2024 and early 2026, weak in 2025. The +10% lock therefore acts as a
sample-specific drawdown repair rather than a general execution mechanism.

The result must not be rescued by choosing another OOS TP or holding horizon.
The next search should treat the underlying pullback-squeeze components as weak
features and seek a distinct conditional interaction that explains why the
same opportunity set works in 2024/2026 but not in 2025.

## Artifacts

- `results/pullback_squeeze_profit_lock_oos_2026-07-15.json`
- `results/pullback_squeeze_profit_lock_manifest_2026-07-15.json`
- `training/search_pullback_squeeze_profit_lock_alpha.py`

Frozen replay command:

```bash
PYTHONPATH=. /home/pakchu/rllm/.venv/bin/python -m \
  training.search_pullback_squeeze_profit_lock_alpha \
  --input-csv /home/pakchu/rllm/data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz \
  --funding-csv /home/pakchu/rllm/data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz \
  --premium-csv /home/pakchu/rllm/data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz \
  --manifest-output results/pullback_squeeze_profit_lock_manifest_2026-07-15.json \
  --output results/pullback_squeeze_profit_lock_oos_2026-07-15.json \
  --open-oos
```
