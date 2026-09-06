# G9 historical barrier clocks

Frozen historical trade-clock export for Fresh Kimchi/FX and annual-refit Rank7.
No portfolio weights are optimized in this artifact.

Market coverage: `2019-12-31 15:00:00` through `2026-05-31 15:00:00` (end-exclusive `2026-05-31 15:05:00`).

| Window | Sleeve | Trades | Long/Short | Return | MDD | Hash | JSONL |
|---|---|---:|---:|---:|---:|---|---|
| 2024 | fresh_kimchi_fx | 30 | 7/23 | 10.6160% | 3.7215% | `0c75a477f28f` | `research/g9_historical_barriers/trades/2024__fresh_kimchi_fx.jsonl` |
| 2024 | frozen_annual_rank7 | 22 | 22/0 | 16.3961% | 3.4631% | `e715c7e91539` | `research/g9_historical_barriers/trades/2024__frozen_annual_rank7.jsonl` |
| 2025 | fresh_kimchi_fx | 17 | 9/8 | 11.9212% | 5.0086% | `eb0e6776b87e` | `research/g9_historical_barriers/trades/2025__fresh_kimchi_fx.jsonl` |
| 2025 | frozen_annual_rank7 | 21 | 21/0 | 16.3620% | 4.9844% | `20f982e3c08a` | `research/g9_historical_barriers/trades/2025__frozen_annual_rank7.jsonl` |
| 2026_prefix | fresh_kimchi_fx | 28 | 20/8 | 9.5597% | 5.5692% | `eaf50ddd5e4b` | `research/g9_historical_barriers/trades/2026_prefix__fresh_kimchi_fx.jsonl` |
| 2026_prefix | frozen_annual_rank7 | 12 | 12/0 | 7.3132% | 4.3007% | `9b86a1ac0a9b` | `research/g9_historical_barriers/trades/2026_prefix__frozen_annual_rank7.jsonl` |

## Integrity

- Annual Rank7 frozen-prefix verification passed: `True`.
- Rank7 annual-reference verification passed: `True`.
- Market frame identity passed: `True`.
- Trade `side` is numeric normalized (`1` long, `-1` short); `side_label` is auxiliary only.
- `exit_kind=open` means max-hold cap at next open; `exit_kind=barrier` means take/stop barrier fill with stop-first execution convention.
