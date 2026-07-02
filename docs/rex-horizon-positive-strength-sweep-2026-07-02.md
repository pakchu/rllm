# REX positive-strength horizon sweep (2026-07-02)

## Why

The regime-family selector showed that REX pockets can work in 2024 but fail in 2025.  Before adding more LLM/RL training, this sweep checks whether shorter horizons increase sample size or whether the useful REX edge remains concentrated in the existing 288-bar hold.

## Code fixes included

- `training/rex_horizon_sweep.py` now matches the positive-strength rule from the pool probe:
  - threshold quantiles are fit on positive train strengths only;
  - candidate rows require `strength > max(0, threshold)`.
- `training/eval_pairwise_candidate_backtest.py` caches `date_to_pos` and OHLC arrays in `market.attrs`.
- `training/event_candidate_pool_probe.py::_simulate_rows` no longer copies the market frame for every replay, so repeated sweeps can reuse the cache.

Verification:

```bash
.venv/bin/python -m py_compile \
  training/eval_pairwise_candidate_backtest.py \
  training/event_candidate_pool_probe.py \
  training/rex_horizon_sweep.py \
  training/event_candidate_regime_family_selector.py
```

and the existing candidate-row tests were invoked directly with `.venv` Python.

## Sweep

Report: `results/rex_horizon_sweep_core_fixed_t2020_2024_v2024_e2025_2026_2026-07-02.json`

Protocol:

- train: 2020-01-01..2024-01-01
- validation/selection: 2024-01-01..2025-01-01
- eval/report-only: 2025-01-01..2026-06-01
- families: REX core pullback/location/compression families
- hold grid: 72, 144, 288 bars
- stride grid: 12, 24 bars
- quantile grid: 0.75, 0.80, 0.85
- selection score uses train + validation only; eval is not used for selection.

## Result summary

Top train+validation selected candidates mostly failed eval:

| rank | family | q | hold | stride | val CAGR/MDD | val trades | eval CAGR/MDD | eval trades | eval p |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `rex_multiscale_location_revert` | 0.85 | 288 | 24 | 4.89 | 141 | -0.63 | 218 | 0.091 |
| 2 | `rex_htf_pullback_resume` | 0.75 | 288 | 24 | 4.18 | 43 | -0.23 | 35 | 0.829 |
| 3 | `rex_htf_context_pullback_resume` | 0.85 | 288 | 24 | 3.83 | 49 | -0.15 | 44 | 0.908 |
| 9 | `rex_htf_deep_pullback_resume` | 0.85 | 288 | 24 | 3.05 | 36 | 2.95 | 27 | 0.095 |
| 10 | `rex_htf_context_pullback_resume` | 0.85 | 288 | 12 | 2.93 | 55 | 0.77 | 49 | 0.452 |

Best eval among the top-20 report rows:

| family | q | hold | stride | val CAGR/MDD | val trades | eval CAGR | eval strict MDD | eval CAGR/MDD | eval trades | eval p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `rex_htf_deep_pullback_resume` | 0.85 | 288 | 24 | 3.05 | 36 | 12.72% | 4.32% | 2.95 | 27 | 0.095 |
| `rex_htf_deep_pullback_resume` | 0.80 | 288 | 24 | 2.77 | 47 | 8.66% | 5.04% | 1.72 | 39 | 0.175 |
| `rex_htf_deep_pullback_resume` | 0.85 | 144 | 24 | 2.05 | 51 | 6.86% | 4.26% | 1.61 | 33 | 0.232 |
| `rex_htf_pullback_reclaim` | 0.85 | 288 | 12 | 2.17 | 64 | 12.15% | 8.65% | 1.40 | 64 | 0.145 |

## Interpretation

1. Shorter horizons did not dominate.  The useful REX candidates are still mostly 288-bar holds.
2. The validation winner (`location_revert`) is a clear anti-signal in eval.  This reinforces the need for an anti-persistence or regime-veto term.
3. The most interesting live candidate is not the validation winner but the more conservative `rex_htf_deep_pullback_resume q=0.85 hold=288 stride=24`.  It remains below the user's target because only 27 eval trades and CAGR/MDD 2.95, but it is close enough to justify targeted diagnostics.
4. The next selector should penalize overly strong validation spikes and favor families with lower train/val drift, lower reversion-regime fragility, and enough expected fold trades.

## Next action

Add a stability-aware selection score:

- penalize validation/train ratio spikes;
- penalize reversion families after adverse pre-fold trend/drawdown regimes;
- require minimum expected fold trades;
- use deep-pullback as the first candidate family for LLM state-card reasoning rather than raw trade choice.
