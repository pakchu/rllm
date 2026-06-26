# Price-action event scan with wavefull market data — 2026-06-26

## Objective

Search for fresh alpha features after path-shape side labels and symbolic action selectors failed to produce deployable returns.

This scan tests causal price-action events built from shifted prior rolling ranges:

- break above / break below
- high sweep reject / low sweep reclaim
- failed breakout / failed breakdown
- mid-range reclaim/reject
- outside range close back inside
- volume expansion variants

All rolling levels use prior bars only; current bar is compared to shifted prior range levels.

## Command artifact

- `results/price_action_event_scan_wavefull_2026-06-26/report.json`

Config:

- market: `data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz`
- train: 2020-01-01 through 2023-12-31
- test: 2024-01-01 through 2025-12-31
- eval: 2026-01-01 through 2026-06-01
- windows: `36,72,144,288,576,2016,4032,8640`
- horizons: `36,72,144,288`
- leverage: 1.0
- rows scanned: 480

## Main result

The best test-ranked events do not survive 2026 eval. Examples:

| event | horizon | side fit on train | test CAGR | test MDD | test ratio | eval CAGR | eval MDD | eval ratio | eval trades |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `pae_w4032_outside_close_back_inside` | 288 | LONG | 25.78% | 17.02% | 1.51 | -30.09% | 21.92% | -1.37 | 30 |
| `pae_w576_failed_breakdown_long` | 144 | LONG | 26.89% | 18.28% | 1.47 | -34.17% | 17.67% | -1.93 | 47 |
| `pae_w4032_break_below` | 36 | LONG | 18.81% | 14.30% | 1.32 | -4.58% | 12.72% | -0.36 | 29 |

Eval-positive events exist, but usually failed or were weak in test. Example:

| event | horizon | test CAGR | test MDD | test ratio | eval CAGR | eval MDD | eval ratio | eval trades |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `pae_w576_reclaim_mid_from_below` | 288 | 1.03% | 44.66% | 0.02 | 94.05% | 12.82% | 7.34 | 67 |
| `pae_w8640_reject_mid_from_above` | 36 | -24.59% | 43.34% | -0.57 | 14.10% | 2.92% | 4.84 | 20 |
| `pae_w8640_reclaim_mid_from_below` | 36 | -19.02% | 34.51% | -0.55 | 10.28% | 3.19% | 3.22 | 20 |

Events positive in both test and eval were still below target:

| event | horizon | test ratio | eval ratio | test trades | eval trades |
|---|---:|---:|---:|---:|---:|
| `pae_w288_break_below_with_volume` | 288 | 0.64 | 0.76 | 293 | 68 |
| `pae_w4032_break_below` | 72 | 0.64 | 0.79 | 79 | 25 |
| `pae_w2016_break_below` | 72 | 0.23 | 0.95 | 127 | 32 |

## Decision

No single price-action event is deployable. The useful finding is feature-level:

- rolling range breaks/reclaims/sweeps clearly produce regime-dependent edge;
- 2026 liked some mid-range reclaim/reject events that 2024-2025 did not;
- fixed train-side mapping is too brittle.

Next use:

1. Promote these events as LLM/RLLM context tokens, not standalone trading rules.
2. Add train/test/eval stability metadata per event family so the model/selector can abstain from historically unstable contexts.
3. Prefer a policy target like “setup quality / path-risk state” over direct LONG/SHORT imitation.
