# OI cost-basis liquidation alpha preflight — 2026-07-13

## Protocol

- Physical data cutoff: `2024-01-01`; no 2024+ rows or positioning source rows are loaded.
- Mechanism: one-bar delayed Binance OI/positioning, path-dependent long/short OI expansion cohort cost basis, underwater cohort pressure, OI contraction, and taker-flow-confirmed liquidation direction.
- 2022 positioning gap handling: 2022 is loaded only for continuity/quarantine diagnostics and is excluded from fit and selection.
- Search grid: H `(288, 864)`, tail `(0.05, 0.1)`, mode `('both', 'long_only', 'short_only')`, hold `(72, 144)` = `24` candidates.
- Execution: next-bar open, 0.5x, 6bp/side (`fee_rate=0.0005`, `slippage_rate=0.0001`), fixed hold, strict OHLC MDD.

## Top result

| name | positive segments | fit ratio | select 2023 ratio | min segment ratio | select trades |
| --- | ---: | ---: | ---: | ---: | ---: |
| oi_cost_basis_liq_h288_tail0.05_long_only_hold144 | 5/5 | 0.89 | 1.42 | 0.49 | 46 |

Admission decision: **fail_pre2024_admission** / **not open for OOS**.  The best pre-2024 candidate is long-only and has 5/5 positive robustness segments, but its fit/select ratios are only about `0.89` / `1.42`, below the required admission gate.

## Top-10 pre-2024 candidates

| name | positive segments | fit ratio | select 2023 ratio | min segment ratio | select trades |
| --- | ---: | ---: | ---: | ---: | ---: |
| oi_cost_basis_liq_h288_tail0.05_long_only_hold144 | 5/5 | 0.89 | 1.42 | 0.49 | 46 |
| oi_cost_basis_liq_h288_tail0.05_long_only_hold72 | 4/5 | 1.69 | 0.65 | -0.04 | 49 |
| oi_cost_basis_liq_h288_tail0.10_long_only_hold144 | 4/5 | 1.29 | 0.62 | -0.18 | 65 |
| oi_cost_basis_liq_h288_tail0.10_long_only_hold72 | 4/5 | 1.89 | 0.07 | -1.00 | 73 |
| oi_cost_basis_liq_h288_tail0.10_both_hold144 | 4/5 | -0.45 | 1.11 | -1.19 | 104 |
| oi_cost_basis_liq_h288_tail0.05_both_hold144 | 3/5 | -0.01 | 0.31 | -0.27 | 77 |
| oi_cost_basis_liq_h864_tail0.10_long_only_hold144 | 3/5 | -0.57 | 0.43 | -1.56 | 53 |
| oi_cost_basis_liq_h864_tail0.05_both_hold72 | 3/5 | -0.61 | -0.41 | -1.62 | 63 |
| oi_cost_basis_liq_h288_tail0.05_both_hold72 | 2/5 | -0.33 | -0.25 | -0.97 | 81 |
| oi_cost_basis_liq_h864_tail0.10_both_hold144 | 2/5 | -0.58 | 0.16 | -1.48 | 86 |

## Direction flip / component ablation on top spec

| variant | positive segments | fit ratio | select 2023 ratio | select trades |
| --- | ---: | ---: | ---: | ---: |
| direction_flip | 0/5 | -0.72 | -0.96 | 46 |
| no_oi_contraction | 5/5 | 2.06 | 1.22 | 89 |
| no_taker_flow | 3/5 | 0.24 | 1.15 | 30 |
| no_cost_basis_pressure | 2/5 | -0.10 | 1.11 | 55 |

## Deterministic verification

- Input rows: `420876`, range `2019-12-31 15:00:00` through `2023-12-31 23:55:00`.
- Max delayed positioning source time: `2023-12-31 23:50:00`.
- Manifest hash: `203857909c7c5a667920b71c55c9e2fa130035f646b32830b013e739b3fc7673`.
- Scan hash: `66d3a51f072e26a9e995ea305144b76cb47765e340022f0a5b9b4c585cc7d276`.
- Protocol assertions: `{"no_2024_rows_loaded": true, "no_2024_source_rows_loaded": true, "physical_cutoff": "2024-01-01", "search_space_count": true, "top_is_not_open_for_oos": true}`.

## Conclusion

The mechanism is economically interpretable but remains a preflight-only discovery.  Because admission fails before any 2024+ data is opened, this alpha is **not admitted to OOS/live evaluation**.
