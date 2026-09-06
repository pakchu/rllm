# Independent short candidate — 2026-09-07

## Decision

384 fixed short settings were searched without any parent-long requirement. The2024 ranking was frozen before exact2025/full2026H1 replay.14 global/family report candidates were retained; negative family rows are diagnostic reports, not accepted alphas.
One fixed finalist (2024 proxy rank2) has positive standalone return in all three periods at6bp/side. None is positive in all three periods at10bp. This is a disabled research candidate, not a validated production short.

## Formula

- Standardized168-hour price momentum < -0.5.
- Six-hour price z-score >0.5: a short rebound inside the weekly downtrend.
- Six-hour aggressive-buy-minus-sell flow / total quote volume < -0.01: selling pressure persists.
- Short at next5m open after a completed hourly decision, independently of any parent portfolio.
- Maximum24h hold,2% take profit,1.5% stop loss; same-bar conflict assumes stop first.
- Fixed1x entry notional; internally nonoverlapping trades; real funding and notional entry/exit costs.

## Exact five-minute results

| Period | 6bp return | MDD | Trades | Win rate | 10bp return |
|---|---:|---:|---:|---:|---:|
| 2024 | 10.39% | 8.98% | 78 | 50.0% | 3.72% |
| 2025 | 3.03% | 13.89% | 114 | 47.4% | -5.95% |
| 2026H1 | 19.14% | 8.36% | 68 | 52.9% | 12.85% |
| recent | 10.13% | 5.73% | 26 | 57.7% | 7.87% |
| extension_since_aug4 | -0.24% | 2.19% | 4 | 50.0% | -0.55% |
| september_only | -0.39% | 2.19% | 1 | 0.0% | -0.47% |

2026H1 is January1–June30. Recent is June1–September5 00:00UTC; September-only covers September1–4. These periods overlap and must not be added together.

## Interpretation / risks

This candidate differs from the previous hedge-only search: it can take profitable short trades while G9 is flat, and uses explicit asymmetric price exits rather than simply trimming current long exposure.
2025 has limited expected margin:3.03% at6bp becomes-5.95% at10bp. Do not describe it as robust to all costs. The new August/September slice is also slightly negative despite the positive June–September aggregate.
The candidate is a retrospective shortlist from exposed periods. No pristine OOS or multiplicity-adjusted significance is claimed. There is no automatic live promotion or original2024 winner replacement.

## Reproduction and evidence

Run from repo root: `python -m training.search_independent_short_candidates` and `python -m training.evaluate_independent_short_september` in the project environment.
All14 finalist2024 proxy terminal returns match the exact ledger at10bp within1e-7. Proxy MDD is only a discovery approximation; reported MDD comes from the five-minute ledger.
`configs/shadow/independent_failed_rebound_short_2026-09-07.json` is disabled. Reports, selection freeze, trades and source hashes are under `research/independent_short_candidates` and `research/independent_short_september`.
