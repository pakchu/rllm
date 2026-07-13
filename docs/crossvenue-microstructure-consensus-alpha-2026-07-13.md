# Cross-venue / microstructure consensus alpha audit (2026-07-13)

## Verdict

Rejected as an alpha. This exact two-feature symmetric agreement/disagreement usage must not be retried without a materially different regime model or data treatment.

## Clean selection protocol

- No REX feature, event, prediction or model entered candidate construction.
- Threshold fit: 2020-2022 only.
- Execution/variant selection: 2023 plus 2023 H1/H2 stability.
- Top-10 manifest was written while market rows from 2024 onward were physically excluded.
- Frozen replay: 2024 test, 2025 eval, 2026 YTD holdout.
- Entry at next 5m open; fixed hold; one position; 0.5x; 6bp/side.
- Strict MDD uses favorable high-water followed by adverse OHLC extreme.
- Each rule combined one Korean local/FX feature with one Binance volume-clock, jump, or liquidity-recovery feature.

## Result

The manifest contained 10 candidates from 320 unique signal masks. No candidate passed the alpha-pool criterion.

| Rank | Pair | Hold | 2024 return / ratio / trades | 2025 return / ratio / trades | 2026 return / ratio / trades |
|---:|---|---:|---:|---:|---:|
| 1 | kimchi-BTC gap 48 × volume-clock flow speed 0.5 agreement | 144 | -13.29% / -0.56 / 151 | -3.17% / -0.20 / 157 | -2.84% / -0.74 / 100 |
| 2 | same pair | 72 | -0.04% / -0.00 / 180 | -2.97% / -0.32 / 189 | -4.07% / -1.40 / 119 |
| 3 | local impulse 48 × volume-clock flow speed 0.5 agreement | 144 | -15.64% / -0.65 / 158 | +11.69% / 1.13 / 157 | -4.08% / -1.00 / 89 |
| 6 | kimchi-BTC gap 48 × volume-clock flow speed 0.25 agreement | 288 | +4.31% / 0.24 / 108 | +48.95% / 4.35 / 141 | -5.93% / -1.23 / 64 |

All exact REX signal-date Jaccards were 0.0, so the candidates were genuinely activation-independent from the reference REX stream. Independence alone did not create stable edge: the relationship changed sign by year.

## Interpretation

The local-market/microstructure relationship is regime-dependent. A fixed symmetric agreement/disagreement rule learned from 2020-2023 does not generalize across 2024-2026. High 2025-only results in some variants are not usable because 2024 and 2026 are negative.

## Artifacts

- `training/search_crossvenue_microstructure_consensus_alpha.py`
- `results/crossvenue_microstructure_consensus_top10_manifest_2026-07-13.json`
- `results/crossvenue_microstructure_consensus_alpha_scan_2026-07-13.json`
