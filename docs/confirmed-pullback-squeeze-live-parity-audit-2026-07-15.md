# Confirmed pullback-squeeze live-parity audit — 2026-07-15

## Verdict

**Reject the previous live-grade interpretation.** The signal still has positive
returns, but it does not satisfy the current `CAGR / strict MDD >= 3` contract
over the full history once replay and live use the same data freshness,
realized funding, and position-wide strict-MDD definition.

No threshold was rescued with 2024–2026. A 28-member exit-overlay family was
evaluated only after physically truncating every source before 2024; it produced
zero qualifiers. The pullback-squeeze may remain a weak feature/mechanism, but
the exact strategy is no longer an accepted standalone alpha.

## What was wrong with the previous result

The earlier `confirmed-pullback-squeeze-alpha-2026-07-15.md` report used:

1. a positional hourly decision clock at minute `:55`;
2. a two-hour premium-index tolerance;
3. no realized funding cash flows; and
4. chronological per-bar adverse marking rather than the repository's current
   position-wide favorable-envelope → adverse-envelope strict MDD.

The live feature contract accepts premium data for only ten minutes. At the old
`:55` clock, **0 of 35,062 pre-2024 decisions** had a fresh premium observation.
The old clock therefore cannot reproduce its premium branch in live trading.

The corrected signal row is labelled `:00`, after the prior hourly premium kline
is complete. Market-derived features are shifted to the completed `:55–:00`
bar, while only boundary-timestamped funding/premium values remain current. The
decision therefore has the full `:00–:05` interval before entry at the `:05`
open; it does not consume the unfinished `:00–:05` market bar.

## Corrected statistics

All rows below include 6 bp per side, realized funding, idle calendar time,
split-contained exits, and position-wide strict MDD.

### Live-parity clock and freshness

| Leverage | Window | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades |
|---:|---|---:|---:|---:|---:|---:|
| 0.5x | Train 2020H2–2022 | +115.24% | 35.84% | 12.79% | **2.80** | 76 |
| 0.5x | Select 2023 | +22.08% | 22.09% | 9.65% | **2.29** | 17 |
| 0.5x | Test 2024 | +20.18% | 20.13% | 5.29% | 3.80 | 17 |
| 0.5x | Eval 2025–2026-05-31 | +19.37% | 13.36% | 5.72% | **2.34** | 30 |
| 0.5x | OOS 2024–2026-05-31 | +43.45% | 16.12% | 5.72% | **2.82** | 47 |
| 0.5x | Full 2020H2–2026-05-31 | +276.91% | 25.14% | 12.79% | **1.97** | 140 |
| 0.9x | Full 2020H2–2026-05-31 | +927.56% | 48.27% | **21.14%** | **2.28** | 140 |

The later windows remain attractive, but they cannot repair the failed train,
selection, and full-history contracts. At the previous 0.9x operating point,
both the 50% CAGR objective and 15% MDD limit are missed.

### Legacy offline clock under the corrected accounting

Even if the old two-hour premium tolerance and `:55` clock are retained, the
current accounting rejects the result:

| Leverage | Window | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades |
|---:|---|---:|---:|---:|---:|---:|
| 0.5x | Full | +301.15% | 26.47% | 12.72% | **2.08** | 139 |
| 0.9x | Full | +1048.52% | 51.08% | **21.19%** | **2.41** | 139 |

This isolates the major discrepancy: the previous MDD (`7.68%` at 0.5x) was
not the current strict definition. Realized funding is a smaller but adverse
effect. On the live-parity full replay at 0.5x, replacing actual funding with
zero funding changes CAGR from `25.14%` to `25.87%` and strict MDD from `12.79%`
to `12.50%`.

## Pre-2024 exit-overlay search

The rule was not retuned on later returns. With all sources physically cut at
`2024-01-01`, 28 fixed execution variants were tested:

- time exits at 24, 36, 48, and 60 hours;
- TP/SL pairs 4%/2%, 4%/3%, 5%/2.5%, 5%/3%, 6%/3%, and 6%/4%;
- 0.5x leverage, next-open entry, stop-before-take ambiguity handling;
- actual funding and the current strict-MDD contract.

Acceptance required at least 60 train trades, 12 selection trades, five trades
in each 2023 half, positive return in every train year and 2023 half, and ratio
at least 3 with strict MDD at most 15% in train, 2023, and combined pre-2024.

**Qualifiers: 0 / 28.** The best row remained the 48-hour time exit:

| Window | CAGR | strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|
| Train | 35.84% | 12.79% | 2.80 | 76 |
| Select 2023 | 22.09% | 9.65% | 2.29 | 17 |
| Combined pre-2024 | 31.77% | 12.79% | 2.48 | 93 |

Because selection failed, no overlay was promoted and later data was not used
to choose an exit.

## Decision

- Do not deploy the old `:55` premium branch.
- Do not use the old 0.9x sizing claim.
- Keep the component weak signals for future interactions, but treat this exact
  strategy as rejected under the current contract.
- Every subsequent alpha search must use a `:00` signal row with prior-bar
  market features and current boundary auxiliary data, ten-minute premium
  freshness, realized funding, and the same strict-MDD engine from the first
  pre-2024 selection run.

## Artifacts

- `training/audit_confirmed_pullback_squeeze_live_parity.py`
- `tests/test_audit_confirmed_pullback_squeeze_live_parity.py`
- `results/confirmed_pullback_squeeze_live_parity_audit_2026-07-15.json`

Reproduce:

```bash
/home/pakchu/rllm/.venv/bin/python \
  training/audit_confirmed_pullback_squeeze_live_parity.py \
  --input-csv /home/pakchu/rllm/data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz \
  --funding-csv /home/pakchu/rllm/data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz \
  --premium-csv /home/pakchu/rllm/data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz
/home/pakchu/rllm/.venv/bin/python -m pytest -q \
  tests/test_audit_confirmed_pullback_squeeze_live_parity.py \
  tests/test_search_specific_pullback_squeeze_alpha.py
```
