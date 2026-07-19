# TGR-12 source-only preregistration — 2026-07-19

## Status

`TGR-12` was rejected before opening any BTC outcome. **No BTC price, funding, return, excursion, PnL, equity, CAGR, or MDD
was opened.** The corrected causal MAD produced only 69 test-year events versus the frozen minimum of 75; the exact policy therefore cannot pass its sequential gate.

## Frozen mechanism

- Source: completed-hour mean ticket and normalized taker flow for six USD-M
  alt perpetuals; no OHLC is present in this source artifact.
- Ticket surprise: robust per-symbol z-score against the strictly prior 720
  hours with at least 672 observations.
- Leaders: the two largest current ticket surprises, ties broken by frozen
  alphabetical symbol order.
- Release: both leader flows agree, their mean absolute flow reaches strictly
  prior q90, and the
  leader-versus-crowd ticket gap reaches strictly prior
  q70, while the
  bottom-four crowd flow remains below its prior median.
- Side: sign of the top-two mean flow.
- Clock: false-to-true onset, entry `+5m`, fixed `12h` hold, one position.

The selected cell maximized mechanism strength among cells passing 2023
source-incidence gates. Future source incidence was opened only after selection
and could not alter the cell.

## Source-only incidence

| Stage | Events | Long | Short | Max month share |
|---|---:|---:|---:|---:|
| train 2023 | 60 | 33 | 27 | 0.167 |
| test 2024 | 69 | 29 | 40 | 0.159 |
| eval 2025 | 79 | 43 | 36 | 0.215 |
| final 2026H1 | 42 | 21 | 21 | 0.310 |

## Clock novelty

- FCIR-12: exact Jaccard `0.0020`, ±6h max near-share `0.1057`
- SQFD-6: exact Jaccard `0.0081`, ±6h max near-share `0.3048`
- OPDR-24: exact Jaccard `0.0000`, ±6h max near-share `0.0720`
- PCBR-12: exact Jaccard `0.0000`, ±6h max near-share `0.1505`
- PSR-30/6: exact Jaccard `0.0000`, ±6h max near-share `0.1520`

## Sequential outcome rule

The strict evaluator and every control would have to be committed before 2023
BTC execution outcomes could be opened. Because the frozen annual test trade
minimum is already impossible from source incidence, no evaluator is frozen and
all BTC outcomes remain sealed. No threshold, side, hold, or control repair is
allowed for this exact policy.
