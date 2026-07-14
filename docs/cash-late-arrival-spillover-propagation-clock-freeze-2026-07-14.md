# CLASP-24 clock freeze plan — 2026-07-14

This stage freezes the selected CLASP-24 support clock before any CLASP outcome
calculation.

## Frozen inputs

- support artifact: `results/cash_late_arrival_spillover_propagation_support_2026-07-14.json`
- expected support SHA-256: `bd26905f7c33360a62c9eb14cef23ba917612e64fc5d83e47e25b50b56db8930`
- selected quantile: `0.75`
- preregistration commit: `29e3983`
- support artifact commit: `aa6fab4`

## Clock invariants

The generator must reproduce the selected q=0.75 primary signal from the causal
pre-2024 source and emit only the frozen schedule schema:

`signal_position, entry_position, exit_position, signal_date, entry_date, exit_date, side, branch, hold_bars`

Validation rejects:

- any schedule column containing outcome tokens such as future, return, PnL,
  funding, high/low/open/close, CAGR, or MDD;
- any entry not exactly one completed five-minute bar after the signal;
- any exit not exactly 24 bars after entry;
- overlapping held intervals;
- non-directional actions;
- non-primary branch names;
- any support clock whose exit reaches `2024-01-01` or later.

The output remains support-clock only. It does not calculate return, strict MDD,
funding, CAGR, or 2024+ behavior.
