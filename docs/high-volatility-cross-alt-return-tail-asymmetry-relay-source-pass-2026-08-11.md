# HVCARTAR-8 source support pass — 2026-08-11

The frozen source evaluator produced 49/73/62/50 train/test/eval/final events.
Every split passed its minimum event count, 20% minority-side share, and 45%
maximum-month-share gate. Minority-side shares were 38.8%, 31.5%, 35.5%, and
38.0%; maximum month shares were 22.4%, 17.8%, 19.4%, and 28.0%.

Only completed official Binance USD-M one-minute candles and causal prior ranks
were opened. Execution prices, funding values, Gross9 clocks, post-entry
returns, and PnL remained sealed. The source clock is therefore authorized only
for the frozen Gross9 novelty comparison; economics remains unauthorized until
that comparison passes.
