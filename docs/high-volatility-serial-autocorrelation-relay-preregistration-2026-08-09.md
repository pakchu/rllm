# HVSAR-12 preregistration — 2026-08-09

## Frozen mechanism

At exact 00:00 and 12:00 UTC boundaries, HVSAR reads only the 144 completed
five-minute BTC returns from the preceding twelve hours. It computes realized
variation and the lag-one Pearson autocorrelation of the ordered returns.

The event is eligible only when realized variation ranks in the causal upper
30% and absolute autocorrelation ranks in the causal upper 25%, each against at
most 270 strict-prior source-valid boundaries with at least 252 observations.
Positive serial dependence follows the completed displacement; negative serial
dependence fades it. Entry is the exact boundary+5m open, gross exposure is
fixed at 0.5x, and the hold is twelve elapsed hours.

## Independence boundary

This is not a repair of the terminal variance-ratio, sign-entropy, causal-memory,
or weekly-momentum candidates. HVSAR uses within-path ordered cross-products,
does not reuse their clocks or controls, and fixes one singleton without a grid.
No candidate incidence, Gross9 rows, execution outcomes, or PnL were opened to
choose the rule.

## Gates and stop rule

Source support must satisfy 8/12/12/8 events, 20% minority-side share, and 45%
maximum month share. Gross9 exact, near-6h, occupied-bar, and signed-exposure
novelty must pass before economics. Economics then run train, test, eval, and
final sequentially under fixed 0.5x quantity, exact funding, 6bp/10bp costs,
full-calendar CAGR, and strict held-path MDD. The first failure is terminal; no
path, rank, sign, hold, clock, subset, threshold, or control repair is allowed.
RV20 q90 remains a post-stage audit only.
