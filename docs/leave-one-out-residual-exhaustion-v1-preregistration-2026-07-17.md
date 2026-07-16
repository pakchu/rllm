# LORE v1 preregistration — 2026-07-17

## Evidence boundary

**Support-only. No LORE post-entry return has been opened.** The physical
selection prefix ends before 2025. Calendar 2025 is opened only after one
2023–2024 policy is committed; 2026 is opened only after that policy passes
2025. Repository-wide human-pristine status is not claimed.

## Orthogonal mechanism

LORE trades six Binance USD-M alt perpetuals, not BTC. It removes a causal
leave-one-out crypto factor from each asset, identifies the residual winner and
loser, and acts only when aggressor flow fails to confirm both residual tails.
It buys the loser and shorts the winner with ex-ante factor-beta-neutral,
gross-one weights. It uses no BTC REX, OI, funding/premium gate, Kimchi/FX,
Markov state, or LLM prediction.

## Frozen policies

| Policy | Residual horizon | Hold |
|---|---:|---:|
| L01 | 6h | 12h |
| L02 | 6h | 24h |
| L03 | 12h | 12h |
| L04 | 12h | 24h |

The signal requires winner residual z >= 1.5, loser residual z <= -1.5,
winner residual-minus-flow z >= 1.0, and loser flow-minus-residual z >= 1.0.
Betas use a shifted 720-hour rolling estimate; residual and flow z-scores use
only the prior shifted 2,160 hours. Every completed hour requires all 12 five-
minute bars for all six symbols, with no fill or nearest join.

## Execution and strict risk

- signal: right edge of a completed hour;
- entry: signal + 5 minutes open;
- exit: entry + fixed 12h or 24h;
- base cost: 6 bp/notional/side; stress: 10 bp;
- exact per-symbol funding for `entry < funding_time <= exit`;
- full-calendar CAGR;
- strict MDD uses global/pre-entry HWM, entry and hypothetical liquidation
  costs, funding debit/credit ordering, and simultaneous conservative
  long-high/short-low favorable then long-low/short-high adverse marks.

## Selection and holdout

2023 fit and 2024 test select at most one of four policies. A policy needs
positive return and ratio >= 1.5 in each year, at least three positive halves,
combined CAGR/strict-MDD >= 3, strict MDD <= 12%, >= 150 combined trades,
10-bp cost survival, and Bonferroni weekly sign-flip p <= 0.10. The frozen
winner then needs 2025 ratio >= 3 and strict MDD <= 10% before 2026 is opened.

## Anti-repair rule

If no policy passes, LORE v1 ends without opening 2025. Direction flip,
threshold changes, pair whitelists, alternate holds, and regime gates are
diagnostics only and cannot rescue the family.

Protocol hash: `18480ed99902cecc126fcd4e5d9f5df40c98e65878bfecfb547e2941084be840`
