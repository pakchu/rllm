# LURI-48 frozen selection result — 2026-07-14

## Decision

**REJECT LURI-48 v1.** The outcome-blind clock had strong structural support,
but the frozen pre-2024 return gate failed decisively. No threshold, direction,
hold, funding endpoint, cost, MDD convention, or control was repaired after
opening the result. Calendar 2024 and later was not read.

- evaluator source commit:
  `cfb40808794a376e9e380d18934396a13a23e6c2`
- evaluator source SHA-256:
  `8e6bcb6920b70f2cff40072590b4d04a0a7d775a7069bb16e406a620defac7d8`
- pre-outcome freeze commit:
  `629af486d3101d280c12d534c6e5e7ed6d2539ee`
- pre-outcome freeze SHA-256:
  `c67b99f8c7ec308c35d57c207363dfa29ff4d60f209b4bf511b64404acc9103b`
- result:
  `results/leveraged_um_inventory_release_handoff_selection_2026-07-14.json`
- result SHA-256:
  `09256c49cfaa37961c46f13b3590127af949f0ffa68d08c2f1a5c379fa991bad`

## Primary statistics

All returns include `0.5x` leverage, 5 bp fee plus 1 bp slippage per notional
side, and every exact realized-funding settlement at the conservatively
inclusive entry/exit endpoints. CAGR uses the full wall-clock split.

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Gross move | Trades | Funding settlements | Cluster p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train 2020–2022 | -28.75% | -10.68% | 29.59% | -0.36 | -7.37 bp | 334 | 180 | 0.9926 |
| full 2023 | -4.07% | -4.07% | 5.08% | -0.80 | +3.63 bp | 98 | 35 | 0.8327 |
| 2023 H1 | -1.75% | -3.49% | 2.80% | -1.25 | +4.26 bp | 45 | 18 | 0.8161 |
| 2023 H2 | -2.37% | -4.64% | 3.45% | -1.34 | +3.08 bp | 53 | 17 | 0.7238 |

Both halves met the frozen minimum event count, so rejection is not caused by
an underpopulated split. Both halves lost money, train strict MDD exceeded 15%,
train/full-2023 ratios were negative, and neither cluster test approached the
one-sided `p < 0.10` requirement.

## Control evidence and diagnosis

The exact direction flip changed mean gross move from `-7.37 bp` to `+7.37 bp`
in train, but from `+3.63 bp` to `-3.63 bp` in 2023. It still lost `-7.98%`
in train and `-7.49%` in 2023. This rejects a simple sign error: the direction
that helped one era failed in the other, and even its favorable train gross
move remained below the frozen 12 bp round-trip underlying break-even hurdle.

No score-bearing control produced a stable profitable alternative. Their
minimum train/full-2023 CAGR/MDD values ranged from `-1.00` to `-0.78`.
Spot-inventory swap and stale-24h marginally exceeded the primary's negative
minimum ratio, causing two additional frozen relative-control failures, but
both controls themselves lost money in both selection windows.

The central failure is therefore economic, not support or implementation:

1. inferred USD-M inventory does not carry a stable release direction across
   train and 2023;
2. the small positive gross moves that appear in one era are below costs;
3. one-bar delay retains the same failure (`-8.92 bp` train, `+3.42 bp` 2023),
   so the issue is not merely a fragile single-bar fill;
4. reverse-time and simultaneous controls are worse, so ordered handoff exists
   structurally but does not imply a profitable four-hour continuation/reversal.

## Verification

- two complete evaluator runs produced the byte-identical result SHA-256;
- runtime was about 11 seconds and peak RSS about 0.68 GB;
- an independent one-off recomputation loaded only the frozen primary clock,
  OHLC, and funding files and separately reproduced, to floating-point
  equality, each window's absolute return, CAGR, strict MDD, ratio, direct
  gross bp, trade count, funding-settlement count, and 100,000-draw weekly
  cluster p-value;
- the evaluator enforced next-open entry, exact 48-bar hold, non-overlap,
  favorable-first adverse-path MDD, funding-debit ordering, and sealed 2024
  boundaries.

## Research consequence

LURI-48 must not be promoted or tuned. The useful retained result is negative:
support/novelty alone can identify a real market microstructure episode while
still providing no tradable directional edge. The next candidate should make
its payoff mechanism conditional on an independently causal state transition,
not merely on inferred inventory release, and must again be selected without
opening returns.
