# FPMR-1 preregistration — 2026-07-18

## Hypothesis

FPMR-1 is a weekly market-neutral alt pair. It buys the alt whose 30-day
beta-residual trend is both strong and improving while its relative 28-day
funding pressure is easing. It shorts the opposite profile. The score is a
fixed equal-coefficient sum of three cross-sectional weak signals; no outcome
threshold is fitted.

## Causal clock

- Price inputs end at Sunday 23:55 UTC.
- The Monday 00:00 funding settlement is allowed only after it is observed.
- Decision is Monday 00:05, entry is the 00:10 open, and exit is the following
  Monday 00:10 open.
- The two legs use gross-one factor-beta-neutral weights.

## Evidence boundary

Earlier cross-alt results are known, but this exact level + rotation - funding
change score has not been evaluated. Support may inspect only timestamps,
predictor values, pair identity, and concentration. The singleton strict
evaluator must be committed and hash-frozen before 2023 post-entry outcomes
are opened. A failed 2023 gate keeps 2024+ sealed.

## Qualification

2023 must have positive absolute return, full-calendar CAGR/strict-MDD at
least 3, strict MDD at most 15%, at least 40 trades, positive halves, positive
10 bp/side stress, and weekly-cluster p <= 0.10. Controls are falsification
only and cannot repair the primary policy.

Protocol hash: `41c33c9a4028e78ae9007e419dcc40c64a66e5309cf1b8ea2623f83262b20cc1`
