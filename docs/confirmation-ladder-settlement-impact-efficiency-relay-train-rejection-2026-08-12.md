# CLSIER-6 train rejection — 2026-08-12

CLSIER-6 passed source support and all Gross9 novelty checks, then failed the
first sequential economic stage under the frozen accounting contract.

- return: +0.8152%
- full-calendar CAGR: 1.6246%
- strict MDD: 1.6395%
- CAGR/MDD: 0.9909 (required 3.0)
- mean gross move: 19.3868 bp (required 20 bp)
- weekly sign-flip p: 0.3310 (required at most 0.1)
- stress return: -0.0684%
- calendar halves: both positive

The frozen evaluator reproduced byte-identically at SHA-256
`9e062c9d8182a4c57e63fa1af2710edc8e177b8b5d73ec3c404f597db9471a6a`.
No test/eval/final outcomes or RV20 audit were opened. CLSIER-6 is terminally
rejected unchanged; its impact formula, clock, side, hold, threshold-free
comparison, and controls cannot be repaired or promoted.
