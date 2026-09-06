# TSPI-12 train economic rejection — 2026-08-09

TSPI-12 is terminally rejected unchanged at the first sequential economic
stage. Test, eval, final, and the post-stage RV20 q90 decomposition were not
opened.

The 26-trade 2023H2 train clock returned `-2.464692%` at 6 bp per notional
side. Full-calendar CAGR was `-4.833148%`, strict held-bar MDD was `4.116193%`,
and CAGR/MDD was `-1.174179`. Mean gross underlying movement in the selected
direction was `-7.235785 bp`; the 10 bp stress return was `-3.476212%`.
Weekly cluster sign-flip p-value was `0.831262`, and both calendar halves were
negative (`-0.267356%`, `-2.203226%`).

- Preregistration SHA-256: `2ed7738d0ed934d7e01a8d0bfd9b1f0c81b0bf75bcf43d3a8b49a7c294aaf633`
- Source-support SHA-256: `8a8777bd439df1476edf7f7bb85c8d58f22174ff7e7502d2dc8db503fd78e7c7`
- Gross9 novelty SHA-256: `85a7c1e3b1b6451fb0e448c4fff86067805ef81849c042dc603b1d32b9857ef7`
- Economic evaluator SHA-256: `6b2c3f934bc9f9973e3ffc6dabc027115a75de18b5fcbfc7f3e3e16b20c1b75f`
- Train result SHA-256: `237551e8648d61b36f5c06e660bc2a34e4ac312eac217912a7a80f7c70c0e527`

An immediate rerun reproduced the train result byte-for-byte. The maximum-run
diagnostic was slightly positive but failed the primary economic standard and
cannot be promoted. No weekday, run transform, side, hold, subset, threshold,
or control repair was attempted.
