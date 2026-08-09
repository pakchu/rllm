# RBEFR-8 train economic rejection — 2026-08-09

RBEFR-8 is terminally rejected unchanged at the first sequential economic
stage. Test, eval, final, and the post-stage RV20 q90 decomposition were not
opened.

The 101-trade 2023H2 train clock returned `-6.116362%` at 6 bp per notional
side. Full-calendar CAGR was `-11.775444%`, strict held-bar MDD was
`10.636014%`, and CAGR/MDD was `-1.107129`. Mean gross underlying movement was
only `0.151343 bp`; the 10 bp stress return was `-9.839013%`. Weekly cluster
sign-flip p-value was `0.773402`. Calendar halves were `0.540864%` and
`-6.621414%`.

- Preregistration SHA-256: `a3611f577e227a2aba9fb6affb13fabe906d42b0da10c2995c78a54c6e9f5e72`
- Source-support SHA-256: `32276081f69150fbe39aebcd2f2a19dd045f219525d3859aa3c4850c1c34ff7f`
- Gross9 novelty SHA-256: `5518e1c5f3900f207b112bb118dc4f90b7785df358d47eaffc4b39a0088bf522`
- Economic evaluator SHA-256: `b8a9fa67bdd767bd628cf81af24cd35ad4518a58ec30ae0244d275710c5aeeb7`
- Train result SHA-256: `4b077c1945271a585456fe21e531ba87d24b9791451e8a71bbaa978148de9659`

An immediate rerun reproduced the train result byte-for-byte. No estimator,
rank, history, onset, impulse, direction, hold, volatility, subset, or control
repair was attempted, and no diagnostic control may be promoted.
