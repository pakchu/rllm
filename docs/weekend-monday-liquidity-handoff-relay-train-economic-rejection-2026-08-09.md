# WMLHR-16 train economic rejection — 2026-08-09

WMLHR-16 is terminally rejected unchanged at the first sequential economic
stage. Test, eval, final, and the post-stage RV20 q90 decomposition were not
opened.

The 8-trade 2023H2 train clock returned `-0.425141%` at 6 bp per notional
side. Full-calendar CAGR was `-0.842162%`, strict held-bar MDD was `2.423365%`,
and CAGR/MDD was `-0.347517`. Mean gross underlying movement in the selected
direction was `1.392373 bp`; the 10 bp stress return was `-0.743496%`.
Weekly cluster sign-flip p-value was `0.595194`. The first calendar half was
positive (`0.404998%`) but the second was negative (`-0.826791%`).

- Preregistration SHA-256: `3f741f7c7750844d214c9b46fe28b123baa52e0cd3ed76e4b4755b7950e06b5c`
- Source-support SHA-256: `0e71d70affd44afa61df08e7cfaf95ffa9d10276720697d021d30c7d1b538c88`
- Gross9 novelty SHA-256: `e52c13b10af68b0d3ea42cc53940125139257a904c1615ef810679e6d11d7b73`
- Economic evaluator SHA-256: `33b0333dbb50a238c3f1cd241b5c99530b3db652fe414b8e9b7eeb5f3fe75d1c`
- Train result SHA-256: `aeef494b38bffd821c2d9332de8efcde9b810f94e393be4accfce1ef3b65188f`

An immediate rerun reproduced the train result byte-for-byte. No liquidity
threshold, weekend/Monday window, direction, entry, hold, volatility, subset,
or control repair was attempted, and no diagnostic control may be promoted.
