# STCR-72 train economic rejection — 2026-08-09

## Decision

STCR-72 is terminally rejected unchanged at the first sequential economic stage. Test, eval, final, and the post-stage RV20-q90 decomposition were not opened.

## Frozen lineage

- Preregistration SHA-256: `096d6f80f18a2b33475016c8f545f3514b6d1400af7ba0ae1f69039f9d3f8f7e`
- Source-support result SHA-256: `d81e7432dc4db84180034a5c8f7128970ddc5d573c68b4a1fd6036cddbce57d7`
- Gross9 novelty result SHA-256: `0628928321a03356661e6812e7822c2079068982b0df3a38e3b939b1c6809979`
- Economic evaluator SHA-256: `9ef4e093545dbf2905313bdb97906440996e212cfa7ad7297b28da9c0c717d4a`
- Train result SHA-256: `0ad039c37ccc5b65ea495a523af5c1dc03bea213ee0488f85ac268a8c4248d81`

## Train evidence

The unchanged 35-trade train clock produced:

- base absolute return: `12.328986%`
- full-calendar CAGR: `25.958955%`
- strict held-bar MDD: `8.168866%`
- CAGR / strict MDD: `3.177792`
- mean gross underlying move: `86.320262 bp`
- 10 bp/side stress absolute return: `10.765931%`
- stress CAGR / strict MDD: `2.641807`

Two mandatory gates failed:

1. UTC-week cluster sign-flip one-sided p-value was `0.151678`, above `0.10`.
2. The first calendar half returned `-0.618949%`, violating the requirement that both halves be positive.

The second calendar half returned `11.805996%`. This concentration is not repairable under the preregistered first-failure rule.

## Reproducibility and stopping rule

An immediate rerun produced the identical train-result SHA-256 shown above. No threshold, side, hold, subset, or diagnostic-control promotion was attempted. The source clock and all predecessor artifacts remain unchanged.
