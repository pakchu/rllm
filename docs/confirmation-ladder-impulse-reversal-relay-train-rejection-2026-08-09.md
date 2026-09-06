# CLIRR-6 terminal train-economic rejection

CLIRR-6 passed source support and every Gross9 novelty gate, then failed the
first strict economic stage. Two train executions reproduced result SHA-256
`9e065d828d1f0cff681f1a11c96114e369a25066bb6c5d1e3291e6c27211beee`.

- base absolute return: `+0.9880%`
- full-calendar CAGR: `+1.9708%`
- strict MDD: `1.5732%`
- CAGR / strict MDD: `1.2527`
- trades: `14` (`8` long / `6` short)
- mean gross underlying move: `26.3855 bp`
- weekly sign-flip p-value: `0.2821`
- stress absolute return: `+0.4249%`
- stress CAGR / strict MDD: `0.5055`
- calendar-half returns: `-0.5040%`, `+1.4996%`

Absolute return, strict-MDD ceiling, gross-move floor, and stress return
passed. The ratio, significance, stress ratio, and both-halves gates failed.
CLIRR-6 is rejected unchanged; test/eval/final, RV20, and any diagnostic
promotion remain unopened. No q80, side, early/late partition, anchor,
confirmation, embargo, or hold repair is permitted.
