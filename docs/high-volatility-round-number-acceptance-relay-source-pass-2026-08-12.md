# HVRNAR-8 source-support pass

The corrected, frozen source-only evaluator passed unchanged and produced
byte-identical artifacts in two independent executions. The first invocation
had stopped after source materialization but before statistics because of an
undefined report-label name; replacing it with the identical preregistered
constant changed no candidate rule or source calculation.

| split | events | long | short | minority share | max month share |
|---|---:|---:|---:|---:|---:|
| train | 54 | 33 | 21 | 0.3889 | 0.2593 |
| test | 224 | 122 | 102 | 0.4554 | 0.1518 |
| eval | 255 | 131 | 124 | 0.4863 | 0.1765 |
| final | 115 | 61 | 54 | 0.4696 | 0.2522 |

All frozen 8/12/12/8 incidence, 0.20 minority-side, and 0.45 monthly
concentration gates passed. No post-entry price, return, PnL, funding, Gross9
row, or economic metric was opened. The unchanged candidate may advance only
to Gross9 structural novelty.
