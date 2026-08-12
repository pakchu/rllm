# CLSIER-6 source-support pass — 2026-08-12

The frozen settlement-impact efficiency relay passed all outcome-blind source
gates unchanged. Event counts were 22/50/32/20 for train/test/eval/final. The
minimum side share was 0.3182 and the maximum monthly share was 0.2727, versus
required limits of 0.20 and 0.45.

The evaluator recomputed each `abs(interval return) / finalized block weight`
from a SHA-bound pre-entry confirmation-ladder cache. No execution prices,
funding, Gross9 rows, post-entry returns, PnL, or RV20 values were opened.
Rerunning the frozen evaluator reproduced all eight generated artifacts
byte-for-byte. CLSIER-6 advances only to Gross9 structural novelty.
