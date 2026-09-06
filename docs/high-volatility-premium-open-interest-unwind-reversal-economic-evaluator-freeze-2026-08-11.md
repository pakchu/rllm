# HVPOIUR-8 economic evaluator freeze — 2026-08-11

The strict sequential evaluator is frozen before opening any execution price, funding PnL, or post-entry return. It includes outcome-blind `load_clock_allow_empty` handling so a legitimate zero-row diagnostic control cannot alter the accounting path after outcomes are visible.

The immutable stage order is train, test, eval, final. Every stage uses fixed 0.5 gross, exact held funding, 6bp/10bp per notional side, full-calendar CAGR, and strict held-five-minute favorable-then-adverse MDD. A failed stage terminates the candidate without repair.
