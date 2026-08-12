# HVOER-8 source-support pass

The frozen causal-label evaluator produced 30/88/61/24 events with minimum
side share 0.40909 and maximum monthly share 0.30000. Every source gate passed.
Only expert labels whose eight-hour exits were observed by each decision were
used; candidate PnL, funding, Gross9 rows, CAGR, and MDD remained unopened.

Immediate replay was byte-identical:

- states `8608450332bc96ff244e5b94ee902285cc21d26a7bff32457d371c78058291a0`
- source manifest `ed2560b4d2f7c58a9ee010375459b62d2bc1ff9f58fff8658334604276cb5e56`
- clock `f1a0e5765cab81427658f8235b1ead37a3a9065f360f11338a3af6bece4b8b52`
- result `b1d654ee70201a3023b1adf8387a608bae2359cd5e289db5ae8662a0db80638f`

This authorizes only a separately frozen Gross9 novelty evaluator.
