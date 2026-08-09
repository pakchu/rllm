# TECC-12 train economic rejection — 2026-08-09

TECC-12 is terminally rejected unchanged at the first sequential economic
stage. Test, eval, final, and the post-stage RV20 q90 decomposition were not
opened.

The 107-trade 2023H2 train clock returned `3.171052%` at 6 bp per notional
side, but full-calendar CAGR of `6.393007%` divided by strict held-bar MDD of
`10.066217%` was only `0.635095`. Mean gross underlying movement was
`18.500533 bp`, below the frozen 20 bp gate. The 10 bp stress return was
`-1.155749%`, and weekly cluster sign-flip p-value was `0.354416`. Calendar
halves were unstable: `-4.762288%` then `8.330040%`.

- Preregistration SHA-256: `35b3912969893bf77886ffc687961a242c4234901232c25c991366a3df19d526`
- Source-support SHA-256: `912355c59cdde6c5e164afbf8e971b6b5ac1bf8d66b73a23f373d457f98b09fb`
- Gross9 novelty SHA-256: `3594049b25a742aef0919b2ff62a601e930ecfccd876600f88c95d25bc938c1f`
- Economic evaluator SHA-256: `57348717c8e51d8ed016c96a61a34ffdd54d45089f04d0dcbd1ba04a3fc227ed`
- Train result SHA-256: `5d4e9d3484ca57c940140f6bdf926f2ffe625a3f05f3781d3f67ef46a6641cc2`

An immediate rerun reproduced the train result byte-for-byte. No rank, history,
onset, block, direction, hold, volatility, subset, or control repair was
attempted, and no diagnostic control may be promoted.
