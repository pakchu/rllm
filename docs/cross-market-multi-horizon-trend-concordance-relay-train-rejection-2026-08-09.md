# CMMTCR-24 train economic rejection

CMMTCR-24 is terminally rejected unchanged at train. Test, eval, final, and
RV20 remain unopened.

The 32-trade 2023H2 clock returned `+3.023957%`; mean gross movement was
`33.469880 bp`, stress return was `+1.708860%`, and both calendar halves were
positive. However full-calendar CAGR/MDD was only `0.777361` versus `3.0`,
stress CAGR/MDD was `0.407152` versus `2.5`, and the weekly sign-flip p-value
was `0.391136` versus `0.10`.

An immediate rerun reproduced the result byte-for-byte. Train SHA-256:
`5adcdaceb14ab348826447e9561e775257ac29ff3c4cd78f95217fb3690b335f`.
No horizon, volatility gate, side, clock, hold, subset, or diagnostic control
was repaired or promoted.
