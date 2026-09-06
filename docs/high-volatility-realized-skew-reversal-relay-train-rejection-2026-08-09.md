# HVRSR-12 train economic rejection

HVRSR-12 is terminally rejected unchanged at the first sequential economic
stage. Test, eval, final, and the post-stage RV20 q90 audit remain unopened.

The 31-trade 2023H2 train clock returned `+0.787173%` at 6 bp per notional
side, but full-calendar CAGR was only `1.568640%`, strict held-bar MDD was
`4.685263%`, and CAGR/MDD was `0.334803`, below the frozen `3.0` minimum.
Mean gross underlying movement was `17.462575 bp`, below `20 bp`; the weekly
cluster sign-flip p-value was `0.409046`; and 10 bp stress return was
`-0.454640%`. Both calendar halves were positive, but that cannot override the
other terminal failures.

An immediate rerun reproduced the train result byte-for-byte. Train result
SHA-256: `6403495e0c30d9d3b0b5f8e7024bec95f90a2b5ad59573ad203ea6cbe4e1c15c`.
No moment formula, rank, direction, clock, hold, subset, or diagnostic control
was repaired or promoted.
