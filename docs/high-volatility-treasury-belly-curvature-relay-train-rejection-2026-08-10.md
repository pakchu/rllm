# HVTBCR-24 terminal train rejection

The frozen HVTBCR-24 train evaluator was run after the economic authorization
artifact was pushed.  The 32-trade 2023H2 clock returned `+1.356886%` at the
base cost, with `21.927751 bp` mean gross movement and positive returns in both
calendar halves.  It nevertheless failed the terminal gates: full-calendar
CAGR/strict-MDD was only `0.380768`, the weekly cluster sign-flip p-value was
`0.392806`, and the 10 bp stress CAGR/strict-MDD was `0.017682`.  Strict MDD
was `7.121108%`; stress absolute return remained barely positive at
`+0.066243%`.

An immediate replay was byte-identical.  Result SHA-256:
`7b2bc6f9afc14b322a78f341a6d5f741dc3bced5e10fae6434fa70edfd66cfa9`.
The candidate is terminal without maturity, curvature, side, threshold,
clock, hold, subset or control repair.  Test, eval, final and RV20 q90 remain
sealed.
