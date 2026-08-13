# HVSOF-8 train rejection

All three source- and Gross9-supported filters were evaluated on train only using
fixed 0.5 gross, exact funding, 6/10bp per side, strict favorable-then-adverse 5m
MDD, and full-calendar CAGR. Test/eval/final remained sealed.

Raw ranking was:

1. `HVSVF-8__FILTERED_BY__HVLZC-8`: -0.2153%, CAGR/MDD -0.0848,
   mean gross move 10.82bp, weekly p 0.5343.
2. `HVRSSR-8__FILTERED_BY__HVLZC-8`: -4.3047%, CAGR/MDD -0.8838.
3. `HVRSSR-8__FILTERED_BY__HVTCCR-8`: -1.5419%, CAGR/MDD -1.5159.

Raw rank one failed positive return, risk-adjusted, gross-move, weekly,
Bonferroni, stress, and both-half gates. HVSOF-8 is therefore terminal with no
winner and no substitution. The result and ten-test suite reproduced twice.
SHA-256:
`856abb431b294adfe0a6737d7d71fb1a78b8cf32a7c5ea6d4c05643cd2344f08`.
