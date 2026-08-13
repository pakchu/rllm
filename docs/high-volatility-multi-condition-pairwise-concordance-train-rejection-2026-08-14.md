# HVMCPAC-8 train economics rejection

The sole source- and Gross9-supported pair,
`CARSC-8__AND__HVTFR-8`, was opened on the train window only. Test, eval, and
final outcomes remain sealed. The strict evaluator and complete 22-test battery
reproduced twice byte-for-byte.

## Train result

- 22 trades (9 long, 13 short)
- base absolute return: +3.4787%
- base full-calendar CAGR: 7.0237%
- strict MDD: 5.5848%
- base CAGR/MDD: 1.2576 (required >=3.0)
- mean gross underlying move: 43.2455 bp
- weekly cluster sign-flip p: 0.18119 (required <=0.10 and train-family raw <=0.01667)
- stress absolute return: +2.5725%
- stress CAGR/MDD: 0.8592 (required >=2.5)
- calendar-half returns: +0.1346%, +3.3396%

The pair passes absolute return, MDD, gross-move, stress-return, and both-half
sign gates, but fails base risk-adjusted return, weekly evidence, Bonferroni
family evidence, and stress risk-adjusted return. Under the preregistered raw
rank-one/no-substitution rule, HVMCPAC-8 is terminally rejected at train. No
other pair may replace it and no component, threshold, clock, side, or hold may
be repaired.

The deterministic train result SHA-256 is
`7a49f8a22747d52f6634efd31b5819aa2b31222bf29fb3eeb6faa623aa2f8916`.
