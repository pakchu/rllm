# Side-token interaction negative result (2026-06-24)

## Purpose

The current ranker has numeric side interactions, but token one-hots are additive and shared across LONG/SHORT. A plausible model-structure improvement was to add `state_token × action_side_sign` interactions so regime tokens can affect LONG and SHORT differently.

## Result

The interaction was tested under the current best protocol:

- PA-ext input
- 6M fit / 3M validation / 3M test
- stats gate
- pair half-life 45d
- light side scaling denominator 0.5

Report: `results/event_candidate_pairwise_walkforward_paext_6m3m3m_decay45_sidescale_d0p5_sidetok_2026-06-24/report.json`

Result:

- CAGR: -4.48%
- Strict MDD: 36.77%
- CAGR/MDD: -0.12
- Trades: 157
- p approx: 0.605

## Conclusion

Side-token interactions are not useful in this form. They likely add sparse capacity and open weak folds without enough stable evidence. The code change was reverted; do not reintroduce this as a default feature expansion without stronger regularization or a separate validation reason.
