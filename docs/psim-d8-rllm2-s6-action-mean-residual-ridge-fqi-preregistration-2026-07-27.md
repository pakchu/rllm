# PSIM-D8-RLLM2-S6 all-action-mean residual Ridge FQI preregistration

Date: 2026-07-27 KST

Status: **PREREGISTERED — no raw market/funding payload or 2021 policy outcome opened**

## Hypothesis

S5 solved S4's direction collapse:

- S5 long / short targets: 44 / 30;
- direction shares: 59.46% / 40.54%;
- action-code permutation mismatches: 0.

It nevertheless failed the unchanged activity gate with 74 non-flat target
rows versus the required 80. S6 does not lower the threshold, calibrate a
Q-margin, or impose a trade quota.

The new single hypothesis is that the remaining abstention comes from the
unconditional flat-action baseline. S6 removes the unconditional mean of
**flat, short, and long** within each current position, forcing Ridge FQI to
learn only source-state-conditional action deviations.

## Fixed transform

For current position \(p\), action \(a\), and all finite reachable 2020 rows:

\[
\mu_{p,a} = \operatorname{mean}_t R_{t,p,a}
\]

\[
\bar{\mu}_p =
\frac{
\mu_{p,\mathrm{flat}}+
\mu_{p,\mathrm{short}}+
\mu_{p,\mathrm{long}}
}{3}
\]

\[
R^*_{t,p,a} =
R_{t,p,a} - \mu_{p,a} + \bar{\mu}_p
\]

This makes the mean residual reward of all three actions equal within each
current position. It is applied to the original S4 strict transition ledger,
not to the S5 residual ledger.

Frozen exclusions:

- no clipping or scaling;
- no threshold or hyperparameter search;
- no target quota;
- no Q-margin calibration;
- no reuse of the observed S5 residual deltas; and
- no 2021 price, funding, reward, PnL, or economic metric.

## Model and policy family

The sole promotable primary is:

```text
semantic_ridge_action_mean_residual_fqi
```

It preserves:

- frozen `google/gemma-4-E4B-it` source embeddings at revision
  `ee0ef6023621cff504d758262d4e04895a5af4a2`;
- frozen 2020-only PCA32;
- Ridge alpha 100 with unpenalized intercept;
- 25 Bellman iterations;
- discount 0.99;
- gross targets 0.0 / -0.5 / +0.5; and
- tie order flat, current target, short, long.

Seven new controls are fixed: action-code permutation, direction flip,
circular-21 reward, within-month shuffled reward, current-position-only,
masked semantic, and metadata/frontmatter-only.

The later familywise family includes all frozen S4, S5, and S6 policies:
25 + 8 + 8 = **41 unique schedules**. Only the S6 primary is promotable.

## Unchanged pre-2021 gate

Before any S6 2021 market/funding access:

- 365 base and 365 delayed primary rows;
- at least 80 non-flat targets;
- at least 20% long and 20% short among non-flat targets;
- delayed target counts equal base;
- exact action-code permutation target identity;
- positive Hamming distance from every degenerate control; and
- at least one target different from the rejected S5 primary.

Any failure terminally rejects S6 without computing 2021 economics.

## Future report-only transfer gate

Only a readiness pass may authorize a separate evaluator. It must require:

- positive 6 bp base, 10 bp stress, and +5m delayed absolute returns;
- positive first-half and second-half returns;
- full-calendar CAGR / strict MDD at least 1.0;
- at least 80 non-flat intervals and 20% exposure in each direction;
- defeat of the strongest required nonsemantic control; and
- weekly familywise max-stat \(p_{\max}<0.25\) over all 41 policies.

2021 is already globally contaminated by unrelated repository research. Any
future result is a policy-specific report-only transfer, not globally pristine
OOS evidence or live-capital authorization.

## Access boundary

At preregistration:

- raw market/funding paths read: 0;
- original 2020 transition rows parsed: 0;
- S6 2021 market/funding paths read: 0;
- S6 2021 rewards/metrics: 0 / 0;
- model loads/forwards: 0 / 0.

Only exact predecessor and frozen-input bytes were hash-verified.

## Immutable artifact

- preregistration:
  `results/psim_d8_rllm2_s6_action_mean_residual_ridge_fqi_preregistration_2026-07-27.json`;
- file SHA-256:
  `3f80a51a001d422e4aea43072b1be157b5a311f14c7f0feb50fe9b36f413f219`;
- manifest hash:
  `e35975dc79e6dd0dd694f0998cb1fee7100c8e0d8d98585c74cf025fb501abdb`;
- predecessor S5 result hash:
  `2331b46e3847d80200c88b3f3522ef35f3f710dead5ea6dca85575f249b0541e`;
- original strict ledger SHA-256:
  `07d465538d84648793ebbf302c54dace38ef71e88b22d7bd3a19fec500b99a7a`.

## Next authorized step

Implement, independently review, commit, and push the exact S6 runner. It may
then parse the original frozen 2020 transition ledger, fit the eight-policy
family, seal 2021 schedules, and run only the outcome-blind readiness gate.
