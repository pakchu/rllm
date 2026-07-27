# PSIM-D8-RLLM2-S5 direction-residual Ridge FQI preregistration

Date: 2026-07-27 KST

Status: **PREREGISTERED — no raw market/funding payload or new economic metric opened**

## Purpose

S4 produced attractive 2020 in-sample statistics but failed before its
policy-specific 2021 evaluation:

- Ridge sealed only 14 short targets among 113 non-flat 2021 targets
  (12.39%);
- ExtraTrees sealed 40 non-flat targets, no short target, and changed two
  targets under a neutral action-code permutation; and
- no primary passed the already preregistered activity/direction readiness
  gate.

S5 tests one narrow repair rather than another hyperparameter search:
**remove the unconditional 2020 long-vs-short reward drift per current
position, then refit one deterministic semantic Ridge FQI**.

S5 keeps the RLLM structure:

- language representation:
  `google/gemma-4-E4B-it`;
- exact revision:
  `ee0ef6023621cff504d758262d4e04895a5af4a2`;
- frozen 2,560-dimensional source embeddings;
- frozen 2020-only PCA32;
- strict counterfactual 2020 transition rewards; and
- offline fitted-Q learning with the current position as a separate input.

No model is loaded, no Gemma forward is run, and no QLoRA is authorized in
this stage.

## Fixed reward transform

For each current position \(p\), using all finite reachable 2020 rows:

\[
\delta_p =
\frac{1}{2}
\operatorname{mean}_t
\left(
R_{t,p,\text{long}} - R_{t,p,\text{short}}
\right)
\]

The residual reward is:

\[
\begin{aligned}
R^*_{t,p,\text{flat}} &= R_{t,p,\text{flat}} \\
R^*_{t,p,\text{short}} &= R_{t,p,\text{short}} + \delta_p \\
R^*_{t,p,\text{long}} &= R_{t,p,\text{long}} - \delta_p
\end{aligned}
\]

This forces the mean long and short reward to be equal within each current
position. It does not manufacture a target quota and does not use any 2021
price, funding, return, reward, or PnL.

Frozen choices:

- no clipping;
- no scaling;
- no reward threshold;
- no delta search;
- no model/hyperparameter sweep; and
- residualize first, then apply the circular or within-month reward controls.

## Single primary

Promotable policy:

```text
semantic_ridge_direction_residual_fqi
```

Fixed FQI:

- estimator: Ridge;
- alpha: 100.0;
- intercept: unpenalized;
- Bellman iterations: 25;
- discount: 0.99;
- actions: flat, short, long;
- account gross: 0.0, -0.5, +0.5;
- tie break: flat, current target, short, long; and
- features: frozen semantic PCA32 plus current-position one-hot.

ExtraTrees is not authorized. The repair is deliberately one-dimensional:
reward-drift removal only.

## Frozen controls

S5 seals eight new schedules:

1. semantic direction-residual Ridge primary;
2. action-code permutation;
3. direction flip;
4. circular-21 residual reward;
5. within-month shuffled residual reward;
6. current-position-only residual Ridge;
7. masked-semantic residual Ridge; and
8. metadata/frontmatter-only residual Ridge.

The existing 25 S4 schedules remain frozen, non-promotable comparison
controls. The later weekly familywise family therefore contains exactly 33
predeclared policies.

## Outcome-blind schedule gate

Before any 2021 market or funding payload may be read, the primary must satisfy
all of:

- exactly 365 base schedule rows and 365 delayed rows;
- at least 80 non-flat targets;
- at least 20% long and 20% short among non-flat targets;
- delayed target counts equal base target counts;
- exact target identity under canonicalized action-code permutation; and
- at least one target different from each degenerate control:
  always-flat, always-long, always-short, persistence, current-position-only,
  masked-semantic, and metadata-only.

Failure is terminal for S5 and creates no 2021 reward or economic metric.
Passing authorizes only a separately preregistered 2021 transfer evaluator.

## Future transfer gate

If and only if the schedule gate passes, a later evaluator may score the
single primary under:

- base cost: 6 bp per changed notional;
- stress cost: 10 bp;
- execution delay: one complete 5-minute bar;
- full-calendar CAGR;
- strict held-path MDD and terminal flatten;
- positive base, stress, and delayed absolute return;
- positive first-half and second-half return;
- CAGR / strict MDD at least 1.0;
- defeat of the strongest required nonsemantic control; and
- shared weekly familywise max-stat \(p_{\max} < 0.25\) over all 33 frozen
  policies.

No 2021 metric may repair or select another S5 policy.

## Global 2021 contamination

S4 itself did not read 2021, and S5 has no policy-specific 2021 metric.
However, this repository already contains an unrelated BCTP 2021 transfer
report. Therefore 2021 is not globally pristine.

Any S5 transfer result must be labeled:

```text
protocol-isolated policy-specific report-only transfer
```

It must not be called a globally clean OOS result or a live-capital
authorization. Globally pristine evidence now requires forward or live data.

## Access boundary at preregistration

- raw market/funding paths read: 0;
- 2020 transition-ledger rows parsed: 0;
- 2021 market/funding paths read: 0;
- 2021 market rows parsed: 0;
- 2021 funding rows parsed: 0;
- 2021 rewards created: 0;
- 2021 economic metrics computed: 0;
- model loads: 0;
- model forwards: 0.

The preregistration hashes the already authorized 2020 outcome-derived ledger
to bind it, but does not parse its rewards.

## Immutable preregistration

- artifact:
  `results/psim_d8_rllm2_s5_direction_residual_ridge_fqi_preregistration_2026-07-27.json`;
- artifact SHA-256:
  `95ebcacf6cf7eaec2d60933fe6a54852a9c3fb74e4d7e50972fe99da9500cace`;
- manifest hash:
  `96fa6b761154beba09fcbdaf826a408b7a5549e067eef9ff68a11c4bed887c54`;
- predecessor S4 readiness result hash:
  `36e81a7a42aa3a06b95e01c5203ec8f363a11b6771bafd28614ce4f055611ba9`;
- frozen 2020 transition ledger SHA-256:
  `07d465538d84648793ebbf302c54dace38ef71e88b22d7bd3a19fec500b99a7a`;
- frozen PCA32 SHA-256:
  `6a01d505ad2531683c9e8e9e0672456daf19a17a59b32d772fc857370210f0a2`.

## Next authorized step

Implement, independently review, commit, and push the exact S5 runner. The
runner may then parse only the frozen 2020 transition ledger, fit the complete
eight-policy family, seal the 2021 schedules, and execute the outcome-blind
readiness gate. It may not read a 2021 market or funding payload.
