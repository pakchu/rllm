# ExtraTrees rank-7 production lifecycle full battery

Updated: 2026-07-16

Canonical clock: **UTC**

Scope: frozen ExtraTrees rank-7 long alpha selected from the pre-2025 top-10 family

> **Current status: research champion, not a live-deployed policy.**
>
> The historical evaluator, frozen manifest, and stability audit exist. A serialized
> production model, exact 40-feature live replay path, atomic model registry, and
> rank-7 execution adapter do not yet exist. Until the blockers in
> [Current deployment blockers](#13-current-deployment-blockers) are closed, this
> document is the target operating contract rather than evidence that live deployment
> is complete.

## 1. Purpose and authority

This runbook fixes the operational answer to three questions:

1. **When and how is the model trained?**
2. **When may the production champion be replaced?**
3. **Exactly when and with which information is inference executed?**

The frozen research evidence remains authoritative for historical claims:

- Selection manifest:
  `results/expanding_extratrees_top10_pre2025_manifest_2026-07-15.json`
- Frozen manifest hash:
  `c6e7d78a328118456eacf70bc42cb12a48f33e26d13edbe21f2edb3aedea4f8e`
- OOS evaluation:
  `results/expanding_extratrees_top10_oos_2026-07-15.json`
- Stability audit:
  `results/expanding_extratrees_rank7_stability_2026-07-15.json`
- Human-readable evidence:
  - `docs/expanding-extratrees-top10-pre2025-2026-07-15.md`
  - `docs/expanding-extratrees-top10-oos-2026-07-15.md`
  - `docs/expanding-extratrees-rank7-stability-2026-07-15.md`

If this runbook conflicts with a frozen artifact, the frozen artifact wins for the
already-reported experiment. Changing the production contract requires a new version,
new manifest, and a fresh full battery; historical artifacts must never be edited in
place.

## 2. Status vocabulary

| Label | Meaning |
|---|---|
| **VERIFIED** | Reproduced by committed code and result artifacts. |
| **OPERATING POLICY** | Required production rule; it may still need implementation. |
| **PROPOSED** | Useful future policy that must not silently change the champion. |
| **BLOCKER** | Live order enablement is forbidden until closed. |

This distinction is important because the current repository proves an annual
walk-forward research result, but it does not yet export or hot-swap a rank-7 live
model.

## 3. Frozen champion contract

### 3.1 Candidate and learner

| Item | Frozen value |
|---|---|
| Side | Long only |
| Tested exposure/leverage | `0.50x` |
| Base events | `funding10_trend70 OR premium20_mom90` |
| Model | Multi-output `ExtraTreesRegressor` |
| Outputs | Exact net trade return, exact adverse excursion risk |
| Seeds | `7, 71, 715, 2026, 71515` |
| Deployment ensemble | Mean of five independent 300-tree models |
| `max_depth` | `2` |
| `min_samples_leaf` | `32` |
| `max_features` | `0.8` |
| Bootstrap | Disabled |
| Risk-adjusted score | `predicted_net - 0.25 * predicted_adverse` |
| Funding score threshold | Fit-source score quantile `0.40` |
| Premium score threshold | Fit-source score quantile `0.55` |
| Adverse-risk cap | Fit-source predicted-risk quantile `0.75` |
| Funding interaction | 7-day range width above fit q20 **or** completed-daily pullback at/below fit q40 |

An individual seed is not a production model. Only one of five individual 300-tree
seeds passed the frozen full-period rule, while the five-seed mean ensemble passed at
300, 1,000, and 2,000 trees. Production must therefore load and average all five
members; silently dropping a member is fail-closed.

The two hourly source events are also frozen:

```text
funding_leg = funding_rate <= -0.0000167
              AND trend_96 >= 0.007485218212390219

premium_leg = premium_index_change <= -0.00023471
              AND htf_1d_return_4 >= 0.0940403008961932
```

`trend_96` is a legacy column name generated under the frozen
`window_size=144` feature configuration. Live code must reproduce the implementation,
not reinterpret the name as permission to change the lookback.

### 3.2 Input contract

The model receives 40 numeric/state features defined by:

- `training/audit_weak_feature_responsibility_stability.py::FEATURE_COLUMNS`
- `training/search_liveparity_state_feature_interactions.py::STATE_FEATURE_NAMES`

The groups are:

- 11 causal Kalman/BOCPD/semi-Markov/source-state features
- 10 REX/completed higher-timeframe price-action features
- 9 macro/derivatives/order-flow features
- 10 nested-barrier and market-braid weak-signal diagnostics

Important implications:

- `open_interest` is not a direct model column, but delayed OI participates in the
  market-braid diagnostics.
- Current position, account balance, unrealized PnL, and portfolio allocation are not
  model inputs. They remain execution/risk gates.
- Missing feature values are imputed with medians frozen on the initial
  `2020-07-01 <= t < 2023-01-01` prefix and clipped to `[-20, 20]`. Kalman,
  BOCPD, and semi-Markov discretization thresholds are anchored to the same frozen
  initial fit regime. These preprocessing values are model artifacts and may not be
  recomputed in the live process or routine annual refit.

### 3.3 Execution and label contract

| Source leg | Maximum hold | Take | Stop |
|---|---:|---:|---:|
| Funding leg | 576 x 5m = 48h | 400 bps | No practical barrier |
| Premium leg | 144 x 5m = 12h | No practical barrier | 300 bps |

Additional fixed rules:

- Decision at signal time `t`; entry at the next 5-minute open `t+1`.
- 6 bps/notional/side total modeled execution cost.
- Realized funding is included.
- Stop wins when stop and take are both touched in the same bar.
- Positions do not overlap.
- A split only owns trades whose exits remain inside that split.
- Strict MDD includes pre-entry HWM and intratrade adverse excursion.

Post-only maker execution that fills materially later than the next 5-minute open is
not automatically equivalent to the backtest contract. Fill delay and slippage must
be logged and included in the shadow/canary battery.

## 4. The three operating clocks

### 4.1 Decision

The initial production policy is deliberately conservative:

| Clock | Cadence | May affect live champion? | Status |
|---|---|---:|---|
| Inference | Every UTC hour at `HH:00` | Yes, creates `TRADE/ABSTAIN` | OPERATING POLICY |
| Data/decision reconciliation | Daily | No; may halt entries | OPERATING POLICY |
| Offline challenger rehearsal | Monthly, after previous UTC month closes | No | PROPOSED |
| Promotion review | Quarterly | Review only under current evidence | OPERATING POLICY |
| Frozen-spec production refit | Annually | Yes, after full battery | VERIFIED shape / export BLOCKER |
| Structural/hyperparameter search | At most annually, separately preregistered | Never without new manifest | OPERATING POLICY |
| Drawdown/drift response | Event driven | Disable or rollback only | OPERATING POLICY |

### 4.2 Why the champion is not replaced monthly yet

The committed evidence uses **annual expanding refits**. There is no committed
monthly-cutoff evaluator proving that monthly retraining preserves leakage, purge,
threshold, event-frequency, and execution behavior.

Therefore:

- A monthly job may build a **challenger** and generate shadow predictions.
- It may not move the production pointer automatically.
- The production champion is replaced at most once per year until a separately
  preregistered monthly walk-forward battery passes.
- Poor recent performance never authorizes emergency retuning. The safe response is
  exposure reduction, entry halt, or rollback.

## 5. Training lifecycle

### 5.1 Cutoff and data eligibility

For a prediction year beginning at cutoff `C`:

1. Fit events must occur at or after `2020-07-01` and strictly before `C`.
2. Both target outputs must be finite.
3. The complete source-owned trade exit must be strictly before `C`.
4. Events whose outcomes cross `C` are purged, even if their final labels are known at
   the wall-clock time the job happens to run.
5. The feature graph must be physically truncated at the declared source cutoff.
6. Model weights, source-specific score/risk/interaction thresholds, and balancing
   weights are fit from the eligible annual prefix only. Frozen preprocessing medians
   and causal-state bucket thresholds remain those from the initial
   `2020-07-01..2022-12-31` fit regime.

Use an expanding history by default. Replacing it with a rolling lookback changes the
learning problem and requires a new candidate family and full battery.

### 5.2 Training sequence

1. **Freeze source snapshot**
   - Record inclusive/exclusive timestamps, row counts, schema, and SHA-256 hashes.
   - Require a complete 5-minute BTC grid across the retained source horizon.
2. **Build the exact live-parity feature graph**
   - Exclude the current market bar at each boundary.
   - Apply the 12 x 5-minute predictor delay.
   - Restore only current `funding_leg` and `premium_leg` source identities.
3. **Freeze immutable event anchors**
   - Candidate events are selected before downstream predictions.
   - Apply the fixed 144-bar anchor cooldown.
4. **Generate exact labels**
   - Use the source-owned exit rules, next-open entry, costs, realized funding, and
     adverse excursion from the same execution engine used by evaluation.
5. **Purge cutoff-crossing labels**.
6. **Balance observations by `(source year, source leg)`**
   - Each group receives equal total fit weight.
7. **Fit all five ensemble members**
   - 300 trees/member for the candidate production artifact.
   - Never replace missing members with duplicated predictions.
8. **Derive fit-only policy values**
   - Funding/premium score thresholds.
   - Funding/premium risk caps.
   - Funding range-width q20 and daily-pullback q40.
   - Reuse and record the frozen initial-prefix medians used for imputation; do not
     refit them at the annual cutoff.
9. **Force deterministic inference mode**
   - Prediction uses `n_jobs=1` even if training uses parallel workers.
10. **Export an immutable candidate artifact**.
11. **Run the full promotion battery** before any production pointer changes.

### 5.3 Required model bundle

A production bundle must contain, at minimum:

```text
rank7-YYYYMMDDTHHMMSSZ/
  manifest.json
  feature_contract.json
  preprocessing.json
  thresholds.json
  model-seed-7.*
  model-seed-71.*
  model-seed-715.*
  model-seed-2026.*
  model-seed-71515.*
  historical-battery.json
  shadow-battery.json
  checksums.sha256
```

`manifest.json` must record:

- model/version ID and parent champion
- Git commit and dirty-worktree status
- source hashes and cutoff
- exact ordered 40-column feature list and feature hash
- fit medians and clipping rule
- learner parameters, seeds, tree count, and library versions
- source-specific score/risk/interaction thresholds
- label/execution contract
- historical schedules and metric hashes
- artifact checksums

The current repository does not yet emit this bundle; this is a deployment blocker.

## 6. Full promotion and replacement battery

Every gate is mandatory. A later gate cannot waive an earlier failure.

### Gate A — Repository and artifact integrity

- Clean Git worktree at the recorded commit.
- Frozen manifest hash and selection result hash match.
- Ordered feature list and feature hash match the candidate bundle.
- Source schemas, time zones, bar labels, and units match training.
- Five distinct seed models are present and checksums pass.
- Model load followed by deterministic test vectors reproduces the export output.

### Gate B — Leakage and chronology

- No feature reads a row after its decision timestamp.
- Current `HH:00` 5-minute market bar is excluded.
- Predictors are delayed exactly 12 x 5-minute rows; only source-leg identities remain
  current.
- Completed 4h/1d/1w candles exclude the candle containing the decision.
- OI is delayed one completed 5-minute bar before market-braid use.
- All fit labels exit before the fold cutoff.
- Selection and later evaluation periods cannot influence hyperparameters, rank, or
  thresholds.
- Extending the source with future rows reproduces the historical prefix exactly.

### Gate C — Unit and integration tests

Minimum targeted suite:

```bash
uv run pytest -q \
  tests/test_select_expanding_extratrees_top10_pre2025.py \
  tests/test_evaluate_expanding_extratrees_top10_oos.py \
  tests/test_audit_expanding_extratrees_rank7_stability.py \
  tests/test_audit_stable_ensemble_conditional_pullback_alpha.py \
  tests/test_audit_weak_feature_responsibility_stability.py
```

Before live enablement, add and pass tests for the production exporter, registry,
historical-vs-live feature replay, scheduler deadline, idempotency, and rollback.

### Gate D — Frozen historical acceptance

The existing acceptance rule is:

- Every full year: absolute return `> 0`, CAGR/strict-MDD `>= 3`, strict MDD `<= 15%`,
  trades `>= 12`.
- Half-year diagnostic: the same return/ratio/MDD requirements, trades `>= 6`.
- Combined period: CAGR/strict-MDD `>= 3`, trades `>= 42`.
- CAGR always uses the complete calendar period, including idle time.

Frozen 300-tree five-seed ensemble evidence:

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| 2023 | 12.86% | 12.87% | 3.12% | 4.13 | 19 |
| 2024 | 16.40% | 16.36% | 3.46% | 4.72 | 22 |
| 2025 | 16.36% | 16.37% | 4.98% | 3.29 | 21 |
| 2026H1 | 7.31% | 18.48% | 4.30% | 4.30 | 12 |
| 2023-2026H1 | 64.04% | 15.59% | 4.98% | 3.13 | 74 |

These are research results, not a promise of live returns.
They use the frozen `0.50x` exposure. Leverage scaling is not part of the accepted
rank-7 contract and requires a complete execution/risk re-evaluation.

### Gate E — Ensemble and numerical stability

- Five-seed mean ensembles at 300, 1,000, and 2,000 trees all satisfy Gate D.
- A fresh repeated run produces identical selected-position and full-result hashes.
- A single-seed pass is not accepted as a substitute.
- Predictions from the serialized 300-tree bundle match in-memory predictions within
  an explicitly versioned numerical tolerance.

### Gate F — Production-only stress tests

These are required before first live deployment but are not proven by the current
rank-7 artifacts:

- 1.5x and 2.0x modeled execution-cost scenarios.
- One- and two-bar late-entry scenarios.
- Premium/funding publication-delay scenarios.
- FX weekday gaps and weekend neutral-fill parity.
- Missing spot/OI/taker rows and recovered-stream scenarios.
- Duplicate scheduler invocation and process restart at the decision boundary.
- Partial fill, maker reprice, cancellation, and rejected-order scenarios.
- Adverse same-bar stop/take ordering.
- Database read replica lag and out-of-order insert scenarios.

Any stress failure must either block deployment or produce an explicit fail-closed
rule that is tested in both replay and live code.

### Gate G — Historical/live feature parity

For a fixed historical interval, capture each live-path feature snapshot and compare it
to batch research output:

- identical timestamps and candidate source legs
- identical ordered feature vectors after imputation/delay
- identical five member predictions and ensemble mean
- identical score/risk thresholds and interaction gate
- identical final `TRADE/ABSTAIN`

Required acceptance is exact equality for categorical/boolean/timestamp values and a
versioned small numeric tolerance for floating-point values. Aggregate metric similarity
is not enough.

### Gate H — Shadow, testnet, and canary

1. **Shadow:** at least 30 calendar days and at least three selected trade decisions.
2. **Testnet:** replay the complete entry/exit lifecycle for at least three selected
   decisions; synthetic forced orders may test plumbing but do not count as policy
   decisions.
3. **Canary live:** start at the smallest approved risk allocation, with no automatic
   leverage increase.
4. Compare expected entry timestamp/price with actual order/fill timestamps and price.
5. Promotion to normal allocation requires operator review of every canary trade.

Sparse signals may make this battery slow. Low frequency is not grounds to shorten it.

## 7. Replacement policy

### 7.1 Normal replacement

A champion may be replaced only when:

- the replacement is the same frozen structure or a separately preregistered new
  strategy version;
- all Gates A-H pass;
- the account is flat and has no open orders;
- the new version is installed alongside, not over, the old version;
- an atomic `champion` pointer change is recorded with operator, timestamp, and reason;
- the previous two passing bundles remain available for rollback.

The model version that opened a position owns that position's exit. Never hot-swap the
decision owner mid-trade.

### 7.2 Scheduled cadence

- **Monthly:** build/replay a challenger after the prior UTC month closes. Shadow only.
- **Quarterly:** review data drift, execution parity, and challenger evidence. No
  automatic production switch under the current annual-only validation contract.
- **Annually:** run the frozen expanding refit and full battery. This is the only normal
  champion replacement cadence currently supported by evidence.
- **Structural search:** separate research branch, separate preregistration, separate
  frozen selection/evaluation manifest. Never mix it into routine refitting.

### 7.3 Emergency action

An emergency event may:

1. disable new entries;
2. preserve/close existing positions according to the owning version and risk policy;
3. roll back to the previous passing bundle;
4. leave the strategy disabled.

It may not authorize ad-hoc retraining, threshold optimization, side reversal, or
recent-window parameter selection.

## 8. Hourly inference timing

### 8.1 Canonical timeline

All scheduling uses UTC. KST is display-only (`UTC + 9h`).

| Time | Required action |
|---|---|
| `HH-1:55:00` to `HH:00:00` | Final source 5-minute interval completes. |
| `HH:00:00` | Decision boundary; do not read the new `HH:00` market bar. |
| `HH:00:00` onward | Wait for source watermarks, query an immutable `as_of=HH:00` snapshot, and build features. |
| Target by `HH:00:30` | Data-readiness decision completed. This is an SLO target, not yet benchmarked. |
| Target by `HH:01:00` | Ensemble prediction and policy decision completed. This is an SLO target, not yet benchmarked. |
| Hard deadline `HH:04:30` | Final order intent must be accepted by the execution bridge. |
| `HH:05:00` | Backtest-equivalent next-open entry boundary. |

If data or inference misses the hard deadline, emit `ABSTAIN_DEADLINE` and wait for the
next UTC hour. Never chase the missed signal at `HH:10` or later.

### 8.2 What the model may know at `HH:00`

- Current `funding_leg`/`premium_leg` identity computed from information timestamped as
  available at the boundary.
- All other 40-column predictors as they existed 12 completed 5-minute rows earlier.
- Completed higher-timeframe candles only.
- The previous completed market bar, never the bar starting at `HH:00`.

The one-hour delay is part of the model, not a generic allowance for stale data. A live
source that is another hour late is invalid, not “covered” by the model delay.

### 8.3 Runtime decision order

```text
verify process/model/checksum
  -> verify UTC boundary and idempotency key
  -> freeze DB as_of timestamp and source watermarks
  -> build live-parity features and data-quality report
  -> compute current funding/premium source event
  -> if no event: ABSTAIN_NO_EVENT
  -> load 1h-delayed predictor vector
  -> predict [net, adverse] with all five 300-tree models
  -> average five predictions
  -> score = net - 0.25 * adverse
  -> apply source-specific score threshold
  -> apply source-specific adverse-risk cap
  -> for funding leg, apply width-q20 OR pullback-q40 interaction
  -> apply account/position/order/notional/manual kill gates
  -> before deadline: submit next-open intent
  -> otherwise: ABSTAIN_DEADLINE
```

Each boundary has one idempotency key, for example:

```text
(strategy_id, model_version, signal_time_utc, source_leg)
```

A retry may recover the same decision; it may not create a second order.

## 9. Data readiness and staleness

### 9.1 Required sources

- Binance BTCUSDT futures 1-minute OHLCV, quote volume, trade count, and taker-buy data
- Binance BTCUSDT funding history/current known funding state
- Binance BTCUSDT premium-index data
- Binance BTCUSDT spot data for market-braid diagnostics
- Binance BTCUSDT open interest, delayed one completed 5-minute bar
- Upbit KRW-BTC and USDKRW for kimchi premium
- DXY synthetic FX components

### 9.2 Fail-closed rules

- The BTC decision grid must have all constituent 1-minute rows for every completed
  5-minute bar used by the feature graph.
- Premium data uses backward-as-of semantics and must not exceed the frozen 10-minute
  tolerance.
- Funding uses only the latest value that was actually published by the decision time;
  historical/live timestamp semantics must match.
- Spot/OI/market-braid diagnostics require exact 5-minute alignment. Missing input does
  not permit a partially computed braid feature.
- Weekday FX values may not be indefinitely forward-filled.
- During the historical FX closure, live behavior must match training: after the short
  as-of tolerance expires, external numerics become neutral rather than retaining a
  stale Friday value.
- Any unknown schema, unit, duplicate timestamp, future timestamp, or backward-moving
  watermark produces `ABSTAIN_DATA_QUALITY`.

The rank-7 feature vector does not include external availability flags. Weekend and
missing-source parity must therefore be tested explicitly rather than inferred from the
model input.

## 10. Runtime logging and monitoring

Every inference attempt, including no-event and failure paths, must persist:

- strategy/model version and all artifact hashes
- scheduler start/end and signal timestamp in UTC
- per-source max timestamp, row count, gap count, and availability
- ordered raw, delayed, imputed, and clipped feature vectors or their content hashes
- source leg, base-event reason, and anchor/cooldown state
- five member predictions, ensemble prediction, score, and risk
- fitted thresholds and every gate result
- final decision and explicit abstention reason
- idempotency key
- position/open-order state before and after execution
- expected entry boundary, order timestamps, fills, fees, funding, exit reason, and PnL

### 10.1 Daily checks

- no duplicate inference keys or orders
- no missed hourly scheduler windows
- source freshness/gap report
- historical-vs-live replay of the prior day
- model/feature/artifact checksum verification
- open position and exchange reconciliation

### 10.2 Performance checks

Do not retune from a handful of sparse trades. Report trailing statistics, but treat
them as model-health evidence only after a meaningful event count.

Immediate safety actions do not need statistical significance:

- feature/replay mismatch: disable new entries
- missing model member or checksum mismatch: disable new entries
- duplicate order or ownership mismatch: disable and reconcile
- strategy drawdown at `1.25 x` frozen all-period strict MDD (about `6.23%`): soft halt
  and review
- strategy drawdown at `1.5 x` frozen all-period strict MDD (about `7.48%`), or any
  configured lower account-level limit: hard halt

These limits control entry enablement; they do not prove that the model is broken and
must not trigger automatic retraining.

## 11. Rollback battery

Rollback must be tested before first live use and after registry/execution changes.

1. Install new and old bundles side by side.
2. Verify both bundles independently.
3. Switch the pointer while flat and confirm only the new version serves inference.
4. Simulate a checksum/data-parity failure.
5. Confirm new entries halt without canceling an owned exit incorrectly.
6. Atomically restore the old pointer.
7. Restart the process and reproduce the same old-version decision.
8. Reconcile exchange position/open orders and record the rollback event.

Never delete the active, previous, or in-shadow bundle. Older failed/intermediate
artifacts should be pruned according to the repository disk policy; WSL usage should
remain below 300 GB.

## 12. Reproduction commands

These commands reproduce research artifacts; they do **not** export a live model
bundle.

```bash
# Rebuild the pre-2025 ranked family and frozen manifest.
uv run python -m training.select_expanding_extratrees_top10_pre2025

# Evaluate the already-frozen top-10 on 2025 and 2026H1.
uv run python -m training.evaluate_expanding_extratrees_top10_oos

# Re-run rank-7 seed/tree-count/determinism stability evidence.
uv run python -m training.audit_expanding_extratrees_rank7_stability
```

The rank-7 audit currently writes canonical result paths directly. Run it only from a
clean worktree and inspect the diff; a reproducibility run should not change the
committed artifacts.

## 13. Current deployment blockers

Live enablement is forbidden until all are closed:

- [ ] Export five fitted 300-tree models plus preprocessing/threshold metadata.
- [ ] Add immutable model registry, checksum verification, atomic pointer, and rollback.
- [ ] Implement the exact 40-column rank-7 feature graph in the live path.
- [ ] Add live completed-hour Kalman, BOCPD, and semi-Markov states.
- [ ] Add live nested-barrier diagnostics with causal previous-row exposure.
- [ ] Query Binance spot BTCUSDT and open-interest data required by market braid.
- [ ] Add market-braid live computation with delayed OI and exact alignment.
- [ ] Prove historical/live feature and decision parity on a frozen interval.
- [ ] Implement hourly UTC scheduler, immutable `as_of`, watermarks, deadlines, and
      idempotency.
- [ ] Implement source-specific rank-7 exits and position ownership in the execution
      bridge.
- [ ] Pass cost/latency/missing-data/restart/partial-fill stress tests.
- [ ] Complete shadow, testnet, canary, and rollback batteries.

The existing `execution/rex_llm_live.py` operates a different REX/RLLM pilot contract.
It must not be relabeled as rank-7 live parity without completing these items.

## 14. Change-control matrix

| Change | Routine refit allowed? | New full selection required? |
|---|---:|---:|
| Add newer rows, preserve expanding history/spec | Yes, after full battery | No |
| Recompute source score/risk/interaction thresholds at annual cutoff | Yes | No |
| Recompute preprocessing medians or causal-state bucket thresholds | No | Yes |
| Monthly live replacement | No | Monthly walk-forward preregistration/evaluation required |
| Change feature definition/order/delay | No | Yes |
| Add or remove a feature, including availability flags | No | Yes |
| Change seed count/tree count/depth/leaf/max-features | No | Yes |
| Change score lambda or quantiles | No | Yes |
| Change event trigger or source exits | No | Yes |
| Change cost/fill/funding assumptions | No | At least complete re-evaluation; usually yes |
| Change leverage, sizing, or portfolio allocation | No | At least complete risk/execution re-evaluation; usually yes |
| Add LLM/RL gate or sizing controller | No | Separate frozen overlay selection and OOS evaluation |
| Roll back to a previously passing identical bundle | Yes | No |

## 15. Operator go/no-go checklist

### Before training

- [ ] UTC cutoff declared and source snapshot frozen.
- [ ] Git commit clean and recorded.
- [ ] Schema/timezone/unit checks pass.
- [ ] Future source rows are physically inaccessible to the fit job.

### Before promotion

- [ ] Gates A-H pass with committed evidence.
- [ ] All five models and metadata checksums pass.
- [ ] Historical/live replay is decision-identical.
- [ ] Shadow/testnet/canary evidence reviewed.
- [ ] Previous two champions are loadable.
- [ ] Account is flat and open-order count is zero.

### Every inference boundary

- [ ] Correct UTC hour and unique idempotency key.
- [ ] Immutable DB `as_of` and source watermarks valid.
- [ ] Current market bar excluded and predictor delay exactly 12 bars.
- [ ] All five member predictions available.
- [ ] Data, model, account, and deadline gates pass.
- [ ] Otherwise emit a specific `ABSTAIN_*`; never infer a replacement value silently.

### After replacement

- [ ] Pointer/version and operator reason recorded.
- [ ] First decision replayed against the candidate artifact.
- [ ] Exchange/account reconciliation clean.
- [ ] Old champion retained and rollback drill still passes.
