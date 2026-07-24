# PIVOT-72 candidate boundary — paired intrinsic venue orderflow topology

## Selection

Select one new source-seen, outcome-unseen policy axis:

**PIVOT-72 — Paired Intrinsic Venue Orderflow Topology Policy**.

PIVOT observes one causally completed relation state after Binance Spot and
USD-M BTCUSDT have each consumed the same fraction of their own strictly prior
normal-day quote activity. It does not require a hand-written conjunction to
declare direction. One compact policy chooses exactly one action:

```text
LONG
SHORT
ABSTAIN
```

The provisional causal chain is:

1. each venue receives its own daily quote-notional target from complete
   strictly prior source days;
2. each venue reaches that target on its own intrinsic clock;
3. after both anchors and one complete computation buffer, deterministic code
   emits a compact state describing clock order and flow-state transitions;
4. one small RLLM ranks long, short, and abstain from those causal relations;
5. a non-abstain action enters at the next permitted five-minute open; and
6. the position exits after exactly 72 five-minute bars.

The `72` suffix reserves a six-hour consequence horizon. The exact source
transform, relation tokens, support floors, reward, cheap baselines, model,
preference/RL method, costs, latency, controls, and qualification gates must be
frozen in a separate mechanism commit before a PIVOT source row, token, model
label, or market outcome is decoded.

This boundary is not an alpha result. It selects a falsifiable state-policy
experiment.

## Why this is not CVICR repair

CVICR-72 was retired unchanged after its exact four-way sequence produced only
nine train events and zero 2023 selection events. Its failed primary required:

```text
q60 clock gap
AND early cross-venue flow conflict
AND leader persistence
AND laggard resolution
```

CVICR then followed the leader's original side. None of those predicates,
thresholds, controls, or side rules may be relaxed under the CVICR identity.

PIVOT changes the predictive object:

- CVICR treated the completed conjunction as an event; PIVOT treats every
  valid paired-anchor day as an opportunity state;
- CVICR owned a deterministic fixed side; PIVOT has no source-owned side and
  exposes long, short, and abstain as policy actions;
- CVICR discarded component states that did not complete its sequence; PIVOT
  represents clock order, disagreement, persistence, alignment, and change as
  simultaneous weak relations;
- CVICR used a q60 gap tail as a gate; PIVOT may use only strictly prior
  ordinal gap context, never that tail as eligibility;
- CVICR had nine accepted clocks; PIVOT reserves the broader paired-anchor
  opportunity clock even when its policy abstains; and
- PIVOT receives a new identity, preregistration, source-support gate,
  policy-learning gate, novelty gate, and immutable failure action.

The CVICR source-support report already disclosed source-only aggregate
incidence:

```text
paired non-tied valid prefixes     954
strictly-prior gap reference       864
gap-only states                    356
initial-conflict-only states       100
late-alignment-only states         232
CVICR primary states                 9
```

That knowledge makes PIVOT source-seen rather than a clean-room discovery.
No CVICR comparator row or market outcome was opened. PIVOT has not decoded
the 954 source rows, their token distributions, entry clocks, directions,
prices, returns, or post-2023 values.

The terminal CVICR report explicitly required a successor to change the state
representation and combine causal weak observations without collapsing them
into the failed conjunction. PIVOT follows that instruction; it does not
reinterpret CVICR as profitable.

## Why this is an RLLM-shaped task

The input is not a raw numeric price forecast. The policy must reason over a
small set of relations such as:

- which venue consumed its normal activity budget first;
- whether the intrinsic gap is ordinary or unusually wide under strictly
  prior history;
- whether venue flows were aligned, opposed, or flat at the first anchor;
- whether each venue persisted, weakened, or reversed by the second anchor;
- whether late consensus is bullish, bearish, divergent, or flat;
- whether the current state is continuing or changing from the immediately
  prior valid paired state; and
- whether the PIVOT sleeve is flat and executable.

Those relations are compositional. No single relation is claimed to be a
strong signal. The hypothesis is that their joint causal grammar changes the
conditional utility of long, short, and abstain.

PIVOT remains one policy model. There is no analyzer/trader pair, free-form
analysis channel, prose chain of thought, or model-generated feature. Numeric
accounting, source validation, scheduling, execution, costs, and risk metrics
remain deterministic code.

The model is justified only after cheap causal baselines show that the frozen
state has out-of-sample learnability. A large language model may not be used
to conceal a state surface that ridge, categorical baselines, and shuffled
controls show to be noise.

## Frozen source family

Primary source:

```text
data/binance_cross_venue_minute_leadership_btc_2020_2023/
  BTCUSDT_cross_venue_minute_leadership_5m_2020-01_2023-12.csv.gz
SHA256 00ab6a55fc7bfeb3012584db5bc97a7d7b98dd995491acfd3f865c6bd41f92cc
```

Manifest:

```text
data/binance_cross_venue_minute_leadership_btc_2020_2023/build_manifest.json
SHA256 544c2945a2b56be478a1edc4abbb93b762bda5afc32cbd0658dd6822ff6b70fa
```

Independent audit:

```text
results/binance_cross_venue_minute_leadership_audit_2026-07-14.json
SHA256 ffe0124ac9c5c0c3f1d1c284b672618cf910dc16cae36e65c1efe79710f039af
```

The source is an exact 420,768-row UTC five-minute grid over
`[2020-01-01, 2024-01-01)`, built from checksum-verified official Binance Spot
and USD-M one-minute archives. Accepted features are available at the
completed five-minute boundary. Source defects remain quarantined; no future,
label, action, reward, PnL, or portfolio column exists.

Official archive and schema references:

- <https://github.com/binance/binance-public-data>
- <https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md#klinecandlestick-data>
- <https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Kline-Candlestick-Data>

The next mechanism must bind exact source hashes and use a column allowlist.
Load-and-drop of price, response, basis, timing-centroid, future, target,
label, reward, action, or PnL fields is forbidden.

## Provisional paired-anchor opportunity

PIVOT may reuse the source coordinate, but not the failed CVICR event.

For complete UTC day `D` and venue `v`, the provisional expected-volume scale
uses the median of complete daily quote notionals in the exact 28 preceding
calendar positions, excluding `D`. Missing positions stay missing; the window
does not widen. At least 21 complete reference days are required.

The provisional target remains one half of that strictly prior scale:

```text
target[v,D] = 0.50 * median_prior_28d_complete_volume[v,D]
```

For each venue, the anchor is the first completed five-minute bar whose
same-day cumulative quote notional reaches its target. Both anchors, the
entire prefix through the later anchor, and the following computation-buffer
bar must be complete and causal.

An exact anchor tie is not a paired-order state. The later anchor must leave
room for the fixed computation buffer, entry, and complete 72-bar hold inside
the same split.

PIVOT does **not** require:

- a minimum or q60 anchor gap;
- early conflict;
- late alignment;
- leader persistence;
- laggard resolution;
- price confirmation;
- funding, premium, OI, Kimchi, DXY, or manual regime confirmation; or
- a deterministic source side.

The next mechanism must either freeze this provisional coordinate exactly or
reject PIVOT before any row is decoded. It may not inspect alternate target
fractions or anchor cutoffs.

## Provisional relation state

The exact token schema is not authorized until the mechanism commit. It must
remain compact and satisfy these boundaries:

Allowed relation families:

1. leader venue and non-tied anchor order;
2. strictly prior ordinal gap state;
3. coarse causal anchor-session relation without date, year, or month;
4. leader and laggard cumulative-flow sign at the early anchor;
5. each venue's early-to-late persistence, weakening, flatness, or reversal;
6. late two-venue consensus/disagreement relation;
7. strictly prior ordinal flow-strength and activity-speed relations;
8. current state versus the immediately previous valid paired state; and
9. current PIVOT sleeve position/executability.

Forbidden model inputs:

```text
raw timestamp or source day
year, month, quarter, row ID, or event ID
raw price, return, basis, funding, premium, OI, Kimchi, DXY
raw quote notional or raw flow magnitude
future path, reward, label, oracle action, PnL, CAGR, or MDD
comparator identity or overlap
post-2023 source or outcome value
free-form analyzer prose
```

Every ordinal threshold must use an exact strictly prior window and exclude
the current state. Future appends must leave every prior token byte-identical.
Rare-level handling must be frozen before incidence; no token may be merged,
deleted, or rebucketed after an outcome is opened.

The mechanism must include venue-swap, sign-mirror, future-append,
missing-prefix, exact-tie, and action-option-order invariance tests.

## Provisional execution

- opportunity origin: first venue anchor;
- state completion: second venue anchor;
- signal availability: one complete five-minute bar after state completion;
- decision/order: after deterministic state validation and policy inference;
- entry: next permitted USD-M BTCUSDT five-minute open;
- actions: `LONG`, `SHORT`, `ABSTAIN`;
- hold: exactly 72 five-minute bars;
- initial exposure: fixed `0.5x` account gross;
- exits: scheduled only; no stop, take-profit, trailing, or model exit;
- candidate cap: at most one opportunity per UTC source day;
- overlap: reserve the full 72-bar interval before the policy action, so
  abstention cannot release a later opportunity;
- split containment: origin, both anchors, buffer, entry, every held bar, and
  exit must stay inside one half-open split; and
- operational position conflict: fail closed to deterministic abstention,
  never ask the model to improvise portfolio netting.

The exact fee and slippage model, entry latency, late-anchor cutoff, and
position ledger are frozen later. No economic value has been inspected to set
them.

## Train, test, eval, and sealed boundary

The temporal roles are:

```text
train       2020-01-01 <= origin < exit < 2022-01-01
test        2022-01-01 <= origin < exit < 2023-01-01
eval        2023-01-01 <= origin < exit < 2024-01-01
sealed      2024-01-01 <= origin
```

The mechanism must freeze one exact policy-development sequence:

1. fit state transforms and policy parameters only from causally available
   train rows;
2. select model family, checkpoint, action floor, and every hyperparameter
   using train-internal causal folds plus the 2022 test only;
3. freeze one policy artifact before any 2023 reward or return is opened;
4. generate 2023 actions without 2023 label updates;
5. pass source, action-clock novelty, and control gates before opening 2023
   outcomes;
6. evaluate 2023 exactly once as fixed-policy eval; and
7. keep 2024+ physically and logically sealed unless every pre-2024 gate
   passes.

No monthly or rolling adaptation is part of PIVOT-72. A later live retraining
schedule, if justified, requires a separate preregistered forward protocol
after static sealed validation. This prevents "continuous learning" from
quietly consuming the eval labels it is supposed to predict.

## Cheap-baseline and RLLM sequence

Before GPU policy training, the frozen state must face:

- always abstain, always long, and always short;
- exact-signature memory;
- shuffled-label and shuffled-reward controls;
- categorical Naive Bayes or an equivalent fixed categorical baseline;
- one regularized linear/contextual value model;
- token-family ablations;
- venue-swap and sign-mirror controls; and
- action-collapse and executed-direction-collapse checks.

If no cheap causal policy has positive 2022 test economics after costs, both
test halves are not positive, or it cannot beat shuffled controls, PIVOT
retires before Gemma training. This is a learnability gate, not permission to
tune the state on 2022.

Only a complete cheap-baseline pass may authorize one small 4-bit
Gemma-family policy. The intended training shape is:

- compact text serialization of frozen causal relation tokens;
- randomized presentation order for `LONG`, `SHORT`, and `ABSTAIN`;
- completion-only action loss;
- mirrored venue/sign examples whose target transforms exactly;
- train-only utility-ranked action pairs;
- supervised action-format warm-up followed by one frozen offline
  preference/RL objective; and
- one action-token score at inference, with no generated rationale required.

The mechanism must freeze the exact base model revision, quantization, LoRA
targets, optimizer steps, checkpoints, memory ceilings, pair construction,
preference loss, seed, and tie-break before any policy training.

The RLLM must beat the strongest frozen cheap causal policy on 2022 without
using 2023. Otherwise it is rejected before eval.

## Qualification objective

The exact machine gates are frozen later, but they must preserve the user's
minimum objective rather than optimize an easier surrogate.

Every report must include:

```text
absolute return
full-calendar CAGR including idle time
held-path strict MDD
CAGR / strict MDD
trade count
long and short count
H1 and H2 results
weekly-cluster significance
cost and delay stress
```

The precommitted 2023 eval gate may not be weaker than:

- positive absolute return;
- `CAGR / strict MDD >= 3.0`;
- `strict MDD <= 15%`;
- at least 60 trades, 20 in each half, and both executed sides represented;
- positive H1 and H2 absolute return;
- one-sided weekly-cluster `p < 0.10`;
- positive base-cost and stressed-cost results;
- no action, direction, token-level, or month concentration collapse; and
- strict improvement over the strongest causal cheap baseline.

If 2023 passes, 2024 and 2025 must be opened sequentially and each must meet a
separately frozen full-year gate. 2026 remains forward reporting and may not
change the policy.

Passing at `0.5x` establishes risk efficiency, not the final CAGR target.
Leverage may be considered only after sealed validation and may never be used
to hide a ratio or strict-MDD failure.

## Novelty and contamination boundary

This source family is not globally pristine. Pre-2024 outcomes have already
been opened for multiple Spot↔USD-M and intrinsic-clock hypotheses. PIVOT may
claim only that its exact paired-state action policy was frozen before its own
outcomes.

The comparator cohort must include at least:

- CVICR primary and its source-only component clocks;
- CATCH and CLASP cross-venue handoff clocks;
- LURI;
- CVTT;
- IVLIR, IVFHR, and IVPLH;
- CARTA's emitted and executed action clocks; and
- the active live portfolio sleeves available at freeze time.

Comparators may not be parsed before PIVOT source support passes. The frozen
policy must generate its 2023 action clock before 2023 returns are opened.
Exact-entry, near-time, occupied-position, signed-side, and incremental
portfolio overlap must be reported.

Novelty failure retires PIVOT before eval outcomes. Comparator thresholds may
not be changed after overlap is seen.

The premium comparator files permanently forbidden by the VARR protocol
breach remain forbidden:

```text
data/premium_compression_breakout_relay_clocks_2020_2026.csv.gz
data/premium_snapback_recenter_clocks_2020_2026.csv.gz
```

## Evidence boundary

This selection used only:

- repository history and prior terminal reports;
- source schema, hashes, manifests, and aggregate audit summaries;
- CVICR's already committed source-only aggregate funnel;
- model/runtime constraints already established in repository tests; and
- official source documentation.

This selection did not decode or compute:

- a PIVOT paired-anchor row or token;
- PIVOT train, test, eval, or sealed incidence;
- token vocabulary, bucket support, signatures, actions, or labels;
- any market price, funding, future return, reward, PnL, CAGR, MDD, or hit
  rate for PIVOT;
- any comparator row or overlap;
- any 2024-or-later source value; or
- any model prediction or checkpoint.

## Mandatory next sequence

1. commit this boundary;
2. freeze one exact PIVOT mechanism and preregistration without decoding a
   PIVOT source row;
3. commit a tested source-support/state builder;
4. run the source-only support and token-stability gate once;
5. retire unchanged on any source or support failure;
6. freeze the cheap baseline, reward builder, action clock, controls, model,
   and evaluator before outcomes;
7. open train and 2022 test only under the frozen evaluator;
8. retire before GPU work if the cheap learnability gate fails;
9. train one small RLLM only if authorized and freeze it before 2023;
10. run novelty without 2023 outcomes;
11. open 2023 eval once only after every prior gate passes; and
12. open sealed years sequentially only after an unchanged pre-2024 pass.

## Bound predecessor evidence

- `docs/binance-cross-venue-minute-leadership-data-design-2026-07-14.md`
- `docs/binance-cross-venue-minute-leadership-data-audit-2026-07-14.md`
- `docs/cross-venue-intrinsic-clock-resolution-boundary-2026-07-24.md`
- `docs/cross-venue-intrinsic-clock-resolution-mechanism-decision-2026-07-24.md`
- `docs/cvicr-source-support-result-2026-07-24.md`
- `docs/causal-adaptive-relational-token-abstainer-preregistration-2026-07-14.md`
- `docs/causal-adaptive-relational-baseline-selection-result-2026-07-14.md`
- `docs/venue-maintenance-extension-release-synthetic-rejection-2026-07-24.md`
