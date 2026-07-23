# OPRR-288 candidate boundary — CBOE option-pressure rank rotation

## Selection

Select one source-seen, candidate-incidence-unseen reserve:
**OPRR-288 — CBOE Option-Pressure Rank Rotation**.

OPRR will test a transition in the cross-sectional ordering of three already
audited CBOE pressure surfaces:

1. volatility-term pressure;
2. tail-hedge pressure; and
3. option-flow pressure.

The provisional economic object is not an option-flow level or a vote. It is a
causal rotation in option pressure relative to both term and tail pressure,
confirmed by contemporaneous own-surface movement and a same-direction change
in the other-surface state. A deterministic composer will own opportunity
timing and direction. A later RLLM may only execute that fixed side or abstain
from compact relation tokens.

The `288` suffix reserves a provisional 288 five-minute-bar consequence
horizon. This file does not freeze the exact rank-position formula, transition
algebra, confirmation rule, tie behavior, side, entry, hold, controls,
source-support gates, comparator battery, or economic evaluator. Those must be
committed before any OPRR state, transition, candidate timestamp, or side is
computed. The horizon is selected prospectively as one complete 24-hour
transmission cycle and matches the already operationally audited CBOE/BTC
session cadence; no OPRR market outcome has been opened.

## Why this candidate is admissible now

The OPRR concept was recorded before the DCLB-864 and SCAF-48 source-incidence
results as:

```text
CBOE relative option-pressure rotation — reserve only
```

That reserve explicitly required:

- a cross-sectional ordinal transition rather than a level threshold; and
- agreement from the option surface's own change so that movement caused only
  by the other two surfaces could not masquerade as option relief or stress.

OPRR activates that pre-existing reserve. It does not use DCLB or SCAF event
counts, side balance, component dominance, gaps, or failed thresholds. Both
later candidates remain retired unchanged.

## Why this is not a CXRT repair

CXRT-288 used simultaneous `RELIEF` / `STRESS` / `NEUTRAL` level votes and an
equal-vote majority. It was retired before comparators and market outcomes
because its source composition failed, including excessive option-only side
reproduction and a long selection same-side run.

OPRR changes the state geometry rather than a CXRT threshold:

- no surface casts a level vote;
- no `0.50`, `0.25`, or `0.75` pressure boundary creates an opportunity;
- no two-of-three or three-of-three majority is used;
- option pressure cannot create a trade from its level or own change alone;
- the option surface must change ordinal position relative to the other
  surfaces; and
- the mechanism must require an independently defined same-direction change
  from the non-option state before it can emit a side.

The option surface is therefore a named rotational participant, not a
standalone direction generator. Its own change will be necessary to validate
the ordinal move but cannot be sufficient to create an OPRR timestamp or side.
The exact multi-surface transition remains to be frozen without decoding OPRR
incidence.

OPRR may not:

- reuse or relax any CXRT support, composition, or novelty threshold;
- use the CXRT majority side, unanimous state, minority state, or pressure
  bucket as an OPRR gate;
- promote `option_only` or any failed CXRT control;
- choose an ordinal transition, confirmation, direction, or horizon from BTC
  outcomes;
- use 2024-or-later CBOE or BTC data during candidate construction; or
- claim that aggregate option volume identifies an investor or directly
  causes BTC demand.

## Frozen source identities

### Volatility term structure

```text
data/cboe_volatility_term_structure_2018_2023/
cboe_vix_term_structure_2018-01-01_2023-12-31.csv.gz
SHA256 6f1b2f7f3a5b1e4d5001d673e6ff54374791879c278248ce27b3d610e4f75dc7
```

Manifest:

```text
data/cboe_volatility_term_structure_2018_2023/build_manifest.json
SHA256 42b2a35ad131bd63574d2adcf684e28766dc3060fa645fc749df10dd3fb27f27
```

### Tail-risk surface

```text
data/cboe_tail_risk_2018_2023/
cboe_tail_risk_2018-01-01_2023-12-31.csv.gz
SHA256 cdde3f8d4bb1e23d00b192f5f9ef759aefba9087be5fd60653e9c02479dfa41a
```

Manifest:

```text
data/cboe_tail_risk_2018_2023/build_manifest.json
SHA256 9ef80ef3034c93d97c5b2a8160b2502527287d570d15f9d7166d631d9866c7bd
```

### Option-flow surface

```text
data/cboe_option_flow_2020_2023/
cboe_option_flow_2020-01-01_2023-12-31.csv.gz
SHA256 35ef106ef01e3abadbcb4a6227187dd1d7cf2722191bd146bac06d08d1684a78
```

Manifest:

```text
data/cboe_option_flow_2020_2023/build_manifest.json
SHA256 0a513b146ad5857d9ab7311e978152c308de64db8ef29c4d463eb07ea503089e
```

The candidate-blind audits froze 1,509 term dates, 1,507 tail dates, and
1,006 option-flow dates through 2023. The later mechanism must retain the
existing causal source rules:

- compute each surface from its own strictly prior history;
- intersect exact source dates only after each surface is available;
- never fill, carry, interpolate, or synthesize a missing CBOE date;
- treat source date `D` values as unavailable for trading on `D`; and
- delay any decision until the first later regular CBOE session fixed by a
  prospective exchange-session calendar; the later session's source-row
  presence or absence may not create or suppress the decision.

This availability wording is intentionally stricter than the earlier CBOE
research clocks. It was corrected before the OPRR mechanism, source incidence,
or market outcome was opened: selecting the next row from a completed archive
would reveal future row membership. The mechanism must freeze the exact
session-calendar rule independently of the three source panels.

These are current historical vintages, not proof of point-in-time revision
history. Live promotion requires forward-vintage capture, parity checks, and
fail-flat behavior.

## Prior-family contamination and no-repair limits

The three source panels, their prior pressure values, and several prior CBOE
candidate outcomes have been opened. OPRR is therefore source-seen and
candidate-incidence-unseen, not a pristine global market holdout.

Prior retired policies include:

- CVTR-1 term-tail rotation;
- CTHD-1 hidden tail-pressure disagreement;
- CIHM-1 option-flow hedge migration; and
- CXRT-288 cross-surface level voting.

The next mechanism must bind their immutable clocks and predeclare novelty
tests before any OPRR incidence is decoded. OPRR may not inherit a prior
candidate's event-tail threshold, state label, side, gate, hold selected from
outcomes, or RLLM checkpoint.

The new falsifiable object must be the **change in relative rank ordering plus
independent multi-surface directional confirmation**. A renamed option delta,
option level, term-tail vote, or CXRT majority is forbidden.

## Mechanism proof required before incidence

The next mechanism commit must define without computing an OPRR feature row:

1. the exact strictly prior pressure construction for all three surfaces;
2. the exact ordinal-position and tie/unavailable behavior;
3. the exact prior-to-current rank-rotation algebra;
4. a non-option confirmation that cannot be satisfied by option data alone;
5. one deterministic side, availability clock, entry, and 288-bar hold;
6. global non-overlap before split containment;
7. option-only, non-option-only, stale, rank-permutation, flip, random, and
   delayed controls;
8. source-support, rotation-composition, and prior-family novelty gates fixed
   before candidate counts; and
9. live fail-flat behavior for missing, revised, duplicated, late, or
   cross-panel-inconsistent source rows.

The evaluator must prove that:

- no current source value enters its own prior rank history;
- no source-date value can trade on the same source date;
- a relative-order change caused only by term/tail movement is rejected;
- an option-only change without non-option confirmation is rejected; and
- ties, missing values, and unavailable prior states fail flat.

## Mandatory comparator cohort

Before OPRR incidence is decoded, the preregistration must hash-bind:

- CXRT-288 primary and its `term_only`, `tail_only`, `option_only`,
  `term_tail_agreement`, stale, flip, random, and delay controls;
- CVTR-1 primary;
- CTHD-1 primary; and
- CIHM-1 primary.

The mechanism must freeze exact-entry overlap, tolerant daily overlap,
same-entry side reproduction, and signed occupied-exposure correlation rules.
Undefined required metrics or comparator drift retires OPRR before outcomes.
No prior control may replace a failed OPRR primary.

## Evidence boundary

This selection unit inspected only:

- committed source audits, manifests, immutable hashes, and source schemas;
- the pre-existing OPRR reserve text;
- prior documented CBOE mechanism definitions and aggregate rejection
  summaries; and
- repository leakage and RLLM architecture constraints.

It did not decode or compute:

- a new term, tail, option, ordinal-position, rotation, confirmation, state,
  side, or OPRR candidate row;
- OPRR annual/monthly counts, gaps, side balance, control reproduction, or
  comparator overlap;
- any OPRR BTC price, funding, return, PnL, CAGR, strict MDD, or hit rate; or
- any 2024-or-later source or outcome.

## Mandatory sequence

1. commit this boundary;
2. freeze one exact rank-rotation mechanism and availability proof;
3. commit an immutable preregistration without decoding OPRR incidence;
4. commit and test an outcome-blind source-support/novelty evaluator;
5. retire OPRR unchanged on any provenance, support, composition, or novelty
   failure;
6. only a complete pass may freeze an economic/RLLM evaluator;
7. open train, selection, and later sealed extensions sequentially, stopping
   at the first failure.

## RLLM boundary

The deterministic composer owns opportunity creation, side, hold, and
leverage. A later RLLM may receive only compact causal relation tokens such as:

- fixed side;
- prior and current option ordinal position;
- one-step rotation magnitude and direction;
- option-own-change agreement;
- non-option confirmation relation;
- source validity and age;
- common-calendar gap bucket; and
- current position state.

Its action set is exactly:

```text
TRADE_FIXED_SIDE
ABSTAIN
```

Raw values, numeric ranks, dates, timestamps, row identifiers, split labels,
BTC prices, funding, future paths, outcomes, rewards, and historical
performance summaries are forbidden. The RLLM may not create a timestamp,
reverse the side, alter the hold or leverage, or infer a split from calendar
identifiers.
