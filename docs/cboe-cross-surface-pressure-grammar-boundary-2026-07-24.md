# CSPG-288 candidate boundary — CBOE cross-surface pressure grammar

## Selection

Select:

**CSPG-288 — CBOE Cross-Surface Pressure Grammar Policy**.

On every causally rank-complete common CBOE source date, one compact policy
will reason over relations among:

1. volatility-term pressure;
2. tail-risk pressure; and
3. option-flow pressure.

The policy chooses:

```text
LONG
SHORT
ABSTAIN
```

The opportunity clock is dense and action-independent. No hand-written vote,
majority, or surface owns direction. The `288` suffix reserves a 24-hour
consequence horizon.

This boundary authorizes only a separate exact mechanism freeze. It opens no
new CSPG source value, token, clock, comparator row, BTC outcome, funding,
return, PnL, model label, or post-2023 row.

## Selection review

The immediately preceding boundary selected `BCRT-72` as a plausible
Bitcoin-ledger relational policy. Before its mechanism was committed or any
BCRT-derived value was opened, two independent adversarial reviews converged
on the same stronger next family:

- dense CBOE cross-surface state grammar;
- then dense intrinsic-volume grammar;
- then SOMA allocation grammar.

The reviews agreed that recent failures were caused first by source/clock
geometry and only later by economics. They recommended dense deterministic
opportunities with learned abstention rather than another sparse conjunction.

BCRT remains a documented reserve. It is not failed or retuned, but is not the
next experiment because:

- the repository already opened poor BATE-family train economics from the same
  confirmed-ledger family;
- base-chain state is highly reproducible but not as orthogonal to prior
  candidate knowledge as an exogenous risk surface;
- a new CBOE grammar uses no Gross-8 sleeve input; and
- the CBOE source already demonstrated 879 rank-complete common dates, enough
  to test a dense policy without weakening a failed event gate.

No BCRT source row, feature, token, incidence, or outcome was opened between
its boundary and this selection.

## Why this is not CXRT or OPRR repair

`CXRT-288` used the same three source families but collapsed them into:

```text
RELIEF / STRESS / NEUTRAL votes
-> at least two nonzero votes
-> deterministic majority side
```

CXRT was retired before comparators and outcomes. Its source-only report
disclosed:

```text
term rows decoded           1509
tail rows decoded           1507
option rows decoded         1006
exact common dates          1006
rank-complete common dates   879
schedulable common dates     878
```

The first failure was excessive same-side run length; option-flow dominance
also collapsed the intended cross-surface composition. Those exact votes,
majority side, run caps, hold controls, and source-support identity remain
retired.

`OPRR-288` required a rare option-rank rotation plus aligned movement. It
produced only 28 globally accepted clocks and was retired before outcomes.
Its event predicates may not be relaxed.

CSPG changes the predictive object:

- every valid rank-complete common source date is an opportunity;
- no vote or deterministic majority is created;
- no source surface owns LONG or SHORT;
- continuous source ranks become compact categorical relations;
- level, change, ordering, dispersion, and prior-state transition remain
  simultaneous weak observations;
- the policy may abstain, but abstention does not release the reserved clock;
- the candidate receives a new state grammar, support gate, learnability gate,
  policy, novelty gate, and failure identity.

This is source-seen and globally CBOE-outcome-seen research. It makes no
clean-room discovery claim. Exact CSPG tokens, action labels, and market
outcomes remain unopened.

## Why this is RLLM-shaped

The model is not asked to forecast a raw price. It must reason over relations
such as:

- whether term pressure is high while tail pressure remains low;
- whether option pressure is leading or contradicting the other surfaces;
- whether all surfaces are compressed near each other or widely dispersed;
- whether pressure is jointly rising, jointly falling, or rotating;
- which surface is the current stress or relief extreme; and
- whether the complete relation topology persists or changes from the prior
  common date.

Each observation is weak. The hypothesis is that their joint causal grammar
changes the conditional utility of LONG, SHORT, and ABSTAIN.

CSPG uses one text-only policy. There is no analyzer/trader pair, free-form
chain of thought, model-generated feature, or model-controlled accounting.
Source validation, ranks, clock, costs, reward, strict MDD, and execution remain
deterministic code.

Cheap causal policies must demonstrate temporal transfer before Gemma is
trained. A language model may not conceal an unlearnable state surface.

## Frozen source families

### Volatility term structure

```text
data/cboe_volatility_term_structure_2018_2023/
  cboe_vix_term_structure_2018-01-01_2023-12-31.csv.gz
SHA256 6f1b2f7f3a5b1e4d5001d673e6ff54374791879c278248ce27b3d610e4f75dc7

data/cboe_volatility_term_structure_2018_2023/build_manifest.json
SHA256 42b2a35ad131bd63574d2adcf684e28766dc3060fa645fc749df10dd3fb27f27
```

Allowed columns:

```text
observation_date
VIX9D_close
VIX_close
VIX3M_close
```

### Tail-risk surface

```text
data/cboe_tail_risk_2018_2023/
  cboe_tail_risk_2018-01-01_2023-12-31.csv.gz
SHA256 cdde3f8d4bb1e23d00b192f5f9ef759aefba9087be5fd60653e9c02479dfa41a

data/cboe_tail_risk_2018_2023/build_manifest.json
SHA256 9ef80ef3034c93d97c5b2a8160b2502527287d570d15f9d7166d631d9866c7bd
```

Allowed columns:

```text
observation_date
SKEW_close
VVIX_close
VIX_close
```

### Option-flow surface

```text
data/cboe_option_flow_2020_2023/
  cboe_option_flow_2020-01-01_2023-12-31.csv.gz
SHA256 35ef106ef01e3abadbcb4a6227187dd1d7cf2722191bd146bac06d08d1684a78

data/cboe_option_flow_2020_2023/build_manifest.json
SHA256 0a513b146ad5857d9ab7311e978152c308de64db8ef29c4d463eb07ea503089e
```

Allowed columns:

```text
observation_date
total_volume
index_call_volume
index_put_volume
index_volume
equity_call_volume
equity_put_volume
vix_call_volume
vix_put_volume
```

The next mechanism must preserve exact allowlists, hashes, independent
strictly-prior histories, exact-date intersection, and cross-panel VIX equality.
Missing dates may not be filled, carried, interpolated, or replaced with zero.

These are frozen current historical vintages, not proof of point-in-time
revision history. Historical use must delay a completed close until a later
exact common CBOE date. Live promotion requires forward raw response capture,
retrieval timestamps, revision alarms, and schema/value parity.

## Provisional source state

The mechanism may reuse the already frozen causal primitives, but not CXRT
votes or side:

```text
front_slope = log(VIX9D_close / VIX_close)
broad_slope = log(VIX_close / VIX3M_close)

skew_level    = log(SKEW_close / 100)
vvix_relative = log(VVIX_close / VIX_close)

institutional_gap =
    log((index_put_volume + 0.5) / (index_call_volume + 0.5))
  - log((equity_put_volume + 0.5) / (equity_call_volume + 0.5))

vix_call_pressure =
    log((vix_call_volume + 0.5) / (vix_put_volume + 0.5))

index_share =
    log((index_volume + 1.0) / (total_volume + 1.0))
```

Each primitive rank must use at most 252 strictly prior same-source
observations and at least 126. The current value is appended only after its
rank is fixed. Option primitives use current-minus-immediately-prior
option-source deltas before rank.

The next mechanism must freeze exactly one compact grammar using:

- coarse term, tail, and option pressure levels;
- current versus prior common-date changes;
- cross-surface order/extremes;
- cross-surface dispersion;
- agreement/disagreement topology; and
- complete topology transition.

It may not use dates, identifiers, raw values, raw ranks, CXRT votes, CXRT
majority side, OPRR rotation eligibility, BTC price, funding, premium, OI,
Kimchi, DXY, outcome, or PnL as model input.

## Provisional clock

For source date `D`:

1. compute all source state using `D` and strictly prior history;
2. let `D_next` be the first later date in the exact common CBOE calendar;
3. treat the state as unavailable before `D_next 09:30`
   America/New_York;
4. reserve the decision/entry at `D_next 09:35` America/New_York;
5. enter at the exact UTC-converted five-minute open;
6. hold exactly 288 five-minute bars; and
7. require the source state, availability, entry, hold, and exit to stay inside
   one split.

Never use the next row's market behavior, existence after the already-known
common calendar, or a synthesized weekend/holiday date to create or suppress a
clock. Opportunities are reserved globally before policy action; abstention
does not release them.

## Alternatives retained

- `BCRT-72`: clean Bitcoin Core production path, but lower immediate
  orthogonality and prior same-family economic failure.
- `DIVA-72`: strongest live parity and density, but higher overlap risk with
  REX/taker/intrinsic-volume sleeves.
- `SCAG-48`: strong macro orthogonality, but harder release/revision parity and
  greater SCAF-family identity risk.
- metadata-only SEC EDGAR topology: broad and causal, but prior semantic
  adapters failed and live issuer semantics are less direct.

## Mandatory stopping rules

1. Commit an exact mechanism and preregistration before deriving one CSPG
   token or incidence.
2. Source support must test density, temporal dispersion, token diversity,
   current-value exclusion, and action-independent reservation before outcomes.
3. Cheap policies must pass a chronological transfer gate before GPU work.
4. Comparator novelty must include tolerant-time overlap and signed exposure
   correlation with Gross-8 and prior CBOE policies before 2023 outcomes.
5. Use 2020–2021 only for fit, 2022 only for selection, and keep 2023 untouched
   by labels, prompt, checkpoint, threshold, token, and policy selection.
6. Any failed gate retires CSPG-288 unchanged.
7. A control diagnoses failure and may not replace the primary.
8. No post-observation change to source, formula, token, clock, hold, reward,
   model, threshold, or support floor is permitted under the same identity.

## Outcome boundary

```text
new CSPG source values read       = 0
CSPG token rows created           = 0
CSPG opportunity clocks opened    = 0
BTC market rows read              = 0
funding rows read                 = 0
comparator rows read              = 0
future-return rows read           = 0
return or PnL fields read         = 0
post-2023 source rows read        = 0
model labels created              = 0
model training runs               = 0
```

Status:

```text
selected_for_mechanism_freeze
```
