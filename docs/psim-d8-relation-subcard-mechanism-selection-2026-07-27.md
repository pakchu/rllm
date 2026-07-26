# PSIM-D8 relation-subcard mechanism selection

Date: 2026-07-27 KST

## Decision

PSIM receives exactly one final source-representation successor:

```text
PSIM_D8_LOGICAL_DAY_CARD_WITH_ORDERED_RELATION_SUBCARDS_V1
```

D8 keeps one logical `DailyCard` per archive schedule and decision day. It
preserves D7's complete ordered relation roster, then publishes an audit-only
manifest describing contiguous model subcard slices of at most 64 relation
units.

This decision authorizes only a D8 source preregistration. It does not
authorize official source execution, model loading, LLM inference, market or
funding access, reward construction, trades, PnL, CAGR, strict MDD, or
outcomes.

## Why D8 is the last source successor

D1 through D7 consumed seven source-engineering rounds. D7 finally
materialized the complete 5,356-event source roster with zero semantic errors,
but failed at the first daily-card construction because legitimate bulk
proposal days exceed the original 64-unit card bound.

The failure is narrow enough to justify one lossless representation repair,
but further source-axis adaptation would become historical source-support
overfitting. Therefore:

```text
D8 source failure => retire PSIM permanently; no D9
```

Even a D8 source pass is not alpha evidence. A separate preregistered
memorization/model stage must pass before market outcomes can be opened.

## D7 evidence binding

The immutable D7 terminal result remains:

```text
result SHA-256
36702b4737f1bb37e901241a96e04f30e77132bb6a18ade1fab277a83f15557e

result_hash
45846070617398860a03f5a401047c95a37c7ba3526c37fbcea5a11687e8658b
```

The read-only D7 cardinality audit established:

| statistic | value |
|---|---:|
| overflowing schedule/day cells | 24 |
| first overflow | 143 relation units |
| maximum | 1,221 relation units |
| frozen cap | 64 |
| network/model/market/outcome access | 0 |

The D7 grammar, transport, source incidence, split, schedule, relation order,
controls, quarantine, and forbidden-access contracts remain unchanged.

## Selected nested-manifest representation

Top-level multiple `DailyCard` rows were rejected because D7's day identity,
control denominator, future-append comparison, and card map all require one
card per `(schedule, decision_at)`.

Instead, a D8 logical card contains:

1. D7's complete ordered `relation_units`;
2. `complete_relation_roster_sha256`;
3. an ordered `relation_subcard_manifest`; and
4. D7's ordinary logical-card payload and hash chain.

The logical card is source/audit material and is never passed wholesale to
the model.

For `N` relation units:

```text
subcard_count = ceil(N / 64)
start(k) = 64 * k
end_exclusive(k) = min(64 * (k + 1), N)
```

Every subcard binds:

- schedule and decision timestamp;
- ordinal and total subcard count;
- start and end offsets;
- relation-unit count;
- exact sliced-payload SHA-256;
- previous subcard hash; and
- complete relation-roster SHA-256.

The subcard chain starts from a deterministic hash of schedule, decision time,
complete roster hash, and `PSIM_D8_SUBCARD_CHAIN_START`. The enclosing logical
card hash then binds the completed manifest. It never creates a circular hash
dependency.

## Losslessness and model boundary

The partition must be:

- contiguous;
- in original D7 order;
- complete;
- non-overlapping;
- without gaps or duplication; and
- exactly replayable from the logical roster.

Dropping, sampling, semantic selection, summarization, cap raising, and
market/outcome-dependent partitioning are forbidden.

The 64-unit value is redefined as:

```text
maximum_model_relation_units_per_subcard
```

It is no longer a limit on the audit-only logical day card. A later model
stage may receive only a verified slice reconstructed from the manifest.
Audit hashes, full logical payloads, proposal/commit/event identities, paths,
blob hashes, receipts, and source text outside the existing model-text
contract remain model-hidden.

Model output aggregation is deliberately unresolved:

```text
UNDECIDED_NOT_AUTHORIZED_BY_D8_SOURCE_PREREGISTRATION
```

It must be fixed before loading a model or opening an outcome.

## Rejected alternatives

### Raise the card cap

Rejected. A new cap would be chosen after observing D7 incidence and would
still permit an impractically large single LLM payload.

### Daily summary aggregation

Rejected. It is lossy and would introduce a new semantic-selection channel
before any model-stage preregistration.

### Multiple top-level daily cards

Rejected. It changes the identity and denominator of inherited daily support,
control sensitivity, future invariance, and hash-chain gates.

## Synthetic mechanism battery

The mechanism battery covers exact relation counts:

```text
1, 64, 65, 70, 143, 1,221
```

Expected subcard counts are:

```text
1, 1, 2, 2, 3, 20
```

It also rejects an empty roster and detects tampered ranges, payload hashes,
subcard chains, and complete-roster hashes.

The probe reads only the committed D7 terminal and forensic artifacts plus
their committed producer hashes. It does not open `/tmp/psim-d7-source`,
historical proposal text, network, model, market, reward, trade, PnL, or
outcomes.

Canonical mechanism evidence:

```text
results/protocol_specification_intent_maturity_d8_mechanism_probe_2026-07-27.json

SHA-256
9c926f1fc44e60e4fcf92679dfd36db8d410220dcbbecec8c71e05bba0076d76

result_hash
3b690e6e11399a12aca41a2ba79f74f5d8642f029dc5241d72d342a6f3706672

scenario_roster_hash
9a718845c1af15904a9d263511c601432d1ae3e2ddd17bad9e9bfb2fbefcc00c
```

## Next boundary

The next unit is a D8 source-only preregistration that binds this mechanism,
inherits every unrelated D7 contract, requires a reviewed implementation and
direct-child seal, and authorizes exactly one fresh `/tmp/psim-d8-source`
attempt only after seal validation.
