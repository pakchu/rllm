# CDLTR-72A outcome-blind preregistration

## Status and predecessor disposition

This document freezes **CDLTR-72A — Cross-Domain Liquidity Transmission
Relay** after the source-mechanism decision, comparator-capability amendment,
and complete comparator-clock freeze. No real CDLTR source value, feature
incidence, event count, BTC market row, funding value, return, PnL, CAGR, or
MDD has been opened.

CDLTR-72 was rejected before preregistration and before incidence because its
signed-exposure contract required directional intervals from artifacts that did
not define them. CDLTR-72A changes only that pre-incidence comparator
capability contract. It inherits the exact source votes, relay, execution,
windows, support gates, controls, and LLM/RL boundary. Component-family
development results are research-seen and cannot validate this interaction;
only shadow observations after final policy freeze may become pristine.

## Bound source contract

The JSON preregistration binds, by path and SHA-256, the mechanism decision,
the CDLTR-72A amendment, this document, all three normalized source panels,
their manifests, and their builders. Only these columns are available to the
later source-only evaluator:

- New York Fed ON RRP:
  `operation_date,result_available_at_utc,total_amount_accepted_usd,source_complete,quarantine_reason`;
- Cboe term structure: `observation_date,VIX9D_close,VIX3M_close`;
- Coin Metrics network:
  `observation_date,available_at,AdrActCnt,TxCnt,TxTfrCnt`.

Preregistration hashes each file and reads at most the CSV header. It does not
parse a source value row. A missing file, symlink, non-repository path, hash
drift, header drift, or unexpected manifest/builder identity fails closed.

## Frozen feature votes

### RRP vote

Use the current normal operation and the operation exactly five normal-operation
slots earlier. Every one of the six slots must be complete and unquarantined;
the implementation may not bridge across a broken slot.

```text
delta = accepted_now - accepted_fifth_prior_slot
LONG  when delta < 0
SHORT when delta > 0
NEUTRAL otherwise
```

The vote becomes causal only at the current row's exact
`result_available_at_utc`.

### Cboe vote

The prior intersection date's close becomes usable at 09:35
`America/New_York` on the next exact date present in the frozen three-index
intersection. No missing date is forward-filled.

```text
LONG  when VIX9D_close < VIX3M_close
SHORT when VIX9D_close > VIX3M_close
NEUTRAL otherwise
```

### Network vote

For each of `AdrActCnt`, `TxCnt`, and `TxTfrCnt`, take the sign of the log change
from the row exactly seven calendar days earlier. All eight dates and every
positive finite value must exist and already be causal.

```text
LONG  when at least two metric signs are positive
SHORT when at least two metric signs are negative
NEUTRAL otherwise
```

## Frozen relay and execution

All ages are elapsed UTC time from actual availability timestamps.

1. RRP and Cboe votes expire exactly 36 hours after their own availability.
2. A macro episode begins only when both unexpired votes enter the same nonzero
   side from an absent or opposite state.
3. Inspect only the first network report strictly after episode onset and no
   later than 36 hours after onset.
4. Emit one event only if that report agrees with the macro side. A neutral,
   opposite, invalid, missing, or late report kills the episode with no retry.
5. Another event requires macro state to leave agreement and re-enter it.

The confirmation `available_at` is the decision time. Entry is
`ceil_to_5m(decision_time) + 5 minutes`, including another five minutes when
decision time is already on a five-minute boundary. Exit is exactly 72 elapsed
hours after entry. Exposure is 0.5x with no stop, take-profit, trailing exit,
leverage search, or overlapping trade. Accept candidates in global
chronological order and skip any event whose entry or exit crosses a split.

## Frozen windows and support gates

- warm-up only: `[2020-01-01, 2021-01-01)` UTC;
- train: `[2021-01-01, 2023-01-01)` UTC;
- selection: `[2023-01-01, 2024-01-01)` UTC;
- every source and outcome at or after `2024-01-01T00:00:00Z` remains closed.

The source-only primary must satisfy all of the following:

- train total at least 60;
- at least 25 events in each train year;
- at least 12 events in every train half-year;
- selection total at least 30;
- at least 12 events in each selection half-year;
- at least 18 LONG and 18 SHORT events in train;
- at least 8 LONG and 8 SHORT events in selection;
- maximum UTC month share at most 0.20 in each split;
- maximum UTC weekday share at most 0.35 in each split.

Any failure rejects CDLTR-72A without repair and without opening BTC outcomes.

## Frozen controls

The support evaluator must retain these diagnostic clocks. None may replace a
failed primary:

1. `macro_only`;
2. `network_only`;
3. `reverse_order`;
4. `one_network_report_delay`;
5. `direction_flip`;
6. `deterministic_random_side`.

To preserve the inherited control exactly, the deterministic random control
retains the predecessor seed string: first SHA-256 byte of
`CDLTR-72|20260721|<entry_time_utc>`, LONG below 128 and SHORT otherwise. The
primary and each control must independently satisfy the same calendar and split
containment rules. Controls are reported only; they cannot repair or select the
primary.

## Complete comparator contract

Preregistration binds only the committed complete sanitized clock and its
manifest:

- `results/cdltr_prior_comparator_views_2026-07-21.csv.gz`;
- `results/cdltr_prior_comparator_views_manifest_2026-07-21.json`.

The clock has exactly 9,985 rows and only:

```text
comparator, capability, decision_time, entry_time, exit_time, side, source_clock
```

The directional interval comparators are ORFR-1, CVTR-1, NTB-7, NWE-8,
chain-activity impulse momentum, DFFB-601, and each of the four frozen FLCC-1
candidate clocks. Every FLCC candidate must pass independently.

NWE-7, the 2023 live-anchor clock, and every prior-microstructure constituent
are timestamp-only. Their `side` and `exit_time` are empty and their
`source_clock` labels are direction-neutral. The evaluator may not invent,
infer, or search a missing side, exit, hold, union, or conflict resolver.

Every nonempty comparator independently fails when either timestamp gate is
exceeded:

```text
decision-date Jaccard > 0.30
fraction of CDLTR dates within +/-1 UTC day > 0.50
```

Every directional comparator independently fails when:

```text
absolute signed occupied-exposure Pearson > 0.40
```

The exposure grid is complete five-minute UTC time, entry-inclusive and
exit-exclusive, with flat 0, LONG +1, and SHORT -1. An empty comparator,
identity mismatch, invalid interval, timestamp-only direction leakage, or
zero-variance directional exposure fails closed. Preregistration reads only
the bundle header; it does not parse a comparator event row or its manifest.

## Outcome and RLLM boundary

Only an unchanged pass of source integrity, primary support, every control
support check, and every novelty gate may authorize a separate strict economic
evaluator. That evaluator must be implemented, tested, committed, and
hash-frozen before opening train. It must stop on train failure and may open
selection only after exact train pass. Full-calendar CAGR includes idle cash;
strict MDD includes the pre-entry high-water mark, intratrade adverse path,
funding, and every cost.

Gemma/RLLM cannot create direction, reorder the relay, change expiry or hold,
or rescue a failed deterministic candidate. Only after deterministic train and
selection both pass may a small Gemma4-class model act as a constrained
`TRADE`/`ABSTAIN` veto. Any later RL reward must penalize strict drawdown and
turnover while preserving the frozen side, timing, and hold.

## Stopping rule

After this document, source-only support and novelty are the sole admissible
next stage. Observed incidence may not change any source field, transform, side
mapping, expiry, order, latency, hold, support floor, comparator identity,
capability, control, or novelty threshold. A failure permanently rejects
CDLTR-72A; a repair requires a new candidate identity and a new preregistration
before incidence is reopened.
