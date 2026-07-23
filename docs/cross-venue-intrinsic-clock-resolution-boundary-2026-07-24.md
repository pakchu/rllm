# CVICR-72 candidate boundary — cross-venue intrinsic-clock resolution

## Selection

Select one new source-seen candidate:
**CVICR-72 — Cross-Venue Intrinsic-Clock Resolution Relay**.

The provisional economic object is an ordered Spot↔USD-M state:

1. Binance Spot and USD-M each advance on their own causal daily notional clock;
2. one venue reaches its prior-day-scaled notional anchor first;
3. at that first anchor, the two venues' cumulative taker-flow signs conflict;
4. by the lagging venue's own anchor, its cumulative flow resolves to the
   leading venue's side; and
5. only after the lagging anchor is complete may the USD-M instrument trade.

The provisional action follows the first venue's cumulative-flow side for a
six-hour consequence horizon. The final target fraction, minimum clock
dislocation, exact resolution predicate, availability buffer, controls,
support floors, novelty cohort, and stopping rules are not frozen here. They
must be fixed in a separate mechanism commit before any CVICR event incidence
is decoded.

## Why this axis was selected

The preceding IVPLH-72 candidate was retired at source support. Its exact
single-venue daily flow-handoff clock failed selection side support and maximum
month concentration before any economic outcome was opened. Reusing or
retuning that event would violate its no-repair boundary.

CVICR changes the observable rather than repairing IVPLH:

- IVPLH compared two calendar-consecutive anchors on one USD-M clock;
- CVICR compares two contemporaneous venue-specific clocks inside one UTC day;
- the event identity is a Spot↔USD-M first-passage race followed by a
  cross-venue flow-resolution transition;
- neither venue's wall-clock arrival time, flow level, nor prior alpha clock is
  sufficient by itself.

The source is operationally mature:

- official Binance Spot and USD-M one-minute kline archives;
- every one of 48 Spot and 48 USD-M monthly payloads checksum-verified;
- 420,768 complete five-minute grid rows over 2020–2023;
- 419,855 feature-valid rows;
- `feature_available_time_utc = date + 5m`;
- no label, target, reward, action, PnL, forward, or future field.

The selected combined source identity is:

```text
data/binance_cross_venue_minute_leadership_btc_2020_2023/
BTCUSDT_cross_venue_minute_leadership_5m_2020-01_2023-12.csv.gz
SHA256 00ab6a55fc7bfeb3012584db5bc97a7d7b98dd995491acfd3f865c6bd41f92cc
```

The source is available in the main checkout and its immutable hash, manifest,
schema, row count, availability rule, quarantine, and audit result were checked
without parsing a CVICR data row in this worktree. Copying or binding the exact
source is authorized only after this boundary is committed.

## Rejected alternatives

### AESS-72 — no-go

An aggregate-event-size sponsorship transition was rejected as the next
standalone alpha. Its principal fields and intended relational representation
overlap too heavily with:

- NETF's event-breadth/capital topology and event-size asymmetry;
- TAAR's event-tail, arrival irregularity, size asymmetry, and price response;
- CARTA's size-asymmetry ranks, topology transitions, and
  `ABSTAIN/FOLLOW/FADE` action set.

The prior aggTrade family also repeatedly produced gross moves too small for
the repository's executable cost model. AESS may later be an auxiliary token
or control, but it is not an independent alpha axis.

### AFCR-72 — reserve only

The six-alt flow conflict-resolution relay remains a logically valid backup,
but its source begins in 2023 and the adjacent FCIR, DTAC, and ticket-gap
families have already exposed substantial source incidence and weak 2023
economic results. It is less clean and less historically deep than CVICR.

### IFDA — permanently excluded

Individual-fill dispersion absorption cannot be reused. Its predecessor source
failed exact raw trade-ID continuity and explicitly requires a new mechanism
and source contract rather than a renamed continuation.

## Prior-family contamination and novelty limits

CVICR is not a pristine global-history discovery. This repository has already
opened pre-2024 outcomes for many Spot↔USD-M and intrinsic-time hypotheses.
Known failures include:

- CATCH-12: dense one-hour cash-to-perp handoff, gross movement approximately
  zero and far below costs;
- CLASP-24: fixed within-bar late-cash propagation;
- LURI-48: inferred multi-hour USD-M inventory release with unstable direction;
- CVTT: fixed within-bar temporal torsion with sub-cost gross movement;
- dual intrinsic price/flow clocks: regime-local 2023 strength but unstable
  pre-2023 fit;
- IVLIR, IVFHR, and IVPLH: single-venue daily equal-notional first-passage
  variants rejected at support.

CVICR therefore may not claim novelty merely from using the words
`leadership`, `handoff`, `intrinsic`, or `first passage`. The later mechanism
must prove that its exact paired daily clock and conflict→resolution sequence
is not:

- a threshold repair of CATCH, CLASP, LURI, CVTT, IVLIR, IVFHR, or IVPLH;
- a static within-bar centroid, basis, return, or lagged-response condition;
- a single-venue flow level with a second venue added as a gate;
- a dense per-bar cross-venue signal; or
- an outcome-selected direction, target fraction, dislocation tail, or hold.

Exact-entry, near-time, and occupied-position comparisons against the frozen
cross-venue and intrinsic-volume predecessor clocks are mandatory before any
economic evaluation.

## Evidence boundary

During this selection unit, only repository history, source schema, immutable
hashes, manifests, audit summaries, file availability, and prior-family
outcomes were inspected.

The following were **not** computed or decoded for CVICR:

- daily venue anchors;
- clock-leader identity;
- anchor-time gaps;
- cumulative Spot or USD-M flow signs;
- conflict or resolution incidence;
- candidate timestamps, sides, annual counts, or calendar concentration;
- comparator overlap;
- post-entry BTC price, funding, return, PnL, CAGR, strict MDD, or hit rate;
- any 2024-or-later source value.

This is an axis-selection document, not a preregistration or alpha result.

## Mandatory sequence

1. commit this boundary;
2. bind the exact source artifacts and freeze one mechanism, including
   direction, daily target construction, latency, hold, controls, source
   support, novelty gates, and failure action;
3. commit a write-once preregistration builder and immutable artifact without
   decoding CVICR incidence;
4. commit and test an outcome-blind source-support/novelty evaluator;
5. retire CVICR unchanged on any source, support, or novelty failure;
6. only a complete pass may authorize a separately committed strict economic
   evaluator;
7. only an unchanged deterministic OOS pass may expose compact causal relation
   tokens to one small RLLM policy.

## RLLM boundary

The deterministic event owns opportunity creation and side. A later RLLM may
receive only causal symbolic relations such as leader venue, clock-gap bucket,
initial conflict, resolution strength, source validity, and current position.
It may choose `TRADE_FIXED_SIDE` or `ABSTAIN`; it may not create a new clock,
reverse the side, alter the hold, or recover sealed outcomes through timestamps
or raw identifiers.

## Bound predecessor evidence

- `docs/binance-cross-venue-minute-leadership-data-audit-2026-07-14.md`
- `docs/binance-cross-venue-minute-leadership-data-design-2026-07-14.md`
- `docs/cash-auction-transfer-catchup-handoff-selection-result-2026-07-14.md`
- `docs/leveraged-um-inventory-release-handoff-selection-result-2026-07-14.md`
- `docs/cross-venue-temporal-torsion-v2-selection-2026-07-16.md`
- `docs/dual-intrinsic-clock-alpha-search-2026-07-14.md`
- `docs/intrinsic-volume-latent-impact-relay-support-rejection-2026-07-23.md`
- `docs/intrinsic-volume-flow-handoff-relay-support-rejection-2026-07-23.md`
- `docs/ivplh-source-support-rejection-2026-07-24.md`
- `docs/individual-fill-dispersion-absorption-source-rejection-2026-07-20.md`
