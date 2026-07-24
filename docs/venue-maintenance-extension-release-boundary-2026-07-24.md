# VMER-2 candidate boundary — venue maintenance extension release

## Selection

Select one new, outcome-unseen candidate:

**VMER-2 — Venue Maintenance Extension Release**.

VMER asks whether the completed release of an **explicitly extended scheduled
exchange maintenance** reveals queued order direction in the first fully
causal BTC bar and carries that direction for two hours.

The provisional causal chain is:

1. an official object is typed by Statuspage as scheduled maintenance, not an
   unscheduled incident;
2. official update text states that practical trading or position-management
   access is materially affected;
3. a later official update explicitly states that the maintenance is delayed,
   extended, taking longer than expected, or otherwise past its expected
   completion;
4. a later update completes restoration of the same capability;
5. deterministic code waits for source readiness and one complete BTC
   revelation bar;
6. the fixed side follows that completed revelation bar; and
7. entry occurs at the next five-minute open for a fixed two-hour hold.

The semantic model may decide only whether the bounded official update prefix
contains a material trading-access maintenance, an explicit unexpected
extension, and completed same-capability restoration. It may not inspect
price, infer a return, choose side, choose a threshold, choose hold, or
generate a trade.

The `2` suffix reserves a two-hour consequence horizon. The next mechanism
must freeze the exact source parser, semantic corpus, model, support floors,
source clock, revelation normalization, controls, and evaluator before any
2020–2023 maintenance incidence or BTC row is opened.

## Why this is not VARR repair

VARR-6 was immutably retired in commit `55226a8` because a diagnostic parsed
1,719 comparator clock rows before causal-market support. VARR produced no
incident incidence, semantic result, market row, displacement, side, return,
or checkpoint.

VMER does not repair VARR:

- VARR used unscheduled incident recovery; VMER accepts only typed scheduled
  maintenance;
- VARR opposed displacement accumulated during impairment; VMER follows a
  complete post-release revelation bar;
- VARR required an impairment onset; VMER requires an explicit unexpected
  schedule extension;
- VARR reserved a six-hour reversal; VMER reserves a two-hour continuation;
- VARR's uncommitted prompt, thresholds, support floors, and draft mechanism
  are discarded; and
- the two comparator artifacts opened during the VARR breach are permanently
  forbidden from VMER:

```text
data/premium_compression_breakout_relay_clocks_2020_2026.csv.gz
data/premium_snapback_recenter_clocks_2020_2026.csv.gz
```

No observed row, count, endpoint timestamp, side, feature value, or coverage
fact from those files may influence VMER.

## Rejected alternative source axes

### EIA-930 ERCOT grid relief

EIA-930 has dense official hourly ERCOT data, but it is not a strict
point-in-time historical ledger. EIA documents that balancing authorities
submit best available values and later correct historical values. The current
API exposes the latest history but no observation-level as-of snapshot,
revision lineage, or release-version endpoint was found.

Therefore a 2020–2023 backfill can reproduce current corrected values but not
prove what a live policy saw at each historical decision time. The axis is
rejected for historical alpha evaluation; it remains suitable only for
forward snapshot collection.

Official evidence:

- [EIA Hourly Electric Grid Monitor — about and data quality](https://www.eia.gov/electricity/gridmonitor/about)
- [EIA API technical documentation](https://www.eia.gov/opendata/documentation.php)
- [Form EIA-930 instructions](https://www.eia.gov/survey/form/eia_930/instructions.pdf)
- [EIA timely grid-monitor data](https://www.eia.gov/todayinenergy/detail.php?id=43295)

### Bitcoin Stack Exchange distress breadth

Stack Exchange API v2.3 exposes public questions, answers, comments, and post
revisions. It does not expose a complete comment-edit history, and the public
data surfaces cannot reconstruct full deleted/private post history. Using only
currently surviving posts would retrospectively filter content that a live
policy could have seen.

The broad distress-recovery source is therefore rejected. A narrower
never-deleted-question study could be descriptive, but it is not admissible as
the complete causal source originally proposed.

Official evidence:

- [Stack Exchange API v2.3](https://api.stackexchange.com/docs)
- [Post revisions by IDs](https://api.stackexchange.com/docs/revisions-by-ids)
- [Comments endpoint](https://api.stackexchange.com/docs/comments)
- [Stack Exchange API throttles](https://api.stackexchange.com/docs/throttle)
- [SEDE/public-dump schema documentation](https://meta.stackexchange.com/questions/2677/database-schema-documentation-for-the-public-data-dump-and-sede)

## Official VMER source axis

Only the official Statuspage surfaces of these exact pages are eligible:

| Venue | Page ID | Page name | Domain |
|---|---|---|---|
| Coinbase Exchange | `bklmvp2c52bl` | `Coinbase Exchange` | `status.exchange.coinbase.com` |
| Kraken | `lfz25gyhcpjf` | `Kraken` | `status.kraken.com` |

Historical enumeration:

```text
https://status.exchange.coinbase.com/history?page={page}
https://status.kraken.com/history?page={page}
```

Typed detail resolution:

```text
https://status.exchange.coinbase.com/api/v2/incidents/{code}.json
https://status.exchange.coinbase.com/api/v2/scheduled-maintenances/{code}.json

https://status.kraken.com/api/v2/incidents/{code}.json
https://status.kraken.com/api/v2/scheduled-maintenances/{code}.json
```

Live consistency:

```text
https://status.exchange.coinbase.com/history.atom
https://status.exchange.coinbase.com/api/v2/scheduled-maintenances.json
https://status.exchange.coinbase.com/api/v2/components.json

https://status.kraken.com/history.atom
https://status.kraken.com/api/v2/scheduled-maintenances.json
https://status.kraken.com/api/v2/components.json
```

Official source/documentation pages:

- [Coinbase Exchange incident history](https://status.exchange.coinbase.com/history)
- [Kraken Status API](https://status.kraken.com/api)
- [Kraken incident history](https://status.kraken.com/history)

The official Statuspage API distinguishes incident lifecycle states from
scheduled-maintenance lifecycle states. Scheduled maintenance progresses
through:

```text
scheduled
in_progress
verifying
completed
```

The recent API is not a historical replacement. Paginated history enumerates
old codes; the exact per-code endpoint type determines whether one code is an
incident or scheduled maintenance.

No mirror, search cache, social repost, third-party outage tracker, or manual
reconstruction may enter the source.

## Pre-candidate source feasibility

Only source warm-up outside the candidate period has been inspected:

- page 40 for each venue established archive reach to August–October 2016;
- page 30 for each venue established a February–April 2019 warm-up page;
- one pre-2020 Coinbase Exchange scheduled-maintenance code returned 404 from
  the incident endpoint and 200 from the scheduled-maintenance endpoint;
- one pre-2020 Kraken scheduled-maintenance code returned the same typed
  404/200 pair;
- one separate pre-2020 Kraken incident whose name contained “maintenance”
  returned 200 only from the incident endpoint; and
- typed scheduled-maintenance objects exposed
  `scheduled_for`, `scheduled_until`, `started_at`, `resolved_at`,
  `incident_updates`, and update-level
  `created_at`, `display_at`, `updated_at`, `status`, `body`, and
  `affected_components`.

This warm-up fixed only transport and schema. It did not set a candidate-period
count, support floor, side, return threshold, or economic parameter.

No 2020–2023 history code, maintenance object, name, update body, component,
timestamp, duration, extension, venue count, or event count has been
enumerated or decoded.

## Source identity and mutable-field boundary

For every retained history code, the next mechanism must request both typed
detail endpoints and require exactly one HTTP 200 and one HTTP 404. Only an
object returned under top-level key `scheduled_maintenance` may enter VMER.
An object returned as `incident` is a typed negative and may never enter the
semantic model, source support, or a trade.

VMER semantic and clock state may use only update-level:

```text
id
incident_id
status
body
created_at
display_at
updated_at
affected_components
```

The object-level name, impact, final status, `scheduled_for`,
`scheduled_until`, `started_at`, `monitoring_at`, `resolved_at`, components,
and `updated_at` are audited but forbidden from the signal. They are mutable
final-object fields without version history and may not rewrite an earlier
causal update prefix.

For update `U`, the intended clocks are:

```text
event_time(U) = max(created_at(U), display_at(U))
available_time(U) = max(created_at(U), display_at(U), updated_at(U))
```

The final text of an update is unknown before its own `updated_at`.
Historical prefixes are replayed by update availability. A later update may
not retrospectively cancel, relabel, or retime a previously eligible prefix.
One maintenance object may emit at most one candidate.

Live inference uses the same update ontology but readiness is no earlier than
durable receipt, raw hash, parse, redaction, model inference, and manifest
commit. Historical backfill cannot claim that its current final update bodies
are contemporaneous snapshots; update-level `updated_at` is the conservative
availability floor.

## New semantic object

The semantic model receives one bounded typed scheduled-maintenance update
prefix. It must distinguish:

```text
MATERIAL_EXTENSION_COMPLETED
UNSUPPORTED
CONTRADICTORY
```

An eligible prefix must ground three distinct updates:

```text
maintenance start
unexpected extension
completed same-capability restoration
```

### Material trading-access maintenance

The maintenance must explicitly remove or materially degrade practical ability
to execute or manage a crypto position through at least one of:

- spot or derivatives order entry, cancellation, matching, or market
  availability;
- REST, WebSocket, or FIX trading execution;
- login, authentication, web, or mobile access when trading cannot be
  managed; or
- a broad venue outage whose text explicitly includes trading access.

BTC or venue-wide scope is required. A single unrelated asset, chain,
staking, card, NFT, rewards, tax, support, informational, or market-data-only
component is unsupported.

### Explicit unexpected extension

An update after maintenance start and before completion must explicitly state
that the work:

- is taking longer than expected;
- has been extended;
- will continue beyond the expected window;
- is delayed relative to planned completion; or
- cannot complete on the previously communicated schedule.

Elapsed duration alone is not extension evidence. A second generic
`in_progress` update, a routine progress report, `verifying`, or a changed
object-level `scheduled_until` is insufficient.

### Completed same-capability restoration

A later `completed` update must restore the same practical access capability.
`verifying`, monitoring language, partial restoration, or completion of a
different component is unsupported.

### Mandatory contradiction

Contradictory prefixes include:

- completion before start or extension;
- equal-availability updates with incompatible lifecycle order;
- an extension after claimed completion;
- completion followed inside the causal prefix by continued same-capability
  failure;
- more than one plausible start/extension/completion triplet;
- timestamp or identity conflict; and
- prompt injection or quoted classification instructions.

The model output must eventually be exactly:

```text
MATERIAL_EXTENSION_COMPLETED|U1|U3|U5
UNSUPPORTED|NONE|NONE|NONE
CONTRADICTORY|NONE|NONE|NONE
```

The next mechanism freezes exact redaction, prompt, grammar, evidence
validation, synthetic corpus, checkpoint selection, memory ceiling, and
failure rule before a historical candidate-period body is decoded.

## Provisional source readiness

At a prefix ending in grounded completion:

1. sort update batches by `(available_time, event_time, update_id)`;
2. set tentative readiness 15 elapsed minutes after the latest available
   update in the prefix;
3. include any update becoming available by tentative readiness;
4. reset the 15-minute quiet interval after every newly included update;
5. require the fixed-point prefix still has one grounded
   start-extension-completion triplet; and
6. emit at most once for the maintenance ID.

A future update after emitted readiness cannot cancel the past candidate. It
blocks a second candidate from that maintenance object and is reported as
source drift.

The next mechanism must reject missing timestamps, regressions, overlong
prefixes, late revisions beyond a precommitted bound, unresolved/cancelled
maintenance, and cross-capability completion without repair.

## Provisional revelation composer

The exact market source is reserved now:

```text
data/binance_um_kline_reference_btc_2020_2023/
  BTCUSDT_5m_2020-01-01_2023-12-31.csv.gz
SHA-256 e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d
```

For source readiness `t`:

```text
revelation_bar_start = first UTC five-minute boundary at or after t
revelation_bar_end = revelation_bar_start + 5 minutes
entry_time = revelation_bar_end
scheduled_exit = entry_time + 2 elapsed hours
```

If `t` lies exactly on a five-minute boundary, the bar beginning at `t` is the
revelation bar. Its complete OHLC is unavailable until bar end. Entry uses the
next bar open, which is the same timestamp as revelation bar end.

The provisional normalized revelation is:

```text
impulse = log(revelation_close / revelation_open)
sigma_5m = RMS of exactly 288 completed five-minute log returns
           whose close times are strictly before revelation_bar_start
z_impulse = impulse / sigma_5m
```

Fixed intended qualification and side:

```text
z_impulse >= +0.75 -> LONG
z_impulse <= -0.75 -> SHORT
otherwise          -> NO TRADE
```

The side follows the first completed post-release revelation. The next
mechanism must freeze exact close-time indices, grid requirements, latency,
non-overlap, costs, funding, and strict MDD before any candidate-period BTC
row is opened.

The hypothesis is not that maintenance predicts price direction. It is that an
unexpectedly extended access constraint can queue imbalanced orders and that
the first completed post-release bar reveals their direction before a short
continuation. A price-only revelation control is therefore mandatory.

## Split and absolute stage gate

| Role | Interval |
|---|---|
| transport/schema/semantic warm-up | before `2020-01-01T00:00:00Z` |
| train | `[2020-01-01, 2023-01-01)` |
| selection | `[2023-01-01, 2024-01-01)` |
| sealed extension | `2024-01-01` and later |

Split membership uses final causal entry time. Entry and fixed exit must both
remain inside one split.

The sequence is absolute:

1. commit this boundary and the exact comparator cohort without parsing any
   comparator row;
2. freeze and pass a synthetic-only semantic gate without historical
   2020–2023 update text;
3. only a synthetic pass may decode historical scheduled-maintenance updates
   and test source/semantic support with market rows exactly zero;
4. only source support may open BTC bars completed by source readiness and the
   one revelation bar, then compute fixed side and source-clock support;
5. only causal-market support may parse the frozen comparator cohort and test
   novelty;
6. only support and novelty may open post-entry BTC bars, funding, future
   returns, PnL, absolute return, CAGR, strict MDD, bootstrap, or reward; and
7. 2024 and later remain sealed throughout.

Any stage breach retires VMER unchanged.

## Frozen outcome-blind control families

Before incidence, freeze these controls:

1. `all_trading_maintenance`: every materially access-affecting completed
   maintenance without extension semantics;
2. `lexicon_extension`: deterministic extension phrases without Gemma;
3. `status_only`: every typed
   `in_progress -> completed` maintenance;
4. `generic_second_update`: every maintenance with at least two progress
   updates regardless of extension meaning;
5. `no_material_scope`: every explicit extension regardless of affected
   capability;
6. `price_only_revelation`: the same revelation threshold on deterministic
   matched timestamps with no maintenance extension;
7. `delay_2h` and `delay_6h`;
8. `exact_side_flip`; and
9. `deterministic_random_side`.

Matched controls, side controls, and delays may construct clocks only after
source/market support; they may not open future returns until novelty passes.

## Frozen prior-clock cohort

Only raw compressed bytes were hashed in this unit. No header, row, timestamp,
side, or event count was decompressed or parsed.

| Family | Artifact | SHA-256 |
|---|---|---|
| cross venue | `data/cross_venue_intrinsic_clock_resolution_clocks_2020_2023.csv.gz` | `9f05b372686805539dbf56fb9b7ea7a8f90f8887d6731e1a8e1b1c1db14d8c0e` |
| intrinsic-volume price lag | `data/intrinsic_volume_price_lag_handoff_clocks_2020_2023.csv.gz` | `2efca3b44b0512a9423da90171f43babcadec2316dc6148796f3e61f98138e80` |
| intrinsic-volume latent impact | `data/intrinsic_volume_latent_impact_relay_clocks_2020_2023.csv.gz` | `523f24a0d955fe99cfb86c62078532c5fc9091234e6669ab9acff2a8f3367788` |
| quantity-lattice cohort | `data/quantity_lattice_cohort_disagreement_evaluation_clocks_2020_2023.csv.gz` | `c699c2d8c462b465579eb4035c76dda96923a4f39663395b371a04e9ad6de4a9` |
| funding divergence | `data/address_funding_divergence_relay_clocks_2021_2023.csv.gz` | `d688c4e4d845cf0a4daaf14b7ecfa6bb4c990bde59602eb9d55ffc7088c6d7b9` |
| semantic issuer breadth | `data/sec_bitcoin_issuer_reactivation_breadth_2020_2023/birb120_support_clocks_2020_2023.csv.gz` | `8f0831120764793a06873dc7ed4e1b97d3deff75d89572e2b4b8f9459bdfea41` |
| macro narrative | `data/federal_liquidity_narrative_sponsorship_relay_clocks_2020_2023.csv.gz` | `3096143d397fc6d8dac639841c96538979772734dcf2fd8157df580f5b297c6c` |
| Treasury settlement | `data/treasury_auction_settlement_collision_carry_2020_2023/tascc72_support_clocks_2020_2023.csv.gz` | `0333ba7f523d86a310e76ac51c15e4d273a1f4fb3e98f5e48dad530ac3696de4` |

The next preregistration must bind these exact raw hashes. It may not add,
remove, replace, or clip a cohort member after source incidence. Comparator
parsing begins only after causal-market support.

## LLM and RLLM boundary

Use a new synthetic-only Gemma 4 adapter. No EBOC adapter, EBOC mixed-class
exception, VARR draft prompt, historical maintenance sentence, market row,
return, or reward may enter adaptation.

The LLM receives only redacted update-prefix text, statuses, and normalized
component-capability phrases. It returns class and grounded update IDs.

Deterministic code owns:

```text
source type
update availability
prefix construction
readiness
revelation bar
volatility normalization
threshold
side
entry
hold
exposure
costs
funding
non-overlap
gates
returns and risk
```

Only if unchanged deterministic VMER passes synthetic, source, causal-market,
novelty, train economics, and selection economics may a train-only RLLM
receive causal symbolic state:

```text
maintenance capability token
extension relation token
extension-to-completion duration bin
revelation magnitude bin
pre-entry volatility bin
current position state
```

Its only actions are:

```text
TRADE_FIXED_SIDE
ABSTAIN
```

It may not create a clock, change side, change hold, change leverage, add a
stop, or train/select on selection or extension rewards.

## Evidence boundary

This unit opened:

- official EIA, Stack Exchange, Coinbase Exchange, Kraken, and Statuspage
  documentation;
- official Statuspage response headers and pre-2020 page/schema probes;
- pre-2020 typed incident-versus-maintenance endpoint results;
- the VARR breach report and prior repository documents needed for novelty;
- raw SHA-256 over the exact comparator cohort;
- the existing audited market-file path and raw hash; and
- no candidate-period source or outcome.

This unit did not:

- enumerate or decode a 2020–2023 VMER history row, maintenance object, update,
  class, grounded triplet, duration, extension, readiness, count, or side;
- decompress or parse a comparator row;
- read a candidate BTC bar, revelation, future bar, funding row, return, PnL,
  absolute return, CAGR, MDD, hit rate, or reward;
- call a model on historical maintenance text; or
- open a 2024-or-later VMER source or outcome.

Current counters:

```text
candidate_history_rows = 0
candidate_detail_objects = 0
candidate_update_bodies = 0
historical_model_calls = 0
comparator_rows = 0
market_rows = 0
future_rows = 0
funding_rows = 0
return_or_pnl_fields = 0
post_2023_candidate_rows = 0
```

## Required next commit

Before candidate-period incidence:

1. freeze exact HTTPS transport, history crawl/termination, type resolution,
   raw hashing, duplicate handling, and sealed traversal;
2. freeze update-prefix replay, revision bounds, redaction, ontology, prompt,
   output grammar, evidence grounding, synthetic corpus, Gemma recipe,
   checkpoint rule, and memory gate;
3. freeze source-only train/selection support floors using only semantic event
   counts, venues, active months, gaps, concentration, lifecycle integrity,
   and extension-to-completion duration;
4. freeze exact completed-bar indices, RMS normalization, `0.75` threshold,
   side, entry, two-hour hold, exposure, costs, funding, non-overlap, and
   causal-market support floors;
5. freeze control construction and novelty thresholds against the exact
   cohort above;
6. freeze economic and bootstrap gates with absolute return, wall-clock CAGR,
   global/pre-entry-high-water strict MDD, and stress costs; and
7. retire without repair on any semantic, source, support, novelty, economic,
   or evidence-boundary failure.
