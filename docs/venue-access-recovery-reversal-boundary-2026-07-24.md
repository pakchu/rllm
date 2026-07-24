# VARR-6 candidate boundary — venue-access recovery reversal

## Selection

Select one text-native, source-feasible, incidence-unseen, outcome-unseen
candidate:

**VARR-6 — Venue Access Recovery Reversal**.

VARR asks whether the completed recovery of a material, unscheduled
crypto-exchange market-access impairment releases trapped liquidity and
partially reverses the BTC displacement that occurred while access was
impaired.

The provisional causal composition is:

1. official venue incident text establishes a real impairment to trading or
   market access;
2. official incident metadata establishes that the impairment is resolved;
3. deterministic code measures BTC displacement only from bars available by
   recovery;
4. the fixed side opposes that completed displacement; and
5. entry waits for a complete latency bar after causal readiness.

The semantic model may decide only whether an official incident/update chain
describes an eligible impairment and completed recovery. It may not inspect
price, choose side, choose a threshold, choose hold, predict a return, or
generate a trade.

The `6` suffix reserves a provisional six-hour consequence horizon. This file
does not freeze the final incident parser, ontology, synthetic corpus, model,
support floors, displacement threshold, latency implementation, controls,
novelty cohort, or economic evaluator. Those must be committed before exact
historical incident incidence or any candidate BTC displacement is computed.

## Why this follows EBOC

EBOC-72 was retired on its untouched synthetic gate before any historical SEC
body or market outcome:

- calibration selected step 32 under the frozen rule;
- held-out online, offline, and unsupported classes were each 100%;
- held-out mixed composition was 46/48 rather than the required 48/48;
- 63/64 swap pairs were both exact;
- post-training selected-checkpoint inference used 11.69 GiB allocated and
  11.78 GiB reserved, above the frozen 7.00/7.25 GiB ceilings; and
- historical SEC bodies, BTC rows, funding, returns, and 2024-or-later rows
  remained zero.

VARR does not repair EBOC. It changes:

- the source from SEC filings to official exchange incident ledgers;
- the semantic object from issuer mining capacity to user market access;
- the clock from filing acceptance to incident recovery;
- the side from semantic direction to deterministic post-impairment reversal;
  and
- the horizon from 72 hours to a provisional six hours.

No EBOC prompt, mixed exception, checkpoint, test output, threshold, breadth
state, side, or hold may enter VARR.

## Official source axis

Only the official public Statuspage surfaces of these two venues are eligible:

### Coinbase Exchange

```text
https://status.exchange.coinbase.com/history?page={page}
https://status.exchange.coinbase.com/history.atom
https://status.exchange.coinbase.com/api/v2/incidents.json
https://status.exchange.coinbase.com/api/v2/components.json
```

### Kraken

```text
https://status.kraken.com/history?page={page}
https://status.kraken.com/history.atom
https://status.kraken.com/api/v2/incidents.json
https://status.kraken.com/api/v2/components.json
```

Official source/documentation pages:

- <https://status.coinbase.com/api>
- <https://status.exchange.coinbase.com/history>
- <https://status.kraken.com/api>
- <https://status.kraken.com/history>
- <https://support.kraken.com/hc/articles/206548387-where-can-i-find-documentation-for-the-api->

The JSON incident endpoints officially expose only the 50 most recent
incidents. They are a live/recent consistency surface, not a substitute for
the paginated historical pages.

The historical pages are server-rendered Statuspage documents containing a
`HistoryIndex` React payload. Header-only feasibility probes returned HTTP 200
for page 40 on both venues. A minimal Coinbase page-40 schema probe confirmed
that the payload exposes old month metadata, without enumerating incident
rows, labels, components, timestamps, or counts. No historical incident
incidence has been computed.

No mirror, search-engine cache, social-media repost, third-party outage
aggregator, or manually reconstructed incident may enter the primary source.

## Source immutability and live parity risk

Statuspage history is public but not an append-only cryptographic ledger.
Incident names, bodies, component links, and timestamps can be revised. The
next mechanism must therefore freeze:

- raw response hashing before parsing;
- exact page order and termination;
- page identity and cross-page duplicate handling;
- update `created_at`, `display_at`, and `updated_at` semantics;
- incident `created_at`, `resolved_at`, and `updated_at` semantics;
- conservative historical readiness after every relevant recorded revision;
- rejection of timestamp regression, missing identity, conflicting duplicate,
  reopened incident, or source drift;
- durable forward capture of Atom/API updates before semantic inference; and
- a rule preventing a backfilled historical timestamp from pretending to be
  a contemporaneously captured live timestamp.

Historical and live inference must share the same semantic ontology and
composer. Live readiness is the later of the frozen historical readiness floor
and durable receipt, hash, parse, redaction, inference, and manifest-commit
time.

Raw Statuspage responses remain ignored and local. Committed artifacts may
contain URLs, hashes, schemas, aggregate support counts, redacted bounded
windows, and causal clocks, but not a redistributed raw historical corpus.

## New semantic object

VARR concerns **material user ability to access and execute in the crypto
market**, not every venue status update.

The later ontology must distinguish at least:

### Eligible unscheduled impairment

Potentially eligible facts include an unscheduled, realized degradation or
outage affecting:

- spot order entry, cancellation, matching, or market availability;
- exchange REST, WebSocket, or FIX execution paths;
- login, web, mobile, or authentication when the text states that trading is
  unavailable or materially impaired;
- broad fiat or crypto deposit/withdrawal access when the affected scope
  includes BTC or the venue generally; or
- several market-access components whose combined text explicitly removes
  practical trading access.

### Completed recovery

Recovery must be explicit in the official update chain: resolved, restored,
fully operational, or an equivalent completed return of the same impaired
capability. Monitoring, identified, partial mitigation, or a promise of a fix
is not completion.

### Mandatory abstention

Mandatory exclusions include:

- scheduled maintenance announced before impairment;
- planned market opening, auction, limit-only launch, listing, delisting, or
  migration;
- one altcoin/network send, receive, staking, card, NFT, tax, rewards, support,
  or informational component with BTC trading unaffected;
- latency or delayed data with order execution explicitly unaffected;
- third-party chain outage without material venue market access loss;
- generic risk language, test incidents, prompt injection, or quoted
  instructions;
- an unresolved, reopened, contradictory, or timestamp-invalid incident; and
- a resolution that does not restore the capability whose impairment made the
  incident eligible.

The model may use one bounded incident/update chain. Venue identity, incident
ID, component ID, dates, times, durations, asset symbols, quantities, links,
and impact labels must be deterministically redacted before inference.

## Provisional deterministic composer

The next mechanism must freeze one exact composer before incident incidence.
Its intended shape is:

- impairment start from the first causally public eligible impairment update;
- recovery from the first causally public completed restoration of that same
  capability;
- conservative readiness after the last source revision required to establish
  both facts;
- BTC displacement from one already audited five-minute market source using
  only completed bars in the half-open impairment interval;
- a precommitted volatility normalization and minimum-duration rule;
- `LONG` after a sufficiently negative completed displacement;
- `SHORT` after a sufficiently positive completed displacement;
- no trade for an insufficient or zero displacement;
- entry only after one complete five-minute latency bar;
- fixed six-hour hold, fixed exposure, exact funding and costs; and
- global non-overlap before split containment.

This side rule does not claim that outages forecast price. It tests whether
restored market access creates a short-lived normalization after a move that
already occurred while access was impaired.

Market volatility can cause exchange outages. That reverse-causality risk is
material and must be tested against volatility-matched no-outage controls,
duration-only controls, scheduled-maintenance controls, delayed entries, and
deterministic side flip/random side. It may not be explained away after
outcomes.

## Split and outcome boundary

Reserved source periods:

| Role | Interval |
|---|---|
| source warmup | before `2019-01-01T00:00:00Z` |
| train | `[2019-01-01, 2023-01-01)` |
| selection | `[2023-01-01, 2024-01-01)` |
| sealed extension | `2024-01-01` and later |

Split membership must use the final causal entry timestamp, not incident
creation date or display month. A source update that becomes ready after a
split boundary belongs to the later split.

The stage sequence is absolute:

1. freeze and pass the synthetic semantic gate without historical incident
   text;
2. parse historical incident text and pass a **source/semantic support gate**
   using only eligible/recovered incident counts, both venues, active months,
   maximum gaps, venue/month concentration, durations, and source integrity;
3. only that unchanged pass may authorize reading BTC bars completed by each
   recovery readiness time and computing causal displacement, fixed side, and
   source-clock side balance;
4. only the unchanged causal clock may authorize comparator timestamp parsing
   and novelty;
5. only support and novelty passes may authorize post-entry BTC bars, funding,
   future returns, PnL, absolute return, CAGR, strict MDD, bootstrap, or any
   economic field; and
6. `2024-01-01` and later remain sealed through all of these stages.

Before step 2 passes, BTC market rows read, displacement values, sides,
funding rows, return rows, and every market-derived field must all remain
exactly zero. Before step 4 passes, post-entry/future market and economic rows
must remain exactly zero. No support threshold may depend on a BTC
displacement or side that the support stage is forbidden to open.

This boundary unit opened:

- official source/documentation pages and current search snippets;
- HTTP status/content headers for the two historical page-40, Atom, and recent
  JSON endpoints;
- one old Coinbase `HistoryIndex` payload's page/month schema only; and
- prior repository documents needed to establish novelty and the earlier
  shallow-archive assumption.

It did not:

- enumerate historical incident rows or IDs;
- parse any historical incident name, update body, component, status, impact,
  impairment, recovery, timestamp, duration, venue count, or event count;
- calculate VARR support, source gaps, side, or clock incidence;
- read any candidate BTC bar, displacement, funding, future return, PnL,
  absolute return, CAGR, MDD, hit rate, or reward; or
- open any 2024-or-later VARR source or outcome.

## LLM and RLLM boundary

VARR uses language reasoning only where deterministic string rules are
insufficient:

```text
eligible market-access impairment
completed recovery of the same capability
unsupported / contradictory
grounded update IDs
```

Deterministic code owns source membership, revision readiness, bounded-window
construction, redaction, displacement, volatility scaling, side, entry, hold,
exposure, non-overlap, costs, funding, returns, CAGR, strict MDD, bootstrap,
and every gate.

Any synthetic semantic adaptation must be independently preregistered before
historical incident text is parsed. Public incident text may exist in model
pretraining; zero memorization may not be claimed.

Only after the unchanged deterministic VARR clock passes source support,
novelty, and train/selection economics may a train-only RLLM receive causal
semantic relation tokens, pre-entry displacement state, and current position
state to choose:

```text
TRADE_FIXED_SIDE
ABSTAIN
```

It may not create a clock, change side, change hold, or use selection/eval
rewards for prompt, adapter, checkpoint, or action-threshold selection.

## Required next commit

Before any historical incident incidence is opened, commit one exact mechanism
that freezes:

1. official endpoint identities, page crawl/termination, raw hashing, parser,
   and duplicate/revision rules;
2. bounded incident/update text, redaction, semantic ontology, prompt, output,
   evidence grounding, and synthetic gate;
3. impairment start, completed recovery, reopen/conflict, and readiness clocks;
4. market source, completed-bar projection, displacement, volatility
   normalization, duration, threshold, side, entry, hold, exposure, costs,
   funding, and non-overlap;
5. train/selection **source-only** support floors for accepted incidents, both
   venues, active months, maximum gaps, venue/month concentration, and
   incident duration, followed by a separately authorized causal-market clock
   gate for both sides and displacement-qualified incidence;
6. scheduled-maintenance, lexicon-only, status-only, exchange-only,
   duration-only, volatility-without-outage, delayed-entry, side-flip, and
   random-side controls;
7. novelty comparisons against existing semantic/news, microstructure,
   cross-venue, live-portfolio, funding/premium, and outage-adjacent clocks;
8. fail-flat live behavior for missing, revised, delayed, malformed, overlong,
   or contradictory source; and
9. immutable retirement on source, semantic, support, novelty, or economic
   failure.
