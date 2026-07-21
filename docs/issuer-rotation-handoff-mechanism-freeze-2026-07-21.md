# IRH-36 issuer-rotation handoff — outcome-blind mechanism freeze

## Decision and sealed boundary

Freeze one new candidate, **IRH-36**, before opening its real source incidence
or any BTC outcome.

IRH-36 asks whether a large, causally finalized rotation between Ethereum USDC
and USDT marks a change in offshore crypto buying-power composition. It is an
event-pair mechanism. It is not a cleaner rerun of the rejected daily
stablecoin-supply breadth rule.

This freeze reads no complete Ethereum event panel, accepted-event count,
BTCUSDT price, funding, future return, label, PnL, absolute return, CAGR, or
MDD. The first implementation may open only the checksum-bound 2020–2023
Ethereum event source and already frozen comparator timestamps. If any source,
support, dispersion, balance, or novelty gate fails, IRH-36 is retired without
changing its tail, window, ratio, direction, hold, or comparator set.

## Why this is a distinct hypothesis

The rejected supply-breadth candidate aggregated several issuers to a daily
count and conditioned expansion/contraction on a completed BTC move. IRH-36:

- uses no BTC state to create a signal;
- uses primary-market contract events rather than daily supply snapshots;
- requires an opposite-signed cross-issuer pair rather than aggregate supply;
- tests issuer substitution, not market-wide stablecoin expansion; and
- becomes knowable only at each event's canonical `available_at` time.

Direct finalized logs repair the old source-vintage defect, but they do not
repair the old economic rule. Failing novelty against the old breadth clock or
the frozen SQFD/SDDR/UCBR clocks rejects IRH-36.

## Frozen source rows

The only eligible rows are:

| Asset | Positive leg | Negative leg |
|---|---|---|
| `usdt_eth` | `issue` | `redeem` |
| `usdc_eth` | `mint` | `burn` |

`destroyed_black_funds` is not customer redemption. It is excluded from pair
construction and acts as a trailing 24-hour veto. `deprecate` remains a source
handoff failure unless a successor contract was separately reviewed before
the event panel was opened.

For every eligible row:

```text
amount_usd = integer(amount_raw) / 1_000_000
event_time = available_at
identity   = (block_hash, transaction_hash, log_index)
```

Events are ordered by `(available_at, block_number, transaction_index,
log_index, identity)`. Event-block timestamps may be retained for audit but
may not drive the signal clock.

## Strictly-prior large-event state

Each of the four `(asset, event)` types has an independent rolling history.
For event `e`, its reference sample contains only same-type rows satisfying:

```text
e_prior.available_at < e.available_at
e.available_at - 365 calendar days <= e_prior.available_at
```

At least 32 prior rows are required. Sort their `amount_usd` values ascending.
For `n` values, the frozen nearest-rank 90th percentile is element
`ceil(0.90 * n) - 1` under zero-based indexing. The current event is large
when its amount is greater than or equal to that threshold. No current-row,
future-row, full-sample, linear-interpolation, or post-2023 quantile may enter
the threshold.

## Frozen pair construction

Process large events in causal order. An event may belong to at most one pair.
When a large event becomes available, look backward 24 hours for unmatched
large events of the required counterpart type:

```text
LONG  = usdt_eth:issue  paired with usdc_eth:burn
SHORT = usdt_eth:redeem paired with usdc_eth:mint
```

The two legs may arrive in either order. If several counterparts are eligible,
choose the one with the latest earlier `available_at`; break ties by canonical
identity. The pair is valid only when:

1. the later minus earlier `available_at` is no more than 24 hours;
2. `min(amount_usd) / max(amount_usd) >= 0.25`; and
3. no `usdt_eth:destroyed_black_funds` row has `available_at` in the closed
   trailing 24-hour interval ending at pair completion.

Pair completion is `max(leg_1.available_at, leg_2.available_at)`. Mark both
legs used only after all three checks pass. A failed candidate does not consume
either leg.

## Frozen execution clock

For pair completion `t`:

```text
latency_bar_start = first UTC five-minute boundary at or after t
entry_time        = latency_bar_start + 5 minutes
scheduled_exit    = entry_time + 36 hours
```

If `t` is exactly on a five-minute boundary, the full bar beginning at `t`
must still elapse. Thus entry is never earlier than five minutes after causal
availability.

Candidate pairs are considered by `(entry_time, pair identities)`. Global
non-overlap is mandatory: accept a pair only when its entry is at or after the
previous accepted scheduled exit. This reservation rule, side, and 36-hour
hold are frozen before any execution price is read.

## Frozen source-only support gates

The support evaluator must fail closed unless all gates pass:

1. **Source integrity**
   - Ethereum mainnet chain ID 1;
   - identical canonical log hash from two independent archive transports;
   - event block hashes match an independently materialized header source;
   - every event is delayed through block `N+64` and covered by finalized head;
   - source manifest reports zero BTC, funding, return, and post-2023 event
     reads.
2. **Incidence**: at least 48 accepted, globally non-overlapping pairs over
   2020–2023.
3. **Year dispersion**: at least 8 accepted entries in each of 2021, 2022, and
   2023. The 2020 warmup cannot substitute for a missing full year.
4. **Calendar concentration**: no UTC entry month exceeds 25% and no UTC entry
   quarter exceeds 45% of accepted pairs.
5. **Side balance**: LONG and SHORT each represent at least 30% of accepted
   pairs.
6. **Tail validity**: all four event types reach the 32-row strictly-prior
   history requirement, and no accepted pair uses a threshold derived from
   fewer rows.
7. **Novelty against secondary-market stablecoin clocks**: against each frozen
   SQFD-6 (`primary`, `no_usdt_lag`, `no_participation`), SDDR-12 `primary`,
   and UCBR-12 `primary` entry clock, exact-entry Jaccard must be at most 0.05
   and maximum bidirectional containment within plus/minus six hours must be
   at most 0.25.
8. **Novelty against rejected supply breadth**: on dates where both clocks are
   defined, exact UTC entry-date Jaccard must be at most 0.20 and maximum
   bidirectional containment within plus/minus one calendar day must be at
   most 0.40.

Comparator clocks must be checksum-bound before overlap is calculated. A
missing comparator, schema drift, duplicate timestamp, side drift, or outcome
field fails closed. Low exact overlap cannot override failed proximity.

## Frozen source-only falsification controls

Controls are diagnostics and cannot replace the primary policy after support
is opened:

- `no_amount_ratio`: remove only the 0.25 amount-balance requirement;
- `no_tail`: keep pairing and veto rules but treat every eligible event as
  large after the 32-row warmup;
- `same_sign_placebo`: pair USDT issue with USDC mint and USDT redeem with USDC
  burn, keeping the primary scheduler and assigning the corresponding sign;
- `stale_6h`: delay each completed primary pair by exactly six hours before
  applying the same five-minute latency and reservation rules;
- `blackfunds_no_veto`: remove only the confiscation veto.

No control incidence may be used to loosen or select a replacement rule.

## Later outcome contract, not yet authorized

Only a full support and novelty pass may authorize a separate, committed,
hash-frozen outcome evaluator. That evaluator must use exact next-open
BTCUSDT USD-M perpetual execution, realized funding, full-calendar CAGR,
strict intratrade position-path MDD, 6 bp base and 10 bp stress cost per
notional side, and sequential train/test/eval/holdout opening. It must report
absolute return together with CAGR, strict MDD, CAGR/MDD, trade count, and
LONG/SHORT count.

Gemma/RLLM is prohibited from creating, retiming, or reversing this clock. It
may be evaluated later, using train-only data, as an abstention or risk-routing
layer only after the deterministic IRH-36 clock demonstrates gross edge above
costs. Failure of deterministic economics retires IRH-36 rather than handing
its direction to an LLM.

## Bound references

- `docs/ethereum-stablecoin-issuance-redemption-source-feasibility-2026-07-21.md`
- `docs/stablecoin-supply-breadth-absorption-frozen-oos-2026-07-16.md`
- `docs/stablecoin-quote-flow-diffusion-support-freeze-2026-07-19.md`
- `docs/stablecoin-denominator-dislocation-support-result-2026-07-20.md`
- `docs/usdt-collateral-breadth-relay-support-rejection-2026-07-20.md`
- `training/build_ethereum_stablecoin_issuance_redemption.py`

