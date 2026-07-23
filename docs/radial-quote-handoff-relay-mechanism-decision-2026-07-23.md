# RQHR-72 mechanism decision — 2026-07-23

## Decision

Preregister one new source-seen, event-incidence-blind singleton:
**RQHR-72 — Radial Quote Handoff Relay**, with a fixed 72 five-minute-bar
(six-hour) hold.

RQHR tests whether directional movement in cumulative depth-weighted average
quotes begins at the 2–3% book radii and is then causally confirmed at the
4–5% radii. The economic claim is a liquidity-repricing handoff:

- positive average-quote skew movement first at the near radii and then at the
  far radii means ask-side liquidity is retreating outward relative to bid-side
  liquidity, so RQHR enters **LONG**;
- the exact negative sequence means bid-side liquidity is retreating outward,
  so RQHR enters **SHORT**.

RQHR is an explicit successor inside the already-tested RNCM average-quote
source family. It was selected only after RNCM failed its frozen source-support
gate, so it is not a pristine or evidence-independent discovery and its result
must not be interpreted as the first test of this source geometry. The new
candidate identity is required because RQHR changes the measured statistic and
causal sequence rather than amending RNCM in place: RNCM used 30-minute changes
in five-minute median radial skew, removed a fitted quote-center response, and
required all four radii to move simultaneously; RQHR uses previously unopened
within-bar `net`, `path`, and `efficiency` fields and a time-ordered near-to-far
relay. It does not reuse an RNCM event, residual, fitted coefficient, or
selected support quantile.

This document freezes source fields, arithmetic, strict-prior thresholds,
arming, confirmation, cancellation, direction, execution, controls, support,
novelty, economic sequence, and stopping rules before any RQHR feature, arm,
terminal, event incidence, comparator overlap, or market outcome is computed.

## Evidence and contamination boundary

The 2023 source panel was built and audited before RNCM. RNCM subsequently read
the panel's quote-center and `skew_2..5_median` columns and opened its own
median-migration source incidence. The viewed RNCM support counts are explicit
contamination: its accepted non-overlapping counts at frozen quantiles 0.995,
0.990, 0.985, and 0.975 were respectively 5, 16, 31, and 39; at 0.975 they
split 17 in H1 and 22 in H2 and failed every quarterly minimum. The exact RQHR
input columns below were created by the source builder but were not loaded by
RNCM's frozen `load_source()` projection. RQHR feature values, arms,
confirmations, events, and incidence remain unopened.

No RNCM, OFR, or other successor candidate in this branch opened a BTC return
for this average-quote source. Nevertheless, the source family and its general
geometry are not pristine. RQHR claims only a new frozen candidate sequence,
not globally pristine discovery.

Frozen history bindings:

- source-axis decision:
  `docs/btc-alpha-source-axis-decision-2026-07-20.md`, SHA-256
  `9a46c534c932efc4c38fa0a0ad168e40f33803527de12b80ac7a49f550c7dadd`;
- source audit:
  `docs/rncm-2023-source-build-audit-2026-07-20.md`, SHA-256
  `bf19382d550bfa1c4bcb6dfec080f4f0a57c64cd256c920dce8dc8158aee4ddb`;
- source builder:
  `training/build_binance_um_book_centroid_2023.py`, SHA-256
  `6021a1ee140500350e8b6bc0e8dae5ca32a84db39039c21d809ca798909a5c24`;
- RNCM preregistration source:
  `training/preregister_residual_notional_centroid_migration.py`, SHA-256
  `733ef4c3aaa823f19c8fe9303d3405def0c86f593c35bb2556a69edc3f67ad6f`;
- RNCM support result:
  `results/residual_notional_centroid_migration_support_2026-07-20.json`,
  SHA-256
  `887c532eb3163cfac47eb9fc2956326f02491b2890e4c0231e084807978577dc`;
- RNCM rejection decision:
  `docs/residual-notional-centroid-migration-support-rejection-2026-07-20.md`,
  SHA-256
  `34764817293d8914d4e4aa3d12d26d998abbfba23f3506d5c26dcbbd85e9c343`.

## Frozen source and causal availability

Source artifacts:

- panel:
  `data/binance_um_book_centroid_btcusdt_2023/BTCUSDT_um_book_centroid_skew_5m_2023.csv.gz`,
  SHA-256
  `c4053ce27d28bebda4137349192b1a940360231469f63edc32bacabb2ce54131`;
- manifest:
  `results/binance_um_book_centroid_btcusdt_2023_manifest.json`, SHA-256
  `d8237c4562d33c12eff162776f723cc5fc94649b69d26a6230e16fc38c52bba1`.

RQHR may read exactly:

- `date`;
- `skew_2_net`, `skew_2_path`, `skew_2_efficiency`;
- `skew_3_net`, `skew_3_path`, `skew_3_efficiency`;
- `skew_4_net`, `skew_4_path`, `skew_4_efficiency`;
- `skew_5_net`, `skew_5_path`, `skew_5_efficiency`;
- `source_complete` and `source_available_at`.

It may not read `center_quote_median`, any `skew_*_median`, external OHLC,
trades, funding, OI, labels, returns, PnL, or post-2023 rows during source
support.

The panel must be the exact 105,120-row 2023 five-minute UTC grid. A complete
row is available only at `date + 5 elapsed minutes`. An incomplete row has no
usable RQHR value, breaks race continuity, and cannot be interpolated,
forward-filled, backward-filled, or skipped inside a race.

For each radius, finite source decimals must satisfy the source-builder algebra
within absolute tolerance `5e-12`:

```text
path >= abs(net)
path == 0  => net == 0 and efficiency == 0
path > 0   => efficiency == abs(net) / path
0 <= efficiency <= 1
```

No clipping, winsorization, fitted transform, price residualization, or source
repair is allowed.

The historical Binance archive is published after the trading day. Bar-close
research availability is conditional on reconstructing equivalent percentage
bands from a live REST snapshot plus correctly sequenced WebSocket update IDs.
Exact historical/live feature parity is a separate production admission gate;
passing research outcomes alone cannot authorize live trading.

## Frozen exact bar features

At each complete bar `t`, let `n_k(t)`, `p_k(t)`, and `e_k(t)` be the source
`net`, `path`, and `efficiency` at radius `k`.

```text
near_sign(t) = common nonzero sign of n_2(t), n_3(t), else 0
far_sign(t)  = common nonzero sign of n_4(t), n_5(t), else 0

near_intensity(t) = (abs(n_2(t)) + abs(n_3(t))) / 2
far_intensity(t)  = (abs(n_4(t)) + abs(n_5(t))) / 2

near_efficiency(t) = min(e_2(t), e_3(t))
far_efficiency(t)  = min(e_4(t), e_5(t))
```

Decimal text, sums, comparisons, and nearest-rank order statistics use exact
decimal/rational arithmetic after the source-algebra tolerance check. Binary
floating point may be used only to replay the already-frozen source-builder
synthetic books, never to rank the real panel.

## Frozen strict-prior thresholds

At each grid row, use only valid feature values from the previous 8,640 grid
rows, excluding current `t`. Activation requires at least 4,032 valid prior
values. Missing rows remain in the calendar window but contribute no value.

Use deterministic nearest-rank order statistics, without interpolation:

```text
near_threshold(t) = rank ceil(0.975 * N)-1 of prior near_intensity
far_threshold(t)  = rank ceil(0.90 * N)-1 of prior far_intensity
```

No alternate window, expanding fallback, fitted transform, or threshold sweep
is permitted.

## Frozen near-to-far race

A near arm occurs only when all are true at complete bar `t`:

1. `near_sign(t)` is `+1` or `-1`;
2. `near_efficiency(t) >= 3/5`;
3. `near_intensity(t) >= near_threshold(t)`;
4. the immediately previous grid row was complete and its near intensity was
   strictly below its own available strict-prior threshold;
5. the far radii are not already confirmed, where confirmed means
   `far_sign(t) == near_sign(t)`, `far_efficiency(t) >= 1/2`, and
   `far_intensity(t) >= far_threshold(t)`; and
6. no race is active.

The arm stores its sign and starts a six-bar elapsed-grid race. It is never
queued or replaced by another arm.

For bars `t+1` through `t+6`:

- **confirmation** occurs when the far radii are confirmed in the stored arm
  sign and the exact cumulative sum of `n_2 + n_3` from the arm bar through the
  current bar remains strictly in the arm sign;
- **cancellation** occurs when the current near radii share the opposite sign
  with `near_efficiency >= 1/2`;
- simultaneous confirmation and cancellation is ambiguous, produces no event,
  and immediately retires the active race;
- an incomplete grid row cancels immediately;
- no terminal by `t+6` is a timeout and produces no event.

A confirmation, cancellation, ambiguity, or timeout bar is terminal-consumed
and cannot arm a new race. Rearming can begin only on the next grid row.

Only confirmation creates an RQHR candidate at the confirming bar:

- positive stored sign: **LONG**;
- negative stored sign: **SHORT**.

Persistence, cancellation, ambiguity, and timeout never trade. Confirmation
age must be an integer from one through six.

## Frozen execution

- signal: confirming row's `source_available_at`;
- entry: signal plus one full five-minute processing-latency bar (not the open
  coincident with source availability);
- exit: entry plus exactly 72 five-minute bars / six elapsed hours;
- notional exposure: 0.5x;
- one chronological reservation on `[entry, exit)`;
- accept only when entry is at or after the previous accepted exit;
- signal, entry, and exit must remain in the same known UTC calendar quarter;
- reservation resets only at a known quarter boundary;
- suppressed candidates are not queued;
- no stop, take-profit, trailing exit, dynamic size, price gate, regime gate,
  direction override, or hold search.

## Frozen source-support gates

Before comparator rows or market outcomes are read, the accepted 2023 clock
must satisfy all of:

- at least 120 events total;
- at least 45 events in each half-year;
- at least 20 events in every quarter;
- LONG and SHORT each at least 35% of events;
- no quarter above 40% of events;
- no UTC month above 15% of events;
- maximum accepted-entry gap at most 21 elapsed days;
- confirmation age at least two bars for at least 20% of all events and at
  least 10% of each half-year's events;
- every event has valid strict-prior thresholds, exact timing, unique entry,
  quarter containment, and global non-overlap;
- zero post-2023 source rows;
- every frozen synthetic fixed-book/moving-band null produces zero raw
  confirmations and zero accepted RQHR events.

Failure of any gate retires RQHR unchanged before novelty and outcomes.

## Frozen source controls

Every control uses the same exact source values, strict-prior radius-specific
thresholds, source availability, processing latency, chronological scheduler,
six-hour hold, and quarter containment unless its stated mechanism differs.
Immediate controls have no race state and independently schedule all qualifying
candidates. Relay controls have independent race state:

1. `simultaneous_near_far`: require primary arm conditions 1–4 and same-sign
   far qualification on that same bar with `far_efficiency >= 1/2` and
   `far_intensity >= far_threshold`; condition 5 is replaced by this
   requirement, and the candidate side is the shared sign;
2. `far_to_near_reverse_relay`: arm on common nonzero far sign with
   `far_efficiency >= 3/5` and `far_intensity >= far_threshold`, only when the
   immediately previous grid row was complete and its far intensity was
   strictly below its own available far threshold, provided the near radii are
   not already qualified in that sign at `near_efficiency >= 1/2` and
   `near_intensity >= near_threshold`. Within six bars, confirm on that near
   qualification while the cumulative `n_4+n_5` remains strictly in the arm
   sign; cancel on opposite common far sign at `far_efficiency >= 1/2`.
   Incomplete, ambiguous, timeout, and terminal-consumed behavior is identical
   to primary;
3. `no_efficiency_relay`: use the primary race after deleting every efficiency
   predicate. Thus arm and confirmation retain signs, crossings, thresholds,
   and cumulative persistence, while opposite common near sign alone cancels;
4. `near_only`: immediately trade every primary near qualification satisfying
   arm conditions 1–4, without primary condition 5 or a race;
5. `far_only`: immediately trade a common nonzero far sign when
   `far_efficiency >= 1/2`, `far_intensity >= far_threshold`, and the
   immediately previous grid row was complete and its far intensity was
   strictly below its own available far threshold;
6. `one_bar_stale`: take the already-built primary confirmation clock, shift
   each signal by exactly one elapsed grid row, preserve its side, require the
   destination row to be complete, use that destination row's
   `source_available_at`, and then rerun latency, quarter containment, and
   non-overlap; a signal shifted outside 2023 or onto an incomplete row is
   dropped rather than searched forward;
7. `five_bar_stale`: apply the same final-signal operation with exactly five
   elapsed grid rows;
8. `quarter_far_triple_permutation`: within each UTC quarter, collect every
   complete row's full algebra-consistent far tuple
   `(n_4,p_4,e_4,n_5,p_5,e_5)`, sort donor tuples by
   `SHA256("RQHR-72|quarter_far_triple_permutation|<quarter>|<donor-date>")`,
   zip them to complete recipient rows in chronological order, leave incomplete
   rows incomplete, preserve recipient availability, and recompute every far
   feature and strict-prior far threshold from the permuted tuples;
9. `deterministic_random_side`, `exact_direction_flip`, `constant_long`, and
   `constant_short` reuse exact accepted primary entries and exits.
   `deterministic_random_side` is LONG when the first byte of
   `SHA256("RQHR-72|deterministic_random_side|<entry-UTC-ISO>")` is even and
   SHORT when odd; the other three respectively negate primary side, set every
   side LONG, or set every side SHORT.

No control may replace the primary. The primary must later beat every listed
control on frozen economics.

## Frozen mechanical nulls

Before reading any real RQHR column, replay the exact fixed-absolute-book
equations in `training/preregister_residual_notional_centroid_migration.py`,
SHA-256
`733ef4c3aaa823f19c8fe9303d3405def0c86f593c35bb2556a69edc3f67ad6f`:
`synthetic_fixed_book_panel()` for `smooth_symmetric`,
`tick_rounded_anchor`, `stepped_asymmetric`, and `missing_rows`, plus
`synthetic_discrete_ladder_panel()` for `discrete_asymmetric_ladder`. The
hash-bound anchor sine periods, tick rounding, best quotes, percentage-band
bounds, fixed ladder prices/quantities, search sides, cumulative depth, and
notional-average equations are normative; no parameter may be re-estimated.

Generate exactly 105,120 grid bars and ten scheduled snapshot slots per bar.
Materialize exactly ten snapshots in every non-suppressed bar. For bar
`b in {0..105119}` and snapshot slot `j in {0..9}`, set timestamp to
`2023-01-01T00:00:00Z + b*5 minutes + j*30 seconds` and evaluate every
hash-bound position-dependent equation at fractional position `b + j/10`.
Periods expressed in five-minute bars remain unchanged. In the missing-row
scenario, evaluate the frozen missing predicate on integer bar `b`, namely
`b % 1009 < 3`, suppress all ten snapshot slots for that bar, retain no usable
RQHR value, and break race continuity. At each retained snapshot, derive radii
2–5 skew exactly as the bound source builder defines:

```text
skew_k = log(ask_average_k / ask_average_1)
       - log(bid_average_1 / bid_average_k)
```

For each radius, aggregate the ten snapshot skews in timestamp order exactly as
the source builder:

```text
net = last - first
path = sum(abs(current - previous))
efficiency = 0 when path == 0 else abs(net) / path
```

Then run the full RQHR feature, race, and scheduler. Binary floating point is
authorized only for replaying these already-frozen synthetic equations and
their source-builder aggregation. No real source row or real RQHR feature may
be read until all five synthetic scenarios pass.

Every scenario must produce exactly zero raw confirmations and exactly zero
accepted events. Any raw confirmation or accepted event rejects RQHR before
the real panel is read. The null parameters may not be changed after a count is
observed.

## Frozen novelty gate

Only a complete source-support and mechanical-null pass may open these exact
outcome-blind 2023 comparator clocks:

| Group | Artifact | SHA-256 | Expected raw rows | Canonical clock SHA-256 | Parser |
|---|---|---|---:|---|---|
| `ccbvfr:primary` | `results/cross_collateral_book_validated_flow_rejection_event_clock_2026-07-18.json` | `79b4838ae634efcff705e028a0ddff8b75d28d79180e3ac89f54b9cab7e5005f` | 144 | `d2cdcad8f57867722c220e32029d0ccbf1f1aa511e5ae590cf43411a588af4bd` | every embedded event's positions, displayed dates, interval, and side |
| `pdf10:primary` | `results/cross_collateral_liquidity_credibility_fracture_event_clock_2026-07-14.json` | `ab8209308619b97880277b95fcc1a2f825b050a603e24b3e2125ddd5bfb226f8` | 591 | `ce1c6ec42434874d97c6b6034f51a73771b27e314da6d37a4f44b0563e6972e2` | replay the frozen PDF-10 support clock, then validate every replayed row |
| `crrc:primary` | `results/cross_venue_radial_refill_compression_event_clock_2026-07-17.json` | `09d2ca954c5c4d06b981575c6b0f0e4dc6b49d8a693da418f3f26e5cc454c835` | 156 | `81e09e3d1d5592f12ce1994077efa279ebf1de4c29a6f5a144060d16ee6b2e9f` | every embedded event's positions, displayed dates, interval, and side |

PDF-10 replay additionally binds:

- `results/cross_collateral_liquidity_credibility_fracture_support_2026-07-14.json`,
  SHA-256
  `9a3001db640ec8041d885645d33f11dd6075276685eb22f8ae3c618363d3099a`;
- `training/preregister_cross_collateral_liquidity_credibility_fracture.py`,
  SHA-256
  `8947050c990b5638f6d8b2e952f252289ddef6c92f85fb13f75001fe721e6e28`.

The other canonical-hash producers are also bound:

- CCBVFR:
  `training/preregister_cross_collateral_book_validated_flow_rejection.py`,
  SHA-256
  `004fa71b1951eff58eca592863cf7ad09e0e36e4749a3e611ce299e1ac3d601f`;
- CRRC:
  `training/qualify_cross_venue_radial_refill_compression.py`, SHA-256
  `96372733a597ca486b52292480ceacde631056054b2d914aa9180024218fa0e7`.

Canonical hash replay is artifact-specific and exact: CCBVFR hashes the full
ordered embedded event dictionaries with sorted JSON keys and compact
separators; PDF-10 hashes ordered `signal_position` and numeric `side`
records from the replayed schedule under the same JSON serialization; CRRC
hashes ordered `signal_position`, `entry_position`, `exit_position`,
numeric `side`, and `hold_bars` records under that serialization.

Before common-window filtering, parse every raw row in each exact group and
validate the artifact protocol, outcome-closed flags, declared event count,
the applicable frozen canonical projection and clock hash, valid side,
positive interval, unique entry, chronological order, global group
non-overlap, quarter containment, and position/date/hold consistency. A parser
may not discard a row before these checks. The legacy JSON display timestamps
are UTC-naive text and are not trusted as timezone evidence. For every embedded
or replayed event, reconstruct each timezone-aware timestamp as
`2023-01-01T00:00:00Z + position * 5 elapsed minutes`, require the displayed
date text to equal that timestamp's UTC-naive rendering, and use only the
reconstructed aware timestamp. Any disagreement or malformed declaration fails
closed.

Bind and apply
`docs/novelty-comparator-common-window-policy-2026-07-23.md`, SHA-256
`928bce6e04fb34001478b4b4ea84156580b661c88a0f0338065a891c009bd580`,
over `[2023-01-01T00:00:00Z, 2024-01-01T00:00:00Z)`.

The report must record, for the candidate and each exact comparator group, raw
rows, fully contained rows, rows before, rows after, and rows crossing either
boundary. Every comparator group is required and must have at least ten fully
contained rows; counts from zero through nine fail closed. For each group:

- exact-entry Jaccard at most 0.10;
- one-to-one RQHR containment within ±12 five-minute bars at most 0.35;
- absolute signed occupied-exposure correlation at most 0.35.

Missing, hash-mismatched, malformed, overlapping, empty in-window, or undefined
required comparisons fail closed. No interval is clipped.

## Strict economic sequence

Only source-support and novelty pass may authorize a separately committed
strict evaluator:

1. train: calendar 2023;
2. immutable source extension and test: calendar 2024 only after train passes;
3. eval: calendar 2025 only after test passes;
4. recent: 2026 only after eval passes.

Each full calendar-year split requires at least 100 trades and at least 25
LONG and 25 SHORT. Train additionally remains subject to its 120-event source
floor. A partial recent-2026 split, whose exact end is hash-frozen before its
source rows are opened, requires at least 40 trades and at least ten per side.

Each split also requires signed net return after all costs and realized funding
strictly above zero, CAGR / strict intratrade MDD at least 3.0, strict MDD at
most 15%, realized funding, 6 bps notional cost per side, positive signed net
return under 10 bps stress, every contained quarter positive, and calendar-week
cluster sign-flip `p <= 0.10`. Train, test, and eval use full-calendar-year
CAGR. Recent uses annualized CAGR over its exact frozen partial interval.
Inactive time remains in both calculations. Primary must beat all frozen
mechanism controls by at least 0.25 in finite CAGR/MDD ratio.

The recent split is frozen now as exactly
`[2026-01-01T00:00:00Z, 2026-07-19T00:00:00Z)`; no later source or outcome row
may enter that verdict.

After eval passes and before portfolio promotion, RQHR must also have absolute
signed occupied-exposure correlation at most 0.35 against every frozen live
portfolio sleeve over their exact common OOS window and improve the frozen
portfolio's return/MDD frontier under unchanged sleeve weights or a separately
preregistered allocation. This portfolio gate cannot rescue a failed standalone
split.

## RLLM boundary

RLLM is unauthorized before deterministic source support, mechanical controls,
novelty, train, and test all pass. A later compact model may only choose
`TRADE_FIXED_SIDE` or `ABSTAIN` from causal bucketed RQHR arm/confirmation and
current position/risk context. It cannot create an event, reverse side, change
size or hold, consume future outcomes, or bypass a gate.

## Stopping rule

Any provenance, source algebra, synthetic-null, incidence, novelty, train, or
later sequential failure retires `RQHR-72` unchanged. A successor requires a
new mechanism, ID, and preregistration. No observed count or outcome may alter
97.5%/90%, 3/5, 1/2, six race bars, 72 hold bars, support floors, side, or
execution.
