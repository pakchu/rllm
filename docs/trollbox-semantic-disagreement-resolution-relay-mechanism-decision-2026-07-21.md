# TSDR-72 — Trollbox Semantic Disagreement-Resolution Relay mechanism decision

## Decision and evidence boundary

The next singleton BTC candidate is **TSDR-72**, a six-hour semantic
disagreement-resolution relay with a fixed six-hour hold.

TSDR uses the already committed, outcome-blind BitMEX Trollbox semantic clock.
It first observes a high-attention window in which at least two independently
sampled participants are bullish and at least two are bearish, with neither
side reaching the frozen two-to-one majority. It then waits for the first later
high-attention window that reaches a two-to-one directional majority. A timely
resolution emits one trade in the new majority's direction.

This decision was made after the repository had already recorded the failure
of TBASR-24. It is therefore not presented as a pristine clean-room discovery.
The design step opened only committed aggregate semantic counts and timestamps;
it did not load a BTC bar, funding mark, return, PnL, equity path, or a 2023-or-
later Trollbox message. The known TBASR performance may not be used to tune,
invert, or repair TSDR after this document.

The source-only viability audit available at decision time showed:

- 5,417 attention windows from 2020-07-01 through 2022-12-31;
- 455 `UNCLEAR` windows with at least two bullish and two bearish participants;
- no market or funding rows loaded by the semantic support artifact; and
- no private message text committed to the repository.

Exact TSDR relay incidence, support gates, comparator overlap, and every market
outcome remain delegated to later, separately committed stages.

## Why this is not a TBASR parameter repair

TBASR-24 asked whether a clear crowd majority that aligned with a completed
one-hour BTC move should be faded for two hours. It failed its frozen train
gate and remains retired. TSDR changes the economic object rather than a
threshold or sign inside that object:

- no BTC displacement or other price-derived setup is allowed;
- a single clear crowd window can never start TSDR;
- the causal origin is a prior, strongly two-sided disagreement window;
- the signal is the first later majority resolution, not the initial crowd;
- the action follows information resolution rather than fading saturation;
- both the resolution deadline and hold are six hours, not TBASR's two hours;
- no TBASR losing trade, return, score, or favorable calendar is selectable;
- no opposite-side TSDR variant may replace a failed primary.

This distinction is falsifiable. If the disagreement path does not add value
over the same resolution windows, the mechanism fails even if a generic crowd
direction happens to make money.

## Frozen source binding

The primary source is exactly:

- semantic clock:
  `results/bitmex_trollbox_semantic_clock_2026-07-20.json`;
- semantic clock SHA-256:
  `af8687564614ec5a1cbd7a1438c908f687af7bd99ceede9539016e5c1b111bd4`;
- semantic clock manifest hash:
  `fdcd9c7c376b18df2799acf24af04a421ca679e27009e6a539888defc7438aa8`;
- semantic support:
  `results/bitmex_trollbox_semantic_support_2026-07-20.json`;
- semantic support SHA-256:
  `2b89f710d59a5c0708d400541defb43d5e292f6d9bdedbe66d6bdcf614d09e94`;
- semantic support result hash:
  `5996b7d7497d6bf5e96343f7ceca766363d58aa34280aea0fdb7b8653a8b1725`;
- attention clock:
  `results/bitmex_trollbox_attention_clock_2026-07-20.json`;
- attention clock SHA-256:
  `5b60016a3d612f8cd29ea4548241daea76b6a6b60759837ab7bfcd60b8727f73`;
- semantic builder:
  `training/preregister_bitmex_trollbox_semantics.py`; and
- semantic builder SHA-256:
  `0a31ac9e888f510742753b6ec74608bb8cbef6783a3771381205e2824dc7799d`.

The semantic clock must declare `market_or_outcomes_opened=false` and
`private_text_committed=false`. The support artifact must declare zero market,
funding, and outcome rows loaded. Any byte, manifest, schema, model revision,
prompt revision, attention-window, or support identity change retires this
version instead of silently rebuilding it.

Only these event fields may be read:

1. `observation_start`;
2. `observation_end`;
3. `entry_earliest`;
4. `exit_time` only to verify and then reject the old two-hour TBASR clock;
5. `crowd_label`;
6. `bullish_participants`;
7. `bearish_participants`;
8. `unclear_participants`;
9. `selected_participants`;
10. `selected_messages`; and
11. `meta_instruction_guarded_messages`.

`contrarian_side` is forbidden except for a schema assertion that it exists;
its value may not affect a TSDR event. Raw/private text, participant names,
message identifiers, model logits, BTC prices, funding, premium, OI, and every
future label are forbidden.

## Frozen semantic definitions

The underlying participant labels are the frozen Gemma2-2B labels already
bound by the semantic artifact. TSDR does not fine-tune, re-prompt, relabel, or
reweight them.

For an attention event `e`:

```text
B[e] = bullish_participants
S[e] = bearish_participants

strong_disagreement[e] = (
    crowd_label == UNCLEAR
    and B[e] >= 2
    and S[e] >= 2
)

clear_bull[e] = crowd_label == BULLISH
clear_bear[e] = crowd_label == BEARISH
```

The frozen semantic builder already defines a clear label as at least two
directional participants and at least a two-to-one majority against the other
direction. TSDR does not recompute the label with another ratio.

Every timestamp is parsed as UTC. Events must be unique and strictly ordered by
`observation_end`. Each observation is exactly five minutes and
`entry_earliest` must equal `observation_end + 5 minutes`. A malformed row,
clock regression, duplicate observation end, non-integer count, negative
count, count-total inconsistency, unknown label, or semantic-contract drift is
a hard failure.

## Frozen first-resolution state machine

Process semantic events once in increasing `observation_end` order.

1. When idle, the first `strong_disagreement` event starts an episode.
2. Freeze its `onset_end` and set
   `deadline = onset_end + 6 hours`.
3. While armed, ignore every later `UNCLEAR` event. It cannot replace, refresh,
   extend, or strengthen the frozen onset.
4. The first strictly later clear event whose `observation_end <= deadline`
   resolves the episode. Later clear events are ineligible.
5. If the first clear event is `BULLISH`, the candidate side is `+1`; if it is
   `BEARISH`, the side is `-1`.
6. If no clear event arrives by the deadline, expire without a candidate.
7. A resolving event terminates the episode whether its candidate is later
   accepted, split-dropped, or overlap-suppressed.
8. When the next observed event is strictly after an expired deadline, expire
   first, then allow that current event to start a new episode if it is a
   `strong_disagreement` event.

There is no score, confidence, plurality magnitude, selected-message count,
participant-count rank, repeated-vote rule, threshold grid, or choice among
later resolutions.

## Frozen execution contract

- decision time: resolving event's `observation_end`;
- entry: resolving event's frozen `entry_earliest`, exactly five minutes later;
- side: `+1` for `BULLISH` resolution and `-1` for `BEARISH` resolution;
- scheduled exit: entry plus exactly 72 five-minute bars / six hours;
- exposure: fixed `0.5x` account notional;
- chronological non-overlap: accept a candidate only when
  `entry >= prior_accepted_exit`;
- overlap action: suppress, never queue or replace;
- split containment: onset, resolution, entry, every held bar, and exit must be
  contained in one declared half-open split; and
- no stop, take-profit, trailing exit, leverage search, regime filter, model
  confidence, or discretionary override.

The six-hour deadline represents same-session semantic resolution. Matching
the hold to that fixed diffusion horizon tests whether the newly consolidated
view propagates for one additional resolution horizon. Neither duration may be
changed after source support or outcomes are observed.

## Frozen research windows

- train: `[2020-07-01, 2022-01-01)` UTC;
- selection: `[2022-01-01, 2023-01-01)` UTC; and
- sealed forward evaluation: every event on or after `2023-01-01` UTC.

The current source contains no sealed-forward events. A 2023-or-later source
extension is forbidden until train and selection both pass under unchanged
code. Before any extension, the official downloader, immutable raw-page hash,
same model files, same prompt, same participant sampling, and live WebSocket
parity contract must be frozen in a new artifact. The extension may not be
used to alter this policy.

## Source-only support gate

Build the complete state machine and each control from the semantic clock
without opening market data. The primary must satisfy every check.

### Train

- at least 150 accepted events;
- at least 45 accepted events in partial 2020 and at least 90 in 2021;
- at least 15 in every represented calendar quarter;
- at least 60 LONG and 60 SHORT events;
- at least 60 active UTC weeks;
- maximum calendar-month share at most 15%; and
- maximum UTC-entry-weekday share at most 22%.

### Selection

- at least 60 accepted events;
- at least 25 in each half-year;
- at least 12 in every quarter;
- at least 25 LONG and 25 SHORT events;
- at least 30 active UTC weeks;
- maximum calendar-month share at most 18%; and
- maximum UTC-entry-weekday share at most 25%.

All source identity, schema, count consistency, ordering, first-resolution,
deadline, split containment, non-overlap, side, latency, and hold assertions
must pass. A failed check retires TSDR-72 before any market row is opened. The
support floor may not be lowered and an adjacent 2/12/24-hour relay may not
replace it.

## Frozen controls

Controls are built from source semantics before market access. Each control
has its own chronological state and non-overlap scheduler unless explicitly
defined on exact primary clocks. None may replace a failed primary.

1. **Initial plurality** — exact primary clocks; side is the onset's bullish-
   minus-bearish sign. A tie uses deterministic SHA-256 side assignment. This
   tests whether the later resolution adds directional information.
2. **Exact direction flip** — exact primary clocks and `-primary_side`.
3. **Deterministic random side** — exact primary clocks; SHA-256 of
   `"TSDR-72-random-side-20260721|" + entry_time` assigns LONG when the first
   digest byte is below 128 and SHORT otherwise.
4. **Clear-after-clear relay** — a clear event starts the six-hour episode;
   the first later clear event resolves it in the later event's direction.
   All deadlines, latency, hold, split containment, and scheduling are exact.
   This is the no-disagreement mechanism control.
5. **Unresolved disagreement** — an onset that receives no clear event by six
   hours emits at `ceil_5m(deadline) + 5 minutes`; deterministic random side is
   used. This tests whether the onset calendar alone carries a spurious edge.
6. **One-hour execution delay** — exact primary onset, resolution, and side;
   shift entry and exit by exactly twelve five-minute bars. A boundary or
   overlap failure drops rather than replaces the trade.

## Novelty and contamination gate

TSDR shares the attention source with retired TBASR, so source identity alone
cannot establish novelty. The later evaluator must therefore publish both:

1. **path novelty:** exact `(onset_end, resolution_end, side)` identity against
   every prior committed semantic strategy; and
2. **execution novelty:** exact entry Jaccard, plus/minus six-hour one-to-one
   match coverage, and signed occupied-exposure correlation against the frozen
   TBASR trade clock and the current live portfolio clocks.

The TBASR clock export may expose only causal origin, entry, exit, and side; it
must not expose returns or performance fields to the source-support stage.
TSDR is rejected before economic evaluation if any of these are true:

- more than 35% of TSDR entries one-to-one match TBASR entries within six
  hours;
- exact entry Jaccard against any live sleeve exceeds 0.20;
- more than 35% of entries match a live sleeve within six hours; or
- absolute signed occupied-exposure correlation against a live sleeve exceeds
  0.40.

Because the hypothesis was generated after a related failed result was known,
train and 2022 selection success are not enough for production. At least one
untouched 12-month post-2022 evaluation and a forward paper-trading interval
are mandatory before portfolio admission.

## Sequential economic gate

Only a complete source-support and novelty pass authorizes a separately
implemented, tested, committed, and hash-frozen evaluator. Open train first;
open 2022 selection only after an exact train pass. No 2023-or-later Trollbox
source or BTC outcome may be opened until selection also passes.

Every opened window must report absolute return, full-calendar CAGR including
idle cash, strict MDD over every held five-minute path including pre-entry HWM,
CAGR/strict-MDD, trades, LONG/SHORT sleeves, exact funding, base/stress cost,
one-hour delay, cluster significance, and all controls.

Primary qualification requires:

- positive absolute return;
- `CAGR / strict MDD >= 3.0`;
- strict MDD `<= 15%`;
- mean gross underlying move at least 30 bp/trade;
- weekly-cluster one-sided sign-flip `p <= 0.10`;
- positive 10 bp/notional/side stress return;
- positive one-hour-delayed return;
- positive return in each contained calendar year and each selection half;
- both LONG and SHORT sleeves positive in train and selection; and
- primary CAGR/strict-MDD at least `0.50` above every mechanism control.

All metrics use the full declared calendar, including time out of market. A
failure retires TSDR-72. No threshold, side, deadline, hold, source subset,
calendar, cost, or model repair is authorized.

## Live parity requirements

The research source came from the official BitMEX chat history endpoint. Live
promotion requires a separately tested adapter that:

1. consumes the official Trollbox WebSocket stream;
2. persists raw messages and receipt time before inference;
3. applies the same language filter, participant sampling, character limit,
   Gemma2-2B model revision, prompt revision, and meta-instruction guard;
4. closes the same five-minute attention window before classification;
5. emits only after all selected participant labels are durable;
6. uses the later of historical synthetic availability and local completion
   time;
7. rejects stale, duplicated, reordered, missing, or model-drifted windows; and
8. fails flat when the source, inference worker, or clock parity is unhealthy.

No production order is allowed merely because the historical evaluator passes.

## Stop condition

The next work unit may implement only the deterministic source-support builder,
its tests, and aggregate support artifact. It must read zero market/funding
rows. If support fails, TSDR-72 is retired. If support passes, the next unit is
a pure-clock comparator export and novelty gate—not an economic backtest.
