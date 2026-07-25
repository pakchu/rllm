# CEFS-D1 boundary — Cboe edge-flip sequence target policy

## Decision

Freeze one candidate identity before decoding any CEFS-D1 source value:

**CEFS-D1 — Cboe Edge-Flip Sequence Policy, a five-state daily
target-position RLLM over twelve primitive, percentile-free relation edges.**

Every exact common Cboe source date becomes an action-independent state after
warm-up. The policy receives five ordered categorical states plus its current
target and chooses exactly one next target:

```text
TARGET_LONG
TARGET_FLAT
TARGET_SHORT
```

This boundary opens no CEFS source value, relation edge, prompt, incidence,
BTC row, funding row, future return, reward, model output, action, trade, PnL,
CAGR, MDD, or post-2023 source row.

## Research boundary and contamination

CEFS-D1 is not a globally pristine discovery. The three Cboe source panels
have been inspected, and several simpler Cboe candidates opened pre-2023
market outcomes:

- CVTR used a scalar term-rotation rule and failed economic Stage 1;
- CTHD used a thresholded tail-disagreement rule and failed economic Stage 1;
- CIHM used a thresholded option-flow score and failed economic Stage 1;
- CXRT used a three-vote majority side and failed source composition;
- OPRR used a sparse rank-rotation conjunction and failed source support; and
- CSPG used twelve fixed ordinal-pressure tokens and failed token stationarity
  before outcomes.

CEFS-D1 claims novelty only for the exact primitive edge representation,
five-state ordered sequence, current-target state, and three-action policy.
It may not reuse or repair any predecessor score, rank, vote, threshold,
trigger, side, token, signature, selected date, or failed gate.

In particular, CEFS-D1 forbids:

- a source-owned long or short side;
- an event threshold or selected subset of common dates;
- expanding, rolling, or absolute percentile ranks;
- pressure levels, stress leaders, relief leaders, or majority votes;
- a hand-coded cross-surface consensus;
- any CSPG token or signature;
- any CVTR, CTHD, CIHM, CXRT, or OPRR policy output; and
- any feature, clock, model, or gate selected from already opened Cboe returns.

## Frozen source identity

### Term-structure panel

Path:

```text
data/cboe_volatility_term_structure_2018_2023/cboe_vix_term_structure_2018-01-01_2023-12-31.csv.gz
```

File SHA-256:

```text
6f1b2f7f3a5b1e4d5001d673e6ff54374791879c278248ce27b3d610e4f75dc7
```

Manifest:

```text
data/cboe_volatility_term_structure_2018_2023/build_manifest.json
42b2a35ad131bd63574d2adcf684e28766dc3060fa645fc749df10dd3fb27f27
```

Allowed columns, in exact physical order:

```text
observation_date,VIX9D_close,VIX_close,VIX3M_close
```

### Tail-risk panel

Path:

```text
data/cboe_tail_risk_2018_2023/cboe_tail_risk_2018-01-01_2023-12-31.csv.gz
```

File SHA-256:

```text
cdde3f8d4bb1e23d00b192f5f9ef759aefba9087be5fd60653e9c02479dfa41a
```

Manifest:

```text
data/cboe_tail_risk_2018_2023/build_manifest.json
9ef80ef3034c93d97c5b2a8160b2502527287d570d15f9d7166d631d9866c7bd
```

Allowed columns, in exact physical order:

```text
observation_date,SKEW_close,VVIX_close,VIX_close
```

### Option-flow panel

Path:

```text
data/cboe_option_flow_2020_2023/cboe_option_flow_2020-01-01_2023-12-31.csv.gz
```

File SHA-256:

```text
35ef106ef01e3abadbcb4a6227187dd1d7cf2722191bd146bac06d08d1684a78
```

Manifest:

```text
data/cboe_option_flow_2020_2023/build_manifest.json
0a513b146ad5857d9ab7311e978152c308de64db8ef29c4d463eb07ea503089e
```

The physical file must retain its complete frozen header. CEFS may decode
only:

```text
observation_date
total_pcr
index_pcr
equity_pcr
vix_pcr
spx_pcr
index_volume
vix_volume
response_sha256
```

Every other option-flow column is validated as present in the frozen physical
header and then ignored without numeric decoding. The exact whole-file and
manifest hashes bind the prior builder's already completed arithmetic and
volume-containment audit; CEFS neither repeats nor weakens those checks.
`response_sha256` is validated only as exact lowercase hexadecimal text and is
forbidden from every relation, token, prompt, control, label, action, or model
input.

### Governing source audits

```text
docs/cboe-volatility-term-structure-source-audit-2026-07-17.md
985799cd9a26217bffb678ae2d5dbfa81070c84edf955bec681de663a3b63c58

docs/cboe-tail-risk-source-audit-2026-07-18.md
706c5839cc5babc7b150d71a139b659c76b2cc5a1a355de61868842000b2847b

docs/cboe-option-flow-source-audit-2026-07-18.md
c182ee2f9078c5bee2d2a0f3ec488105980a1c54651b8b651db2e3af96278f8f
```

The source loader must verify every bound path, file hash, manifest hash,
audit hash, schema, row order, date grammar, uniqueness constraint, and every
numeric invariant of the explicitly allowed CEFS fields before producing a
state. It must not parse a forbidden option-flow field merely to repeat an
invariant already bound by the exact frozen file hash.

## Exact source parsing

`observation_date` must be exactly ten ASCII bytes in `YYYY-MM-DD` form and
round-trip through strict Gregorian parsing. Dates must be unique and strictly
increasing within each panel.

Every allowed numeric field is parsed from its original decimal text with
`decimal.Decimal`; binary floating point is forbidden. Numeric text must be a
finite, strictly positive plain decimal without exponent notation, sign
prefix, surrounding whitespace, comma, or underscore. Volumes must also be
exact positive base-ten integers.

The CEFS date set is the exact sorted intersection of the three panel date
sets. No missing date is synthesized, carried, interpolated, substituted, or
zero-filled. The two independently frozen `VIX_close` values must be
decimal-exact on every common date. A mismatch rejects CEFS-D1.

The frozen source history is physically bounded at 2023-12-31. Any accepted
row after 2023-12-31, any hidden path substitution, or any attempt to load a
2024-or-later source file rejects the source stage.

## Exact comparison function

For finite exact decimals `left` and `right`:

```text
compare(left, right) =
    LOWER   when left < right
    EQUAL   when left = right
    HIGHER  when left > right
```

No epsilon, rounding, fitted threshold, rank, clipping, standardization, or
missing-value branch is permitted.

For two positive ratios `a/b` and `c/d`, compare them only by exact positive
cross multiplication:

```text
compare_ratio(a, b, c, d) = compare(a*d, c*b)
```

Division is forbidden in relation construction.

## Twelve primitive relation edges

Let `C` be one exact common source date and `P` its immediately preceding
exact common source date. The first common date has no edge state.

The exact edge order is:

1. `TERM_FRONT_LEVEL`
2. `TERM_BACK_LEVEL`
3. `TERM_FRONT_CHANGE`
4. `TERM_BACK_CHANGE`
5. `TAIL_SKEW_CHANGE`
6. `TAIL_VOLVOL_CHANGE`
7. `FLOW_TOTAL_PCR_CHANGE`
8. `FLOW_INDEX_PCR_CHANGE`
9. `FLOW_EQUITY_PCR_CHANGE`
10. `FLOW_VIX_PCR_CHANGE`
11. `FLOW_SPX_PCR_CHANGE`
12. `FLOW_VIX_SHARE_CHANGE`

Their formulas are:

```text
TERM_FRONT_LEVEL =
    compare(VIX9D_close[C], VIX_close[C])

TERM_BACK_LEVEL =
    compare(VIX_close[C], VIX3M_close[C])

TERM_FRONT_CHANGE =
    compare_ratio(
        VIX9D_close[C], VIX_close[C],
        VIX9D_close[P], VIX_close[P]
    )

TERM_BACK_CHANGE =
    compare_ratio(
        VIX_close[C], VIX3M_close[C],
        VIX_close[P], VIX3M_close[P]
    )

TAIL_SKEW_CHANGE =
    compare(SKEW_close[C], SKEW_close[P])

TAIL_VOLVOL_CHANGE =
    compare_ratio(
        VVIX_close[C], VIX_close[C],
        VVIX_close[P], VIX_close[P]
    )

FLOW_TOTAL_PCR_CHANGE =
    compare(total_pcr[C], total_pcr[P])

FLOW_INDEX_PCR_CHANGE =
    compare(index_pcr[C], index_pcr[P])

FLOW_EQUITY_PCR_CHANGE =
    compare(equity_pcr[C], equity_pcr[P])

FLOW_VIX_PCR_CHANGE =
    compare(vix_pcr[C], vix_pcr[P])

FLOW_SPX_PCR_CHANGE =
    compare(spx_pcr[C], spx_pcr[P])

FLOW_VIX_SHARE_CHANGE =
    compare_ratio(
        vix_volume[C], index_volume[C],
        vix_volume[P], index_volume[P]
    )
```

These edges are unsigned observations. `HIGHER` is not intrinsically bullish
or bearish. No deterministic component may aggregate them into a score,
consensus, source side, confidence, or trade permission.

## Five-state categorical sequence

An edge state at common date `C_i` requires `C_(i-1)`. A CEFS sequence ending
at `C_i` contains edge states for:

```text
C_(i-4), C_(i-3), C_(i-2), C_(i-1), C_i
```

Therefore the earliest sequence requires six common source rows
`C_(i-5)..C_i`. The five relative state labels, in exact serialized order,
are:

```text
EARLIEST
EARLY
MIDDLE
LATE
CURRENT
```

Within each state, the twelve edges appear in the exact frozen edge order.
The canonical primary prompt contains sixty edge lines followed by one
position line:

```text
RELATIVE_STATE.EDGE_NAME=LOWER|EQUAL|HIGHER
POSITION=TARGET_LONG|TARGET_FLAT|TARGET_SHORT
```

The literal `|` notation above describes alternatives and never appears in a
prompt. Lines use ASCII, `\n`, and one final newline. No blank line is
permitted.

Prompts may contain only:

- the five relative state labels;
- the twelve edge names;
- `LOWER`, `EQUAL`, or `HIGHER`;
- `POSITION`;
- the three target labels; and
- fixed task/instruction text that is hash-frozen after source support and
  before rewards.

Raw or formatted numbers, dates, weekdays, months, years, ranks, source paths,
hashes, prices, returns, funding, PnL, equity, drawdown, split names,
checkpoint names, model confidence, and hidden reasoning are forbidden.

The source stage does not know the realized current target. For every complete
action-independent schedule row it must therefore serialize exactly three
position-conditioned prompt templates, ordered:

```text
TARGET_FLAT
TARGET_LONG
TARGET_SHORT
```

These templates are source-language objects, not selected actions. The later
economic runner must choose exactly the template matching the deterministic
current target before inference.

## Causal decision clock

Let a sequence end at source observation date `D`. Define `D+1` and `D+2` as
the next one and two **calendar** dates, not later business, exchange, common,
or source-row dates.

The state from `D` has this future-row-independent clock:

```text
source available = calendar D+1 09:30 America/New_York
decision          = calendar D+1 09:35 America/New_York
entry/rebalance   = calendar D+1 09:35 America/New_York
scheduled exit    = calendar D+2 09:35 America/New_York
```

Timezone conversion must use Python `zoneinfo.ZoneInfo("America/New_York")`.
Ambiguous or nonexistent local times reject the row; these fixed morning
times are expected to be unambiguous. The UTC result must align exactly to a
five-minute boundary.

Weekend and holiday entries are valid because BTCUSDT trades continuously.
The presence or absence of any later Cboe row cannot create, suppress, extend,
or move an already formed schedule. This deliberately reuses the corrected
future-row-independent Cboe availability clock as causal infrastructure; it
does not claim clock novelty over CSPG.

No stale source carry, same-date entry, late backfill, queued trade, or
bar-time rounding is permitted. The completed `D` state is used only for its
fixed `[D+1 09:35, D+2 09:35)` interval and is then stale.

Every interval is reserved before any position or model action is known.
Intervals from consecutive source dates meet exactly at rebalance. A later
interval after a source-date gap begins from flat. `TARGET_FLAT`, inference
failure, missing market execution, or external portfolio conflict cannot
release or move another interval.

## Target-position transition

At each valid entry/rebalance:

1. mark the old target through the rebalance open;
2. settle all funding marks with the later frozen evaluator's exact
   half-open convention;
3. obtain one of the three next targets;
4. change only the required notional from old target to next target;
5. charge costs on absolute changed notional; and
6. hold the new target until the scheduled exit.

When a consecutive source-date interval starts exactly at the scheduled exit,
the exit and next entry are one direct target-to-target rebalance. When no
reserved interval starts at the scheduled exit, the target changes to
`TARGET_FLAT`.

The source stage computes no action. It stores only the action-independent
state and schedule. Unknown tokens, invalid source, missing runtime artifacts,
model errors, timeouts, non-finite logits, stale live input, or an invalid
action force `TARGET_FLAT`.

No stop, take-profit, trailing exit, pyramiding, dynamic leverage, source
threshold, or model-created holding period is allowed. Exact leverage, cost,
funding, reward, accounting, and terminal-flatten rules must be frozen after
source support and before any market outcome is opened.

## Chronological roles

Source-only support may inspect CEFS tokens and schedules across 2020–2023,
but no market outcome:

| Role | Complete entry/rebalance interval |
|---|---|
| train/development | 2020-01-01 through 2022-01-01 |
| test/checkpoint selection | 2022-01-01 through 2023-01-01 |
| untouched candidate eval | 2023-01-01 through 2024-01-01 |
| sealed extensions | 2024, then 2025, then 2026-YTD |

An interval belongs to a role only when both entry and scheduled exit are
inside its half-open wall-clock boundary. Earlier source states may warm the
sequence but cannot create cross-boundary PnL. Ranks do not exist and no
history resets at a role boundary.

No 2023 outcome may choose a source field, edge, sequence length, prompt,
reward, algorithm, hyperparameter, checkpoint rule, action threshold, cost,
or control. No 2024 source or outcome may be opened unless the unchanged 2023
gate passes.

## Frozen source-support gates

The source-support runner must evaluate these gates in order and stop at the
first failure. Later-gate statistics remain uncomputed and absent after a
failure.

### Gate 1 — authority and forbidden access

- worktree clean at the committed runner revision;
- every bound file and document hash exact;
- only the three frozen pre-2024 source panels opened;
- zero market, funding, future-return, reward, model, action, PnL, CAGR, MDD,
  comparator-action, and post-2023 source accesses.

### Gate 2 — schema and chronology

- exact physical headers and allowlists;
- exact decimal grammar and positivity;
- unique strictly increasing dates in every panel;
- exact common-date intersection;
- exact term/tail VIX equality on every common date;
- common coverage exactly `2020-01-02` through `2023-12-29`;
- exactly `1,006` common dates; and
- maximum consecutive common-date gap at most ten calendar days.

### Gate 3 — schedule support

- at least `920` complete reserved intervals across 2020–2023;
- at least `230` complete intervals in each entry year 2020, 2021, 2022, and
  2023;
- at least `50` complete intervals in every calendar quarter;
- zero overlapping intervals;
- every interval exactly follows the frozen next-calendar-day, fixed
  twenty-four-hour clock;
- deleting or appending later source rows cannot create, suppress, or move an
  already formed interval; and
- no role-crossing interval.

### Gate 4 — primitive edge support

For the two level edges, in train/development, test, and eval separately:

- at least two of `LOWER`, `EQUAL`, and `HIGHER` occur; and
- no one level exceeds `98%`.

For each of the ten change edges, in train/development, test, and eval
separately:

- both `LOWER` and `HIGHER` have at least `10%` share;
- no one level exceeds `88%`; and
- `EQUAL` may be absent.

### Gate 5 — state diversity and stability

Separately for every entry year:

- at least `40` distinct current twelve-edge signatures;
- largest current-signature share at most `15%`;
- at least `80%` of complete five-state sequence signatures are unique; and
- largest five-state sequence share at most `2%`.

For every edge and every level, the absolute share difference between
train/development and test is at most `25` percentage points, and between
train/development and eval is at most `25` percentage points.

### Gate 6 — source-only controls

For every complete primary schedule row and each of its three ordered
position-conditioned templates, produce:

1. `reverse_sequence` — reverse the five state blocks;
2. `stale_current` — replace `CURRENT` with a duplicate of `LATE`;
3. `group_order_rotation` — serialize flow, term, then tail edge groups inside
   every state while preserving edge names and values;
4. `within_group_value_rotation` — swap the two term level values, swap the
   two term change values, swap the two tail values, and rotate the six flow
   values left by one;
5. `term_only` — replace tail and flow values with `MASKED`;
6. `tail_only` — replace term and flow values with `MASKED`;
7. `flow_only` — replace term and tail values with `MASKED`; and
8. `current_only` — replace every non-current edge value with `MASKED`.

`MASKED` is valid only in a control prompt and never in a primary prompt.
Controls retain the same schedule and position line.

Required source-only differences:

- every mask control differs from primary on every row;
- `group_order_rotation` differs bytewise from primary on every row;
- `reverse_sequence` differs bytewise on at least `95%` of rows;
- `stale_current` changes at least one semantic edge value on at least `35%`
  of rows; and
- `within_group_value_rotation` changes at least one semantic edge value on at
  least `50%` of rows.

These checks prove control construction, not predictive novelty.

### Gate 7 — determinism and append replay

- two clean executions produce byte-identical token, schedule, control, and
  report artifacts;
- rebuilding every prefix ending at each year boundary reproduces all prior
  states and schedules byte-for-byte;
- appending a synthetic valid future common row cannot change any prior edge,
  sequence, schedule, or control; and
- row order, dictionary order, locale, host timezone, and gzip timestamp
  cannot change output bytes.

Any gate failure retires CEFS-D1 unchanged before outcomes. No threshold,
category merge, edge drop, edge addition, sequence change, date exception,
clock change, source substitution, or control repair is allowed.

## Source-stage artifacts

Only after a committed implementation contract, committed runner/tests, and a
committed execution seal may the source runner decode CEFS values.

A pass may atomically publish:

```text
data/cboe_edge_flip_sequence_policy_source_2020_2023.csv.gz
data/cboe_edge_flip_sequence_policy_controls_2020_2023.csv.gz
results/cboe_edge_flip_sequence_policy_source_support_2026-07-25.json
```

A rejection may publish only:

```text
results/cboe_edge_flip_sequence_policy_source_rejection_2026-07-25.json
```

Pass artifacts may contain source dates, UTC schedule timestamps, categorical
edges, relative sequence slots, and control identifiers. They may not contain
decoded source numbers, model targets, actions, market outcomes, returns, or
economic metrics.

## RLLM boundary after source support

Only a complete source pass authorizes a separate evaluator/model freeze.
That later freeze must:

- keep deterministic arithmetic and execution outside the model;
- use one small locally deployable language model with a non-generative
  three-action value head;
- train only from deterministic, causally available transition rewards;
- freeze model family, tokenizer, adapter method, seeds, optimizer, context,
  checkpoint selection, and logit-bias calibration before test outcomes;
- include simple linear, tree, exact-memory, sequence-reversed, stale-current,
  single-surface, constant-target, random-target, and shuffled-reward
  controls;
- compare signed exposure and action changes against retired Cboe policies and
  the frozen live portfolio;
- require incremental value beyond the strongest deterministic baseline; and
- stop sequentially at train/development, test, eval, 2024, 2025, and
  2026-YTD.

The LLM is used only for conditional deduction across ordered weak relations.
It may not generate features, arithmetic, explanations, free-form orders, or
hidden chain-of-thought.

## Live boundary

The frozen historical Cboe pages are current-vintage research sources, not a
proof of historical point-in-time publication. A historical pass is therefore
candidate-specific transaction evidence only.

Before shadow or live admission, a separately committed live adapter must:

- capture every official response with retrieval time and content hash;
- reproduce the frozen schema and edge language;
- use the active Cboe calendar and methodology versions;
- prove source publication before the 09:30 availability boundary;
- fail flat on missing, late, revised, duplicated, stale, or schema-drifted
  data;
- never carry a state across an expected no-data/error session;
- replay historical and forward-captured overlap byte-for-byte; and
- demonstrate shadow parity before orders.

Until that adapter passes, every CEFS live action is `TARGET_FLAT`.

## Outcome boundary at this commit

```text
CEFS source values decoded       = 0
CEFS edge states created         = 0
CEFS schedules created           = 0
post-2023 source rows read        = 0
BTC market rows read              = 0
funding rows read                 = 0
future returns computed           = 0
rewards computed                  = 0
models trained                    = 0
actions selected                  = 0
trades simulated                  = 0
PnL/CAGR/MDD computed             = 0
```

## Mandatory next sequence

1. commit this boundary;
2. generate and commit one machine-readable preregistration without opening
   source values;
3. freeze one source-support implementation contract;
4. implement and test the source runner using synthetic fixtures only;
5. commit a one-shot execution seal;
6. execute source support once;
7. retire CEFS-D1 unchanged on the first failure; and
8. only a complete pass may freeze economic/RLLM evaluation.
