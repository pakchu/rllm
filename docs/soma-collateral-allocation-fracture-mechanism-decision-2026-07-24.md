# SCAF-48 mechanism decision — SOMA collateral-allocation fracture

## Decision

Preregister one source-seen, outcome-unseen singleton:
**SCAF-48 — SOMA Collateral-Allocation Fracture**.

SCAF measures how the cross-sectional allocation of SOMA securities-lending
demand changes relative to available inventory, awards, unmet demand, and
accepted lending fees. Four bounded weak relations compose the side. No single
component, aggregate scarcity level, fitted model, or outcome-selected
threshold creates an opportunity.

This document freezes source allowlists, causal batching, formulas, numerical
conventions, direction, execution, controls, source-support gates, SLCS
specificity/novelty gates, failure action, and the future RLLM boundary before
any SCAF feature or incidence is decoded.

## Bound pre-incidence documents

Boundary:

```text
docs/soma-collateral-allocation-fracture-boundary-2026-07-24.md
SHA256 faade722ffa8f7ce67db50cb34e55d371a31c9d3770e96f1b9507e8470b340d3
```

Schema amendment:

```text
docs/scaf-normalized-source-schema-preincidence-amendment-2026-07-24.md
SHA256 8e6a2ef1be5c5e93c5e998cb8b9d7a9ddf0a3e931042d6d0f47238af0b39b5d2
```

The amendment replaces unavailable operation-type history with strict
`available_at_utc` batches and replaces the impossible operation-type RLLM
token with batch-component agreement/disagreement.

## Frozen source identities and allowlists

Operation panel:

```text
data/new_york_fed_securities_lending_2019_2023/
new_york_fed_securities_lending_operations_2019_2023.csv.gz
SHA256 99eb8c37c05417789dfad7452c7b2ddc5b6b640078b87451f1c945158af77906
header SHA256 c0d63795e5e53cef816c50472c6941069cb018f30ad1f745f250daa0fa6b9200
```

Exact operation allowlist:

```text
operation_id
operation_date
available_at_utc
total_par_submitted
total_par_accepted
```

Detail panel:

```text
data/new_york_fed_securities_lending_2019_2023/
new_york_fed_securities_lending_details_2019_2023.csv.gz
SHA256 27178d8738cb50c4e6c13f1e5940fcfdf4009e6979b006c42fb86fb399d0716d
header SHA256 9f4d54dff4b9c9f0c47c0a85e0bf245276e5a3cb764b3c084017f679586b76dd
```

Exact detail allowlist:

```text
operation_id
operation_date
available_at_utc
cusip
par_submitted
par_accepted
weighted_average_rate
actual_available_to_borrow
```

Build manifest:

```text
data/new_york_fed_securities_lending_2019_2023/build_manifest.json
SHA256 58b9eb56728065d919978b8969e9bbb4bcb291f723a290d22045fe2ca3da2019
```

Every CSV is loaded with only its exact allowlist, string dtype, empty-string
preservation, and no inferred NA values. Reading `security_description`,
holdings, theoretical availability, outstanding loans, settlement/maturity
dates, notes, raw API JSON, BTC, funding, returns, labels, or PnL is forbidden
in the source-support evaluator.

## Source validation and batch completeness

Parse exact decimals; binary floating-point input parsing is forbidden.
Identifiers must be nonempty exact strings. Dates must be exact ISO dates.
Timestamps must be timezone-aware RFC 3339 UTC values.

An operation is complete only when:

1. its identity is unique;
2. every joined detail has the exact same operation date and availability;
3. `(operation_id, cusip)` is unique;
4. submitted, accepted, and actual-available values are finite and
   nonnegative;
5. accepted never exceeds submitted;
6. a positive accepted amount has a finite nonnegative fee;
7. a zero accepted amount may have only an empty or finite nonnegative fee;
8. detail submitted and accepted sums exactly reconcile to operation totals;
   and
9. at least one detail atom exists.

A causal batch contains every complete operation with one exact
`available_at_utc`. Its atom identity is `(operation_id, cusip)`; equal CUSIPs
across operations are not merged. The batch is complete only when every
operation assigned to that timestamp is complete.

The batch additionally requires strictly positive:

- total submitted amount;
- total actual available amount;
- total accepted amount; and
- total accepted-fee mass.

Any failure invalidates the entire batch and resets transition continuity. The
next complete batch establishes a new baseline and cannot trigger. Only the
following complete batch at a strictly later availability may compare with it.
No same-timestamp operation can enter another's reference state.

## Deterministic numerical convention

Atoms are sorted lexicographically by exact `(operation_id, cusip)`.
All amount arithmetic and reconciliation use arbitrary-precision decimal
values.

For distributional logarithms only:

1. convert exact normalized shares to IEEE-754 binary64 in sorted atom order;
2. use natural logarithms and `math.fsum`;
3. define `0 * log(0 / m) = 0`;
4. reject a nonfinite intermediate or result; and
5. convert the result through its shortest decimal string and quantize to
   exactly `1e-12` with decimal `ROUND_HALF_EVEN`.

Every stored component lies in `[0, 1]`. A value outside that interval by more
than `1e-12` rejects the batch; a value within that tolerance is clipped before
quantization.

For two normalized distributions `a` and `b`, define:

```text
m_i = 0.5 * (a_i + b_i)

JSD(a, b)
  = [0.5 * sum_i a_i * ln(a_i / m_i)
     + 0.5 * sum_i b_i * ln(b_i / m_i)]
    / ln(2)
```

## Four bounded weak components

For atom `i` in complete batch `t`:

```text
S_i = par_submitted_i
A_i = actual_available_to_borrow_i
W_i = par_accepted_i
R_i = weighted_average_rate_i when W_i > 0, else 0

p_i = S_i / sum_j S_j
q_i = A_i / sum_j A_j
r_i = W_i / sum_j W_j
F_i = W_i * R_i
f_i = F_i / sum_j F_j
```

The components are:

### 1. Demand/inventory mismatch

```text
inventory_mismatch[t] = JSD(p, q)
```

Higher values mean submitted demand is distributed differently from lendable
inventory.

### 2. Award distortion

```text
award_distortion[t] = JSD(p, r)
```

Higher values mean accepted awards are distributed differently from submitted
demand.

### 3. Unmet-demand mass

```text
unmet_demand_mass[t] = sum_i p_i * I(W_i == 0)
```

This is demand-weighted rationing, not a count breadth and not SLCS demand
breadth. Compute it as the exact decimal ratio:

```text
sum_i S_i * I(W_i == 0) / sum_i S_i
```

inside a local decimal context with precision `80` and
`ROUND_HALF_EVEN`, then quantize to exactly `1e-12` with
`ROUND_HALF_EVEN`. No binary float enters this component.

### 4. Fee-allocation distortion

```text
fee_distortion[t] = JSD(r, f)
```

Higher values mean accepted fee mass is concentrated differently from accepted
notional. The New York Fed lending fee is not interpreted as a repo rate.

## Transition algebra and fixed direction

Compare every quantized current component with the corresponding quantized
component in the immediately previous complete, strictly earlier causal batch.

For each component:

```text
UP   = current > previous
DOWN = current < previous
FLAT = current == previous
```

The batch relation is:

```text
FRACTURE = count(UP) >= 3
RELIEF   = count(DOWN) >= 3
NEUTRAL  = otherwise
```

`FRACTURE` and `RELIEF` are mutually exclusive. No magnitude threshold,
rolling rank, fitted coefficient, volatility scale, calendar regime, or market
state is allowed.

Fixed side:

- `FRACTURE`: collateral allocation is becoming more mismatched, rationed, and
  security-specific — **SHORT BTC**;
- `RELIEF`: substitution and allocation are broadening — **LONG BTC**;
- `NEUTRAL`: abstain.

Every complete consensus batch is a raw opportunity. Persistence is not a
separate trigger; global reservation determines whether a raw opportunity can
enter the accepted clock.

## Exact execution

- canonical signal timestamp: UTC
  `YYYY-MM-DDTHH:MM:SSZ`; a source timestamp with fractional seconds rejects
  the batch;
- primary `signal_id`:
  lowercase hexadecimal SHA-256 of UTF-8
  `SCAF-48|<canonical_signal_timestamp>|<FRACTURE_OR_RELIEF>`;
- signal availability: current causal batch `available_at_utc`;
- entry: `ceil_to_5m(signal) + 5 elapsed minutes`, including an exact-grid
  signal;
- exit: exactly 48 elapsed hours / 576 five-minute bars after entry;
- policy notional for later economics: fixed `0.5x`;
- no stop, take-profit, trailing exit, dynamic size, reverse, or price gate.

Sort raw opportunities by `(entry_time, signal_available_time, signal_id)`.
Globally reserve `[entry, exit)` before assigning splits. Accept only when
`entry >= previous_accepted_exit`; suppressed opportunities are not queued.
Entry and exit must both be contained in one split.

Complete-batch and valid-transition coverage assigns a row to a window by its
signal availability. Raw-opportunity statistics include only opportunities
whose hypothetical entry and 48-hour exit are fully contained in that window.
Accepted-clock statistics use the same containment after global reservation.

Frozen windows:

- source warmup/baseline: 2019;
- train: `[2020-01-01T00:00:00Z, 2023-01-01T00:00:00Z)`;
- selection: `[2023-01-01T00:00:00Z, 2024-01-01T00:00:00Z)`;
- sealed source and outcomes: 2024 onward.

## Frozen controls

Each control owns an independent raw clock and independent global reservation
with the same availability, latency, 48-hour hold, and split containment.

Component controls:

1. `inventory_mismatch_only`: UP short, DOWN long, FLAT abstain;
2. `award_distortion_only`: same rule;
3. `unmet_demand_mass_only`: same rule;
4. `fee_distortion_only`: same rule.

Composition controls:

5. `mean_change_without_consensus`: quantize the arithmetic mean of the four
   already quantized components to `1e-12` with `ROUND_HALF_EVEN`, compare its
   current and previous values, use UP as short and DOWN as long, and abstain
   on equality;
6. `two_of_four_without_opposition`: at least two UP and zero DOWN is short;
   at least two DOWN and zero UP is long;
7. `one_batch_stale`: apply the previous valid transition vector at the
   current batch availability;
8. `five_batch_stale`: apply the transition vector five valid transitions old;
9. `within_batch_demand_permutation`: keep `q`, `r`, `f`, and the zero-award
   mask fixed, but deterministically permute `p` over sorted atoms; then
   rebuild all four components and transitions.

Stale-control history is local to one uninterrupted valid-batch segment.
Invalid batches clear both stale queues. At the current availability,
`one_batch_stale` uses the immediately prior transition vector and
`five_batch_stale` uses the fifth prior vector. Insufficient history or a
stored NEUTRAL vector abstains.

The exact permutation is:

1. reject NUL in `operation_id` or `cusip`;
2. list source `p` values in lexicographic `(operation_id, cusip)` order;
3. for every destination atom compute SHA-256 over the byte concatenation
   `b"SCAF-48\0" + timestamp_ascii + b"\0" + operation_id_utf8 + b"\0" +
   cusip_utf8`;
4. order destinations by `(digest_bytes, operation_id, cusip)` ascending; and
5. assign source value `j` to destination `j`.

Permutation components compare only with the previous permutation components
inside the same uninterrupted valid-batch segment. Digest collisions therefore
have an exact lexical tie-break.

Economic-side controls on the exact accepted primary clock:

10. `exact_direction_flip`;
11. `deterministic_random_side`, using the low bit of
    `SHA256("SCAF-48|<primary_signal_id>|RANDOM_SIDE")`, where the angle
    brackets are replaced by the lowercase primary signal-id value;
12. `constant_long`;
13. `constant_short`.

No control may replace the primary after source or outcome evidence opens.

For random side, interpret the 32 digest bytes as one unsigned big-endian
integer: even is LONG and odd is SHORT. The string is exact UTF-8 with no
newline. Direction-flip, random, constant-long, and constant-short controls
reuse the exact accepted primary entries/exits and do not run a second
reservation. Every other control uses an independent reservation. A
source-clock control `signal_id` is the lowercase SHA-256 of UTF-8
`SCAF-48|<control>|<canonical_signal_timestamp>|<LONG_OR_SHORT>`.

## Source-support gates

Batch/transition coverage uses complete batches and valid transitions assigned
by signal availability. Raw-consensus statistics use split-contained
hypothetical raw opportunities before reservation. Every other source-support
and composition statistic uses accepted, split-contained clocks unless its
definition below explicitly names the raw-primary denominator. Empty
denominators fail. The first false check in the listed order is the frozen
first failure.

UTC is the sole calendar basis for support gates. Convert accepted entry times
to UTC calendar dates before assigning year, half, quarter, month, active
month, maximum-gap, or same-side-run order. Split containment continues to use
the exact UTC timestamp windows.

### Batch/transition coverage

- train complete causal batches: at least 700;
- selection complete causal batches: at least 220;
- train valid transitions: at least 690;
- selection valid transitions: at least 215;
- invalid batches and continuity resets: report exactly;
- raw consensus share among valid transitions in each split: 10% through 65%.

### Accepted primary clock

Train:

- at least 120 events;
- at least 30 events in each of 2020, 2021, and 2022;
- at least 30 long and 30 short;
- at least 30 active calendar months;
- maximum UTC calendar-date entry gap at most 30 days;
- maximum month share at most 12%;
- maximum quarter share at most 25%; and
- maximum same-side run at most 12.

Selection:

- at least 35 events;
- at least 15 events in each half of 2023;
- at least 7 events in each quarter;
- at least 10 long and 10 short;
- at least 10 active calendar months;
- maximum UTC calendar-date entry gap at most 30 days;
- maximum month share at most 20%; and
- maximum same-side run at most 8.

Every component, composition, stale, permutation, flipped, random, and
constant-side control must be nonempty in train and selection.

## Relational-composition and specificity gates

Apply separately in train and selection:

- each of the four components agrees with the primary raw side on 55% through
  95% of raw primary opportunities;
- exact four-of-four unanimity share: 10% through 85%;
- exact three-of-four share: at least 15%;
- each component-only accepted clock exact-entry same-side reproduction of
  accepted primary: at most 80%;
- `mean_change_without_consensus` reproduction: at most 90%;
- each stale-control reproduction: at most 75%;
- deterministic-random-side reproduction: at most 60%;
- permutation exact-entry Jaccard: at most 50%; and
- permutation exact-entry same-side reproduction: at most 65%.

For component agreement, the denominator is every split-contained raw primary
opportunity before reservation. An UP component agrees with a FRACTURE/SHORT
opportunity; a DOWN component agrees with a RELIEF/LONG opportunity. FLAT and
the opposite direction both count as disagreement.

Four-of-four and exact-three-of-four shares use that same raw-primary
denominator. Four-of-four means all four component directions agree with the
primary relation. Exact three-of-four means exactly three agree; the fourth may
be FLAT or opposite.

Control reproduction uses accepted primary as denominator and requires an
exact equal entry timestamp plus equal side. Exact-entry Jaccard uses accepted
entry sets and equals `intersection / union`; an empty union fails.

## Frozen SLCS same-source novelty battery

Comparator artifact:

```text
results/soma_lending_collateral_scarcity_clocks_2026-07-23.csv.gz
SHA256 b3fe0dc8c895f9a8974cdf08b5bed9d58ff693b8aca7ed59c224627de930a948
header SHA256 45a24e800b79a30047ffeb5f45c69cf4817262e57b0af1cf5e046332536e5e94
```

Read only:

```text
control
entry_time
exit_time
side
```

Exact comparator groups:

```text
primary
demand_intensity_only
weighted_fee_only
carry_intensity_only
demand_breadth_only
mean_without_consensus
same_sign_without_magnitude
```

Use the exact common window:

```text
[2020-01-01T00:00:00Z, 2024-01-01T00:00:00Z)
```

Only fully contained intervals count. Every comparator group must contain at
least 20 rows. For each group SCAF must satisfy all:

- exact-entry Jaccard at most 0.25;
- maximum-cardinality one-to-one ±1-calendar-day entry Jaccard at most 0.50;
- same-entry same-side reproduction using SCAF as denominator at most 0.30;
  and
- absolute signed five-minute occupied-exposure Pearson at most 0.35.

Undefined correlation, malformed rows, overlapping intervals inside one group,
or a failed row floor retires SCAF. Comparator value rows remain unopened until
all source-support and relational-composition gates pass.

Exact-entry Jaccard is:

```text
intersection_count / (SCAF_count + comparator_count - intersection_count)
```

Duplicate entries in either clock fail.

For the one-calendar-day metric:

1. convert entry timestamps to `America/New_York` calendar dates;
2. connect a SCAF/comparator pair when their absolute local-date difference is
   at most one;
3. sort each side by `(entry_time, signal_id)` for SCAF and by
   `(entry_time, original_row_number)` for the comparator;
4. compute maximum-cardinality bipartite matching using deterministic
   augmenting paths that visit left rows and adjacent right rows in those
   orders; and
5. report `matched / (SCAF_count + comparator_count - matched)`.

The matching identity can vary under ties but the maximum cardinality and gate
value cannot.

For same-entry same-side reproduction, the denominator is all contained SCAF
rows. A row matches only an exact equal comparator entry timestamp and exact
equal `LONG`/`SHORT` side.

Signed occupancy uses every five-minute cell
`[t, t + 5m)` over the exact common window. Entry and exit must lie on that
grid. Each fully contained interval writes `+1` for LONG or `-1` for SHORT into
every cell in `[entry, exit)`; idle cells are `0`. Any self-overlap fails.
Pearson correlation is computed over the entire equal-length cell arrays,
including idle cells, and the absolute value is gated. A constant array,
nonfinite result, or unequal grid fails.

## Failure action and evidence order

The strict sequence is:

1. commit this mechanism;
2. commit a tested immutable preregistration without SCAF source incidence;
3. commit and hash-bind an outcome-blind source-support/novelty evaluator;
4. open exact operation/detail allowlists;
5. retire unchanged on any provenance, causality, support, composition, or
   SLCS novelty failure;
6. only a complete pass may freeze a strict economic/RLLM evaluator; and
7. open train, selection, and later immutable source extensions sequentially.

No failed threshold, component, transition, direction, latency, hold, or
control may be repaired under SCAF-48.

## Live fail-flat parity

Live operation must capture and hash the exact official response used for each
normalized batch. A missing operation/detail, schema drift, duplicate,
reconciliation failure, timestamp regression, post-capture revision, or
feature mismatch:

1. invalidates every not-yet-submitted order from that batch;
2. permits no new or reverse order;
3. while flat, remains flat and halts SCAF;
4. while positioned, submits one reduce-only market flatten request, records it
   as an operational fail-safe rather than a policy exit, and halts SCAF; and
5. requires an explicit audited restart from a newly established complete
   baseline batch.

Historical source-support/economic clocks retain the frozen 48-hour exit; the
live emergency flatten is a safety boundary, never backfilled as alpha.

## Future economic and RLLM boundary

No economic evaluator is authorized by this document. A later evaluator must
use full-wall-clock CAGR, strict pre-entry/global-high-water plus intratrade
MDD, exact funding, executable costs, stress costs, clustered significance,
positive required subperiods, and component/control falsification. The
deterministic baseline must satisfy `CAGR / strict MDD >= 3.0` before RLLM.

Only after unchanged deterministic train and selection passes may one small
RLLM receive:

- four `UP/DOWN/FLAT` component tokens;
- `FRACTURE/RELIEF` and three-versus-four agreement;
- prior symbolic relation;
- batch validity;
- current position, fixed side, and time in position; and
- a fixed risk-budget token.

The action set is exactly:

```text
TRADE_FIXED_SIDE
ABSTAIN
```

Raw component values, amounts, rates, CUSIPs, operation IDs, dates,
timestamps, BTC prices, future paths, split labels, ranks, rewards, and
evaluated outcomes are forbidden. RLLM may not create an opportunity, reverse
the side, change entry/exit/leverage, or recover calendar identity.

## Evidence boundary

This mechanism used only committed source schema/audit facts, immutable hashes,
the outcome-blind SLCS source-support result, the outcome-blind DCLB result,
and pre-incidence reasoning. It did not decode either normalized source CSV or
the SLCS comparator clock and did not compute a SCAF component, causal batch,
transition, candidate count, side count, overlap, market return, PnL, CAGR,
strict MDD, or hit rate.
