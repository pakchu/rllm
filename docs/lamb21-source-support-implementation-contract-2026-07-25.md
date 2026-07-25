# LAMB-21 source-support implementation contract — 2026-07-25

## Authority

This contract implements, but cannot alter:

```text
docs/post-tracer-alpha-mechanism-audit-2026-07-25.md
SHA256 7394cd096d92b5469eb625605faaa8f53c49fc486b921269a1b2da0b08afbf9e

docs/liquidity-aware-microstructure-braid-boundary-2026-07-25.md
SHA256 a412b3e7dcfad625e0cdccd3a1941055bc5e104f4693b21ddf24b4fa1aa7c654

training/preregister_liquidity_aware_microstructure_braid.py
SHA256 5e65e0806628c08691591fe9179022c9b840fb6d399b25f168c7bfb4fdf69c36

tests/test_preregister_liquidity_aware_microstructure_braid.py
SHA256 557ed94e7863e279f23b2bc36445c173622b99a121f57f6301b888c98ba540bd

results/liquidity_aware_microstructure_braid_preregistration_2026-07-25.json
SHA256 4ac8bf8f2d54120130c49a90f3d40a5cfaf141673525cb54df4b5333c01290e6
manifest hash be035126c30f35c425563ee5b8d8d81c57b64c50dd072e8e7ae9b6acc1fd939e
```

The preregistration producer is sealed at:

```text
commit 32f97c8d74e2598c9858da32b7eb203b690da0b4
script SHA256 1fb3b7f39fe418e9c160a3035cbb63a8f65cb72119a49d36c93fc5528c37e10c
```

The later stability fix does not rewrite that historical producer claim. It
only verifies the sealed bytes from Git and prevents recreation by a different
HEAD.

## Authorized implementation

Exact runner:

```text
training/build_lamb21_source_support.py
```

Exact tests:

```text
tests/test_build_lamb21_source_support.py
```

Exact outputs:

```text
data/lamb21_source_support/token_support.csv.gz
results/lamb21_source_support_2026-07-25.json
```

The runner and tests must be committed together after synthetic-only tests.
The first real joint source incidence may be decoded only from a clean commit
where both files were last changed by `HEAD`. The official report records
`HEAD` plus SHA-256 for the audit, boundary, preregistration source/test/JSON,
this contract, runner, and runner tests.

The token file and report are write-once. Gzip uses UTF-8, LF, and `mtime=0`.
A byte-identical rerun verifies the existing files; drift fails closed.

## Evidence boundary

The implementation may decode only the four exact preregistered source
projections. It must never import an execution-kline module, funding module,
future-return builder, reward builder, model trainer, checkpoint selector,
trade simulator, or portfolio evaluator.

The report contains counters initialized before loading:

```text
source_value_rows_decoded
joint_state_rows_built
execution_market_rows_opened
funding_rows_opened
future_return_rows_opened
reward_rows_built
model_rows_built
trades_built
pnl_values_computed
cagr_values_computed
mdd_values_computed
post_2023_source_rows_opened
```

Only the first two may become nonzero. Every other counter must remain zero.
Any forbidden counter increment aborts before output.

## Physical loading

Before `pandas.read_csv`, the runner verifies:

- exact path;
- full compressed-file SHA-256;
- gzip `mtime=0`;
- exact physical first-line bytes and header SHA-256;
- exact build-manifest bytes;
- source end before `2024-01-01`; and
- the frozen allowlist is unique and contained in the physical header.

Each source is loaded as:

```text
pandas.read_csv(
  exact_path,
  usecols=exact_allowlist,
  dtype="string",
  keep_default_na=false,
  na_filter=false,
)
```

Columns are then reordered to the exact allowlist. Loading all columns and
dropping `qlcd_score`, `max_ms_score`, or other forbidden fields afterward is
not allowed.

## Source validation

### H.4.1

- release date, observation date, and availability are unique/increasing;
- observation date is strictly before release date;
- availability is timezone-aware and its New York date equals release date;
- every level is finite and strictly positive;
- every release and availability is before 2024; and
- `h41_delta` uses only the immediately prior physical release.

### ON RRP

- operation date and availability are unique/increasing;
- `source_complete` is exact lowercase `true|false`;
- complete rows have blank quarantine reason and finite, nonnegative amount
  and integer counterparty counts;
- accepted counterparties do not exceed participating counterparties;
- incomplete rows have nonblank quarantine reason and blank numeric values;
- every operation and availability is before 2024; and
- a delta exists only when current and immediately prior physical rows are
  complete. An incomplete row breaks the segment.

### Five-minute sources

Both micro sources must reproduce the exact canonical five-minute UTC grid:

```text
[2020-01-01T00:00:00Z, 2024-01-01T00:00:00Z)
frequency 5 minutes
420,768 rows
```

At every timestamp their five validity flags must be byte-equivalent after
exact boolean parsing.

For each source independently:

```text
base_complete = (source_observed OR verified_zero_volume_empty)
                AND NOT source_gap_day

post_gap_quarantine =
  rolling_any(
    shift_one_bar(NOT base_complete),
    exact window 24 bars
  )

source_complete = base_complete AND NOT post_gap_quarantine
```

Observed and verified-empty flags are mutually exclusive. All projected
numeric cells are finite. Nonobserved rows have all projected numeric values
zero.

Quantity-lattice validation additionally requires:

- integer counts, quantities, and signed quantities;
- observed count and total quantity strictly positive;
- unsigned quantities nonnegative;
- absolute signed coarse/fine quantities no larger than paired unsigned
  quantities; and
- coarse plus fine quantity no larger than total quantity.

Cascade validation additionally requires:

- integer count and transaction-millisecond fields;
- observed count, price, quote notional, and maximum-group notional strictly
  positive;
- `date_ms <= first_ms <= last_ms < date_ms+300000`;
- collision and maximum-group notional no larger than bar quote notional; and
- absolute maximum-group signed notional no larger than its unsigned notional.

## Canonical process and state build

Canonical boundaries are every UTC `00:00`, `08:00`, and `16:00` from
`2020-01-01T00:00:00Z` through `2023-12-31T16:00:00Z`.

For each boundary:

1. select exactly 96 micro rows in `[B-8h,B)`;
2. require both sources to have identical timestamps and valid rows;
3. select the latest H.4.1 availability `<=B`, age at most 10 elapsed days,
   with an immediately prior physical release;
4. select the latest complete ON-RRP availability `<=B`, age at most 5
   elapsed days, with an immediately prior complete row in the same segment;
5. compute only the primitives named in the boundary;
6. validate all denominator, sign, share, and coherence identities; and
7. mark the boundary `core_source_valid`.

The first 2020 boundary is expected to lack its pre-source micro window and
remains an invalid wall-clock boundary. No calendar row is dropped to repair
support.

## Strict-prior ranks and tokens

Ranks use the last 270 prior valid canonical boundaries and require 180.
Current values are excluded. Quantiles use NumPy/pandas linear interpolation
at exact fractions `0.33` and `0.67`, with the frozen inclusive tie rule.
Invalid boundaries never enter a reference.

The six ranked primitives, eleven token fields, exact vocabularies, mappings,
safety line, and transition rules come only from the boundary. A current
invalid/rank-unready boundary emits the full safety line. A valid current
state whose immediately prior canonical state is invalid preserves current
primitive tokens but uses both mixed transition values.

`sequence_ready` means:

- current state is rank-ready; and
- twenty immediately prior canonical state rows exist.

Those prior rows may contain safety lines. Calendar time is never compressed.

The primary token output header is exactly:

```text
boundary_time
core_source_valid
rank_ready
sequence_ready
h41_impulse
rrp_impulse
macro_sponsorship
macro_age
lattice_relation
lattice_concentration
cascade_impact
cascade_intensity
micro_braid
macro_transition
micro_transition
```

No raw level, rank, rank numerator, source release date, price, return
magnitude, reward, action, or outcome is written.

## Controls

Every control rebuilds affected primitives, ranks, transitions, and token
serialization independently:

1. `h41_stale_one_release` uses one additional prior physical H.4.1 release;
2. `rrp_stale_one_operation` uses one additional prior complete operation
   inside the same segment;
3. `lattice_cohort_swap` swaps coarse/fine signed and unsigned quantities
   before aggregation;
4. `cascade_delay_37` uses the Source-D row 37 five-minute positions earlier
   inside the same UTC month; the first 37 positions are control-invalid and
   never borrow across a month; and
5. `macro_relation_mask` replaces all four macro fields on every history line
   and then sets the macro transition to its mixed value.

The two stale controls are causal diagnostic states. On a primary
sequence-ready boundary they may use a value older than the primary freshness
cap, but never a future value. If the additional predecessor does not exist,
the control line is safety-invalid. Controls never replace the primary and
cannot authorize economics.

Each full control token serialization must differ from primary and from every
other control.

## Append replay

Append replay uses eight fixed availability cutoffs:

```text
2020-06-30T23:59:59Z
2020-12-31T23:59:59Z
2021-06-30T23:59:59Z
2021-12-31T23:59:59Z
2022-06-30T23:59:59Z
2022-12-31T23:59:59Z
2023-06-30T23:59:59Z
2023-12-31T23:59:59Z
```

For each cutoff, create physical in-memory prefixes:

- H.4.1 `available_at_utc <= cutoff`;
- ON RRP `result_available_at_utc <= cutoff`; and
- micro `date+5m <= cutoff`.

Rebuild from each prefix. Every already formable primary primitive, macro
selection, rank, transition, readiness flag, and token row must equal the
corresponding full-build row byte-for-byte. Comparing a full build to itself
or merely filtering a full result is forbidden.

## Gate order and denominator

The official runner evaluates exactly:

1. protocol/source/header/manifest/type/clock validation;
2. annual exact micro-grid join at least 99.0%;
3. annual core-valid share at least 95.0%;
4. sequence-ready counts: 2020 at least 750; 2021–2023 each at least 1,000;
   every post-warm-up quarter at least 225;
5. every full post-warm-up quarter forced-flat share at most 8.0%;
6. every field has at least two categories with at least 3.0% annual share,
   and no category exceeds 94.0%;
7. each year has at least 5.0% liquidity-support and 5.0%
   liquidity-restrict states;
8. each year has at least 10.0% aggregate micro-buy and 10.0% aggregate
   micro-sell states;
9. each year has at least 7.5% cascade follow-through and 7.5% absorption;
10. each year has at least 120 exact eleven-field signatures and no signature
    exceeds 10.0%;
11. every adjacent-year per-field Jensen-Shannon divergence is at most 0.30;
12. all five controls are mutually distinct and differ from primary;
13. all eight true append replays are byte-identical; and
14. every forbidden counter is zero.

Gates 6–11 use only sequence-ready, current-core-valid rows in the named UTC
boundary year. Safety rows and position are excluded. JSD uses the full frozen
vocabulary, base-two logarithms, and zero contribution for absent mass.

Gate evaluation is short-circuiting for authorization but the report may
materialize all source-only diagnostics already computed without opening new
evidence. No failed gate may be repaired.

## Report decision

Exact decisions:

```text
pass
  authorize_stage_0_5_reward_evaluator_freeze

fail
  retire_lamb21_unchanged_before_rewards
```

A pass is source support only. It is not an alpha, trade, model, return, CAGR,
MDD, or deployability result.
