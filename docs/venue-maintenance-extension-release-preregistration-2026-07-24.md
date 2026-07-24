# VMER-2 machine preregistration

## Decision

Freeze the outcome-blind machine contract for **VMER-2 — Venue Maintenance
Extension Release** and authorize exactly one synthetic-only Gemma 4 E2B LoRA
run.

This unit does not authorize:

- fetching or decoding a 2020–2023 Statuspage history row or maintenance body;
- creating a historical VMER class, clock, revelation, side, or trade;
- parsing a comparator row;
- reading a BTC bar, funding row, future return, PnL, or reward; or
- opening a 2024-or-later candidate source row or outcome.

Machine artifact:

```text
results/venue_maintenance_extension_release_preregistration_2026-07-24.json
SHA256 de5cc97ddd7b1c3bfb155a1d3e3cd11e501e43148047c6e8fd9b4d48100e5809
contract e583ce62b13516d7825c3454a226fe4dc990a693d5ecaebba40bdc97e416c303
manifest 05ebb83b40a3d2e9e0b0776d22e78f8282e229fd07970e2e4352a3ffd08e09ac
```

Implementation:

```text
training/preregister_venue_maintenance_extension_release.py
SHA256 2d457a0fecce6490ce7d0dbf59c50bb72f66ce762cb036706035c2607d38f06f
```

## Frozen source boundary

Only the official Coinbase Exchange and Kraken Statuspage pages bound in the
candidate boundary are eligible. The contract fixes:

- HTTPS GET, TLS 1.2 minimum, identity transfer encoding, 30-second timeout,
  three attempts, and backoffs of 0, 2, and 8 seconds;
- history pages 27 down through 11 on the frozen 2026-07-24 archive layout;
- raw SHA-256 and receipt manifest before parsing;
- raw month-object slicing and a year check before JSON decoding, so 2019 and
  2024 boundary-month rows are not materialized;
- exact `(venue, code)` duplicate suppression and retirement on a conflicting
  duplicate;
- requests to both per-code typed endpoints, with exactly one HTTP 200 and one
  HTTP 404 required;
- incident objects as typed negatives whose nested incident/update body is
  never JSON-decoded or materialized;
- `scheduled_maintenance` as the only eligible top-level object;
- page-ID verification and exact update-field allowlisting; and
- update availability at
  `max(created_at, display_at, updated_at)`, a 30-day revision-age ceiling,
  and a 15-minute fixed-point quiet interval.

Object-level planned dates, final state, resolution fields, components, and
object `updated_at` remain forbidden from the signal because current objects
can encode later revisions.

All four synthetic ledgers and the machine artifact are write-once. Any
existing target path causes failure before a byte is rewritten.

## Synthetic semantic contract

The model may emit only:

```text
MATERIAL_EXTENSION_COMPLETED|U1|U3|U5
UNSUPPORTED|NONE|NONE|NONE
CONTRADICTORY|NONE|NONE|NONE
```

For the material class, deterministic validation requires three distinct,
ordered, existing update labels with statuses `in_progress`, `in_progress`,
and `completed`. Prompt-injection language deterministically fails closed as
`CONTRADICTORY`.

Synthetic datasets:

| Split | Rows | Per class | SHA-256 |
|---|---:|---:|---|
| train | 384 | 128 | `c1e6a9f6f667451e741f4982989f24565e9983b720cc9f644f8ebfb61d5e98be` |
| calibration | 144 | 48 | `5b16f0ab7ca24eff3199f2de768a127f415286d84693510fea9790c3d9353d2e` |
| adversarial | 144 | 48 | `a610ff7d04b8a55d1c0104ecacc2ebc1a31207d5be3dd62b03e30fcdfe6ef721` |
| swaps | 96 | 32 | `150662f4e8de89f91db8c79cccdbfc6b8af3d6a93345d95172dac92af24e7ec2` |

Train, calibration, and test template families are disjoint. The 48 swap
pairs differ in synthetic venue/date/quantity/link surfaces but become
byte-identical after deterministic redaction.

## Gemma adaptation

Frozen base:

```text
google/gemma-4-E2B-it
revision 3e22461f65e89153144f8adb70e3b8c2cc9845a7
```

All local base files and runtime versions were hash-verified. The adapter is
restricted to text-language-model `q_proj`, `k_proj`, `v_proj`, and `o_proj`:

```text
LoRA rank 8
alpha 16
dropout 0.05
trainable parameters 2,678,784
NF4 double quantization
BF16 compute
```

The run uses completion-only causal loss, AdamW, learning rate `1e-4`, weight
decay `0.01`, batch one, gradient accumulation eight, four warmup steps,
gradient clipping at 1.0, and exactly 48 optimizer steps. Only steps 12, 24,
36, and 48 may be checkpoints.

Calibration selects lexicographically by:

1. highest exact class-plus-grounding count;
2. highest minimum per-class exact share;
3. lowest malformed count; and
4. lowest checkpoint step.

The unopened adversarial/swap gate requires at least 98% exactness in every
class, zero malformed outputs, every guarded prompt-injection row exact, at
least 98% swap-pair exact invariance, and strict improvement over the base
checkpoint. A failure retires the unchanged candidate.

## Frozen downstream gates

No threshold below may be changed after source incidence:

- source-only minimums: 12 train and 4 selection events, 8 and 3 active
  months, both venues with at least 3 train events each, fixed concentration
  and maximum-gap ceilings, and 100% lifecycle integrity;
- causal-market minimums: 8 train and 3 selection qualified events, at least
  40% retention, 288 strictly prior completed five-minute returns for RMS
  normalization, and `|z_impulse| >= 0.75`;
- execution: follow revelation side, enter the next five-minute open, hold
  exactly two elapsed hours, unit exposure, global non-overlap, no stop or
  take profit, 6 bps per side base cost, 10 bps stress cost, and exact funding;
- novelty: the eight raw-hash-bound comparator families, six-hour near window,
  maximum 50% near overlap, maximum 10% exact-entry overlap, and at least
  60-minute median nearest distance;
- economics: positive absolute return, wall-clock CAGR, global/pre-entry-high-
  water strict MDD at or below 20%, and CAGR/strict-MDD at least 3.0 in both
  train and selection under base cost;
- stress: positive return, strict MDD at or below 25%, and ratio at least 2.0;
  and
- 10,000 stationary event-return bootstrap replications with fixed seed.

The two premium comparator artifacts exposed during the retired VARR process
are explicitly forbidden and absent from the VMER cohort.

The machine contract also fixes every support denominator and ordering:
source concentration uses all split-contained source events before market
qualification; qualification retention uses all source events as denominator
and threshold-qualified, globally nonoverlapping clocks as numerator; and
novelty uses each comparator family's explicitly filtered primary
`entry_time`. Exact overlap is candidate containment, near overlap is
maximum-cardinality one-to-one matching, and nearest distance is calculated
per candidate within the same split. No family is pooled away.

Any later RLLM may train and select checkpoints only on precommitted
train-period causal folds. Selection-period and sealed-extension rewards are
both forbidden for training, tuning, ranking, or checkpoint selection.

## Evidence boundary

The generated artifact records:

```text
candidate history rows                 0
candidate detail objects               0
candidate update bodies                0
historical model calls                 0
comparator rows                        0
BTC market rows                        0
future rows                            0
funding rows                           0
return or PnL fields                   0
2024+ candidate rows                   0
synthetic rows                       768
synthetic model calls                  0
```

The next permitted unit is implementation and testing of the frozen
synthetic-only trainer/evaluator. Only after that code is committed may the
single 48-step run start.
