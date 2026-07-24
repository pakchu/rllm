# EBOC-72 machine preregistration

## Decision

The deterministic synthetic adaptation contract for **EBOC-72 — EDGAR
Bitcoin Operational Capacity Transition** is frozen and authorizes exactly one
synthetic-only Gemma 4 E2B LoRA run.

It does **not** authorize:

- fetching or decoding any historical SEC filing body;
- creating a historical EBOC label, clock, or signal;
- parsing any comparator row or timestamp;
- reading BTC price, funding, return, PnL, or reward data; or
- opening any 2024-or-later source or outcome.

Machine artifact:

```text
results/sec_edgar_bitcoin_operational_capacity_preregistration_2026-07-24.json
SHA256 5ce80db5875e282a1d173489bb910026b53f073f9c783c5140ee99fe3d72605d
contract da227af07d68626974dd69c8934fa5258d3ea14f697b5a4e7960a5a460dda391
manifest 76fe7387078dfd0c8783c889cdf81685ac2b3de8aa16c8854fb96b8c7a82901e
```

Generator and composer implementation:

```text
training/preregister_sec_edgar_bitcoin_operational_capacity.py
SHA256 f383c1938e7ddc1757806f583b0541567c7226c8b3ecb4925beff2f62c685a67
```

## Synthetic data

| Split | Rows | Per class | SHA-256 |
|---|---:|---:|---|
| train | 512 | 128 | `207c98d9a453e0913f5072732d81136a335a621120725d84f8734c66c7939630` |
| calibration | 128 | 32 | `718ea047bcddd4268afc267df54b81a26d5c01ece30386ace1fb58e5d9a28a86` |
| adversarial | 192 | 48 | `2e2c4fbcbf4894a034852f164db95721db840f416fa6e7f6f9fb81c8f491a14c` |
| swaps | 128 | 32 | `2f25aa4c0ce34d8581bff84bf34c3df58110c08918de3234dd30ecc874033cc2` |

The four classes are balanced:

```text
CAPACITY_ONLINE
CAPACITY_OFFLINE
UNSUPPORTED
MIXED
```

Every non-swap row has a unique redacted window. Train, calibration, and test
template families are disjoint, and no redacted train window is byte-identical
to a calibration or test window. The decision-bearing sentence templates also
have zero exact overlap across train, calibration, and test. The 64 swap pairs
change only synthetic
entity, ticker, date, and quantity surfaces and must reduce to identical
redacted windows and outputs.

The adversarial split contains:

- 8 deterministic prompt-injection guards that permit zero model calls;
- 12 EBCT balance-sheet negative controls;
- 12 BPAX customer-access negative controls; and
- 16 hard attribution, negation, completion, or temporal negatives.

## Gemma adaptation

Frozen base:

```text
google/gemma-4-E2B-it
revision 3e22461f65e89153144f8adb70e3b8c2cc9845a7
```

The local base files and runtime were hash/version verified. The adapter is
restricted to text-language-model `q_proj`, `k_proj`, `v_proj`, and `o_proj`
modules:

```text
target regex .*language_model.*\.(q_proj|k_proj|v_proj|o_proj)$
rank 8
alpha 16
dropout 0.05
trainable parameters 2,678,784
NF4 double quantization, BF16 compute
```

Training is completion-only causal loss with AdamW, learning rate `1e-4`,
weight decay `0.01`, batch one, gradient accumulation eight, four warmup
steps, cosine decay, gradient clipping at 1.0, and exactly 64 optimizer steps.
Only steps 16, 32, 48, and 64 may be checkpoints.

Calibration selects the checkpoint lexicographically by:

1. highest exact class-plus-evidence count;
2. highest minimum per-class exact share;
3. lowest malformed count; and
4. lowest checkpoint step.

The adversarial and swap splits remain unopened until checkpoint selection.
Any synthetic-gate failure retires the exact adapter without prompt, parser,
data, LoRA, checkpoint, memory, or threshold repair.

## Deterministic composer tests

The preregistration code also locks:

- fixed evidence roles: previous `S1`, target `S2`, next `S3`;
- same-time, same-issuer duplicate suppression;
- same-time, same-issuer directional-conflict suppression;
- a non-resetting 21-day issuer cooldown with the exact 21-day boundary
  eligible;
- no mutual observation among equal-ready-time issuers;
- lexicographic same-batch signal tie-breaking;
- one complete five-minute latency bar before entry;
- a fixed 72-hour hold; and
- global non-overlap.

## Evidence boundary

The artifact records:

```text
historical SEC bodies opened       0
historical windows created         0
historical semantic model calls    0
BTC market rows read               0
funding rows read                  0
future-return rows read            0
comparator rows parsed             0
2024+ source rows read              0
synthetic rows created            960
synthetic model calls               0
```

The next permitted unit is implementation and test of the frozen
trainer/evaluator. Only after that code is committed may the one synthetic-only
64-step run begin.
