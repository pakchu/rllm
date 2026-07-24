# EBOC-72 synthetic rejection

## Decision

**Retire EBOC-72 unchanged.**

The exact preregistered Gemma 4 E2B adapter failed its untouched synthetic
gate. Historical SEC filing transport, semantic inference, novelty, market
data, and economics remain unauthorized. Prompt, ontology, examples, LoRA,
checkpoint, memory limits, and thresholds may not be repaired.

Result:

```text
results/sec_edgar_bitcoin_operational_capacity_synthetic_gate_2026-07-24.json
SHA256 12cc1bf61d1dc422a1f4cf7500cd4f53f3e765bd00d65dafc54fd630026fa8d9
manifest 7e9d769347b7c5b1dccc7550164141df21b759f23931989d6b95cc7a9a99d605
```

## Run

| Item | Result |
|---|---:|
| base | `google/gemma-4-E2B-it` |
| base revision | `3e22461f65e89153144f8adb70e3b8c2cc9845a7` |
| train rows | 512 |
| optimizer steps | 64 |
| checkpoints | 16, 32, 48, 64 |
| selected checkpoint | 32 |
| training elapsed | 230.59 s |
| complete run elapsed | 994.92 s |
| first-step mean loss | 0.652267 |
| last-step mean loss | 0.000284 |
| training peak allocated | 16.85 GiB |
| training peak reserved | 19.30 GiB |

Calibration was perfect at steps 32, 48, and 64. The frozen tie-break selected
the earliest, step 32:

| Step | Exact | Share | Malformed |
|---:|---:|---:|---:|
| 16 | 99 / 128 | 77.34% | 0 |
| 32 | 128 / 128 | 100.00% | 0 |
| 48 | 128 / 128 | 100.00% | 0 |
| 64 | 128 / 128 | 100.00% | 0 |

No final-test row or output participated in checkpoint selection.

## Untouched final gate

Adversarial:

| Class | Exact | Share | Required |
|---|---:|---:|---:|
| `CAPACITY_ONLINE` | 48 / 48 | 100.00% | at least 95% |
| `CAPACITY_OFFLINE` | 48 / 48 | 100.00% | at least 95% |
| `UNSUPPORTED` | 48 / 48 | 100.00% | at least 97% |
| `MIXED` | 46 / 48 | 95.83% | **100%** |
| overall | 190 / 192 | 98.96% | at least 95% |

The two mixed failures were:

```text
adversarial:MIXED:015 expected MIXED|NONE, got CAPACITY_OFFLINE|S2
adversarial:MIXED:016 expected MIXED|NONE, got UNSUPPORTED|NONE
```

Swap:

```text
64 / 64 pairs invariant
63 / 64 pairs both exact
```

The failed pair was the same mixed template family:

```text
swap:MIXED:015:v0 expected MIXED|NONE, got CAPACITY_OFFLINE|S2
swap:MIXED:015:v1 expected MIXED|NONE, got CAPACITY_OFFLINE|S2
```

All eight prompt-injection rows were rejected with zero model calls. All
twelve EBCT balance-sheet and twelve BPAX customer-access controls were
`UNSUPPORTED`.

## Memory failure

The frozen inference ceilings also failed:

| Metric | Observed | Ceiling |
|---|---:|---:|
| peak allocated | 11.69 GiB | 7.00 GiB |
| peak reserved | 11.78 GiB | 7.25 GiB |

Training remained below its 24 GiB allocated and reserved ceilings.

This is not reinterpreted as a 3060 Ti-compatible pass. The selected adapter
was evaluated in the exact post-training, multi-checkpoint process required by
the contract, and that process exceeded the frozen live-inference budget.

## Evidence boundary

The entire run used generated synthetic text only:

```text
historical SEC bodies opened       0
historical windows created         0
historical semantic model calls    0
BTC market rows read               0
funding rows read                  0
future-return rows read            0
comparator rows parsed             0
2024+ source rows read              0
```

Therefore EBOC has no return, CAGR, strict MDD, trade count, or alpha claim.
The failure occurred before source incidence and before every economic field.

## Research implication

The adapter learned direct online/offline and abstention semantics, but
perfect calibration did not generalize to all held-out mixed compositions.
The failure is exactly why the mixed and swap gates were frozen at 100%.

The next alpha search must move to a new semantic object or source/mechanism,
not patch this mixed family, loosen memory, choose step 48/64 after seeing the
test, or open historical outcomes. EBOC checkpoints may be deleted after this
rejection result and their hashes are committed.
