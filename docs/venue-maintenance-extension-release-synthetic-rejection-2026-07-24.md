# VMER-2 synthetic rejection

## Decision

**Retire VMER-2 unchanged.**

The exact preregistered Gemma 4 E2B adapter failed its untouched synthetic
gate. Historical Statuspage transport, historical semantic inference,
novelty, BTC market data, comparators, and economics remain unauthorized.
The ontology, prompt, synthetic examples, LoRA schedule, checkpoint choice,
memory limits, and pass thresholds may not be repaired after observing the
sealed synthetic result.

Result:

```text
results/venue_maintenance_extension_release_synthetic_gate_2026-07-24.json
SHA256 a6c94ae35400e20043a877459381321d26b9b30c8532e8f62a4528676505427b
manifest f36195ada72d50aee82333ad52b142a33f7a4a90060d94604c6077831987b5a5
```

## Run

| Item | Result |
|---|---:|
| policy | `VMER-2` |
| base | `google/gemma-4-E2B-it` |
| base revision | `3e22461f65e89153144f8adb70e3b8c2cc9845a7` |
| train rows | 384 |
| optimizer steps | 48 |
| checkpoints | 12, 24, 36, 48 |
| selected checkpoint | 36 |
| training elapsed | 168.43 s |
| complete run elapsed | 1875.38 s |
| first-step mean loss | 0.976826 |
| last-step mean loss | 0.010174 |
| training peak allocated | 18.41 GiB |
| training peak reserved | 25.32 GiB |

The frozen calibration rank selected step 36 using exact count, minimum
per-class exact share, malformed count, and then earliest step:

| Step | Exact | Share | Minimum class | Malformed |
|---:|---:|---:|---:|---:|
| 12 | 102 / 144 | 70.83% | 37.50% | 0 |
| 24 | 121 / 144 | 84.03% | 72.92% | 0 |
| 36 | 133 / 144 | 92.36% | 85.42% | 0 |
| 48 | 132 / 144 | 91.67% | 85.42% | 0 |

No adversarial or swap row was opened before checkpoint selection.

## Untouched final gate

Adversarial:

| Class | Exact | Share | Required |
|---|---:|---:|---:|
| `MATERIAL_EXTENSION_COMPLETED` | 48 / 48 | 100.00% | at least 98% |
| `CONTRADICTORY` | 44 / 48 | 91.67% | **at least 98%** |
| `UNSUPPORTED` | 41 / 48 | 85.42% | **at least 98%** |
| overall | 133 / 144 | 92.36% | — |

Swap:

```text
48 / 48 pairs invariant
46 / 48 pairs both exact
92 / 96 rows exact
```

The adapter over-classified several unsupported windows as contradictory.
It also accepted three contradictory windows as completed material
extensions and mapped one contradictory window to unsupported. The swap
errors were two unsupported template families, with both orderings producing
the same wrong contradictory result. Order invariance therefore passed while
the frozen 98% pair-exactness gate failed.

All thirteen prompt-injection guard rows were exact with zero model calls.
The selected adapter produced strict-parseable output for all 240 final rows
and strictly beat the unadapted base, 225 exact rows versus 93. Those partial
successes do not override any failed conjunctive gate.

Failed checks:

```text
adversarial_contradictory_exact
adversarial_unsupported_exact
swaps_unsupported_exact
swap_exact
training_peak_reserved
```

## Memory failure

| Metric | Observed | Ceiling |
|---|---:|---:|
| training peak allocated | 18.41 GiB | 24.00 GiB |
| training peak reserved | 25.32 GiB | **24.00 GiB** |
| inference peak allocated | 11.72 GiB | 13.00 GiB |
| inference peak reserved | 11.87 GiB | 13.25 GiB |

Training reserved memory exceeded the preregistered ceiling. Inference stayed
within its separate live-inference ceilings.

## Evidence boundary

The complete run used generated synthetic text only:

```text
candidate history rows opened       0
candidate detail objects opened     0
candidate update bodies opened      0
historical semantic model calls     0
BTC market rows read                0
funding rows read                   0
future-return rows read             0
return/PnL fields read              0
comparator rows parsed              0
2024+ candidate rows read           0
```

VMER-2 therefore has no return, CAGR, strict MDD, trade-count, novelty, or
alpha claim. Failure occurred before every historical source and economic
field.

## Research implication

The adapter learned the positive completion class and exact output grammar,
but did not robustly separate contradictory from unsupported compositions.
This is a semantic-boundary failure, not evidence that scheduled maintenance
extensions lack market impact.

The next search must use a genuinely different causal object or mechanism.
It may not patch VMER-2, lower the 98% gate, choose a checkpoint after reading
the final splits, or open historical outcomes. The four rejected checkpoints
were deleted after their file manifests and hashes were preserved in the
committed result.
