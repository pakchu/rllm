# ECRL-1 synthetic gate: immutable failure record

## Decision

ECRL-1 is **retired**. Its single preregistered Gemma 4 E2B QLoRA run did not
pass the frozen synthetic gate. No repair run, threshold change, checkpoint
substitution, historical SEC body access, market-data access, reward access, or
economic evaluation is authorized.

The immutable result is:

```text
results/edgar_claim_relation_language_synthetic_gate_2026-07-25.json
file SHA-256 9d872a023097cb321b1e8c5488c9a4209c5fe75bc66b6e6e97a2bfe321daa483
self-hash   afd25fbcc5982416d7847156b7cd17e2b657a101f4a63c0cb7e6fcec319dbe6f
```

Every outcome-boundary counter is zero.

## Frozen run

- base: `google/gemma-4-E2B-it` at revision
  `3e22461a4911541c564711279de12583ff9845a7`
- adaptation: 4-bit NF4/double-quantized QLoRA, rank 16, alpha 32,
  dropout 0.05, BF16
- optimization: 4,096 synthetic rows, 256 optimizer steps, gradient
  accumulation 16
- candidate checkpoints: steps 64, 128, 192, and 256
- selected by the frozen calibration ranking: step 192
- selected adapter size: 21,466,545 bytes
- selected adapter was not copied to `selected/`, because the final gate failed

Calibration exact/parse results:

| step | exact | parse |
|---:|---:|---:|
| 64 | 460/512 (89.84%) | 493/512 (96.29%) |
| 128 | 475/512 (92.77%) | 501/512 (97.85%) |
| 192 | 482/512 (94.14%) | 503/512 (98.24%) |
| 256 | 482/512 (94.14%) | 503/512 (98.24%) |

Step 192 won the preregistered tie-break over step 256.

## Final gate

The model passed the 98% aggregate exactness requirement:

```text
1,241 / 1,248 = 99.4391% exact
1,246 / 1,248 = 99.8397% parsed
```

It also passed all field-code minimums, all key-family minimums, all 32 guard
rejections with zero model calls, all 16 relation-contrast groups, the memory
caps, adapter-size cap, and filesystem cap.

Four frozen checks failed:

| check | observed | required |
|---|---:|---:|
| parse/evidence validity | 1,246/1,248 | 1,248/1,248 |
| swap invariance | 255/256 pairs | 256/256 |
| both swap variants exact | 253/256 pairs | at least 254/256 |

The adversarial split was 735/736 exact with 736/736 parsed. The swap split was
506/512 exact with 510/512 parsed. Diagnostic scenario misses were concentrated
in `CONDITIONAL_UP` (76/78 exact) and `REVERSE_UP` (73/78 exact), but aggregate
exactness itself still passed.

Peak CUDA memory remained within the preregistered limits:

```text
training allocated 18,115,151,872 bytes
training reserved  23,129,489,408 bytes
inference allocated 12,589,342,208 bytes
inference reserved  12,664,700,928 bytes
```

## Consequence

This result shows that the relation language was learnable at high synthetic
accuracy, but not robust enough for the deliberately exact evidence and
surface-invariance boundary. It is not alpha evidence and cannot justify
historical or economic testing. The contingent historical-support builder was
deleted uncommitted. Subsequent alpha work must use a genuinely independent,
precommitted mechanism rather than repair ECRL-1 after observing this failure.
