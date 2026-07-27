# PSIM-D8-RLLM2 base memorization gate result

Date: 2026-07-27 KST

Status: passed; source-feature construction only is authorized

## Verdict

The official one-shot base memorization gate completed on:

```text
197ba160c5231ca11e9228bc73574bb157903dad
```

Result:

```text
ACCEPT_PSIM_D8_RLLM2_BASE_MEMORIZATION_GATE_SOURCE_FEATURES_ONLY
```

The exact pretrained Gemma 4 E4B model did not recover proposal identity
above the preregistered corrected significance threshold in Ethereum,
Bitcoin, or the combined sample.

| Family | Correct | Trials | Accuracy | Exact one-sided p | Reject |
|---|---:|---:|---:|---:|---|
| Ethereum | 10 | 64 | 15.625% | 0.2746260401 | no |
| Bitcoin | 5 | 64 | 7.8125% | 0.9150077906 | no |
| Combined | 15 | 128 | 11.71875% | 0.6450678052 | no |

Chance is 12.5%. Rejection required `p < 0.01 / 3`, or approximately
0.0033333.

This passes only the pretrained-memory leakage gate. It is not an economic
alpha result and still authorizes no market access.

## Complete execution evidence

- 128 unique cases scored;
- 64 Ethereum and 64 Bitcoin;
- all logits for A-H finite;
- generated or decoded answer text: none;
- minimum input: 159 tokens;
- maximum input: 10,291 tokens;
- total measured forward time: 27.2975 seconds;
- maximum single forward: 1.2191 seconds.

The model had a strong output-code prior:

| Code | Predictions |
|---|---:|
| A | 1 |
| B | 8 |
| C | 7 |
| D | 12 |
| E | 15 |
| F | 5 |
| G | 12 |
| H | 68 |

This does not invalidate the gate because the true codes were frozen at
exactly 16 occurrences each across the combined sample. A constant or biased
code preference therefore remains calibrated to the 1/8 null. The observed
combined accuracy was 15/128, slightly below chance.

## Actual runtime

- model class: `Gemma4ForConditionalGeneration`;
- quantized: true;
- model device: `cuda:0`;
- first parameter device: `cuda:0`;
- observed `hf_device_map`: `{}`;
- empty advisory mapping accepted: true;
- CUDA allocation immediately after load: 9,323,639,296 bytes;
- peak allocated VRAM: 19,078,689,792 bytes;
- peak reserved VRAM: 25,492,979,712 bytes;
- preregistered allocated cap: 32,212,254,720 bytes;
- model load time: 6.2737 seconds;
- GPU: NVIDIA GeForce RTX 5090.

This confirms that the RLLM1 failure was the rejected advisory-map assertion,
not CPU or disk offload.

## Immutable artifacts

- attempt:
  `results/psim_d8_rllm2_base_memorization_gate_attempt_2026-07-27.json`
  - SHA-256:
    `e91b4c58797bd78d5062dff2c07d4363d8d897c8c3291620486f9c02aad42ea0`
  - attempt hash:
    `b83c227d38a959a6ae2405700b5ea7b268e13a958c7b7c8282108e8169a2c759`
- result:
  `results/psim_d8_rllm2_base_memorization_gate_2026-07-27.json`
  - SHA-256:
    `0abf3b5babe9e398e97721ddcc3e29b6d23cc742345cd5f804e78d507982818f`
  - result hash:
    `8debfe4b37a6be1f65b306cce5b1408bf21a01a7f316254e4b42c2529a851ce3`
- raw execution log:
  `results/psim_d8_rllm2_base_memorization_gate_2026-07-27.log`
  - SHA-256:
    `45103890f0b95561665f1afcd40487b196c4254df27d8c4871b0f5d6cc80a34f`
- executed runner SHA-256:
  `1188853c7df02459e388fb2e133a87656e7755b37dcb73d09c5c35fa24e66c4c`

## Access boundary

- market paths read: none;
- funding paths read: none;
- market rows parsed: 0;
- funding rows parsed: 0;
- rewards created: 0;
- economic metrics computed: 0;
- test outcomes opened: false;
- eval outcomes opened: false.

## Next authorized stage

Construct and seal source-only selected-subcard features, frozen Gemma
embeddings, and the relation teacher. Market outcomes remain closed until
those source-only artifacts and their controls are committed and reviewed.

Fresh verification:

- dedicated immutable-result tests: `6 passed`;
- RLLM1-terminal/RLLM2-lineage battery: `21 passed` in 7.68 seconds;
- independent artifact/statistics/CUDA verifier: `VERIFIED (PASS)`;
- WSL usage after execution: approximately 298 GiB.
