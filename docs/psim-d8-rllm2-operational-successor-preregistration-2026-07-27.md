# PSIM-D8-RLLM2 operational-only successor preregistration

Date: 2026-07-27 KST

Status: frozen source-only successor; no RLLM2 model inference or market
outcome opened

Candidate: `PSIM-D8-RLLM2`

## Why a successor is valid

`PSIM-D8-RLLM1` is terminal and will not be rerun. Its one official attempt
loaded all 2,076 tensors but failed during scorer construction because the
runner incorrectly required a populated `hf_device_map`. It produced:

- zero model forwards;
- zero challenge predictions;
- zero challenge statistics;
- zero market or funding reads;
- zero rewards or economic metrics.

Therefore no model response, proposal-recovery result, or market outcome is
available to tune against. A separately named and preregistered operational
successor can correct that runtime assertion without changing the scientific
test.

## Immutable predecessor

- RLLM1 execution commit:
  `ce9ba77782ff0cc34411d60dc1ba7def5bea707f`
- RLLM1 terminal-record commit:
  `8ec8d4711900f405a206b1980a51fdcd582a1415`
- RLLM1 runner SHA-256:
  `931a9d5a888e4e821023a177790915a0b632762fb23fc90c17268fcf08119b5d`
- RLLM1 attempt SHA-256:
  `a325fb09286cf921e5b9e1d65e4655a03bde11058aa09a6fe0cd5d1fc79c3179`
- RLLM1 failure SHA-256:
  `02728096681f058144c12090cfa5876a973fb1cbd5146e35d59e2aa260dca812`
- RLLM1 failure result hash:
  `b0a40fa9904dd9b7877b3b64c9f382999d0b24a75a4edbcc687ccfe8b424fe69`

## Unchanged scientific contract

RLLM2 inherits the complete RLLM1 scientific payload under contract hash:

```text
59a7c1dd03155d8552614e4886087ca1dd08db4cc8c8953257c2f6f68d28af23
```

Unchanged:

- frozen D8 authority and selected-subcard selector;
- exact `google/gemma-4-E4B-it` revision and file hashes;
- runtime versions and 4-bit NF4/BF16 configuration;
- all redaction, source, and policy prompts;
- 128-case memorization roster;
- every true ID, decoy ID, and A-H mapping;
- case-roster hash
  `5065cd58322aee8f38f11ec2c4a186fb1a7ba8133aa2b2bb0182f67322a8bf39`;
- one-forward, no-generation scoring;
- exact one-sided binomial families and `1/8` chance;
- Bonferroni threshold `p < 0.01 / 3`;
- prompt limit, VRAM cap, chronology, controls, and economic gates.

There is no source resampling, prompt repair, threshold change, model swap, or
market-informed change.

## Sole operational delta

Both versions explicitly pass:

```python
device_map={"": 0}
```

RLLM1 incorrectly required `hf_device_map` to be nonempty afterward.

RLLM2 instead requires:

1. exact model class `Gemma4ForConditionalGeneration`;
2. `is_quantized == true`;
3. `model.device == cuda:0`;
4. the first model parameter is on `cuda:0`;
5. if `hf_device_map` is nonempty, every target resolves only to CUDA device
   zero and none resolves to CPU, disk, or meta;
6. an empty advisory `hf_device_map` is accepted.

The actual model forward remains the final placement proof.

## New one-shot paths

- attempt:
  `results/psim_d8_rllm2_base_memorization_gate_attempt_2026-07-27.json`
- result:
  `results/psim_d8_rllm2_base_memorization_gate_2026-07-27.json`

The output path is not configurable. A clean pushed HEAD is required, and the
attempt sentinel is created before weight loading. Any failure after that
sentinel consumes RLLM2.

## Preregistration artifact

- path:
  `results/psim_d8_rllm2_operational_successor_preregistration_2026-07-27.json`
- SHA-256:
  `85ede8a56393b11f4f1ced7e304adb3c2639132c1f0b008ed973aae92af9ef54`
- manifest hash:
  `c9b8a7527d90e8de3b1aeadac834c4b9d7a97bc3358c08256f79fa24fc18266c`

This preregistration reads only the three committed RLLM1 JSON artifacts. It
loads no model and opens no market, funding, reward, test, or eval payload.

## Next authorized action

Implement, test, independently review, commit, and push the RLLM2
operational-only runner. Only then may its one official base memorization
attempt execute.
