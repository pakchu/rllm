# PSIM-D8-RLLM1 base memorization gate terminal failure

Date: 2026-07-27 KST

Status: terminal rejection; rerun prohibited

## Result

The official `PSIM-D8-RLLM1` base memorization attempt was launched from the
clean, pushed commit:

```text
ce9ba77782ff0cc34411d60dc1ba7def5bea707f
```

The source/tokenizer preflight passed and the one-shot attempt sentinel was
created. All 2,076 weight tensors then loaded. Before the scorer constructor
returned or any challenge forward began, the runner raised:

```text
RuntimeError: frozen model device map changed: {}
```

Consequently:

- model forwards started: 0;
- challenge predictions created: 0;
- challenge statistics computed: false;
- market/funding rows read: 0;
- rewards or economic metrics computed: 0;
- official success/result artifact created: false.

There is no return, CAGR, strict MDD, trade count, or alpha claim.

## Root cause

The runner explicitly loaded the entire model on CUDA device zero with
`device_map={"": 0}`. In this single-device path, Transformers did not
populate the optional `hf_device_map` advisory attribute. The runner
incorrectly required that advisory mapping to contain `{0}` and rejected the
observed empty dictionary after weight loading.

This was an operational validation defect, not a challenge result. It exposed
no proposal prediction and no market outcome.

## Immutable evidence

- attempt artifact:
  `results/psim_d8_rllm1_base_memorization_gate_attempt_2026-07-27.json`
- attempt artifact SHA-256:
  `a325fb09286cf921e5b9e1d65e4655a03bde11058aa09a6fe0cd5d1fc79c3179`
- attempt hash:
  `db2e1d7c5ce0bc7dbb061ff6f3e1d4a674d018db16dbf04a7509e04566d3a609`
- executed runner SHA-256:
  `931a9d5a888e4e821023a177790915a0b632762fb23fc90c17268fcf08119b5d`
- raw failure log:
  `results/psim_d8_rllm1_base_memorization_gate_failure_2026-07-27.log`
- raw failure log SHA-256:
  `2a3ef2ce55b8b668d41e5e7097168a1a619986e39c23e2087c59c2ef8fdc71ae`
- terminal failure artifact:
  `results/psim_d8_rllm1_base_memorization_gate_failure_2026-07-27.json`
- terminal failure artifact SHA-256:
  `02728096681f058144c12090cfa5876a973fb1cbd5146e35d59e2aa260dca812`
- terminal failure result hash:
  `b0a40fa9904dd9b7877b3b64c9f382999d0b24a75a4edbcc687ccfe8b424fe69`

## Scientific handling

The attempt marker is retained and `PSIM-D8-RLLM1` will not be rerun,
repaired, resampled, or silently relabeled.

A successor is permissible only as a separately preregistered candidate that:

1. preserves the exact D8 authority, model revision, source roster, prompts,
   candidate mappings, statistical tests, and market chronology;
2. changes only the erroneous runtime-placement assertion;
3. records this terminal failure as predecessor evidence;
4. creates separate one-shot attempt and result paths;
5. opens no market payload before passing its own base memorization gate.
