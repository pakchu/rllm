# EBCT-72 Gemma 4 synthetic gate — rejected

## Decision

Retire the exact `EBCT-72` singleton before opening any SEC filing body or BTC
outcome. The frozen Gemma 4 extraction contract failed three of fifteen
model-generated controls. The preregistration forbids changing the prompt,
schema, parser, model, quantization, redaction, or label mapping after this
gate, so there is no repair run.

- result:
  `results/sec_edgar_bitcoin_constraint_synthetic_gate_2026-07-21.json`;
- result SHA-256:
  `04e0e032531f95761fe63b24454a763b09e5c6f9a7d3b4ace6f88ac6fa2a14f8`;
- result manifest:
  `a2f8b7044b5102415ae8b9a4ffa8b3815da2f38817b1ecf808ba4f6e45ba90c4`;
- runner:
  `training/run_sec_edgar_bitcoin_constraint_synthetic_gate.py`;
- runner SHA-256:
  `f727215882c1717f7d23215f83f04f8cad6ac01400eb5ea17d90b7ab9c600f87`.

This is a model-interface rejection, not an economic result.

## Results

| Check | Result |
|---|---:|
| literal controls | 17 |
| deterministic guard controls | 2 / 2 pass |
| Gemma 4 calls | 15 |
| exact label + role | **14 / 17** |
| parse + quote-valid model outputs | **12 / 15** |
| entity/date/amount swap invariance | pass |
| complete gate | **fail** |

The exact failing rows were:

1. `completed_sale`: expected
   `BTC_CONSTRAINT_DRAW / BTC_SALE`, but Gemma returned `BTC_SALE` in both
   `label` and `role`; the strict label parser correctly rejected it.
2. `pledged_collateral`: expected
   `BTC_CONSTRAINT_DRAW / BTC_PLEDGE`, but Gemma returned `BTC_PLEDGE` in both
   fields; the parser rejected it.
3. `mixed_draw_buffer`: expected `UNSUPPORTED / NONE`, but Gemma selected the
   sale fragment and again returned `BTC_SALE` as both label and role.

The other fourteen cases matched the frozen expected role. Both symmetric
prompt-injection strings were rejected by the deterministic guard without a
model call, and the two identity/date/amount variants rendered the same
redacted text and produced the same result.

## Runtime and 3060 Ti finding

The pinned [`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it)
revision loaded in 4-bit NF4 on one RTX 5090. Observed peak CUDA allocation was
`9,448,872,960` bytes and peak reservation was `9,508,487,168` bytes. That
exceeds an 8 GiB RTX 3060 Ti before live-process headroom, so this exact
Transformers multimodal loading path is **not suitable for the 3060 Ti 8 GB**.
Google describes E4B as 4.5B effective / 8B with embeddings on the official
[Gemma 4 page](https://developers.google.com/edge/litert-lm/models/gemma-4),
but does not guarantee Transformers 4-bit VRAM use.

An initial operational attempt failed before model construction because this
PyTorch build rejected integer device `0` for `reset_peak_memory_stats`. No
model inference or semantic result existed. The runner was changed only to set
the visible CUDA device and call the same statistics API without an argument;
the frozen model, prompt, cases, parser, and decision rules did not change.
The second attempt produced the committed semantic result above.

## Data boundary

```text
SEC filing bodies opened       = 0
historical semantic rows       = 0
BTC market rows                = 0
funding rows                   = 0
future-return/PnL fields       = 0
2024+ source rows              = 0
```

No profitability, absolute return, CAGR, strict MDD, or trade statistic exists
for EBCT-72. The frozen SEC source remains reusable for a genuinely new,
separately preregistered mechanism, but EBCT-72 itself is retired.
