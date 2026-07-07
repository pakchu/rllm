# REX selector Gemma4 LoRA load report on RTX 3060 Ti

Date: 2026-07-07
Host GPU: NVIDIA GeForce RTX 3060 Ti, 8 GiB VRAM
Driver after reboot: 580.159.03
CUDA reported by `nvidia-smi`: 13.0
PyTorch runtime: 2.9.0+cu128

## Summary

After reboot, CUDA itself is healthy, but the current REX selector checkpoint does
not reliably load and run inference on the 8 GiB RTX 3060 Ti.

The blocker is not the LoRA adapter size. The adapter is only about 202 MB
(`adapter_model.safetensors`). The blocker is the Gemma4 E4B base model loading
path used by the selector: `google/gemma-4-E4B-it` is exposed as a Gemma4
multimodal architecture with text, vision, and audio configuration/components,
not as a small text-only causal LM in this environment.

The live selector should therefore be treated as unavailable on this host until
one of the remediation paths below is implemented and verified.

## What works

CUDA and the driver are now normal after reboot:

```text
nvidia-smi: OK
Driver Version: 580.159.03
GPU: NVIDIA GeForce RTX 3060 Ti
VRAM: 8192 MiB
```

Both system Python and the project `uv` environment see CUDA:

```text
torch 2.9.0+cu128
cuda available: True
device count: 1
device 0: NVIDIA GeForce RTX 3060 Ti
```

The REX selector checkpoint exists:

```text
checkpoints/rex_regime_thesis_range_kimchi_label_gemma4_s32_2026-07-03/
  adapter_config.json
  adapter_model.safetensors   # ~202 MB
  tokenizer.json
  tokenizer_config.json
  chat_template.jinja
  checkpoint-32/
```

## What fails

### 1. Normal live path: `AutoModelForCausalLM(... device_map="auto")` + PEFT

The current live code reaches:

```python
base = AutoModelForCausalLM.from_pretrained(..., device_map="auto")
model = PeftModel.from_pretrained(base, adapter_dir)
```

This fails before inference with:

```text
TypeError: unhashable type: 'set'
```

Root cause observed during inspection:

```python
type(base._no_split_modules) == set
base._no_split_modules == {
    "Gemma4AudioLayer",
    "Gemma4VisionEncoderLayer",
    "Gemma4TextDecoderLayer",
}
```

Current PEFT/accelerate dispatch expects hashable module-class entries and
trips over the Gemma4 `_no_split_modules` representation.

### 2. Normalizing `_no_split_modules` gets past the first error but still fails

Converting the set to a list moves past the `unhashable type: 'set'` error, but
PEFT then fails while updating the offload index:

```text
KeyError: 'base_model.model.model.model.audio_tower.layers.0.feed_forward1.ffw_layer_1'
```

This indicates the model is already in a CPU/disk offload state and PEFT adapter
injection is not clean for the Gemma4 multimodal module tree on this host.

### 3. 4-bit single-GPU load still OOMs on 8 GiB

A 4-bit bitsandbytes attempt with the whole model on GPU failed:

```text
torch.OutOfMemoryError: CUDA out of memory.
Tried to allocate 5.25 GiB.
GPU total capacity: 7.66 GiB
```

This shows that even quantized loading is tight or infeasible with the current
full Gemma4 checkpoint/wrapper and PEFT target set on an 8 GiB card.

### 4. CPU/GPU mixed offload loads slowly but forward is not usable

A mixed CPU/GPU offload attempt with an offload folder eventually loaded PEFT,
but it took about 169 seconds and the forward pass failed:

```text
KeyError: 22
```

The stack trace points into Gemma4 shared key/value attention state handling
(`shared_kv_states`). This path is not acceptable for live trading even if it
could be made to run, because the load/inference latency would be too high.

### 5. Direct `Gemma4ForCausalLM` class is not a clean fix

Trying `transformers.models.gemma4.modeling_gemma4.Gemma4ForCausalLM` directly
avoided some multimodal tower attributes, but the checkpoint key structure did
not match cleanly. The loader reported many missing/unexpected keys and PEFT
failed during 4-bit adapter injection:

```text
AttributeError: 'Parameter' object has no attribute 'compress_statistics'
```

This is not currently a safe live workaround.

## Why LoRA is not the main cause

LoRA adds trainable low-rank adapter weights on top of a base model. In this
checkpoint the adapter file is around 202 MB, which is small relative to the
base Gemma4 E4B model and runtime memory overhead.

The practical VRAM pressure comes from:

- Gemma4 E4B base weights
- Gemma4 multimodal wrapper components/configuration
- PEFT adapter injection/dispatch
- activation and attention/KV runtime memory
- CPU/disk offload coordination when the model does not fit fully on GPU

So the issue is better described as:

> The current Gemma4 E4B LoRA selector checkpoint is not compatible with reliable
> live inference on the 8 GiB 3060 Ti using the current transformers/PEFT/accelerate
> loading path.

Not:

> LoRA made the model much larger than the base model.

## VRAM expectation

Based on observed behavior:

- 8 GiB: not reliable with current checkpoint/loading path
- 12 GiB: may be possible with careful quantization/offload, but should be verified
- 16 GiB+: likely practical for live selector inference
- 24 GiB: preferred for stable low-latency operation

These are operational expectations, not formal model requirements.

## Impact on live trading

The live portfolio process is running with `--rex-selector-adapter-dir`, but the
selector is lazy-loaded only when a REX candidate reaches the selector path.
Because the current selector config is fail-closed by default, a load/inference
failure at signal time will block an otherwise valid REX candidate.

Operationally, until this is fixed, the REX LLM selector should be considered a
potential fail-closed blocker on this host.

## Remediation options

1. **Use a larger GPU for the current checkpoint**
   - Minimum practical target: 16 GiB VRAM.
   - Re-run the same forced candidate-logprob inference test before enabling live.

2. **Create a text-only selector artifact**
   - Use a text-only base model/checkpoint architecture.
   - Avoid Gemma4 multimodal wrapper and audio/vision tower dispatch.
   - This is the cleanest long-term path for a low-latency live selector.

3. **Use a smaller selector model**
   - Train or distill to a smaller text-only model that fits 8 GiB reliably.
   - Must preserve the bounded `TRADE`/`ABSTAIN` contract.

4. **Remote selector inference**
   - Run the current selector on another GPU host and call it over a bounded local/API interface.
   - Must fail closed or explicitly fail open according to the live risk decision.

5. **Disable or fail-open the selector temporarily**
   - This is an explicit trading/risk decision, not a technical fix.
   - If chosen, document the divergence from research assumptions.

## Minimal regression check for future fixes

A valid fix must pass all of the following on the target live host:

1. `nvidia-smi` succeeds.
2. `torch.cuda.is_available()` is true in the `uv` environment.
3. The REX adapter loads without CPU/disk offload errors.
4. A forced candidate-logprob call over labels `ABSTAIN` and `TRADE` completes.
5. GPU memory usage and inference latency are recorded.
6. The live process can execute one cycle without selector errors.

