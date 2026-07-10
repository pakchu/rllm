# Gemma4 12B coder llama.cpp / REX LoRA fine-tune note (2026-07-10)

## Target

Evaluate `yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF` as a REX selector candidate and complete a local fine-tuning smoke path.

## Installed / downloaded

- `llama.cpp`: `/home/pakchu/tools/llama.cpp`, CUDA build, commit reported by binary as `07d9378`.
- Binaries built: `llama-cli`, `llama-server`, `llama-finetune`, `llama-export-lora`.
- GGUF: `models/gguf/gemma4-coding-Q4_K_M.gguf` (~6.9 GiB).
- HF safetensors base: `models/hf/gemma4-12b-coder-raw/model.safetensors` (~23 GiB).
- Patched local Transformers config: `models/hf/gemma4-12b-coder-patched`.

## llama.cpp result

- `llama-cli` loaded Q4_K_M on RTX 5090; observed prompt throughput around 411 tok/s and generation around 105 tok/s before interruption.
- `llama-finetune` against the Q4 GGUF failed in ggml backward graph construction:
  `GGML_ASSERT(!node->view_src || node->op == GGML_OP_CPY || ...) failed`.
- Interpretation: current llama.cpp finetune path is still WIP/FP32-oriented and did not support this quantized Gemma4 GGUF for training.

## Working fine-tune path

Transformers 5.7.0.dev0 contains Gemma4 classes but does not recognize the upstream config's `gemma4_unified` model type directly. A local patched config maps:

- `model_type`: `gemma4_unified` -> `gemma4`
- `text_config.model_type`: `gemma4_unified_text` -> `gemma4_text`
- architecture: `Gemma4ForCausalLM`

`training/train_text_sft.py` now supports:

- `--lora-target-modules` with comma list, `all-linear`, or `regex:<pattern>`.
- `--completion-only-loss/--no-completion-only-loss`.

Final command used language-model-only LoRA targets to avoid Gemma4 audio/vision wrappers and disabled completion-only loss because TRL's completion mask was off by one token with this tokenizer:

```bash
PYTHONPATH=. python -m training.train_text_sft \
  --model-name models/hf/gemma4-12b-coder-patched \
  --train-jsonl data/rex_regime_thesis_range_kimchi_label_train_2021_2024.jsonl \
  --output-dir checkpoints/rex_regime_thesis_gemma4_12b_coder_patched_lora_s16_256balanced_4bit_textloss_2026-07-10 \
  --max-samples 256 --sample-mode balanced --max-seq-length 1024 \
  --max-steps 16 --per-device-train-batch-size 1 --gradient-accumulation-steps 8 \
  --learning-rate 2e-5 --lora-r 8 --lora-alpha 16 --lora-dropout 0.05 \
  --lora-target-modules 'regex:^model\.language_model\.layers\.[0-9]+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))$' \
  --no-completion-only-loss --load-in-4bit --seed 42
```

## Fine-tune output

Final adapter:

`checkpoints/rex_regime_thesis_gemma4_12b_coder_patched_lora_s16_256balanced_4bit_textloss_2026-07-10`

Training evidence:

- Rows: 256 balanced (`TRADE=128`, `ABSTAIN=128`).
- Steps: 16.
- Runtime: 128.6s.
- Train loss: 4.236.
- Loss trend: first logged step 4.699, final logged step 3.980.
- Mean token accuracy rose roughly 0.391 -> 0.428.
- Adapter file: `adapter_model.safetensors` (~65.7 MiB).

## Smoke validation

Adapter reload with 4bit base succeeded. Candidate-logprob smoke on two train rows wrote:

`results/gemma4_12b_coder_rex_lora_s16_4bit_smoke_2026-07-10.json`

Both rows predicted `ABSTAIN` despite `TRADE` targets, so this is an installation/fine-tuning path validation only, not a trading-performance win.

## Caveats

- This model's upstream config/checkpoint naming is version-sensitive. Prefer keeping the patched local config until Transformers supports `gemma4_unified` directly.
- The patched architecture still emits missing/unused audio/vision warnings. The LoRA regex intentionally targets only `model.language_model.layers.*` projections.
- Completion-only loss is currently disabled for this checkpoint due TRL/tokenizer mask mismatch; fix before any serious selector evaluation.
- No backtest performance claim is attached to this 16-step smoke adapter.
