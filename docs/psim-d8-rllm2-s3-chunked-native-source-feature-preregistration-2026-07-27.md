# PSIM-D8-RLLM2-S3 chunked-native source-feature preregistration

Date: 2026-07-27

Status: **preregistered; no model output or market/outcome access**

## Purpose

S1's one-pass SDPA operator exceeded device memory on the frozen 29,727-token
policy prompt. S2 proved that a fixed-512 causal-cache scan is not numerically
equivalent to that one-pass operator for multi-chunk prompts. Both stages are
terminal.

S3 does not relax S2 thresholds and makes no one-pass-equivalence claim. It
defines the fixed-512 cache scan as a new, chunked-native source
representation before any 2020 or later outcome is opened.

## Frozen operator

The exact Gemma4 model, revision, 4-bit quantization, tokenizer, chat template,
1,461 source rows, prompts, redaction, selector, and A-F relation mapping stay
fixed.

Each exact prompt is tokenized once, without truncation, then scanned in
contiguous 512-token chunks with:

- `use_cache=true`;
- the exact prior returned cache;
- absolute original position IDs;
- the complete prefix attention mask;
- no multimodal token type IDs; and
- no generation or decoded trading text.

The embedding is the final chunk's final hidden token converted to float32.
Relation logits apply the frozen LM head and final-logit softcap to that hidden
token, then retain only the fixed A-F token IDs.

## Pre-market repeatability and capacity gate

The ten source-only cases preregistered before S2 execution are reused only as
an output-independent coverage roster. The complete roster and capacity case
are run once under each of two sequential, independently loaded model
instances. The first model is destroyed and CUDA is cleared before the second
load. For both policy and relation prompts, the two S3 scans must have:

- exact frozen token counts and reconstruction;
- an independent cache reset between repeats;
- finite `(2560,)` float32 embeddings;
- byte-identical embedding SHA-256 hashes;
- finite or canonical-NaN relation logits;
- byte-identical canonical relation-logit SHA-256 hashes; and
- identical predicted relation codes.

The original long row 341 is also scanned twice for both prompts. It must meet
the same exact-repeatability contract and stay at or below 30 GiB peak
allocated VRAM. Gate outputs are discarded and may not be reused by full
extraction, which uses a fresh third model load.

## Failure and resume

Any repeatability, capacity, placement, memory, source-integrity, or artifact
failure terminally rejects S3. Resume is forbidden before the gate passes.
After a pass, only a contiguous hash-verified prefix of fresh row-granular S3
checkpoints may resume; an ambiguous in-flight row is terminal.

Final source artifacts are staged and hash-verified. The pass result is
published last and is the only artifact allowed to open 2020 train outcomes.
No S1/S2 model output or checkpoint may be read.

## Frozen identity

- Preregistration SHA-256:
  `7edb7eeef115e0579099f9aa990648f1db1cde28053867c0ae1edb6c60deb196`
- Manifest hash:
  `263f0476ce887cad7b5e9c0174e02b6f714a8ce54029ef5788a6179ad1c396f3`
- Repeatability roster hash:
  `d960f57da0d5e8b37004643a20b39caa5a569278bd0385238e2845ce41585bcf`
- Capacity case hash:
  `5875ce6259a15bfc226d6b12ab50c7d6da7d4866011a7b87222a4e5e491c0ab0`
