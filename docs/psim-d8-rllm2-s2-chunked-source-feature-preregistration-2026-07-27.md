# PSIM-D8-RLLM2-S2 chunked source-feature preregistration

Date: 2026-07-27

Status: **preregistered; no model output or market/outcome access**

## Why S2 exists

S1 is terminal and is not resumed. Its exact one-pass SDPA operator failed on
the frozen 29,727-token policy prompt at source row 341 after completing 341
rows. It opened no market, funding, reward, or outcome payload.

S2 is a new source-only operational successor. It reuses no S1 checkpoint or
partial model output. The model, revision, quantization, tokenizer, source
roster, redaction, prompts, A–F relation mapping, and later economic contract
remain unchanged.

## Sole operational delta

The exact chat-rendered prompt is tokenized once and scanned in contiguous
**512-token** chunks. The size is fixed from the frozen Gemma4 text
`sliding_window=512`; it is not tuned from any model or market result.

For every chunk:

- `use_cache=true`;
- the exact cache returned by the prior chunk is supplied;
- position IDs equal the original absolute token positions;
- the attention mask is the complete prefix through the chunk end;
- text-only multimodal token types are omitted;
- no token is truncated, summarized, reordered, or generated.

The source embedding is the final chunk's last hidden token. Relation logits
use the exact frozen LM head and final-logit softcapping on that same last
hidden token. Cache state is discarded between logical prompts.

## Pre-market equivalence gate

Before full extraction, ten fixed source rows compare one-pass and chunked
operators for both policy embeddings and relation-teacher logits. The roster
spans short prompts, deterministic rank points, and prompts nearest 512, 1024,
2048, 4096, and 8192 tokens.

Required per case:

- exact token reconstruction;
- embedding cosine similarity at least `0.99999`;
- embedding RMS absolute delta at most `0.01`;
- embedding maximum absolute delta at most `0.05`;
- identical relation code;
- relation mean absolute delta at most `0.01`; and
- relation maximum absolute delta at most `0.03`.

The original OOM row 341 is then processed chunked as a capacity-only gate.
Both prompts must complete within the 30 GiB allocated-memory cap. Its gate
outputs are not reused in full extraction.

## Failure and resume

Any post-attempt equivalence, capacity, placement, memory, source-integrity,
token, or artifact failure terminally rejects S2. S2 may not resume before an
equivalence/capacity pass artifact exists.

After the gates pass, full extraction uses fresh row-granular S2 checkpoints.
Only a contiguous verified prefix may resume. An ambiguous in-flight row is
terminal rather than replayed.

Final artifacts are staged and hash-verified. The source-seal pass is published
last and is the only S2 artifact that may authorize opening 2020 train outcomes.
2021, 2022, 2023, and all market/economic payloads remain closed during S2.
