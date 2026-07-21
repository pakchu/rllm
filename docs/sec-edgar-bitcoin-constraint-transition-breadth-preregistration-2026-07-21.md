# EBCT-72 — SEC Bitcoin constraint-transition breadth preregistration

## Decision

Freeze one singleton text-native candidate: **EDGAR Bitcoin Constraint
Transition Breadth (`EBCT-72`)**. The only next authorized operation is a
synthetic Gemma 4 extraction and memorization-control battery. Filing bodies,
historical semantic labels, BTC prices, funding, returns, PnL, and every
2024-or-later row remain unopened.

- executable contract:
  `training/preregister_sec_edgar_bitcoin_constraint_transition_breadth.py`;
- preregistration artifact:
  `results/sec_edgar_bitcoin_constraint_transition_breadth_preregistration_2026-07-21.json`;
- artifact SHA-256:
  `a9c55b98202b341ffb51bede731e5d2a2281d3851fbee604670868ea47470405`;
- contract hash:
  `7c52d3c0b6c5b2869ab90723d864e73f0f06097f657a67167f8557b420465e48`;
- manifest hash:
  `e2553638d6df1be9e9fd034f5fae837d38e0525bd48e34d74aaffc7b52cc4566`.

This is not an alpha or profitability claim. It freezes the hypothesis before
the first document body or market outcome is opened.

## Economic hypothesis

Public-company filings can disclose a change in how issuers use Bitcoin on
their balance sheets. A single filing is stale and issuer-specific; the
candidate therefore acts only when **three distinct issuers** change into the
same state within ten calendar days:

- `BTC_CONSTRAINT_DRAW`: completed or binding sale, pledge, use of sale
  proceeds, or forced Bitcoin liquidity;
- `BTC_CONSTRAINT_BUFFER`: completed non-Bitcoin financing that preserves
  Bitcoin, collateral release, explicit retention, or accumulation;
- mixed, planned, generic-risk, third-party, or unsupported language cannot
  change issuer state.

The model does not choose a trade. Numeric code maps buffer breadth to long and
draw breadth to short. This is intentionally different from sentiment
classification and direct LLM return prediction.

## Frozen official source

The source is the previously audited SEC EDGAR 8-K/6-K metadata clock for
2018–2023:

- 3,496 exact full-text document hits;
- 2,493 non-amendment accessions eligible for later semantics;
- 308 CIKs and 992 acceptance days;
- official `acceptanceDateTime` is the causal clock;
- amendments are retained for audit but cannot emit.

The source artifact and audit are hash-bound. Official SEC references are the
[EDGAR APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces),
[full-text search](https://www.sec.gov/edgar/search/index.html), and
[fair-access guidance](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data).
Only official SEC archive documents may supply body text.

## Frozen Gemma 4 runtime

The extractor is Google's official
[`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it):

- revision `ee0ef6023621cff504d758262d4e04895a5af4a2`, the official main SHA
  observed on 2026-07-20;
- every runtime-used config, processor config, template, tokenizer, and weight
  file SHA-256 is embedded in the artifact;
- `AutoModelForMultimodalLM`, text-only input, remote code disabled;
- bitsandbytes 4-bit NF4, double quantization, FP16 compute;
- one GPU only (`device_map={"": 0}`), eager attention, batch one;
- at most 1,536 input tokens and 160 generated tokens;
- greedy decoding with no sampling.
- one user message, `add_generation_prompt=true`, and
  `enable_thinking=false`; decode only the generated suffix with special tokens
  retained, then validate the pinned processor's parsed final content.

Google describes E4B as a dense model with 4.5B effective parameters and 8B
including embeddings on its official
[Gemma 4 model page](https://developers.google.com/edge/litert-lm/models/gemma-4).
Hugging Face documents
[4-bit bitsandbytes inference](https://huggingface.co/docs/transformers/quantization/bitsandbytes).
The exact local 15.99 GB weight file was verified without duplicating it; the
new upstream revision reused the same weight hash and changed only small
template/tokenizer metadata. Disk use stayed at 292 GB, below the requested
300 GB ceiling.

An RTX 3060 Ti 8 GB is **not yet certified**. Batch-one, short-context 4-bit
inference is plausible because the quantized weights are several GB, but
runtime overhead and KV cache leave little margin. A measured 8 GB smoke test
is required before live deployment. That feasibility statement is an
engineering inference, not a vendor VRAM guarantee.

## LLM boundary and anti-memorization controls

The LLM receives one short, redacted evidence window at a time: a Bitcoin-hit
paragraph plus its immediate neighbors. Action-term hits are processed first,
with at most 16 windows per accession. It never receives accession, CIK,
ticker metadata, detected issuer ticker, company-name metadata,
acceptance/file date, URL, market value, price, return, funding, position,
reward, or later context.

Before rendering, deterministic code masks issuer aliases, corporate-suffix
names, exchange-labeled/dollar-prefixed/supplied issuer tickers, URLs, emails,
accessions, CIKs, dates, currencies, amounts, and percentages. An unidentified
standalone body symbol or paraphrased issuer name can still survive, so this is
a reduction—not elimination—of identity leakage. The model must return exactly:

```json
{"label":"BTC_CONSTRAINT_DRAW","role":"BTC_SALE","quote":"exact redacted substring"}
```

Supported output is accepted only when the role belongs to the label and the
quote is an exact contiguous substring of the redacted input. Malformed or
hallucinated output fails closed. Opposing supported windows make the whole
accession mixed and unable to change state.

Public EDGAR prose may have existed in model pretraining, so the project does
**not** claim zero memorization. The fixed defenses are identity/date/number
redaction, quote grounding, entity-swap invariance controls, and keeping all
direction logic outside the model.

## Causal state and event construction

1. Internally sort by acceptance UTC then accession and reject duplicate keys.
2. Parse CIKs numerically and use the smallest numeric CIK in a co-filer set as
   one zero-padded stable issuer key; one filing
   can never count as several issuers.
3. The first supported state initializes an issuer. Unsupported and same-state
   filings emit nothing.
4. A change between draw and buffer emits one raw transition.
5. An episode starts with the first unused transition. Each issuer can count
   once. It resolves on causal arrival of the third same-class issuer if the
   opposite class has at most one issuer within ten calendar days.
6. Resolve buffer to long and draw to short, then clear the episode so a
   transition cannot be reused.

2018–2020 is state warm-up, 2021–2022 is train/support, 2023 is selection, and
2024+ is sealed.

## Frozen execution and evaluation

- historical readiness: SEC acceptance plus 15 minutes;
- live readiness: the later of that floor or durable local
  receipt/parse/redaction/inference completion;
- entry: first five-minute BTCUSDT perpetual open at or after another five
  minutes, rounded upward to the grid;
- fixed 72-hour hold, 0.5x exposure, no stop, no take-profit, global
  non-overlap;
- reserve in entry order and count only signals whose entry is at or after the
  prior accepted exit;
- exact realized funding, 6 bp per side base cost, 10 bp stress cost;
- full calendar time, including warm-up and idle cash, for absolute return and
  CAGR;
- global/pre-entry-HWM strict MDD includes entry cost, every held five-minute
  path, funding, virtual adverse exit cost, and actual exit.

## Sequential gates

### 1. Synthetic model gate

The model must pass 17 hash-frozen literal cases covering exact factual
draw/buffer, planned-action, generic-risk,
third-party, mixed-evidence, symmetric prompt-injection, entity-swap, and
date/amount-redaction controls. A failure retires this exact prompt/model
contract before EDGAR bodies are opened.

### 2. Historical semantic support

Before any BTC row, exact JSON parsing must be at least 98%, supported quote
matching at least 99%, role/label consistency 100%, and entity-swap
label/role invariance at least 95%. Train requires at least 120 directional
accessions, 60 raw transitions, and 36 breadth events; 2023 requires at least
50 directional accessions, 24 transitions, and 18 breadth events. Both states,
both sides, issuer breadth, active-month coverage, and concentration caps are
also frozen in the artifact.

### 3. Outcome-free novelty

The event clock is compared against six hash-pinned prior families: semantic,
miner cadence, microstructure, the live portfolio, network/fee, and regional
FX. Reject before economics if any comparator exceeds exact-entry Jaccard
0.10, ±1-day match coverage 0.35, or absolute signed occupied-exposure
correlation 0.35.

### 4. Economics

Only a complete support and novelty pass opens 2021–2023 BTC outcomes. Both
2021–2022 and 2023 must independently have positive absolute return,
`CAGR / strict MDD >= 3`, strict MDD no greater than 15%, and positive
10-bp-stress return. 2023 additionally requires positive absolute return in
both halves, at least 18 trades, and both sides. Controls include exact side
flip, deterministic random side, one-business-day delay, generic mention
breadth, level-only clustering, duplicate-issuer breadth, and stressed costs.

Any failed frozen gate retires EBCT-72. No prompt, model, sign, window,
breadth, hold, latency, threshold, or redaction repair is permitted after the
corresponding data are opened.

## Current boundary

```text
filing bodies opened       = 0
Gemma semantic calls       = 0
semantic labels created    = 0
BTC market rows read       = 0
funding rows read          = 0
future-return/PnL fields   = 0
2024+ source rows read      = 0
```

Next: run only the frozen synthetic Gemma 4 gate.
