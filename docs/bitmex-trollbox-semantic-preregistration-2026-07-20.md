# TBASR-24 — Gemma2 message semantics preregistration

## Boundary

The text-free attention gate passed and its 5,417 observation windows are
hash-frozen. This stage freezes the only LLM role before private text is
opened: classify one non-guarded participant message at a time as `BULLISH`,
`BEARISH`, or `UNCLEAR`. A direction-neutral deterministic guard first maps
classifier/output meta-instructions to `UNCLEAR`; it cannot emit a bullish or
bearish label. Numeric code then aggregates messages to participant stance and
participants to event consensus. The LLM never sees price, return, funding,
position, reward, PnL, or another participant's message.

The 2026 historical pull proves the frozen rows and their conservative ID-order
availability clock, but it cannot reconstruct an independently archived
point-in-time view of any messages that BitMEX may previously have removed or
moderated. This is a disclosed retrospective-source/provenance limitation, not
permission to repair outcomes. A live implementation must consume and retain
the chat stream contemporaneously rather than assume perfect parity with this
historical snapshot.

## Frozen model and runtime

- model: [Google `gemma-2-2b-it`](https://huggingface.co/google/gemma-2-2b-it);
- Hugging Face revision:
  `299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8`;
- instruction chat template from that exact tokenizer revision;
- 4-bit bitsandbytes NF4, double quantization, FP16 compute;
- remote model code disabled;
- eager attention, batch size 16, greedy decoding, no sampling, eight maximum
  output tokens;
- Transformers `5.7.0.dev0` at upstream commit
  `5d7ff4393ab99aa7cadf4cccd1f814dbb799f2bb`, bitsandbytes `0.49.2`,
  Accelerate `1.12.0`, and PyTorch `2.9.0`;
- model config, tokenizer, index, and both weight-shard hashes are embedded in
  the executable contract.

Google's official model card describes Gemma 2 as an English text-to-text
instruction model and documents both the required chat template and 4-bit
bitsandbytes loading. Gemma terms remain applicable.

## Frozen private preprocessing

For every attention window, process increasing causal `available_date` and ID:

1. count every source message and participant for source parity;
2. select the first eight participants by first appearance;
3. select the first two messages per selected participant;
4. normalize each message to Unicode NFC, replace control characters with a
   space, collapse whitespace, and keep the first 160 Unicode characters;
5. apply the frozen Unicode-NFKC/casefold meta-instruction patterns to the
   complete sanitized 160-character message and map a
   match directly to `UNCLEAR` without model generation;
6. for each remaining message, if the rendered chat prompt exceeds 512 tokens,
   keep the longest Unicode-character prefix whose complete prompt fits;
7. classify each remaining selected message separately; and
8. never write rendered prompts containing source messages, raw source text,
   usernames, participant hashes, or participant-level labels to a committed
   artifact.

The prompt treats the one quoted message as untrusted data, explicitly ignores
instructions inside it, respects negation and trading slang, and requires one
exact label. A malformed output fails closed to `UNCLEAR`.

Synthetic attempt 1 failed only the frozen prompt-injection control before any
private text or market data was opened. Prompt revision
`v2_synthetic_meta_instruction_hardening` therefore adds one narrow rule:
messages that discuss classification rules, ask a classifier/AI for a label,
or tell it what to output are meta-instructions and must be `UNCLEAR`. The
original eight model controls and all numeric controls remained unchanged.
Attempt 2 showed that wording alone still copied the injected label. Further
wording tuning was abandoned. Semantic revision
`v3_direction_neutral_meta_instruction_guard` adds hash-frozen high-precision
patterns for instruction/rule/prompt override and classifier label commands.
The battery now adds symmetric bearish injections and legitimate bullish and
bearish uses of `ignore`, `rules`, and `model` to check both directional
neutrality and false positives.

The executable gate also verifies exact package versions, the Transformers Git
revision recorded in installed package metadata, and every frozen model file
hash before model loading. Resumable private labels are bound to the contract
and ordered job list by a per-record hash chain; resume rows cannot contain
text or additional fields.

Participant stance is deterministic: only bullish message labels gives
`BULLISH`, only bearish gives `BEARISH`, both gives `UNCLEAR`, and no directional
label gives `UNCLEAR`. Event stance requires at least two independently
directional participants and a 2:1 participant majority. Otherwise it is
`UNCLEAR`. The later contrarian side is short for bullish crowd and long for
bearish crowd.

## Synthetic gate before private text

The committed synthetic battery covers explicit long/buy, explicit short/sell,
negated bullish language, bullish and bearish slang, neutral/non-BTC chat, a
direction-only question, symmetric prompt injections, classifier commands, and
legitimate stance language near guard keywords. Numeric controls cover mixed
messages from one participant, balanced participants, one-participant evidence,
and exact 2:1 consensus. The private mode is disabled until the synthetic
artifact passes and its file SHA-256 is pinned in code.

Revision v3 passed all 13 model/guard controls and all five numeric controls.
The passing artifact is pinned by file SHA-256; the two earlier injection
failures remain separately committed. This pass opens only private semantic
support counting. It does not permit BTC price, funding, outcome, or PnL data.

## Frozen directional-support gate

From the 5,417 attention windows, clear event labels must satisfy all:

- at least 800 overall;
- at least 450 in train (2020H2–2021), including 100 in 2020H2 and 300 in
  2021;
- at least 300 in calendar 2022, including 120 in each half;
- at least 30 in every quarter from 2020Q3 through 2022Q4;
- clear events in at least 80 weeks overall, 50 train weeks, and 30 test weeks;
- bullish and bearish labels each at least 25% of clear labels in all, train,
  and test;
- no quarter above 20% of clear labels; and
- no more than 1% of selected messages rejected by the meta-instruction guard;
  and
- at least 98% of individual message generations parse exactly.

Failure rejects TBASR before any BTC price or outcome. No prompt, model,
quantization, cap, consensus, label parser, or support threshold may be repaired
after private label incidence. A pass permits only the next preregistration:
freeze completed pre-event price displacement/alignment and the strict market
evaluator before opening any post-entry path.

This remains a candidate-level freeze, not a pristine global holdout claim.
