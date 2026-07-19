# BitMEX Trollbox attention-saturation mechanism decision — 2026-07-20

## Decision

The next standalone BTC candidate is **TBASR-24 — Trollbox Attention
Saturation Reversal**. It will detect an unusual burst of independent English
BitMEX Trollbox participants after a completed BTC displacement, use one small
frozen LLM only to summarize the crowd's expressed direction, and trade
against a sufficiently coherent crowd for exactly two hours.

This is a mechanism/source decision only. It opens no complete chat history,
event incidence, LLM classification distribution, post-entry return, PnL, or
2023+ row. The API probes were limited to endpoint schema, first/last
timestamps, the 500-row cap, and forward pagination behavior. Six returned
message examples were visible during that probe, but no subsequent BTC outcome
was joined or inspected.

## Why this uses the LLM where it is strong

The numeric layer will do only arithmetic it can audit reliably:

- message count;
- distinct-participant count;
- repeat-author concentration;
- a strictly prior activity reference; and
- the completed pre-event BTC move.

The LLM will not predict a return, size a position, calculate a threshold, or
read future price. Its single role is semantic compression of a frozen set of
already-published chat messages into `BULLISH`, `BEARISH`, or `UNCLEAR`, with a
short evidence span. That is where language modeling can add information over
numeric features: trader slang, negation, price-level claims, mixed statements,
and context-dependent direction.

The final action remains deterministic. A source-only event needs an unusual,
multi-user attention burst, a completed material BTC displacement, and a
directional LLM consensus aligned with that displacement. The policy then
fades the completed crowd direction. `UNCLEAR` never trades.

This is deliberately one analyzer-like semantic head feeding a rule/RL-ready
state, not the discarded two-large-LLM analyzer/trader architecture.

## Official source and live parity

BitMEX documents the Chat API as Trollbox data and exposes
`GET https://www.bitmex.com/api/v1/chat`. The English global channel is
`channelID=1`; the endpoint supports an ID cursor, chronological order, and up
to 500 rows per request. The official WebSocket documentation exposes `chat`
on `wss://ws.bitmex.com/realtimePlatform`, providing the live counterpart.

Official references:

- [BitMEX Chat API](https://docs.bitmex.com/api-explorer/chat)
- [Get chat messages](https://docs.bitmex.com/api-explorer/chat-get)
- [BitMEX WebSocket API](https://www.bitmex.com/app/wsAPI)
- [BitMEX API overview](https://docs.bitmex.com/api-explorer/bitmex-api)
- [BitMEX Terms of Service](https://www.bitmex.com/terms)
- [BitMEX Privacy Notice](https://static.bitmex.com/documents/Bitmex_Privacy_Notice_2025.pdf)

A source probe found that the currently exposed English history begins at
`2020-03-13 08:49:12.370 UTC` and remains live in July 2026. History is
therefore long enough for a pre-2023 selection prefix and later sequential
years, but it is not a point-in-time vintage archive. Production eligibility
will require a forward WebSocket shadow that stores first-seen timestamps.

## Privacy and data-use boundary

Chat text and usernames are more sensitive than market bars. The source
pipeline must therefore:

- keep raw responses ignored and local;
- never commit usernames or raw messages;
- hash participant identifiers with a study-specific salt before aggregation;
- cap repeated messages from one participant inside an event;
- commit only aggregate support counts, source hashes, model/prompt hashes,
  class labels, and event clocks without message text;
- never reproduce chat excerpts in reports; and
- treat current BitMEX terms and privacy notices as binding source conditions.

The repository does not grant a license to redistribute BitMEX chat data.

## Why this is not the rejected attention family

The Wikimedia candidate was daily, broad public attention with a 36-hour
publication delay and only four total diagnostic trades. TBASR observes a
derivatives venue's own trader conversation at event time, requires multiple
independent participants, operates intraday, and uses language direction rather
than pageview magnitude alone.

It is also not another liquidation/OI/funding feature. Chat attention is a
human information-and-positioning surface. The completed price displacement is
directional context, while the event trigger and semantic state come from an
independent source.

## Frozen research sequence

1. Commit this decision before fetching the complete pre-2023 chat prefix or
   calculating attention incidence.
2. Freeze a resumable private downloader that streams English messages,
   validates monotonic IDs/timestamps, hashes the raw canonical stream, and
   emits privacy-preserving five-minute aggregates.
3. Freeze an **attention-only** source-support gate before reading complete
   source incidence. No message semantics or market outcome may enter this
   stage.
4. Reject without repair if attention bursts lack train/test/calendar support.
5. Only after an attention pass, freeze the exact small Gemma model revision,
   quantization, prompt, decoding, participant cap, and synthetic semantic
   controls. Then run an outcome-blind directional-support gate.
6. Reject without repair if `BULLISH`/`BEARISH` coverage or side balance is
   insufficient. Do not tune the prompt on BTC returns.
7. Commit and hash-freeze one strict train/test evaluator before parsing any
   post-entry five-minute path or funding mark.
8. Use 2020H2–2021 as train and calendar 2022 as test. Both must pass positive
   absolute return, full-calendar CAGR/strict-MDD at least 3, strict MDD at
   most 15%, stress cost, delayed entry, mechanism controls, and clustered
   statistical significance.
9. Only a complete pre-2023 pass can fetch 2023+. Open 2023, 2024, 2025, and
   2026 sequentially and stop at the first failure.

No outcome-trained LoRA or RL step is admissible until the frozen base semantic
feature demonstrates incremental pre-2023 value. If it does, a later
train-only policy-optimization stage can use the LLM state without changing
the sealed semantic extractor.

The branch is globally contaminated by prior BTC research. This sequence can
support only a candidate-level frozen claim, not a pristine global human
holdout claim.
