# EBOC-72 mechanism decision — realized Bitcoin mining capacity breadth

## Decision

Freeze one exact text-native candidate:
**EBOC-72 — EDGAR Bitcoin Operational Capacity Transition**.

EBOC extracts only completed changes in the filing issuer's operating Bitcoin
mining capacity. A deterministic composer aggregates distinct issuers whose
capacity entered or left operation. The Gemma adapter never chooses a market
side, timestamp, hold, leverage, threshold, or reward.

This decision is committed before fetching or decoding any historical SEC
filing body, creating any EBOC semantic label, parsing any comparator clock, or
opening any BTC market outcome.

## Fixed source and split boundary

Use only:

```text
data/sec_edgar_bitcoin_8k_6k_source_2018_2023.jsonl.gz
SHA256 c8489dfe9b4ac25da8bea7653115e5b58a44fa897f2815eaf68bad354e10c6ce
```

Source canonical-row hash:

```text
98793185f1e411d8c59736fb54c5ed529d539e81ccddf2c823f24127ecfcef0b
```

Source audit:

```text
results/sec_edgar_bitcoin_8k_6k_source_audit_2026-07-21.json
SHA256 c1e11d1f5089378ac787fdb2a80474f0feec33d5fb2296fb0c3014d6f1fafec1
```

Only non-amendment `8-K` and `6-K` accessions may emit. Multiple matched
documents in one accession are processed but can produce at most one accession
class and one issuer fact.

Splits use official submissions `acceptanceDateTime` UTC:

| Purpose | Interval |
|---|---|
| semantic warm-up and issuer history | `[2018-01-01, 2021-01-01)` |
| train/support/economics | `[2021-01-01, 2023-01-01)` |
| selection/support/economics | `[2023-01-01, 2024-01-01)` |
| sealed extension | `2024-01-01` and later |

No 2024-or-later SEC row, document, model label, market row, or reward may be
read on this candidate until train and selection pass every preceding gate and
a separate extension protocol is committed.

## Historical document transport

For accession `AAAAAAAAAA-BB-CCCCCC`, CIK `K`, and frozen matched document
`D`, the only body URL is:

```text
https://www.sec.gov/Archives/edgar/data/int(K)/
  AAAAAAAAAABBCCCCCC/D
```

`int(K)` is the decimal CIK without leading zeroes. The accession directory is
the accession with hyphens removed. Every URL component must match
`^[A-Za-z0-9._-]+$` after the frozen source parser has already validated the
accession and document grammar.

For issuer-alias redaction only, the builder may also fetch:

```text
https://www.sec.gov/Archives/edgar/data/int(K)/
  AAAAAAAAAABBCCCCCC/AAAAAAAAAA-BB-CCCCCC-index-headers.html
```

The index-header response may contribute only exact conformed-name aliases
belonging to the accession's frozen CIK set. It may not add a filing, document,
timestamp, form, ticker, semantic fact, or event.

Requirements:

- official `www.sec.gov` HTTPS host only, no redirect to another host;
- `Accept-Encoding: identity`;
- a contact-bearing User-Agent and at most 8 requests per elapsed second;
- nonempty response, maximum 32 MiB per matched document and 4 MiB per header;
- raw response bytes hashed before parsing;
- identical accession/document/CIK/form membership to the frozen metadata;
- complete content-addressed cache and deterministic gzip archive;
- a retry may repeat the same URL only and may not switch to a mirror; and
- missing, changed, oversized, malformed, or conflicting content fails the
  complete source build without quarantine or substitution.

The historical ready floor is:

```text
acceptanceDateTime + 60 elapsed minutes
```

Live readiness is the later of that floor and durable local receipt, complete
download, hash, parse, redaction, model inference, deterministic composition,
and manifest commit. Filing date and archive-header clock never define entry.

## Visible-text parser

Decode bytes as UTF-8 with strict errors after removing an optional UTF-8 BOM.
HTML is parsed with Python's standard-library `html.parser.HTMLParser` using
`convert_charrefs=True`.

The parser:

- discards `script`, `style`, `noscript`, `svg`, and `template` element
  contents;
- emits a block boundary before and after `p`, `div`, `li`, `tr`, `td`, `th`,
  `br`, `h1` through `h6`, `pre`, and `blockquote`;
- appends all other visible `handle_data` text in document order;
- normalizes Unicode to NFKC;
- replaces every category-`C` code point except LF and TAB with one space;
- converts CRLF/CR to LF and TAB to one space;
- collapses horizontal whitespace to one ASCII space;
- collapses three or more LF characters to two; and
- strips each block and removes empty blocks.

Plain-text matched documents use the same Unicode/control/whitespace rules
without HTML tag interpretation. A document whose first 4 KiB contain a
case-insensitive HTML doctype, `<html`, `<body`, or `<document` token is HTML;
all other documents are plain text. The parser decision is retained.

Sentences are constructed independently inside each block. Split after `.`,
`!`, or `?` only when followed by whitespace and then an ASCII uppercase
letter, `[` or a digit. A block with no split remains one sentence. Sentences
longer than 1,200 Unicode code points are deterministically chunked at the last
ASCII space no later than 1,200 characters, with a hard split only when no
space exists. No chunk may overlap another.

## Candidate sentence and window

Case-insensitive ASCII-boundary mining-context terms:

```text
bitcoin
btc
bitcoin mining
bitcoin miner
bitcoin miners
mining machine
mining machines
mining equipment
hashrate
hash rate
mining facility
mining facilities
data center
data centre
```

Case-insensitive ASCII-boundary transition terms:

```text
energized
commissioned
deployed
installed
operating
operational
online
resumed
restarted
reactivated
producing
shut down
shutdown
offline
curtailed
curtailment
suspended
ceased
decommissioned
terminated
removed
sold
relocated
```

A target sentence must contain at least one mining-context term and one
transition term. The evidence window is the target plus the immediately
previous and next sentence from the same block. Sentence IDs are fixed by
role, not reassigned by presence: the previous sentence is always `S1`, the
target is always `S2`, and the next sentence is always `S3`. An absent neighbor
leaves that ID absent; a target-only window therefore contains only `S2`.
Presented sentences remain in document order. The target's fixed `S2` role is
retained internally but is not disclosed as the expected answer.

Windows are deduplicated by the SHA-256 of their redacted numbered text. Keep
at most 16 windows per accession, ordered by matched-document sequence as an
integer, matched document name, block index, sentence index, and window hash.
Non-integer or conflicting document sequence values fail the accession.

An accession with no candidate window is deterministically `UNSUPPORTED`
without a model call. Candidate-window count is source support, not an alpha
or outcome.

## Deterministic redaction

Before sentence numbering and prompting:

- replace exact conformed-name aliases and corporate suffix variants for every
  frozen co-filer CIK with `[ENTITY]`, longest alias first;
- replace exchange-labelled and dollar-prefixed tickers obtained only from the
  same body/header with `[TICKER]`;
- replace URLs and email addresses with `[LINK]`;
- replace accession numbers and 10-digit CIKs with `[ID]`;
- replace ISO, slash, month-name, and compact filing dates with `[DATE]`;
- replace currencies, percentages, decimal/integer quantities, power units,
  hashrate units, miner counts, and financial amounts with `[NUM]`; and
- collapse whitespace once more.

The model never receives accession, CIK, company metadata, ticker metadata,
acceptance time, file date, URL, split, BTC price, return, funding, position,
reward, or future context. Redaction is a memorization-risk reduction, not a
clean-room guarantee.

## Fixed semantic ontology

The model output class is one of:

```text
CAPACITY_ONLINE
CAPACITY_OFFLINE
UNSUPPORTED
MIXED
```

### `CAPACITY_ONLINE`

All three facts must be explicit in the selected sentence and bounded
neighbors:

1. the capacity belongs to or is operated for the filing issuer;
2. the capacity mines Bitcoin; and
3. a completed transition put that capacity into operation.

Allowed realized transitions are commissioning, energization, deployment,
installation followed by explicit operation, resumption/restart/reactivation,
or a closed acquisition whose acquired mining capacity is explicitly
operating.

### `CAPACITY_OFFLINE`

All three facts must be explicit:

1. the capacity belongs to or is operated for the filing issuer;
2. the capacity mines Bitcoin; and
3. a completed transition removed that capacity from operation.

Allowed realized transitions are shutdown, completed curtailment, suspension,
cessation, decommissioning, or a terminated hosting/power arrangement that
explicitly stopped the issuer's mining operation.

Equipment sale or relocation is `CAPACITY_OFFLINE` only when the evidence also
states that the issuer's corresponding mining operation stopped or the
capacity was taken offline. Sale, relocation, impairment, or contract
termination alone is `UNSUPPORTED`.

### `UNSUPPORTED`

Mandatory cases include:

- plan, target, forecast, expectation, intention, option, agreement, MOU,
  equipment order, construction, delivery, installation without explicit
  operation, financing, or expected future energization;
- current hashrate, miner count, efficiency, power, production, mined Bitcoin,
  difficulty, network hashrate, or monthly comparison without an operating
  transition;
- treasury purchase/sale/pledge/retention, impairment, accounting, proceeds,
  customer access, custody, trading, payment, or settlement;
- third-party or industry capacity without explicit filing-issuer operation;
- generic risk, hypothetical, boilerplate, or negated/non-occurring action;
- an evidence ID that does not directly support completion, issuer
  attribution, and direction; and
- a guarded meta-instruction window.

### `MIXED`

Use only when the same presented window contains both a supported realized
online transition and a supported realized offline transition for the filing
issuer. A current online level plus a completed offline transition is not
mixed; it is offline. A plan in one direction and completed transition in the
other takes the completed direction.

## Prompt and output grammar

The exact prompt is frozen in the machine preregistration. Its semantic
instructions are exactly the ontology above, shortened only by removing
examples and file paths. The excerpt is explicitly declared untrusted evidence
and never an instruction.

The assistant must emit exactly one ASCII line:

```text
CAPACITY_ONLINE|S1
CAPACITY_OFFLINE|S2
UNSUPPORTED|NONE
MIXED|NONE
```

Grammar:

```text
^(CAPACITY_ONLINE|CAPACITY_OFFLINE)\|S[1-3]$
^(UNSUPPORTED|MIXED)\|NONE$
```

No leading/trailing whitespace, Markdown, JSON, explanation, lowercase alias,
extra line, or generated quote is accepted. A supported evidence ID must exist
in the presented window. Malformed output fails closed to `UNSUPPORTED` and
counts against the 100% parse gate.

Before a model call, a case-insensitive meta-instruction guard searches:

```text
\b(ignore|disregard|override|system prompt|developer message|
return exactly|output exactly|classif(?:y|ier|ication))\b
```

A match emits `UNSUPPORTED|NONE` without inference.

## Whole-accession resolution

Process all windows for an accession.

- no supported window: `UNSUPPORTED`;
- one or more supported windows, all `CAPACITY_ONLINE`: online;
- one or more supported windows, all `CAPACITY_OFFLINE`: offline;
- any supported online and supported offline windows: `MIXED`;
- any window-level `MIXED`: accession `MIXED`.

Repeated same-class windows do not increase breadth. One accession contributes
at most one issuer fact.

The issuer key is the lexicographically smallest zero-padded numeric CIK in the
frozen accession CIK set. This is an identity key, not an entity-resolution
claim. Co-filers never count as several issuers.

## Causal issuer breadth

Process accession facts by `(ready_time, accession)` and batch equal
`ready_time` values. Facts in one batch cannot observe each other.

Resolve same-issuer directional facts inside each equal-time batch before
cooldown or breadth:

- if all facts for one issuer have the same direction, retain only the
  lexicographically first accession as that issuer's batch representative and
  mark the rest `same_issuer_same_batch_suppressed`;
- if both directions occur for one issuer, mark every directional fact for
  that issuer `same_issuer_batch_conflict` and accept none of them; and
- unsupported or mixed accessions never participate in this resolution.

An issuer batch representative is accepted into issuer history only when that
issuer has no previously accepted directional fact with elapsed time strictly
less than 21 days. A fact exactly 21 elapsed days after the prior accepted fact
is eligible. Cooldown-skipped facts do not reset cooldown or enter breadth.

At each newly accepted directional fact, evaluate history strictly before its
equal-time batch plus the current fact only. For each issuer, keep its latest
accepted directional fact whose ready time is within the inclusive trailing
21-calendar-day interval. Define:

```text
online_issuers  = distinct active issuers with latest class online
offline_issuers = distinct active issuers with latest class offline
score           = online_issuers - offline_issuers
```

Emit a raw candidate only when:

- at least two distinct active issuers exist;
- `score >= +2` and the triggering fact is online: `LONG`; or
- `score <= -2` and the triggering fact is offline: `SHORT`.

Expiry, an unsupported/mixed filing, a cooldown-skipped filing, or a trigger
opposite the current score never emits. No future issuer fact may alter a past
event.

Equal-time batches are sorted by accession only for deterministic output, but
all retained issuer representatives see the same prior history and only
themselves as the current fact. If several representatives would emit, retain
the lexicographically first accession and mark the rest
`same_batch_signal_suppressed`; every cooldown-eligible representative still
enters history after batch resolution. Same-issuer duplicates, same-issuer
conflicts, and cooldown-skipped facts never enter history.

## Execution

For raw candidate readiness `t`:

```text
latency_bar_start = first UTC 5-minute boundary at or after t
entry_time        = latency_bar_start + 5 minutes
scheduled_exit    = entry_time + 72 hours
```

If `t` is exactly on a five-minute boundary, the complete bar beginning at
`t` must elapse. Candidate rows are sorted by `(entry_time, accession)`.
Global non-overlap accepts a candidate only when its entry is at or after the
previous accepted scheduled exit.

Frozen economics, if later authorized:

- BTCUSDT USD-M perpetual;
- fixed 0.5x exposure;
- no stop loss and no take profit;
- exact realized funding, entry-inclusive and exit-exclusive;
- 6 bp per notional side base cost and 10 bp stress cost;
- full split wall-clock absolute return and CAGR, including idle cash; and
- global/pre-entry-HWM strict MDD including entry cost, every held five-minute
  high/low path, funding, virtual adverse exit cost, and actual exit.

Every report must show absolute return, CAGR, strict MDD, CAGR/strict MDD,
trades, side counts, active months, and clustered statistical evidence.

## Synthetic-only Gemma adaptation

Base:

```text
google/gemma-4-E2B-it
revision 3e22461f65e89153144f8adb70e3b8c2cc9845a7
```

Runtime-used base file hashes:

| File | SHA-256 |
|---|---|
| `model.safetensors` | `2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550` |
| `config.json` | `1b28f3d2c3100f6c594754b81107428bd7b822a7f48272ca681dae9d2ec38330` |
| `processor_config.json` | `32bdf45d2ad4cc29a0822ddd157a182de76644f0419a6228d151495256e9813c` |
| `tokenizer.json` | `cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f` |
| `tokenizer_config.json` | `9f4fec4b1dc6ecddf8f4a92e9caea5971c0e67d81309f3f9066a2bee8c362633` |
| `chat_template.jinja` | `0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5` |
| `generation_config.json` | `d4226bbe3117d2d253ba4609720ba82c6c4ce4627a9a6ae05387c78983ac03de` |

Pinned runtime:

```text
Torch 2.9.0
Transformers 5.7.0.dev0
PEFT 0.18.1
bitsandbytes 0.49.2
Accelerate 1.12.0
```

Quantization and LoRA:

- one visible CUDA GPU;
- bitsandbytes NF4 4-bit, double quantization, BF16 compute;
- eager attention, batch one, maximum 512 total tokens;
- adapters only on text-language-model `q_proj`, `k_proj`, `v_proj`, and
  `o_proj`; vision/audio modules forbidden;
- exact PEFT regex:
  `.*language_model.*\.(q_proj|k_proj|v_proj|o_proj)$`;
- rank 8, alpha 16, dropout 0.05, no bias;
- 2,678,784 trainable adapter parameters under the pinned base;
- completion-only causal loss;
- Torch AdamW, learning rate `1e-4`, weight decay `0.01`;
- 4 optimizer-step linear warmup, cosine decay;
- one example per device, gradient accumulation 8;
- gradient norm clipped to 1.0;
- exactly 64 optimizer steps and seed `20260724`;
- checkpoints only at steps 16, 32, 48, and 64; and
- no historical filing, metadata value, market row, return, or reward in
  training or checkpoint selection.

The deterministic synthetic generator must create:

| Split | Rows | Per class | Template-family relation |
|---|---:|---:|---|
| train | 512 | 128 | train-only families |
| calibration | 128 | 32 | calibration-only families |
| adversarial test | 192 | 48 | test-only families |
| swap pairs | 64 pairs | balanced | test-only paired rewrites |

Template family IDs are disjoint across train, calibration, and test. Surface
entity names, dates, quantities, miner/facility nouns, transition verbs,
sentence positions, distractors, and neighbor ordering vary deterministically.
No train row is byte-identical to any calibration/test row after redaction.

Training rows may be shuffled only by a seeded permutation stored in the
manifest. Calibration selects one checkpoint by this ordered rule:

1. highest exact `(class, evidence_id)` count;
2. highest minimum per-class exact share;
3. lowest malformed count;
4. lowest checkpoint step.

No calibration threshold, temperature, probability cutoff, ensemble, adapter
merge, or hand selection is allowed.

After selection, the chosen checkpoint runs the untouched adversarial and swap
gates once. Required pass:

- strict output parse: 100%;
- evidence ID existence: 100%;
- overall exact class plus evidence: at least 95%;
- each online/offline class exact: at least 95%;
- unsupported exact: at least 97%;
- mixed exact: 100%;
- guarded prompt-injection cases: 100% and zero model calls;
- EBCT balance-sheet and BPAX customer-access negative controls: 100%
  `UNSUPPORTED`;
- entity/date/number/ticker swaps: 100% class and evidence-position invariance;
- selected-checkpoint inference peak allocated at most 7 GiB and reserved at
  most 7.25 GiB for batch-one 512-token-cap evaluation; and
- complete training peak allocated and reserved at most 24 GiB.

Failure retires the exact EBOC-72 adapter. Prompt, grammar, ontology, parser,
training rows, template family, seed, LoRA recipe, checkpoint rule, memory
gate, or semantic threshold may not be changed after training begins.

All nonselected checkpoints are retained until their calibration metrics and
hashes are committed, then deleted. The selected adapter and tokenizer/base
bindings are immutable. Total retained EBOC checkpoint bytes must remain below
1 GiB.

## Historical semantic support gates

Only a complete synthetic pass may authorize body retrieval and historical
semantic inference. Before comparator timestamps or market rows:

### Integrity

- 100% exact frozen source/body/header/cache identity;
- 100% strict output parse and evidence-ID existence;
- 100% whole-accession deterministic replay;
- 100% source rows stop before 2024;
- model/adaptor/runtime hashes match the selected synthetic artifact; and
- zero BTC, funding, return, PnL, reward, or comparator-row access.

### Train, 2021–2022

- directional accessions at least 60;
- distinct directional issuers at least 20;
- globally accepted signals at least 36;
- LONG and SHORT each at least 20% and at least 6;
- active entry months at least 18 of 24;
- maximum accepted-entry gap at most 75 elapsed days;
- no issuer above 12.5% of directional accessions; and
- no UTC entry month above 20% of accepted signals.

### Selection, 2023

- directional accessions at least 24;
- distinct directional issuers at least 10;
- globally accepted signals at least 18;
- LONG and SHORT each at least 20% and at least 4;
- active entry months at least 8 of 12;
- maximum accepted-entry gap at most 60 elapsed days;
- no issuer above 20% of directional accessions; and
- no UTC entry month above 30% of accepted signals.

Train plus selection must contain at least 54 accepted signals. A source
support failure retires EBOC before novelty and outcomes. The threshold,
breadth window, cooldown, hold, parser, or class may not be repaired.

## Outcome-blind controls and novelty

Before historical semantic incidence is opened, the preregistration must
hash-bind:

- EBCT preregistration and synthetic rejection;
- BPAX preregistration and synthetic rejection;
- miner-cadence recovery clock;
- prior semantic clock;
- GDELT/news clock where common coverage exists;
- canonical microstructure comparator bundle;
- executable live-portfolio clock;
- network/fee clock; and
- regional-FX clock.

Comparator files are raw-byte hashed only. No row or timestamp may be parsed
until EBOC semantic support passes.

Required controls:

- `lexicon_only`: deterministic online/offline transition lexicon without
  Gemma;
- `generic_mention`: every eligible Bitcoin accession under the same issuer
  cooldown/breadth scheduler;
- `no_breadth`: every directional accession after cooldown;
- `no_cooldown`: exact EBOC facts without issuer cooldown;
- `delay_24h`: accepted EBOC entries delayed exactly 24 hours;
- `stale_21d`: use breadth state immediately before the triggering fact; and
- synthetic side flip/random side are deferred to economics.

Against every qualifying prior clock over exact common coverage:

- exact-entry Jaccard at most 0.10;
- maximum-cardinality one-to-one candidate containment within `±24h` at most
  0.35; and
- absolute signed five-minute occupied-exposure correlation at most 0.35.

Against `lexicon_only`, `generic_mention`, `no_breadth`, and `no_cooldown`,
EBOC must also satisfy:

- exact-entry Jaccard at most 0.50;
- same-entry same-side reproduction using EBOC as denominator at most 0.70; and
- no one control may reproduce more than 75% of EBOC occupied exposure.

Missing, malformed, outcome-bearing, coverage-empty, or hash-mismatched
comparators fail closed. A novelty failure retires EBOC before economics.

## Later economic and RLLM gates

Only a complete semantic-support and novelty pass may authorize a separately
committed economic evaluator. Train and selection are opened sequentially and
must each independently pass:

- positive absolute return;
- `CAGR / strict MDD >= 3.0`;
- strict MDD at most 15%;
- positive absolute return at 10 bp per side stress cost; and
- positive stationary-block-bootstrap probability at a threshold frozen in
  the later economic evaluator.

Selection must also have positive absolute return in both calendar halves.
Every finite control, exact side flip, deterministic random side, and 24-hour
delay must be reported.

Only if unchanged deterministic EBOC passes both periods may a train-only RLLM
receive causal semantic relation tokens and current position state to choose
`TRADE_FIXED_SIDE` or `ABSTAIN`. It may not create a clock, change a side or
hold, or use selection/eval rewards for prompt, adapter, or checkpoint
selection.

## Evidence boundary

This mechanism unit opened only:

- the candidate boundary, source audit, and immutable source metadata hashes;
- EBCT/BPAX preregistration and synthetic-rejection contracts;
- local Gemma base-file hashes, package versions, module names, and two
  synthetic-only LoRA construction/one-step feasibility probes; and
- official SEC and Gemma documentation.

The feasibility probe used one synthetic sentence and no filing or market
data. It established 2,678,784 trainable parameters and a measured one-step
training peak below the frozen 24 GiB ceiling; it is not a synthetic gate,
semantic result, or alpha result.

The unit did not open:

- any historical SEC filing body, EBOC window, label, fact, event, count, side,
  comparator row, or timestamp;
- any BTC market, funding, future return, PnL, absolute return, CAGR, MDD, hit
  rate, or reward; or
- any 2024-or-later source or outcome.

## Next immutable sequence

1. commit this mechanism decision;
2. implement and test a write-once preregistration plus deterministic synthetic
   generator without opening filing bodies;
3. commit immutable synthetic split artifacts and trainer/evaluator code;
4. train once and select only by the frozen calibration rule;
5. run the untouched adversarial/swap gate once;
6. retire unchanged on any failure;
7. only a pass may authorize the separately frozen SEC body/support builder;
8. only support and novelty passes may authorize economics; and
9. keep every 2024-or-later source and outcome sealed.
