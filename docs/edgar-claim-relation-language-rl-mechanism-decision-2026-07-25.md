# ECRL-1 — EDGAR Claim-Relation Language RL Mechanism Decision

## Decision

Freeze one new RLLM mechanism:

**ECRL-1 — EDGAR Claim-Relation Language + QR-DQN**

ECRL-1 uses one small language model to compare an issuer's prior and current
Bitcoin disclosures. The model emits a compact, quote-grounded relation state.
A separate deterministic feature composer aggregates many weak relation states,
and a small QR-DQN learns whether the combined state warrants short, flat, or
long exposure.

This is not an analyzer/trader pair. There is one LLM, and the LLM never sees a
price, return, position, reward, trade side, hold, leverage, CAGR, or MDD. It
performs only natural-language entailment, contradiction, temporal-status, and
attribution reasoning.

This decision is committed before:

- fetching or decoding the historical SEC corpus;
- creating an ECRL historical pair or label;
- training or invoking the synthetic adapter;
- opening a comparator clock;
- reading BTC, funding, price, return, reward, or performance data; or
- opening any 2024-or-later SEC row or market outcome.

## Why this source, not another unproven archive

The repository already passed a source-only audit for the exact SEC EDGAR
Bitcoin 8-K/6-K panel:

```text
data/sec_edgar_bitcoin_8k_6k_source_2018_2023.jsonl.gz
SHA-256 c8489dfe9b4ac25da8bea7653115e5b58a44fa897f2815eaf68bad354e10c6ce
canonical-row hash
98793185f1e411d8c59736fb54c5ed529d539e81ccddf2c823f24127ecfcef0b
```

The audit found:

```text
2,493 eligible non-amendment accessions
308 distinct CIKs
992 distinct acceptance days
2018-2020: 407 accessions
2021-2022: 1,399 accessions
2023: 687 accessions
```

The official acceptance timestamp is already causally frozen and all source
support/concentration gates passed. The audit opened no BTC outcomes.

Official SEC references:

- [SEC EDGAR APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)
- [EDGAR full-text search](https://www.sec.gov/edgar/search/index.html)
- [Accessing EDGAR data](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data)
- [SEC webmaster reuse and EDGAR timing FAQ](https://www.sec.gov/about/webmaster-frequently-asked-questions)
- [Correct or delete a filing](https://www.sec.gov/submit-filings/filer-support-resources/how-do-i-guides/correct-or-delete-filing)

The SEC states that its government-created content and public EDGAR filing
content are free to access and reuse. Original and amended filings remain
available, and the acceptance timestamp provides the replay clock. This is a
stronger foundation than opening another mutable newsroom or current-calendar
source.

## Novelty and separation from retired EDGAR candidates

Three prior SEC mechanisms were retired before historical bodies or market
outcomes:

- EBCT-72 classified absolute balance-sheet constraint states;
- BPAX-120 classified absolute customer-access expansion/retraction; and
- EBOC-72 classified absolute mining-capacity online/offline states.

ECRL-1 does not repair any of those ontologies or prompts. Its primitive is the
**relation between a prior claim and a current claim**:

- did the current disclosure fulfill a prior plan;
- reverse a prior plan or warning;
- repeat a comparable rendered prior claim or state;
- introduce a genuinely new realized or planned state; or
- remain incomparable, risk-only, third-party, or mixed?

The relation object is cross-domain: inventory, mining capacity, customer
access, financing/collateral, and operational exposure are examples used in
clean-room synthetic training, not separate alpha candidates. The LLM emits a
weak semantic state, not a trade direction.

## Frozen source and chronological split

Only the hash-pinned 2018-2023 source artifact is authorized.

| Stage | UTC acceptance interval | Use |
|---|---|---|
| issuer/text history warm-up | `[2018-01-01, 2021-01-01)` | prior-state construction only |
| RL optimization | `[2021-01-01, 2022-07-01)` | reward-bearing training |
| fixed validation diagnostic | `[2022-07-01, 2023-01-01)` | report only; no hyperparameter or checkpoint selection |
| untouched test | `[2023-01-01, 2024-01-01)` | one economic test |
| sealed extension | `2024-01-01` and later | forbidden until all 2023 gates pass |

Amendments remain audit-only and cannot emit. One filing may contribute at most
one pair state. A co-filer set uses the smallest numeric CIK as its stable issuer
key, matching the frozen source conventions.

## Historical body transport and visible text

The only historical body URL is the official SEC archive member already frozen
by accession, CIK, and document name:

```text
https://www.sec.gov/Archives/edgar/data/{decimal_cik}/
{accession_without_hyphens}/{frozen_document_name}
```

The only additional request authorized for issuer-name redaction is the
official SEC index-header member for the same frozen accession:

```text
https://www.sec.gov/Archives/edgar/data/{decimal_cik}/
{accession_without_hyphens}/{accession_with_hyphens}-index-headers.html
```

Only exact `CONFORMED NAME` and `FORMER COMPANY` header values may be consumed
from that member. The header cannot add a filing, timestamp, form, ticker,
document, semantic fact, or pair. It is receipt/hash bound exactly like the
filing body. Missing, conflicting, oversized, or malformed headers fail the
complete build; they cannot silently weaken redaction.

Transport requirements:

- `www.sec.gov` HTTPS only;
- exact frozen accession/document/CIK membership;
- contact-bearing User-Agent and at most eight requests per elapsed second;
- `Accept-Encoding: identity`;
- no mirror or substituted document;
- 32 MiB maximum per document;
- content-addressed raw cache and deterministic gzip archive;
- strict hash/receipt manifest; and
- complete-build failure on missing, changed, oversized, or malformed content.

Historical ready time is:

```text
official acceptanceDateTime UTC + 60 elapsed minutes
```

Live ready time is the later of that floor and durable receipt, complete
download, parsing, redaction, inference, composition, and manifest commit.

The visible-text parser is the already frozen standard-library SEC parser:

- optional UTF-8 BOM removal, then strict UTF-8;
- `HTMLParser(convert_charrefs=True)`;
- script/style/noscript/svg/template removal;
- deterministic block boundaries;
- Unicode normalization and control/whitespace cleanup; and
- deterministic sentence splitting inside blocks.

No human, agent, or external model may read historical body text. Only the
committed extractor process may transform it.

## Prior/current pair construction

Eligible filings are totally ordered by:

```text
(acceptanceDateTime UTC, accession, document sequence, document name)
```

For equal acceptance timestamps, every accession in the same-issuer timestamp
batch sees only the snapshot that existed strictly before that timestamp. Batch
members cannot observe one another. After the complete batch is processed, the
lexicographically greatest eligible accession becomes the issuer's next prior
snapshot. This rule applies before any semantic model call.

For every eligible current accession:

1. find the immediately preceding eligible accession for the same stable issuer;
2. require both accessions to contain at least one frozen Bitcoin-hit document;
3. process all matched documents but emit at most one pair;
4. extract sentence windows centered on case-insensitive whole-word `bitcoin`;
5. merge overlapping hit sentence windows inside each document;
6. retain the hit sentence plus one preceding and one following sentence;
7. order windows by source document sequence and sentence order;
8. cap prior text at 768 model tokens and current text at 1,024 model tokens;
9. keep earliest windows under the cap; never rank with market information; and
10. drop a pair if either side has no retained sentence after strict parsing.

Sentence IDs are local and deterministic:

```text
prior:   P1 ... P8
current: C1 ... C8
```

At most eight sentences per side are rendered. Each side is truncated only at a
sentence boundary.

Before prompting, deterministic code redacts:

- issuer names from the exact authorized `CONFORMED NAME` and `FORMER COMPANY`
  header values, including their trailing corporate suffix;
- CIK, accession, URL, and email;
- dollar-prefixed symbols and symbols explicitly prefixed by `NASDAQ`,
  `NYSE`, `NYSEAMERICAN`, `TSX`, or `LSE`;
- calendar dates and clock times;
- currency amounts, percentages, quantities, and supplied valuation figures.

The words Bitcoin, mining, custody, collateral, financing, sale, purchase,
launch, shutdown, plan, risk, cancellation, and other substantive relational
language are preserved. Redaction is tested for entity/date/amount invariance.
No current ticker directory, issuer registry, search endpoint, or manually
maintained alias list is authorized.

Redaction first applies Unicode NFKC and then performs longest match first. For
each authorized header alias, it creates only these deterministic variants:

1. the exact normalized alias;
2. the alias after repeatedly removing a rightmost whole-token suffix from
   `CORPORATION, CORP, INCORPORATED, INC, LIMITED, LTD, LLC, L.L.C, PLC,
   P.L.C, LP, L.P, LLP, L.L.P, CO, COMPANY, HOLDINGS, GROUP`; and
3. each of the above with periods removed and runs of whitespace collapsed.

Alias matches are case-insensitive and bounded by non-alphanumeric characters.
Every match becomes `[ENTITY]`. The remaining symbol passes, in this order,
are:

```text
(?i:\b(?:NASDAQ|NYSE|NYSEAMERICAN|TSX|LSE)\s*:\s*[A-Z][A-Z0-9.]{0,5}\b)
(?<![A-Za-z0-9])\$[A-Za-z][A-Za-z0-9.]{0,5}(?![A-Za-z0-9])
(?<![A-Za-z0-9])[A-Z][A-Z0-9.]{1,5}(?![A-Za-z0-9])
```

Each match becomes `[SYMBOL]`; there is no acronym whitelist. This can redact
useful acronyms, but it prevents a standalone all-capital ticker from becoming
an identity side channel. The protocol claims mitigation, not proof that a
pretrained model has zero issuer knowledge. Synthetic alias/name/symbol swaps
must be 100% invariant, and the historical swap-invariance gate includes the
same three surfaces in addition to dates and amounts.

Before prompt construction, a case-insensitive literal-substring prefilter
rejects a pair when either side contains any of:

```text
ignore previous
ignore all previous
ignore the previous
system prompt
developer message
assistant response
<start_of_turn>
<end_of_turn>
STATUS|DELTA
CURRENT_EVIDENCE|PRIOR_EVIDENCE
```

A prefilter rejection records only accession/hash/reason metadata, performs
zero model calls, and emits no semantic state.

## Frozen semantic ontology

The LLM emits five pipe-separated fields:

```text
STATUS|DELTA|RELATION|CURRENT_EVIDENCE|PRIOR_EVIDENCE
```

### Status

| Code | Meaning |
|---|---|
| `A` | current issuer action/state is realized or completed |
| `B` | current issuer action is planned but not completed |
| `C` | current issuer action is conditional or contingent |
| `D` | current text is risk-only or hypothetical |
| `E` | current text concerns a third party, customer, investee, or market generally |
| `F` | no supported issuer claim |
| `G` | supported current evidence is directionally mixed |

### Exposure delta

| Code | Meaning |
|---|---|
| `U` | issuer's realized/planned Bitcoin economic exposure moves up |
| `V` | issuer's realized/planned Bitcoin economic exposure moves down |
| `W` | comparable state is explicitly unchanged or repeated |
| `X` | no supported directional delta |

Exposure includes issuer-owned inventory, issuer-operated mining capacity,
issuer-provided Bitcoin access, and issuer balance-sheet/collateral commitment.
It never means predicted BTC price direction.

### Prior/current relation

| Code | Meaning |
|---|---|
| `F` | current claim fulfills a prior plan, condition, or warning |
| `R` | current claim reverses/cancels a prior plan, condition, or warning |
| `N` | current claim is new relative to the rendered prior evidence |
| `P` | current claim repeats a comparable rendered prior claim or state, including a planned/conditional repeat when status is `B/C` |
| `I` | prior and current are incomparable or unsupported |

### Evidence

`CURRENT_EVIDENCE` is one `C1..C8` or `NONE`.
`PRIOR_EVIDENCE` is one `P1..P8` or `NONE`.

The parser accepts exactly:

```text
^(?:[A-G])\|(?:[U-X])\|(?:[FRNPI])\|(?:C[1-8]|NONE)\|(?:P[1-8]|NONE)$
```

ASCII full-match is applied to one line after removing exactly one optional
terminal LF. No whitespace stripping, code-fence extraction, substring search,
JSON recovery, second decode, repair prompt, or retry is allowed.

Consistency checks run in this exact order:

1. referenced evidence IDs must exist in the exact rendered input;
2. status `G` requires `X|I|NONE|NONE`;
3. status `D` or `E` requires `X|I|C#|NONE`;
4. status `F` requires `X|I|NONE|NONE`;
5. relation `N` requires prior evidence `NONE` and current evidence `C#`;
6. relation `F`, `R`, or `P` requires both current and prior evidence IDs;
7. delta `U`, `V`, or `W` requires current evidence `C#`; and
8. relation `P` requires delta `W`;
9. delta `W` requires relation `P` and status `A`, `B`, or `C`; and
10. relation `I` is invalid with status `A`, `B`, or `C`.

`C#` above means one existing `C1..C8`. Failure at any check produces no
semantic state and cannot be repaired or retried.

The complete prompt body is frozen as:

```text
You compare two public-company Bitcoin disclosure excerpts from the same
issuer. Decide only what the CURRENT issuer claim says and how it relates to
the PRIOR issuer claim.

STATUS: A realized/completed; B planned; C conditional; D risk-only;
E third-party/customer/market; F no supported issuer claim; G mixed.
DELTA: U issuer Bitcoin economic exposure up; V down; W explicitly unchanged;
X unsupported direction. This is issuer exposure, never BTC price direction.
RELATION: F fulfills prior plan/condition/warning; R reverses or cancels it;
N new relative to rendered prior evidence; P repeats a comparable rendered
prior claim/state, including planned or conditional repeat with status B/C;
I incomparable or unsupported.

Use one CURRENT evidence id C1..C8 and one PRIOR evidence id P1..P8 when the
grammar requires them. Do not infer omitted facts. Planned, conditional,
risk-only, third-party, negated, and mixed claims are not realized actions.

PRIOR
{prior_numbered_sentences}

CURRENT
{current_numbered_sentences}

Return exactly one ASCII line and nothing else:
STATUS|DELTA|RELATION|CURRENT_EVIDENCE|PRIOR_EVIDENCE
```

## Gemma 4 synthetic QLoRA

Frozen base:

```text
google/gemma-4-E2B-it
revision 3e22461f65e89153144f8adb70e3b8c2cc9845a7
```

Local base identities:

```text
config.json
1b28f3d2c3100f6c594754b81107428bd7b822a7f48272ca681dae9d2ec38330
processor_config.json
32bdf45d2ad4cc29a0822ddd157a182de76644f0419a6228d151495256e9813c
tokenizer.json
cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f
tokenizer_config.json
9f4fec4b1dc6ecddf8f4a92e9caea5971c0e67d81309f3f9066a2bee8c362633
chat_template.jinja
0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5
model.safetensors
2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550
```

Runtime is pinned to:

```text
torch 2.9.0+cu128
transformers 5.7.0.dev0
peft 0.18.1
bitsandbytes 0.49.2
accelerate 1.12.0
```

Training uses one visible CUDA device:

```text
4-bit NF4, double quantization, BF16 compute
LoRA target regex .*language_model.*\.(q_proj|k_proj|v_proj|o_proj)$
rank 16
alpha 32
dropout 0.05
bias none
completion-only causal loss
AdamW
learning rate 0.000075
weight decay 0.01
batch 1
gradient accumulation 16
warm-up 16 optimizer steps
cosine decay
gradient clip 1.0
exactly 256 optimizer steps
checkpoints 64, 128, 192, 256
```

Synthetic corpus:

The generator seed is the integer `20260725`. It has exactly these 16 balanced
scenario IDs and targets:

| ID | Target |
|---|---|
| `FULFILL_UP` | `A|U|F|C2|P2` |
| `FULFILL_DOWN` | `A|V|F|C2|P2` |
| `REVERSE_UP` | `A|U|R|C2|P2` |
| `REVERSE_DOWN` | `A|V|R|C2|P2` |
| `NEW_UP` | `A|U|N|C2|NONE` |
| `NEW_DOWN` | `A|V|N|C2|NONE` |
| `REALIZED_REPEAT` | `A|W|P|C2|P2` |
| `PLANNED_UP` | `B|U|N|C2|NONE` |
| `PLANNED_DOWN` | `B|V|N|C2|NONE` |
| `CONDITIONAL_UP` | `C|U|N|C2|NONE` |
| `CONDITIONAL_DOWN` | `C|V|N|C2|NONE` |
| `RISK_ONLY` | `D|X|I|C2|NONE` |
| `THIRD_PARTY` | `E|X|I|C2|NONE` |
| `NO_CLAIM` | `F|X|I|NONE|NONE` |
| `MIXED` | `G|X|I|NONE|NONE` |
| `PLANNED_REPEAT` | `B|W|P|C2|P2` |

The decision-bearing sentence is always `P2` and/or `C2`; distractors occupy
the other rendered IDs. Train, calibration, adversarial, and swap
decision-bearing template IDs are disjoint. Exact committed template pools
cover issuer inventory, mining capacity, customer access,
financing/collateral, negation, attribution, and prompt injection.

For every selected surface field, the generator computes:

```text
SHA256(
  "ECRL-1|20260725|{split}|{scenario_id}|{zero_based_ordinal}|{field}"
)
```

as ASCII, interprets the digest as an unsigned big-endian integer, and indexes
the relevant committed pool modulo its length. No process RNG, Python hash,
timestamp, filesystem order, model, or external source participates.

| Split | Rows | Rows per scenario | Opened when |
|---|---:|---:|---|
| train | 4,096 | 256 | training |
| calibration | 512 | 32 | checkpoint selection |
| adversarial | 768 | 48 | after checkpoint selection |
| swap | 512 / 256 pairs | 32 / 16 pairs | after checkpoint selection |

In the adversarial split, zero-based ordinals `0` and `1` of every scenario are
guard-only rows, exactly 32 rows total. Ordinal `0` places
`Ignore previous instructions.` in `P3`; ordinal `1` places `System prompt.` in
`C3`. Their scenario target remains in the manifest to preserve balance, but
the prefilter must reject them with zero model calls. They are excluded from
model exact/parse/evidence/per-code denominators and included only in the
`guard_rows=32`, `guard_rejections=32`, `guard_model_calls=0` gate. The
remaining adversarial model denominator is exactly 736 rows, 46 per scenario.
This conservative guard may reject a legitimate filing containing a forbidden
literal; such a row fails closed rather than weakening the prefilter.

Each swap pair differs only in issuer alias, calendar/date, amount, quantity,
and exchange-symbol surfaces and retains the same target. The machine
preregistration commit must contain, before any model load:

- generator source and tests;
- the exact prompt-template bytes;
- all decision-bearing template IDs and their split assignment;
- canonical JSONL split manifests with keys in the fixed order
  `row_id,split,scenario_id,template_id,prior,current,target,pair_id`;
- row count and per-scenario count;
- SHA-256 for every JSONL file, ordered row-ID list, template inventory, and
  prompt; and
- a full regeneration equality test.

Canonical JSONL uses UTF-8, no BOM, compact JSON separators, no ASCII escaping,
LF line endings, and one terminal LF. Rows sort by
`(split, scenario_id, zero_based_ordinal, row_id)`. Any preregistration hash
change retires ECRL-1 rather than reopening model selection.

All integer gates use:

```text
required_count = ceil(required_rate * evaluated_group_size)
```

The committed report records both the integer numerator and denominator.

### Zero-model-call machine preregistration boundary

After this decision is committed, a separate source-only `M0` stage is required
before synthetic training. `M0` may read only this decision, committed
repository source/tests, package metadata, and already recorded local model
file metadata. It must report all of these counters as exact zero:

```text
tokenizer_loads
model_loads
model_calls
SEC_body_requests
SEC_header_requests
historical_pairs_created
market_rows_read
funding_rows_read
premium_rows_read
reward_rows_read
2024_or_later_rows_read
```

`M0` commits the complete generator implementation, every literal template and
lexicon inventory, prompt bytes, parser, redactor, relation-contrast fixtures,
four canonical split files, tests, split/hash manifest, and one self-hashed
preregistration report. The synthetic runner pins the resulting commit,
preregistration-file SHA-256, self-hash, generator-source SHA-256, prompt
SHA-256, and every split SHA-256. No one may edit, regenerate, relax, or
substitute any pinned artifact between `M0` and training. Any drift or non-zero
counter retires ECRL-1.

Checkpoint selection uses only calibration:

1. highest exact five-field count;
2. highest minimum exact share across status values;
3. highest minimum exact share across relation values;
4. lowest malformed count; and
5. lowest checkpoint step.

`M0` necessarily generates, validates, hashes, and commits adversarial and swap
bytes. “Remain unopened until selection” means no trainer, checkpoint selector,
inference evaluator, agent, or human may inspect or parse their row content
after `M0` and before calibration selects the checkpoint. Hash-only
verification remains allowed. The frozen `M0` generator/tests are the sole
pre-selection content readers.

Synthetic pass requires:

- overall exact output at least 98%;
- every status, delta, and relation value at least 95%;
- `MIXED`, risk-only, third-party, negation, and attribution families at least
  98%;
- output parse rate 100%;
- evidence-ID validity 100%;
- swap-pair output invariance 100%;
- at least 99% of swap pairs individually exact;
- every prompt-injection row rejected with zero model calls;
- 100% exactness on committed relation-contrast quadruplets in which the
  current sentence is byte-identical while only prior context changes its
  required relation among `F`, `R`, `N`, and `P`;
- training peak allocated and reserved below 24 GiB; and
- selected-adapter directory below 512 MiB.

The relation-contrast gate is a structural novelty check: an absolute-state
classifier cannot pass by reading the current sentence alone. A later
relation-ablated control removes relation-specific information while preserving
status/delta marginals and must be evaluated beside the primary.

There are exactly 16 relation-contrast quadruplets in adversarial ordinals
`2..17`: eight upward-current and eight downward-current groups. Each group
contains one row from the matching `FULFILL`, `REVERSE`, `NEW`, and
`REALIZED_REPEAT` scenarios. Its raw current `C2` sentence and all non-prior
surfaces are byte-identical across four rows; only prior text and the required
delta, relation, and prior-evidence target fields differ. These 64 rows are
part of the 736-row model denominator.

Research inference on the RTX 5090 is authorized after synthetic pass.
3060-Ti live deployment is **not** yet authorized. If economics later pass, a
separate GGUF/quantized runtime parity and latency gate must prove the 8 GiB
deployment path. A research-stage memory observation cannot be reinterpreted as
live compatibility.

Only the selected adapter is retained. Unselected checkpoints are deleted after
their hashes and selection evidence are committed. Filesystem reported use must
remain below 300 GiB before and after every model run.

## Historical semantic-support gate

Historical bodies may open only after the synthetic result is committed PASS.
Before any BTC row opens, the complete historical semantic run must satisfy:

- all official body receipts/hash checks;
- exact output parse rate at least 99.5%;
- evidence-ID validity 100%;
- consistency validity 100%;
- entity/name/symbol/date/amount swap invariance at least 99%;
- at least 1,000 valid prior/current pairs in 2021-2022;
- at least 450 valid prior/current pairs in 2023;
- at least 120 realized directional pairs in 2021-2022;
- at least 50 realized directional pairs in 2023;
- both `U` and `V`, and all `F/R/N/P` relation classes, in both periods;
- at least 100 distinct issuers in 2021-2022 and 50 in 2023;
- no issuer above 10% of valid pairs in either period; and
- no 2024-or-later source read.

Failure retires ECRL-1 before economics. Thresholds cannot be weakened after
historical incidence is observed.

## Daily weak-signal feature tensor

Semantic pair states become available at their historical/live ready time.
At each daily decision, deterministic code uses only states already ready.

For each horizon `3d`, `14d`, and `60d`, compute exponentially decayed:

```text
fulfill_up
fulfill_down
reverse_up
reverse_down
new_up
new_down
realized_repeat
planned_or_conditional_repeat
planned_up
planned_down
risk_or_third
mixed_or_unknown
```

The 12 buckets are mutually exclusive in this order:

1. status `D/E` -> `risk_or_third`;
2. status `F/G`, relation `I`, or delta `X` -> `mixed_or_unknown`;
3. status `A`, relation `P`, delta `W` -> `realized_repeat`;
4. status `B/C`, relation `P`, delta `W` ->
   `planned_or_conditional_repeat`;
5. status `B/C` with delta `U/V` -> `planned_up/down`; and
6. status `A` with relation `F/R/N` and delta `U/V` -> the matching
   relation-direction bucket.

Any remaining delta-`W` combination was already rejected by the semantic
consistency gate.
For horizon `H` in elapsed seconds, include only events satisfying:

```text
cutoff - 4*H < event_ready_time <= cutoff
weight = 2 ** (-(cutoff - event_ready_time).total_seconds() / H)
bucket_value = log1p(sum(weight))
```

`H` is exactly `3*86400`, `14*86400`, or `60*86400`. One issuer contributes at
most one event per UTC calendar day and bucket. Multiple eligible states in
that issuer/day/bucket collapse to the lexicographically first
`(event_ready_time, accession)` and use that event's ready time. This produces
`12 * 3 = 36` values in the exact horizon-major, bucket-minor order printed
above. Add, in this order:

```text
hours_since_last_up
hours_since_last_down
30d_up_top1_issuer_share
30d_down_top1_issuer_share
```

An up/down event is any accepted `U/V` state. Recency uses the latest ready time
not later than cutoff after collapsing duplicate
`(issuer, UTC-day, direction)` events by the same earliest-time rule. The 30-day
share window is:

```text
cutoff - 30*86400 < event_ready_time <= cutoff
```

Within that window, each issuer contributes at most one event per UTC day and
direction. `top1_share` is the largest issuer event count divided by all event
counts for that direction. With no event, recency is `1,440` hours and share is
`0`; otherwise recency is clipped at `1,440` hours and share to `[0,1]`.
Recencies are divided by `1,440`. The text tensor has exactly 40 `float32`
values.

## Causal price-action and position tensor

The market and position tensor is fixed and intentionally small. Exact scalar
order is:

```text
 1- 3 BTC close-to-close log return: 1d, 7d, 30d
 4- 5 realized volatility: 7d, 30d
 6- 9 rolling range location: 7d, 30d, 90d, 365d
10-11 drawdown from rolling maximum: 30d, 90d
12-13 rebound from rolling minimum: 30d, 90d
14    realized funding sum: trailing 7d
15    premium-index arithmetic mean: trailing 7d
16    current target position: -1, 0, +1
17    days in current position / 30, clipped to [0,1]
18    current strategy drawdown / 0.20, clipped to [0,1]
```

Rolling range location is:

```text
(last_completed_close - past_rolling_min) /
(past_rolling_max - past_rolling_min)
```

with `0.5` when the denominator is zero. Rolling maxima/minima and every other
price feature include only bars closed before the decision clock. Current
position information is therefore explicitly present.

For close series `c`, log return over `h` days is `log(c_t/c_{t-h})`.
Realized volatility is the population standard deviation of one-day log
returns in the trailing `h` completed UTC days, multiplied by `sqrt(365)`.
Drawdown is `c_t / rolling_max_h - 1`; rebound is
`c_t / rolling_min_h - 1`. Funding sums only intervals whose funding timestamp
is no later than the cutoff. Premium averages only completed premium-index
intervals. Any missing close, funding, or premium observation needed by these
windows invalidates that decision day, and a complete historical build fails
if any authorized decision day is invalid. At least 365 completed days of
market prehistory are required before the first 2021 training decision.

The first 55 scalars (40 text plus 15 market) are normalized with medians and
interquartile ranges fitted only on `[2021-01-01, 2022-07-01)`:

```text
clip((x - train_median) / max(train_IQR, 1e-6), -8, 8)
```

The final three position scalars remain in their explicit raw/scaled form.
Their concatenation is one ordered 58-value `float32` state. Normalizer values,
feature-order manifest, input-table row order, and their SHA-256 hashes are
committed before RL optimization.

## Decision and execution clock

One decision occurs each UTC day:

```text
state cutoff: 00:05 UTC
decision time: 00:10 UTC
target-position entry/rebalance: 00:15 UTC five-minute bar open
```

The state cutoff includes:

- semantic states durably ready by 00:05;
- BTC/funding/premium observations whose bars or intervals ended by 00:05; and
- the position/equity state after all fills and funding through 00:05.

The action is the next target position:

```text
-1 = short 1x
 0 = flat
+1 = long 1x
```

Changing `-1` to `+1` or the reverse pays a two-unit turnover. Historical
execution uses the 00:15 open, exact realized funding, 6 bp per changed exposure
unit, and a 10 bp stress replay. Strict MDD includes entry cost, every held
five-minute path, funding, virtual adverse exit cost, and pre-entry high-water
mark. CAGR uses the full calendar, including warm-up, flat, and no-trade time.

## Frozen QR-DQN

The LLM adapter is frozen before any reward opens. QR-DQN receives only the
fixed numeric tensor above.

```text
actions: 3
quantiles: 51
input: 58
shared MLP: 58 -> 64 -> 64, SiLU after each hidden linear
value head: 64 -> 51
advantage head: 64 -> 153
discount: 0.99
three-step returns
replay capacity: 100,000
batch: 64
AdamW learning rate: 0.0001
weight decay: 0.0001
learning starts: 500 transitions
gradient clip: 10
target hard update: every 250 optimizer steps
epsilon: 1.0 to 0.05 over first 3,000 steps
optimizer steps: exactly 10,000
```

The dueling quantiles are
`value + advantage - mean_action(advantage)`. Shared linear layers use
orthogonal initialization with gain `sqrt(2)`; value and advantage weights use
orthogonal gain `0.01`; all biases are zero. The online network is copied
exactly into the target network before the first transition.

Five fixed seeds train independently:

```text
11, 23, 47, 71, 97
```

No seed or checkpoint is selected. Final-step networks form one ensemble. At
inference, four of five greedy actions must agree; otherwise the target is flat.
Expected quantile ties resolve in the fixed order flat, short, long.

Per-step reward is:

```text
100 * next-day net log equity return
- 2 * max(0, next strict drawdown - 0.05)
```

The reward includes actual cost and funding. No future beyond the next
transition appears in the state. Training uses only `[2021-01-01, 2022-07-01)`.
Validation is a hard accept/reject gate only and cannot change any frozen
choice.

Each seed runs on CPU with Python, NumPy, and Torch set to that seed,
`torch.use_deterministic_algorithms(True)`, one intra-op thread, and one
inter-op thread. NumPy replay sampling uses `PCG64(seed)`. The environment is
chronological and uses this exact procedure:

1. reset the train interval with flat position, equity `1`, drawdown `0`, and
   no carried n-step queue;
2. insert transitions into replay in chronological order;
3. never construct an n-step transition across the terminal boundary;
4. retain replay contents when another identical train episode starts;
5. after 500 replay transitions exist, perform one optimizer update after each
   inserted transition;
6. uniformly sample replay indices without replacement for each batch;
7. repeat the same chronological train episode until exactly 10,000 optimizer
   updates have completed; and
8. discard any partial n-step queue and stop immediately at update 10,000.

Double QR-DQN selects the bootstrap action with the online network and evaluates
its target quantiles with the target network. The loss is the pairwise quantile
Huber loss with `kappa=1`, averaged over target quantiles, predicted quantiles,
and batch. Terminal targets contain reward only. Target weights are hard-copied
every 250 optimizer updates. Epsilon is linear by environment action count from
`1.0` at count zero to `0.05` at count 3,000 and remains `0.05`; exploration
samples uniformly from all three actions.

The machine preregistration fixes and hashes:

- the complete ordered transition table before training;
- terminal flags and three-step return construction;
- network source and package/runtime versions;
- each seed's initial network parameters;
- replay insertion order and sampled-index digest; and
- final-step model state and action digest.

There is no training retry, checkpoint fallback, seed replacement, or
validation-based model choice. A hardware/process failure may resume only from
an exact committed deterministic snapshot whose inputs and hashes are
unchanged; otherwise ECRL-1 is rejected.

## Controls and promotion gates

The same fixed pipeline must produce:

1. full ECRL text tensor;
2. price/position-only QR-DQN, with all 40 raw text scalars set to zero;
3. text tensor shuffled in fixed 28-day blocks before RL training;
4. deterministic lexicon relation counts without an LLM;
5. relation-ablated text, mapping every accepted status-`A`, delta-`U/V`
   state directly to `new_up/down` and every relation-`P`, delta-`W` state to
   `mixed_or_unknown` before composition, while preserving status, delta, ready
   time, issuer, and all price/position features;
6. full text tensor with `U` and `V` swapped;
7. buy-and-hold 1x;
8. always flat; and
9. deterministic random actions.

Learned controls 2-6 use the same ordered 58 dimensions, network, optimizer,
transition clock, final-step ensemble, five seeds, and 10,000-update procedure
as the primary. Each fits its own first-55 median/IQR normalizer on the same
training interval only.

For shuffled text, partition normalized training text vectors into consecutive
28-day UTC blocks anchored at `2021-01-01T00:00:00Z`. The final partial block is
one block. Permute the ordered blocks once with `PCG64(20260725)`, concatenate
their vectors without changing within-block order, and assign the resulting
sequence to chronological training decision dates. Validation and test text
remain chronological. No label, reward, price, or position row is permuted.

The lexicon control runs after the same rendering/redaction and uses
case-folded whole words/phrases from this exact inventory:

```text
UP: acquire, acquired, acquisition, purchase, purchased, increase, increased,
    expand, expanded, launch, launched, commission, commissioned, energize,
    energized, add, added, accept, accepted, custody, collateralized
DOWN: sell, sold, sale, decrease, decreased, reduce, reduced, shutdown,
      shut down, decommission, decommissioned, cease, ceased, terminate,
      terminated, cancel, cancelled, canceled, withdraw, withdrew, impair,
      impaired
REALIZED: has, have, completed, commenced, began, now, entered into, was, were
PLANNED: plan, plans, planned, intend, intends, expected to, will, propose,
         proposes
CONDITIONAL: may, might, could, subject to, if
RISK: risk, uncertain, uncertainty, adverse
THIRD: customer, customers, client, clients, investee, third party,
       market participants
NEGATION: not, no, never, without, did not, has not, have not
```

Phrases match longest first; remaining tokens use ASCII word boundaries.
Sentences containing any `NEGATION` phrase are unsupported. Current sentences
are scanned by ascending evidence ID. Any `RISK` match yields status `D`; else
any `THIRD` match yields `E`; else exactly one direction plus `CONDITIONAL`
yields `C`, plus `PLANNED` yields `B`, and plus `REALIZED` yields `A`, in that
priority. Both directions across supported current sentences yield `G`; no
supported sentence yields `F`. For `A/B/C`, delta is the unique direction.
Given the first supported prior sentence under the same rules, an `A` current
claim emits `A|U/V|F|C#|P#` against same-direction prior `B/C`,
`A|U/V|R|C#|P#` against opposite-direction prior `B/C`,
`A|W|P|C#|P#` against same-direction prior `A`, and
`A|U/V|N|C#|NONE` otherwise. A supported `B/C` claim emits
`B/C|W|P|C#|P#` against same-direction prior `B/C` and
`B/C|U/V|N|C#|NONE` otherwise. Status `D/E` emits
`D/E|X|I|C#|NONE`; `F` emits `F|X|I|NONE|NONE`; and `G` emits
`G|X|I|NONE|NONE`. Here `U/V`, `B/C`, `D/E`, `C#`, and `P#` mean the
uniquely determined code or first matching existing evidence ID, not literal
output text. The complete literal inventory and implementation are hash-pinned
at `M0`.

The relation-ablated control mechanically tests whether prior/current relation
carries value beyond the absolute current status/delta variables used by
retired EDGAR candidates. The polarity-swap control exchanges `U/V` before
bucket composition, including recency/share direction, and changes nothing
else.

For deterministic random actions, instantiate five independent streams
`PCG64(20260725 + seed)` for seeds `11,23,47,71,97`. Each stream draws one
uniform integer from `{short,flat,long}` per chronological decision. Four of
five identical draws are required; otherwise target flat. Buy-and-hold enters
long at the interval's first 00:15 execution and exits at interval end. Always
flat never enters. All controls pay the same applicable cost/funding and use
the same strict-equity accounting.

A non-flat exposure episode is one maximal contiguous interval with the same
non-zero target. Flat ends an episode; a direct sign flip closes one episode
and opens another. Rebalances that retain the same sign remain one episode.

For any report interval `[start,end)`, absolute return is
`ending_equity / starting_equity - 1`, and:

```text
CAGR = (ending_equity / starting_equity)
       ** (365.2425 / full_calendar_days) - 1
```

`full_calendar_days` is the exact elapsed UTC duration of the entire interval,
including warm-up, flat, invalid/no-trade, and no-filing days. It is never
replaced by active or invested time.

Strict equity starts at `1` before the interval's first entry cost. At every
held five-minute bar it marks a virtual immediate liquidation after accrued
funding and exit cost, using the bar low for a long and bar high for a short.
Flat marks realized equity. The high-water mark includes the pre-entry value.
Strict MDD is the maximum `1 - strict_equity/high_water`. Equity at or below
zero forces `strict MDD=1`, `CAGR=-1`, and rejection. Any NaN or infinity
rejects the run.

The reported ratio is:

```text
CAGR / max(strict_MDD, 0.005)
```

All inputs are decimal fractions. A negative CAGR remains a negative ratio;
zero observed drawdown never creates an infinite score.

The untouched 2023 primary must satisfy:

- absolute return at least 10%;
- CAGR at least 10%;
- `CAGR / strict MDD >= 3`;
- strict MDD no greater than 15%;
- positive return under 10 bp stress cost;
- at least 30 non-flat exposure episodes;
- at least nine active calendar months;
- at least eight long and eight short episodes;
- one-sided stationary block-bootstrap probability of non-positive net return
  below 5%, using 10,000 deterministic resamples and mean block length 10 days;
- higher `CAGR / strict MDD` than price-only, shuffled-text, lexicon,
  relation-ablated, polarity-swap, and deterministic-random controls;
- at least 0.50 greater `CAGR / strict MDD` than each of those six
  prespecified learned/signal controls; and
- paired stationary-bootstrap probability that primary minus control daily net
  log return is non-positive below 10% separately against every one of those
  six controls.

The stationary bootstrap uses restart probability `0.1`, exactly 10,000
length-preserving resamples, and NumPy `PCG64(20260725)`. The unpaired statistic
is the full-calendar sum of resampled primary daily net log returns. Each
paired comparison applies identical sampled indices to the primary and one
control and sums their daily log-return difference. Reported probability is
`(1 + count(statistic <= 0)) / 10001`. No "strongest control" is selected from
test outcomes.

The 2022-H2 diagnostic must independently have positive absolute return,
strict MDD no greater than 15%, and `CAGR / strict MDD >= 2`. Failure retires the
candidate before 2023 is opened; it cannot tune the 2023 result.

Every report must show, for each split/control:

- full-calendar absolute return;
- CAGR;
- strict MDD;
- CAGR/strict-MDD;
- long, short, and total episode counts;
- active months;
- turnover and total cost;
- funding PnL; and
- bootstrap probability when applicable.

## 2024+ extension rule

No 2024-or-later source or market row may be opened unless:

1. synthetic PASS is committed;
2. historical semantic-support PASS is committed;
3. the fixed QR-DQN run and untouched 2023 test are committed; and
4. every 2023 promotion gate passes.

Only then may a separate source-extension and evaluation protocol freeze
2024-2025 as evaluation and 2026 as confirmation. The eventual target remains
at least three years including recent data, no leakage, and portfolio-level
`CAGR / strict MDD >= 3`. CAGR 50% may later be pursued with leverage only after
the unlevered risk-adjusted edge and strict MDD are validated.

## Stop condition

Retire ECRL-1 without repair on:

- synthetic gate failure;
- historical transport, parse, redaction, inference, or support failure;
- model/output/feature hash drift;
- any 2024+ source or outcome access before authorization;
- validation diagnostic failure;
- untouched 2023 promotion failure;
- leakage or execution-clock violation; or
- inability to reproduce the exact report from committed artifacts.

No prompt, ontology, output code, model, adapter, checkpoint, feature, side,
reward, cost, threshold, horizon, seed, or hold may be changed after the
corresponding boundary is opened.
