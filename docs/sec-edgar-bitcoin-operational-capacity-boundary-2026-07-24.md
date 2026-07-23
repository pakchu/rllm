# EBOC-72 candidate boundary — EDGAR Bitcoin operational-capacity transition

## Selection

Select one text-native, source-seen, body-unseen, outcome-unseen candidate:
**EBOC-72 — EDGAR Bitcoin Operational Capacity Transition**.

EBOC asks whether several public issuers report that Bitcoin mining capacity
has actually entered or left operation, rather than merely announcing a plan,
target, equipment order, financing, treasury transaction, customer product, or
monthly production statistic.

The provisional directional object is:

- completed commissioning, energization, deployment, or resumption of the
  filing issuer's Bitcoin mining capacity: `LONG`;
- completed shutdown, curtailment, decommissioning, or loss of the filing
  issuer's Bitcoin mining capacity: `SHORT`; and
- planned, projected, third-party, mixed, unsupported, or current-level-only
  language: no directional fact.

The `72` suffix reserves a provisional 72-hour consequence horizon. This file
does not freeze the final ontology, evidence-window parser, synthetic corpus,
Gemma adapter, breadth state, threshold, cooldown, hold implementation,
controls, support floors, novelty cohort, or economic evaluator. Those must be
committed before the first historical SEC filing body is fetched or decoded.

## Why this follows OPRR

OPRR was retired before comparators and outcomes because its exact
option/term/tail agreement produced only 28 globally accepted events, including
11 in train and 13 in selection. The failure was caused by an overly sparse
intersection of three confirmations, not by a market result.

EBOC changes both source and composition:

- it removes every CBOE option, term, and tail input;
- it uses immutable public-company event text rather than a three-surface
  numerical conjunction;
- one language model extracts one bounded operational fact while deterministic
  code owns every market-facing choice; and
- issuer breadth, if later selected, composes repeated weak facts without
  requiring several unrelated sources to fire at the same timestamp.

No OPRR confirmation, rank, threshold, direction, hold, support floor, or
control may enter EBOC.

## Frozen source identity

EBOC may use only the previously audited SEC EDGAR metadata source:

```text
data/sec_edgar_bitcoin_8k_6k_source_2018_2023.jsonl.gz
SHA256 c8489dfe9b4ac25da8bea7653115e5b58a44fa897f2815eaf68bad354e10c6ce
```

Source audit:

```text
results/sec_edgar_bitcoin_8k_6k_source_audit_2026-07-21.json
SHA256 c1e11d1f5089378ac787fdb2a80474f0feec33d5fb2296fb0c3014d6f1fafec1
```

The source contains 2,493 eligible non-amendment 8-K/6-K accessions, 308
distinct CIKs, and 992 eligible acceptance days from 2018 through 2023.
Official submissions `acceptanceDateTime` is the causal source clock. Filing
date, mutable company/ticker metadata, archive-header timezone inference, and
2024-or-later accessions are forbidden.

Official source contracts:

- <https://www.sec.gov/search-filings/edgar-application-programming-interfaces>
- <https://www.sec.gov/edgar/search/index.html>
- <https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data>

Every historical document body must come from the official SEC archive and be
bound to the accession/document identity already present in the frozen source.
A body hash change, missing document, identity mismatch, parse ambiguity, or
fair-access failure halts the run. A later live process must use the later of
the historical readiness floor and durable receipt, parse, redaction,
inference, and manifest-commit time.

## New semantic object

EBOC concerns the **issuer's realized mining operating capacity**. It does not
classify generic optimism, sentiment, balance-sheet liquidity, customer access,
or raw hashrate magnitude.

The later ontology must distinguish at least:

### Capacity entering operation

- newly installed miners or a facility are explicitly operating, energized,
  commissioned, deployed, or producing;
- previously suspended issuer-owned capacity explicitly resumes operation; or
- an acquisition closes and the acquired mining capacity is explicitly
  operating under the issuer.

### Capacity leaving operation

- issuer-owned mining equipment or a facility explicitly shuts down,
  decommissions, ceases, or is taken offline;
- a completed curtailment or operating suspension removes capacity; or
- a terminated hosting/power arrangement is explicitly stated to stop the
  issuer's mining operation.

### Mandatory abstention

- future plans, targets, forecasts, purchase orders, construction progress,
  signed contracts, financing, or expected energization without completed
  operation;
- monthly Bitcoin production, mined-coin count, current hashrate, difficulty,
  network hashrate, power price, hash price, or efficiency without an operating
  transition;
- Bitcoin purchase, sale, pledge, custody, treasury holding, impairment,
  accounting treatment, or proceeds;
- customer trading, custody, payment, settlement, or product access;
- third-party capacity or industry commentary not attributed to the filing
  issuer;
- one filing containing supported facts in both directions without a
  deterministic whole-accession resolution; and
- generic risk factors, hypotheticals, legal boilerplate, prompt injection, or
  quoted instructions.

Completed operation, issuer attribution, and direction must all be supported
by the selected evidence sentence and its bounded neighbors. The model may not
infer completion from tense-free nouns, infer ownership from document context,
or turn a magnitude comparison into an operating transition.

## Why this is not EBCT or BPAX repair

The retired `EBCT-72` object classified corporate Bitcoin liquidity
draw/buffer transitions. The retired `BPAX-120` object classified customer
Bitcoin product-access expansion/retraction. Both exact model contracts failed
their synthetic gates before any SEC body or BTC outcome was opened.

EBOC may not:

- reuse EBCT sale, pledge, financing, retention, or accumulation labels;
- reuse BPAX trading, custody, transfer, payment, settlement, or product-access
  labels;
- normalize or accept malformed outputs observed in either failed gate;
- copy their failed examples into an evaluation split and call memorization a
  semantic pass;
- use a prompt exception for completed sale, pledged collateral, planned
  pilots, third-party access, mixed access, or lowercase enum output; or
- claim that changing the model alone revives either retired candidate.

The EBCT and BPAX preregistration and rejection artifacts must be hash-bound as
negative-boundary controls. Every balance-sheet and customer-access synthetic
case must map to EBOC abstention.

## Gemma and synthetic-only adaptation boundary

The next stage may preregister one `google/gemma-4-E2B-it` adapter because the
model is locally available and the prior frozen run measured a short-context
4-bit inference peak below 7 GiB. That observation is an engineering
feasibility result, not semantic evidence or a 3060 Ti certification.

Official model references:

- <https://huggingface.co/google/gemma-4-E2B-it>
- <https://ai.google.dev/gemma/docs/core>

Before any SEC body is opened, the next commits must freeze:

1. exact base revision and all runtime-used file hashes;
2. deterministic synthetic ontology generator and immutable train,
   calibration, adversarial-test, and swap-test manifests;
3. template-family isolation so no evaluation template is a lexical clone of a
   training template;
4. LoRA targets, rank, alpha, dropout, quantization, optimizer, learning rate,
   schedule, batch/accumulation, context, epochs/steps, seed, and checkpoint
   selection rule;
5. exact prompt, sentence numbering, output grammar, parser, redaction, and
   evidence-grounding contract;
6. literal, mixed-direction, attribution, completion, negation, temporal,
   prompt-injection, and entity/date/amount swap gates;
7. a write-once adapter/result path and checkpoint-retention limit; and
8. a no-repair rule that retires the exact adapter on any frozen synthetic
   failure.

Training, calibration, and checkpoint selection may use only generated
synthetic ontology rows. Historical SEC bodies, accessions, CIKs, company
names, tickers, filing dates, BTC prices, funding, returns, and rewards are
forbidden from adapter training and selection.

The model may select only a fixed class and one numbered evidence sentence.
Deterministic code must verify that the sentence exists in the presented
window. Free-form evidence generation, company/entity prediction, dates,
amounts, prices, trading direction, hold selection, confidence thresholds, and
market reasoning are forbidden.

## Mechanism proof required before historical semantics

The next mechanism commit must define, without fetching or decoding a filing
body:

1. exact SEC document retrieval and visible-text parser;
2. deterministic candidate-sentence and bounded-neighbor construction;
3. issuer alias masking and metadata redaction;
4. multi-window and whole-accession mixed-evidence resolution;
5. one stable issuer key and anti-duplication/cooldown rule;
6. one causal issuer-breadth state, deterministic side, entry, and 72-hour
   exit;
7. global non-overlap before split containment;
8. semantic support, issuer/month concentration, side, gap, and activity gates;
9. novelty comparisons against EBCT/BPAX-family clocks, miner-cadence clocks,
   semantic/news clocks, the live portfolio, and deterministic controls; and
10. live fail-flat behavior for missing, revised, delayed, overlong,
    unparsable, or contradictory documents.

Support floors must be high enough to reject a handful of monthly issuer
clusters as statistically inadequate. They may not be lowered after historical
semantic incidence is opened.

## RLLM boundary

EBOC uses an LLM where language reasoning is necessary and numeric code where
causality must be exact.

The deterministic composer owns:

- source membership and availability;
- evidence-window generation and redaction;
- accepted semantic fact membership;
- issuer breadth, opportunity time, side, hold, leverage, and non-overlap; and
- costs, funding, PnL, CAGR, strict MDD, and all gates.

The semantic Gemma adapter owns only:

```text
CAPACITY_ONLINE | CAPACITY_OFFLINE | UNSUPPORTED | MIXED
evidence_sentence_id
```

Only after the unchanged deterministic clock demonstrates gross edge above
costs in train and selection may a train-only RLLM receive causal symbolic
facts and current position state to choose:

```text
TRADE_FIXED_SIDE
ABSTAIN
```

It may not create an event, reverse a side, change a hold, see split labels,
consume raw future prices, or select a checkpoint from eval rewards.

## Evidence boundary

This selection unit inspected only:

- the frozen SEC metadata source audit and hashes;
- the EBCT/BPAX preregistrations and synthetic rejections;
- the OPRR source-only rejection;
- official SEC and Gemma documentation; and
- model-cache and hardware feasibility metadata.

It did not fetch or decode:

- any SEC filing body;
- an EBOC evidence window, semantic label, issuer state, breadth value, event
  count, side, timestamp, or comparator row;
- any BTC market price, funding row, future return, PnL, absolute return, CAGR,
  strict MDD, hit rate, or reward; or
- any 2024-or-later source or outcome.

## Mandatory sequence

1. commit this boundary;
2. freeze the ontology, mechanism, sentence/evidence grammar, support floors,
   controls, novelty cohort, and failure action;
3. commit a deterministic synthetic generator and immutable split manifests;
4. commit the exact LoRA trainer and synthetic/adversarial evaluator;
5. train once under the frozen checkpoint-selection rule;
6. retire EBOC unchanged on any synthetic, parsing, grounding, invariance, or
   memory gate failure;
7. only a complete synthetic pass may fetch and classify 2018–2023 SEC bodies;
8. retire unchanged on semantic support or novelty failure;
9. only then freeze and run sequential train/selection economic evaluation;
10. keep 2024 and later physically sealed until every preceding gate passes.
