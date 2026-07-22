# BIRB-120 source-support rejection — 2026-07-23

## Decision

**Reject `BIRB-120-SOURCE-FAMILY-SEEN` at source support. Do not open BTC
outcomes and do not repair this identity.**

The frozen source-only builder was executed twice and reproduced byte-identical
artifacts:

- report:
  `results/sec_bitcoin_issuer_reactivation_breadth_support_2026-07-23.json`;
- report SHA-256:
  `752e77022e8d670084327680da4e8d60d753a344dcbef754b592493ffa9bfec6`;
- report manifest:
  `ed0336f1328735973331eec99848c774e0d11fd6c672c9dd4fdbd893438f779d`;
- source/control clock:
  `data/sec_bitcoin_issuer_reactivation_breadth_2020_2023/birb120_support_clocks_2020_2023.csv.gz`;
- clock SHA-256:
  `8f0831120764793a06873dc7ed4e1b97d3deff75d89572e2b4b8f9459bdfea41`.

The first run completed in 0.18 seconds at about 30 MiB maximum RSS.

## What the source showed

The audited source contained 3,496 exact document hits, 2,543 unique
accessions, 2,493 eligible non-amendment accessions, and 308 issuers. Only 38
issuer events met the frozen `>=365` elapsed-day reactivation definition.

After the seven-day, three-distinct-issuer first passage and fixed 120-hour
global nonoverlap, only **two** primary events remained:

| Split | Events | Distinct constituent issuers | Coverage |
|---|---:|---:|---|
| train 2020–2022 | 2 | 6 | one event in 2020Q1, one in 2021Q1 |
| selection 2023 | 0 | 0 | none |

The train gap was 423.19 days. Month and quarter concentration were each 50%.
The primary failed train count, every-train-year, active-quarter, issuer,
concentration, gap, selection count, selection issuer, and both 2023-half
gates. Identity-integrity checks passed.

The fixed breadth controls reinforce that the mechanism is structurally too
sparse rather than narrowly missing one threshold:

| Clock | Total accepted | Train | 2023 |
|---|---:|---:|---:|
| primary threshold 3 | 2 | 2 | 0 |
| threshold 2 | 4 | 3 | 1 |
| threshold 4 | 0 | 0 | 0 |
| single reactivation | 25 | 17 | 8 |
| first-ever birth breadth | 30 | 26 | 4 |
| any-mention breadth | 34 | 30 | 4 |
| repeat-filer breadth | 32 | 25 | 7 |

The source-specificity controls passed, but that cannot rescue an
unidentifiable primary. Several dense comparators also exceeded the ±12-hour
novelty-containment cap at 50%; with only two primary events this is one matched
event and is secondary to the decisive support failure.

## Interpretation

The economically interesting object—multiple dormant issuers independently
returning after a full annual cycle—is too rare in the 8-K/6-K Bitcoin-hit
stream. Lowering the breadth threshold, shortening the 365-day absence, or
loosening concentration would not validate BIRB; it would define a new and more
generic attention clock after observing this exact incidence.

The result does retain one useful source fact for future clean-room design:
single issuer reactivations have enough raw incidence (25 accepted events,
including 8 in 2023), but their direction cannot be inferred from metadata.
Any future use therefore needs a genuinely new preregistered semantic or
cross-source confirmation object, not a BIRB threshold repair.

## Outcome boundary

The run read the frozen SEC metadata values and hash-bound comparator
timestamps only. It fetched no filing bodies and read zero BTC market, funding,
future-return, or post-2023 source rows. It computed no PnL, CAGR, or MDD and
made no network or subprocess calls. Calendar 2024 onward remains sealed for
this family.
