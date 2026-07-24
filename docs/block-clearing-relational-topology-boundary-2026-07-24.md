# BCRT-72 candidate boundary — block-clearing relational topology

## Selection

Select one new source-seen, outcome-unseen policy axis:

**BCRT-72 — Block-Clearing Relational Topology Policy**.

BCRT observes a causally delayed twelve-hour Bitcoin blockspace state. It
combines several individually weak, price-independent relations between block
cadence, weight use, fee burden, transaction packing, witness discount, and
UTXO-set pressure. One compact policy chooses exactly one action:

```text
LONG
SHORT
ABSTAIN
```

The provisional causal chain is:

1. aggregate only a complete twelve-hour UTC block bucket;
2. delay that bucket by a conservative fixed publication embargo;
3. compare its blockspace statistics only with strictly prior source buckets;
4. emit a compact categorical state describing levels, disagreements, and
   transitions among the weak blockspace relations;
5. let one small RLLM rank long, short, and abstain without a source-owned
   direction; and
6. enter only after one complete five-minute latency bar and exit after exactly
   72 five-minute bars.

The `72` suffix reserves a six-hour consequence horizon. The exact aggregation,
rank history, token grammar, reservation clock, support floors, reward,
baselines, model, preference/RL method, costs, controls, and qualification
gates must be frozen in a separate mechanism commit before any BCRT-derived
source value, token, incidence, label, market outcome, or post-2023 row is
opened.

This boundary is not an alpha result. It selects a falsifiable state-policy
experiment.

## Why this is not a BATE, UFCP, FETD, BFRT, or WCTR repair

Prior Bitcoin base-chain candidates disclosed that the source family is
causally usable, but their exact predictive objects were different:

- `BATE-288` was a sparse, fixed-direction throughput-elasticity event with a
  24-hour hold. Its 2020–2022 economics were rejected with 81.97% strict MDD
  and 0.17 CAGR/MDD. Its thresholds, side, packet size, latency, and hold may
  not be repaired.
- `UFCP` selected a daily UTXO/fee polarity axis and froze deterministic
  directional tests. BCRT does not inherit a UFCP side or threshold.
- `FETD-288`, `BFRT-288`, and `WCTR-288` were fixed-rule tail events. They were
  retired at source support and may not be relaxed or promoted through their
  controls.
- `QLCD` used Binance aggregate-quantity cohorts rather than the confirmed
  Bitcoin ledger and later failed train economics.

BCRT changes the predictive object:

- every valid delayed twelve-hour bucket is an opportunity state rather than
  only a tail event;
- no source relation owns a long or short side;
- no conjunction is required for eligibility;
- weak source relations remain simultaneous categorical inputs instead of
  being collapsed into one hand-written score;
- abstention is a learned action, but it does not release the globally reserved
  opportunity clock;
- the hold is six hours rather than the failed 24-hour BATE/FETD/WCTR hold; and
- BCRT receives a new identity, preregistration, source-support gate,
  learnability gate, novelty gate, and immutable failure action.

Known BATE train outcomes make this source family globally outcome-seen. BCRT
therefore makes no clean-room discovery claim. No BCRT token, clock, label,
return, PnL, or 2023 outcome has been opened.

## Why this is an RLLM-shaped task

BCRT is not a raw numeric price forecast. The policy must reason over
compositions such as:

- whether blocks are arriving quickly while remaining unusually full;
- whether fee burden agrees with or diverges from weight utilization;
- whether transaction packing agrees with witness discount;
- whether UTXO-set expansion occurs under high or low fee pressure;
- whether dispersion across blocks is broadening or compressing; and
- whether the current topology persists, rotates, or reverses relative to the
  immediately prior valid state.

No single relation is claimed to be strong. The hypothesis is that their joint
causal grammar changes the conditional utility of long, short, and abstain.

BCRT remains one policy model. There is no analyzer/trader pair, free-form
chain of thought, model-generated feature, or model-controlled accounting.
Source validation, rank construction, scheduling, costs, reward calculation,
risk metrics, and execution remain deterministic code.

A language model is justified only if frozen cheap causal baselines demonstrate
transferable structure and the RLLM improves over them without using 2023 for
selection. Failure at any preregistered gate retires BCRT-72 unchanged.

## Frozen source family

Primary normalized source:

```text
data/bitcoin_utxo_fee_block_stats_2020_2023.csv.gz
SHA256 8d5160b59bd104819adb34752f2bd5b01e9c07d7aa108188abcd9e6d323d102f
```

Source manifest:

```text
results/bitcoin_utxo_fee_block_stats_source_manifest_2026-07-20.json
file SHA256 ceb096b7bfeaf309729c4cab9f5cd4d2e0d526e261b1e460e9d03cd4f24e8084
manifest_hash 98a84b0bd0338300f62eaa047b87498cc5a8d9505a03f6bd1912d1deb9564e8c
```

Independent basic-field reference:

```text
data/bitcoin_block_summaries_2020_2023.csv.gz
SHA256 1f8d1c0153717f2c18d8fa6f09428c780a850b2aecc8fb42bc497e16e68e1833
```

The manifest already discloses a contiguous 213,095-block source from height
610,691 through 823,785, entirely before 2024, with linked parent hashes and
all basic fields cross-checked against the independent block-summary artifact.
Those aggregate source facts are prior knowledge, not BCRT incidence.

The allowed source columns are limited to:

```text
height
id
previousblockhash
timestamp
mediantime
tx_count
size
weight
total_fees
total_inputs
total_outputs
utxo_set_change
```

The mechanism must use an explicit allowlist and fail closed on schema or hash
drift. Load-and-drop of market price, mark/index price, funding, premium, OI,
future return, label, action, reward, PnL, portfolio, or post-2023 data is
forbidden.

The research transport is the frozen public Mempool REST cache. Production
must reproduce the same consensus-derived fields from an owned Bitcoin Core
node and record actual first-seen times. The live clock must use the later of
actual causal availability and the frozen conservative historical clock.

Official source references already bound by the source manifest:

- <https://mempool.space/docs/api/rest>
- <https://developer.bitcoin.org/reference/rpc/getblockstats.html>

## Provisional opportunity clock

The mechanism may aggregate completed blocks into exact twelve-hour UTC buckets
using miner-reported header timestamps, but it may not treat those timestamps
as publication times.

The next mechanism must freeze one conservative clock before source values are
decoded. The permitted clock family is:

1. require a complete retained twelve-hour bucket and the confirmations or
   successor rows frozen by the mechanism;
2. make the bucket unavailable no earlier than its UTC end plus 48 hours;
3. round availability upward to a five-minute boundary;
4. consume one additional complete five-minute latency bar;
5. enter only at the following five-minute open;
6. reserve opportunities globally in chronological order; and
7. require the full 72-bar hold to remain inside one split.

The first and last source buckets needed for incomplete aggregation,
confirmation, latency, or hold must be discarded. Missing or discontinuous
source history rejects the state rather than widening a window.

## Provisional relation families

The exact formulas and categories are not authorized until the mechanism
commit. The mechanism may select one compact grammar from these source-only
families:

1. block cadence;
2. block-weight utilization;
3. transaction packing per unit weight;
4. fee burden per unit weight;
5. UTXO-set change per transaction;
6. witness discount derived from size and weight;
7. within-bucket load or fee dispersion;
8. cadence versus utilization;
9. utilization versus fee burden;
10. packing versus witness discount;
11. UTXO pressure versus fee burden;
12. current topology versus the immediately prior valid topology; and
13. current BCRT sleeve position and executability.

Every ordinal category must use a strictly prior fixed-length reference with a
minimum-history floor. Date, year, month, price, return, funding, position PnL,
portfolio state, prior model action, future source row, and post-entry path are
forbidden model inputs.

## Alternatives considered but not selected

### SEC EDGAR metadata topology

The frozen SEC source is broad and causal: 2,493 emittable accessions across
992 event days through 2023. It remains a viable future axis. It is not selected
now because previous semantic adapters failed before historical execution, and
a metadata-only policy would still require issuer-identity controls and
cross-sectional semantics that are less direct to reproduce live than Bitcoin
Core block statistics.

### OFR preliminary repo segmentation

The preliminary-vintage OFR panel is usable with restrictions. Three recent
mechanisms nevertheless collapsed under sparse or concentrated intersections.
A future candidate should use a new single-segment geometry, not another
all-confirmed event. It is not selected now because release-vintage operations
and revision monitoring add production risk while providing fewer policy
states.

### New York Fed SOMA securities-lending allocation

The official panel is dense and complete, with 182,616 CUSIP-detail rows.
Prior failures were mechanism failures rather than source failures. It remains
the strongest macro alternative, but its conservative next-midnight release
clock and CUSIP-distribution semantics make it better suited to a separate
slow regime sleeve than the next compact six-hour RLLM policy.

### Federal-liquidity and CBOE cross-surface successors

These families retain plausible weak structure, but recent candidates exposed
substantial overlap with existing macro/option objects and source-composition
collapse. They remain secondary until BCRT is resolved.

## Mandatory stopping rules

1. Commit an exact mechanism and preregistration before deriving any BCRT
   feature or incidence.
2. Run source continuity, causal-clock, state-support, temporal-dispersion, and
   action-independent reservation checks before market outcomes.
3. Compare tolerant-time containment and signed exposure correlation with all
   available Gross-8 sleeves before economic promotion.
4. Open one frozen 2020–2022 development evaluator only after source and
   novelty pass.
5. Keep 2023 untouched by model, threshold, prompt, checkpoint, or control
   selection.
6. A failed source, learnability, economic, delayed-entry, cost-stress, strict
   MDD, or novelty gate retires BCRT-72 unchanged.
7. Controls diagnose failure and may not replace the primary after inspection.
8. No threshold, side, token, latency, hold, reward, model, or support repair is
   permitted after its corresponding information is opened.

## Outcome boundary at this commit

This decision uses only prior documents, disclosed source schemas, hashes, and
aggregate source-audit counts.

```text
BCRT-derived feature values read = 0
BCRT token rows created          = 0
BCRT opportunity clocks opened  = 0
BTC market rows read             = 0
funding rows read                = 0
future-return rows read          = 0
return or PnL fields read        = 0
post-2023 source rows read       = 0
model labels created             = 0
model training runs              = 0
```

Selection status:

```text
selected_for_mechanism_freeze
```
