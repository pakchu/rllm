# UFCP-1 preregistration — 2026-07-20

Status: **frozen before real UFCP event incidence or any post-entry outcome was
opened**.

## Singleton

For each completed UTC source day `D`:

```text
edges         = sum(total_inputs + total_outputs)
fee_burden    = log(sum(total_fees) / edges)
utxo_polarity = sum(utxo_set_change) / edges
```

Both features receive a strict-prior 180-source-day empirical midrank with 120
days required.  The exact midrank is
`(count(prior < current) + 0.5*count(prior == current)) / prior_count`.

- long: `fee_rank >= 0.75` and `polarity_rank >= 0.75`;
- short: `fee_rank >= 0.75` and `polarity_rank <= 0.25`.

Every eligible day is considered in chronological order.  A source day needs
at least 72 blocks, positive fee/edge totals, no missing usable source day, and
six hash-linked successor blocks.

## Causal execution

- source day `D` is unavailable before `D+2 00:00 UTC`;
- entry is `D+2 00:05 UTC`, after one complete five-minute latency bar;
- hold is 288 five-minute bars / one day;
- exposure is 0.5x;
- base cost is 6 bp/notional/side and stress cost is 10 bp/notional/side;
- funding is exact, entry-inclusive/exit-exclusive, with fixed entry quantity;
- signals reserve a chronological non-overlapping clock.

Header `firstSeen`, pool identity, unconfirmed mempool state, and post-entry
blocks are forbidden.  The deliberately delayed daily schedule is mandatory
because historical node receipt time is unavailable.

## Outcome-blind support gate

- 2021-2022 train: at least 60 events and 24 in each year;
- 2023 selection: at least 24 events and 10 in each half;
- each side: 25-75% in train and separately in selection;
- largest month: at most 15% in train and separately in selection;
- every usable UTC source day has at least 72 blocks and no day is missing.

The support stage may read the frozen confirmed-ledger source values and build
clocks.  It may not read market OHLC, funding, returns, CAGR, or drawdown.

## Frozen performance gate

Both 2021-2022 train and calendar-2023 selection must independently have:

- positive absolute return;
- `CAGR / strict MDD >= 3.0`;
- strict MDD at most 15%;
- weekly-cluster one-sided sign-flip `p <= 0.10`;
- mean gross move at least 30 bp; and
- positive absolute return at 10 bp/notional/side stress cost.

Calendar 2021, 2022, 2023-H1, and 2023-H2 must each be positive.  Long-only and
short-only slices, plus a one-bar-delayed execution, must be positive in train
and selection.  CAGR uses the entire declared wall-clock window including idle
cash.  Strict MDD uses the global/pre-entry high-water mark, favorable-before-
adverse held OHLC/funding extremes, and entry/exit/hypothetical-liquidation
costs.

Controls are direction flip, constant-long/constant-short on the same clock,
topology-only, low-fee mirror, seven-day stale state, year/side-stratified
random clock, and one-bar delayed entry.  Component controls passing the full
primary gate reject the claimed fee-topology mechanism.  No threshold, side,
hold, or latency repair is allowed after an outcome opens.

## Frozen anchors

- exact source CSV SHA-256:
  `8d5160b59bd104819adb34752f2bd5b01e9c07d7aa108188abcd9e6d323d102f`;
- exact source manifest file SHA-256:
  `ceb096b7bfeaf309729c4cab9f5cd4d2e0d526e261b1e460e9d03cd4f24e8084`;
- preregistration source commit:
  `45ac487c681f8ef9542380189d0a202d09a9dae3`;
- preregistration source SHA-256:
  `10311773ae2e9baac1b27d537bcab7cf5b8687d7dc84a0dd1b0bcd8d2673fa05`;
- preregistration artifact:
  `results/utxo_fee_clearing_polarity_preregistration_2026-07-20.json`;
- artifact file SHA-256:
  `160efdd2eb857c47a80ec0ed4a976a659a1ee3dd3c930093d197798e619d65c9`;
- artifact canonical manifest hash:
  `95cd5911171b033923603d5d845949df6a7ef28020f2591c41ab1e0d1293da5b`;
- policy hash:
  `9945815de1e3f88ab1d59e2dec7dc7923294c91807cea8071f10e55c60a0daef`.

The artifact records zero source CSV values read, zero market/funding/return
rows, and `outcomes_opened=false`.  The next admissible step is to implement
and commit the exact outcome-blind support builder, then open source incidence
once.  Train returns remain closed until a separate strict evaluator is
implemented and hash-frozen.
