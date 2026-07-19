# TBASR-24 strict evaluator preregistration — 2026-07-20

## Frozen hypothesis

An unusually broad BitMEX Trollbox attention burst is tradable only when the
frozen Gemma2 crowd direction agrees with a **completed, material BTC move**.
That combination represents crowded short-horizon extrapolation; TBASR-24
fades the crowd for exactly two hours. `UNCLEAR` never trades.

This contract was written before parsing any BTC OHLC or funding row for this
candidate. The branch is globally contaminated by earlier BTC research, so a
pass can support only a candidate-level frozen sequence—not a pristine global
holdout claim.

## One specification, no grid

There is one policy and no mutable threshold, hold, leverage, or cost choice.

### Completed displacement

Binance BTCUSDT USD-M five-minute bar timestamps are bar-open times. For an
event ending at `observation_end`:

- target end: the close of the bar opened at `observation_end - 5m`;
- target start: the open at `observation_end - 60m`;
- displacement: `d_i = ln(close_i / open_{i-11})`, exactly 12 completed bars;
- strictly-prior reference: overlapping absolute 60-minute displacements whose
  **end timestamps** are in `[target_start - 28d, target_start)`;
- implementation: 8,064-reference rolling linear 90th percentile shifted by
  **13 final-bar indices**;
- latest reference: final bar `i-13`, whose close is
  `target_start - 5m`; therefore no reference path touches the target move;
- material: `abs(d_i)` is at least that one frozen 90th percentile.

`BULLISH` must align with `d_i > 0`; `BEARISH` must align with `d_i < 0`.
The action is the already-frozen semantic `contrarian_side`. This means the
LLM supplies a semantic alignment gate, not numeric prediction, sizing, or
threshold selection.

### Entry, exit, overlap, and costs

- Base entry: open at `observation_end + 5m`. The intervening market bar is
  complete before entry.
- Base exit: open exactly 24 bars / two hours later.
- Leverage: 1.0x.
- Base cost: 6 bp of notional per side.
- Delayed stress: the same accepted event identities, entry and exit both
  shifted five minutes, still 24 bars, at 10 bp per side.
- Split containment: both base and delayed-stress exits must be strictly before
  the split end.
- Non-overlap: stable chronological greedy selection after all primary filters;
  accept a candidate only when `entry >= previous accepted exit`.

## Controls

1. **Direction flip** — exact accepted primary clocks, opposite side.
2. **Deterministic random side** — exact accepted primary clocks; side is the
   parity of the first hexadecimal nibble of
   `SHA256("TBASR-24|random-side|<UTC-entry>")`.
3. **Semantic-alignment ablation** — every clear, material-displacement event,
   without requiring semantic/price direction agreement; retain the semantic
   contrarian side and apply its own chronological greedy non-overlap.

Any non-finite control ratio fails the mechanism-margin gate. The primary
ratio must exceed every control by at least 0.25.

## Sequential windows

| stage | full calendar | permitted price history |
|---|---|---|
| train | `[2020-07-01, 2022-01-01)` | 2020-01 through 2021-12, including causal warm-up |
| test | `[2022-01-01, 2023-01-01)` | prior history plus 2022, only after train passes |

The freeze may read committed semantic clocks and source manifests, but it may
not parse or hash execution-data bytes, build a price-conditioned schedule, or
simulate a result. Train loading verifies and decodes only the monthly market
files before 2022. Test loading is blocked before any test-row parse unless the
stored train report passed every frozen gate. Full broad-container SHA checks
may occur only inside an outcome stage and reveal no decoded future value;
reports disclose that identity-only hash access separately from decoded rows.

If train fails, calendar 2022 remains sealed for this candidate. No failed
gate can be repaired by changing a parameter.

## Strict accounting

- full declared calendar CAGR, including warm-up and every idle interval;
- global high-water mark initialized before the first entry;
- entry cost;
- every held five-minute bar in favorable-then-adverse OHLC order;
- virtual exit cost at every adverse mark;
- exact Binance funding time and rate with the frozen official containing-8h
  mark-price-open proxy from the committed funding artifact;
- interior funding credits and debits symmetric;
- at exact entry/exit, credits dropped and debits retained;
- every visited funding settlement mark updates the strict path even when a
  boundary credit is dropped; and
- actual exit cost.

## Frozen gates — each of train and test

- absolute return > 0;
- full-calendar CAGR / strict MDD >= 3.0;
- strict MDD <= 15%;
- trades >= 80 train / 40 test;
- longs and shorts each >= 20 train / 10 test;
- active weekly clusters >= 40 train / 25 test;
- weekly-cluster two-sided sign-flip `p <= 0.10` after base costs and funding;
- mean gross underlying move >= 20 bp;
- every contained half-year absolute return > 0;
- delayed-stress absolute return > 0 and CAGR / strict MDD >= 2.5;
- delayed stress preserves the primary trade count; and
- minimum primary-minus-control CAGR/MDD margin >= 0.25.

The weekly test uses UTC ISO entry weeks. It enumerates all Rademacher sign
vectors through 20 clusters; otherwise it uses NumPy `default_rng(20260720)`
for 20,000 deterministic draws and the preregistered plus-one p-value.

## Stop rule

Only a complete train pass may open test. Only a complete pre-2023 pass may
justify a separately frozen 2023+ source/semantic extension. No LoRA, RL, or
outcome-trained language policy is admissible before the base semantic feature
passes this contract.
