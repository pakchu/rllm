# CSPG-288 source-support retirement

Decision: **retire `CSPG-288` unchanged before market outcomes**.

The frozen source-support implementation was committed at `831f925` and then
executed once. It decoded only the preregistered CBOE predictor allowlists.
BTC market rows, funding rows, comparator rows, returns, PnL, CAGR, MDD,
post-2023 rows, model labels, and model training all remained unopened.

## Frozen artifacts

| Artifact | SHA-256 |
|---|---|
| `results/cboe_cross_surface_pressure_grammar_support_2026-07-24.json` | `ee52aa25d9a1a870eb5f52974e4ab5720fd34d5abac4bd5819900ce2c0db9a1c` |
| `data/cboe_cross_surface_pressure_grammar_clocks_2020_2023.csv.gz` | `33a87fe5cb9c99e9a8b4ecc0699f5746387fabfc5ea434f4adf96615374aacdb` |

Report manifest:

```text
07d8ecf98bf78221d6385648c8b551a860cc7dbc3911d15e091fe75b401267ac
```

## What passed

The causal source and calendar support were sufficient:

- 1,006 exact common source dates;
- 879 rank-complete common states;
- 878 token-ready and globally reserved opportunities;
- no overlap suppression;
- 375 train opportunities in 2020–2021;
- 123 opportunities in the 2020 warm-up half-year;
- 250 opportunities in each of 2021, 2022, and 2023;
- all required 2021–2023 months, halves, quarters, month concentration, and
  maximum-gap gates passed;
- real-prefix future-append invariance passed;
- corrected next-calendar-day New York clock and global non-overlap passed; and
- all source-support checks passed.

The failure is therefore not insufficient sample count.

## Terminal token-support failure

The first frozen failure was:

```text
train:tail_level:LOW_share_min
```

Twenty-four token-support checks failed. The dominant structural failures were:

1. **Tail state nonstationarity**
   - train tail levels: `HIGH 61.6%`, `MID 31.2%`, `LOW 7.2%`;
   - 2022: `LOW 90.4%`, `MID 8.8%`, `HIGH 0.8%`;
   - 2023: `HIGH 77.6%`, `MID 19.6%`, `LOW 2.8%`.
2. **Tail change collapse**
   - `SAME` occupied `86.93%`, `95.2%`, and `90.4%` in train, 2022, and
     2023 respectively;
   - both directional change classes missed their frozen 8% floor.
3. **Exact grammar concentration**
   - largest exact twelve-token signature shares were `17.6%`, `32.8%`, and
     `26.0%`, versus the frozen 5% maximum.
4. **Additional collapse**
   - train option `MID` share was `80.27%`;
   - 2023 option `MID` share was `83.6%`;
   - 2022 term-change `SAME` share was `81.2%`;
   - 2022 compressed dispersion and unison agreement were only `2.0%` and
     `2.8%`;
   - train topology `RESET` was only `1.87%`; and
   - 2022 relief leadership was overly concentrated in `TAIL`.

The fixed global `1/3` and `2/3` pressure-level grammar does not provide a
stable, sufficiently expressive categorical state across years. This is a
source-state failure, not a trading-performance disappointment.

## Outcome boundary

The report freezes all of these at zero:

```text
BTC market rows
funding rows
comparator rows
future-return rows
return or PnL fields
PnL/CAGR/MDD values
post-2023 rows
model labels
model training runs
network calls
```

No CSPG return, trade, accuracy, CAGR, or MDD result exists.

## Consequence

The preregistration forbids threshold, token, clock, support-floor, or hold
repair after this failure. CSPG-288 is terminal and must not be rescued by
changing quantile boundaries, dropping tail tokens, merging categories, or
opening outcomes.

Further alpha research must start under a new candidate identifier and a new
observable/state machine. It may use this failure only as source-level design
evidence: fixed absolute rank buckets can collapse under multi-year
distribution shifts, while dense support alone does not establish a learnable
grammar.
