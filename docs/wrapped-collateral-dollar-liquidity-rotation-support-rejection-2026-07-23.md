# WCDR-2016 source-support rejection — 2026-07-23

## Verdict

**Retire WCDR-2016 before opening BTC outcomes.**

The fixed wrapped-collateral/dollar-liquidity divergence produced a real,
causal source clock, but it failed four preregistered support gates. No BTC
OHLC, funding, future return, PnL, absolute return, CAGR, strict MDD, or
post-2023 contract event was opened. Profitability statistics are therefore
`N/A`, not zero.

The result cannot be repaired by shortening the hold, lowering support floors,
removing the 6-hour delay, widening actor concentration, changing lookbacks,
or reversing direction.

## Frozen primary support

| Split | Trades | LONG | SHORT | Largest month | Longest same-side run | Distinct WBTC actors |
|---|---:|---:|---:|---:|---:|---:|
| Train 2021-2022 | 44 | 18 | 26 | 9.09% | 9 | 19 |
| Selection 2023 | 34 | 6 | 28 | 14.71% | **28** | 19 |

Subperiod counts:

| Subperiod | Trades | Frozen minimum | Pass |
|---|---:|---:|:---:|
| 2021 | **14** | 20 | no |
| 2021 H1 | **5** | 8 | no |
| 2021 H2 | 9 | 8 | yes |
| 2022 | 30 | 20 | yes |
| 2022 H1 | 18 | 8 | yes |
| 2022 H2 | 12 | 8 | yes |
| 2023 H1 | 14 | 8 | yes |
| 2023 H2 | 20 | 8 | yes |

Failed gates:

1. train total: 44, required 50;
2. each train year: 2021 had 14, required 20;
3. each train half-year: 2021 H1 had 5, required 8;
4. longest same-side run: selection had 28, maximum 10.

All split side-count, calendar-month concentration, and distinct-actor gates
otherwise passed. The selection failure is structural rather than a marginal
count miss: 28 consecutive seven-day positions shared the same SHORT source
state, so the apparent 34-trade sample did not represent 34 independent
regime observations.

## Source-only controls

| Clock | Total trades | Exact-entry Jaccard vs primary | Interpretation |
|---|---:|---:|---|
| Primary | 78 | 1.000 | rejected |
| Direction flip | 78 | 1.000 | exact clock, opposite side |
| WBTC-only contrarian | 133 | 0.082 | WBTC component is much broader |
| USDC-only direct | 156 | 0.093 | USDC component is nearly continuous |
| Same-sign direct | 85 | 0.000 | disjoint alternate mechanism |
| Stale seven days | 78 | 0.431 | slow states persist after a full week |
| Count-sign consensus | 52 | 0.548 | stricter state becomes more one-sided |
| Within-year amount permutation | 80 | 0.082 | amount placement materially changes clock |
| Deterministic random side | 78 | 1.000 | exact-clock outcome control only |

The controls confirm why threshold repair is inappropriate. Both individual
source components are broad, but their opposite-sign conjunction removes too
much 2021 support while becoming nearly continuously SHORT in 2023. The
seven-day stale clock retaining 78 trades further shows that this construction
mostly captures a persistent source regime rather than a fresh transmission
event.

## Integrity evidence

- preregistration commit: `501a767`
- preregistration manifest hash:
  `267fae61f29caa3117846349c6346e14cbd041e0ed121249c2f9fcdf8f37bf4f`
- support implementation SHA-256:
  `3cbc2c5c06629775b240337dcdbaaa92c91e4410da0702787d1fba4f0f2d53c3`
- support clock:
  `data/wrapped_collateral_dollar_liquidity_rotation_2021_2023/`
  `wcdr2016_support_clocks_2021_2023.csv.gz`
- support clock rows: 818 across primary and eight controls
- support clock SHA-256:
  `241d96a64a654ba2faeda2d4a8460131269acf21d0bbbf31177d35d1ecd63b3c`
- support report:
  `results/wrapped_collateral_dollar_liquidity_rotation_support_2026-07-23.json`
- support report SHA-256:
  `df3101a973c514c7c8297e7132b6c9d95fe7d10adf234dc5b5b29c497e972c35`
- support report manifest hash:
  `0a28128c820c1f5baf73c7653901056d3803e9bc0cce54b29a03afc7051ef600`

Every state used `available_at`, a six-hour extra source delay, half-open
lookbacks, next five-minute entry, exact seven-day exit, split containment, and
global non-overlap. Source rows were bound to the independently replayed
Ethereum artifacts.

The stablecoin artifact contains two causally available rows after the sealed
boundary. Support protocol v2 stops at the first `2024-01-01` timestamp
sentinel before parsing its event fields. It loaded 266,360 physical rows and
265,583 eligible USDC values strictly before the boundary, with zero
post-2023 contract-event values loaded; the one timestamp-only stop sentinel
is recorded separately in the audit.

## Research implication

Do not create `WCDR-2016B` by loosening this clock. A later candidate must use
a genuinely different information geometry. The most useful surviving fact is
that **WBTC turnover can act as a sparse materiality clock while stablecoin
flow supplies direction**; that is different from WCDR's persistent
opposite-sign regime and must receive a new identity, explicit disclosure that
source incidence was seen, and a fresh pre-outcome freeze.
