# Federal Liquidity Component Concordance (FLCC-1) preregistration — 2026-07-17

## Hypothesis

An H.4.1 net-liquidity impulse should be more informative when several weak
sources of public USD liquidity agree rather than when one large accounting
movement dominates:

1. Federal Reserve assets expand;
2. the Treasury General Account releases cash;
3. Federal Reserve reverse-repo liabilities release cash.

FLCC follows a positive net-liquidity tail with a BTC **LONG** and a negative
tail with a **SHORT**, but only when at least two of those three component ranks
agree. This signal uses no BTC price, exchange volume, OI, perpetual carry,
kimchi premium, FX, REX, or existing alpha state.

## Exact source and availability

- Source audit: `docs/federal-reserve-h41-net-liquidity-source-audit-2026-07-17.md`
- Frozen panel SHA-256: `224883dad01b9d7f17d52eb87f3d7ef9890c8dd055a6c36577a534d2afe69621`
- Clock ledger SHA-256: `7ebb0450422d9265e46c596e0b6415b6a8816c66f5e0cbb9ccda14ca6cb4c67c`
- Official release archive: <https://www.federalreserve.gov/releases/h41/>

The source panel already delays availability to 16:35 America/New_York, five
minutes after the Fed's stated 16:30 release time. FLCC waits one more complete
five-minute bucket and enters at `available_at_utc + 5 minutes`.

## Exact feature family

For horizon `h` in `{4, 8}` releases:

```text
asset_impulse       = A[t]   - A[t-h]
tga_release         = -(TGA[t] - TGA[t-h])
rrp_release         = -(RRP[t] - RRP[t-h])
net_impulse         = N[t]   - N[t-h]
N[t]                = A[t] - TGA[t] - RRP[t]
```

Each impulse receives an exact integer midrank against the previous 104
impulses, excluding the current row. The numerator is
`2*count(prior < current) + count(prior == current)` over denominator 208.

The four frozen candidates are:

| Candidate | Horizon | Lower numerator | Upper numerator | Approx. tails |
|---|---:|---:|---:|---:|
| FLCC-H4-Q60 | 4 | 83 | 125 | 40% / 60% |
| FLCC-H4-Q65 | 4 | 72 | 136 | 35% / 65% |
| FLCC-H8-Q60 | 8 | 83 | 125 | 40% / 60% |
| FLCC-H8-Q65 | 8 | 72 | 136 | 35% / 65% |

At least two centered component ranks must share the net-rank direction.
Breadth 3 was rejected before outcomes because it left almost no 2023 support.
The source-only density screen inspected horizons 4/8/13, tails 0.55–0.85,
and breadth 2/3; this support inspection is disclosed as in-sample and is not
alpha evidence.

## Source-only support

No market outcome was opened for these counts.

| Candidate | Train 2020–22 | L/S | 2020 | 2021 | 2022 | Sealed 2023 support | L/S | H1/H2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FLCC-H4-Q60 | 108 | 59/49 | 34 | 33 | 40 | 27 | 13/14 | 19/8 |
| FLCC-H4-Q65 | 99 | 53/46 | 33 | 31 | 34 | 23 | 13/10 | 16/7 |
| FLCC-H8-Q60 | 97 | 50/47 | 32 | 22 | 41 | 28 | 13/15 | 18/10 |
| FLCC-H8-Q65 | 94 | 49/45 | 31 | 21 | 40 | 22 | 13/9 | 15/7 |

The directional imbalance inside individual years is a feature of the macro
liquidity regime, not a manually selected side. The whole train and sealed
2023 clocks both contain meaningful LONG and SHORT support.

## Execution and costs

- Entry: signal availability + one five-minute bar.
- Exit: five calendar days / 1,440 five-minute bars after entry.
- Position: fixed 0.5x account gross.
- Non-overlap: ignore rather than queue a colliding event.
- Base implementation cost: 6 bp/notional/side.
- Stress cost: 10 bp/notional/side.
- Funding: exact BTCUSDT settlements on `[entry, exit)`.
- CAGR: full wall-clock split including idle cash.
- Strict MDD: pre-entry/global high-water, costs, funding, held OHLC adverse
  path, and hypothetical liquidation.

## Controls

1. `net_only`: net tail without component breadth.
2. `component_concordance_only`: component tails without net tail.
3. `direction_flip`: same entries, opposite actions.
4. `one_release_delay`: same feature and side at the next release.
5. `random_side`: hash-fixed random action on the same entries.

The primary must beat both mechanism components. A fully qualifying flipped,
delayed, or random control falsifies interpretation.

## Sequential outcome protocol

### Stage 1 — 2020 through 2022 only

All four candidates are evaluated. Each must satisfy, among other fixed gates:

- positive absolute return overall and in 2020, 2021, and 2022;
- CAGR / strict MDD at least 3.0;
- strict MDD at most 15%;
- weekly cluster sign-flip `p <= 0.025` (Bonferroni for four candidates);
- at least 90 trades, 40 per side, and 20 per calendar year;
- positive return at 10 bp/notional/side stress cost;
- primary ratio strictly above both mechanism controls.

The exact winner is ranked by minimum annual ratio, then overall ratio, stress
return, and lexical candidate ID. If none qualifies, FLCC-1 is rejected and
2023 outcomes remain unopened.

### Stage 2 — sealed 2023

Only the exact Stage-1 winner may run. Required gates include positive absolute
return in both halves, CAGR/strict-MDD at least 3.0, strict MDD at most 15%,
weekly sign-flip `p <= 0.10`, at least 20 trades and 7 per side, stress-cost
profitability, and component-control dominance. There is no fallback or repair.

2024, 2025, and 2026 YTD remain sealed in all cases.
