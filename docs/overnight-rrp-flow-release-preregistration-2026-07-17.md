# Overnight RRP Flow Release (ORFR-1) preregistration — 2026-07-17

## Hypothesis

ORFR-1 treats an unusually large daily New York Fed overnight reverse-repo
award as a temporary USD-liquidity absorption and an unusually small award as
a relative liquidity release.

The [New York Fed's official FAQ](https://www.newyorkfed.org/markets/rrp_faq.html)
explains that ON RRP settlement moves cash from reserve liabilities into the
Fed's reverse-repo liability. ORFR therefore maps an upper-tail cash-absorption
innovation to **SHORT BTC** and a lower-tail release innovation to **LONG BTC**.

The signal contains no crypto price, return, volume, taker flow, OI, funding,
premium, Kimchi/FX, REX, on-chain, options, or existing-alpha state. Source
orthogonality is not profitability evidence; return orthogonality is deferred
until the standalone economic gates pass.

## Frozen source and clock

- Source audit:
  `docs/new-york-fed-overnight-rrp-source-audit-2026-07-17.md`
- Official API documentation:
  <https://markets.newyorkfed.org/static/docs/markets-api.html>
- Panel SHA-256:
  `49f67ed44b7eb81fd35c17a8209cf14d6a8019d7e9f77fce8c343d1a7fb66b27`
- Preregistered clock SHA-256:
  `9f09bc88c9661441a33cee724e59524f57c0b021abff0fe81263e1a341b7b7b7`

Each normal afternoon operation becomes available at its official close time
plus 15 minutes in `America/New_York`. ORFR waits one more complete five-minute
bucket before entry. Morning small-value exercises are excluded. Later-updated
archive rows expose no amount, reset the local baseline, and emit no signal.

## Frozen feature and direction

For complete operation `t`:

```text
A[t] = log1p(total_amount_accepted_usd[t] / 1e9)
M[t] = median(A[t-5:t])                 # current t excluded
X[t] = A[t] - M[t]
R[t] = strict-prior midrank of X[t]
       against exactly 104 previous valid innovations
```

- `R[t] <= 0.125`: **LONG**;
- `R[t] >= 0.875`: **SHORT**;
- otherwise: abstain.

The current innovation is ranked before it is appended. No forward amount,
market return, or same-row fitted statistic enters the signal.

## Source-only support screen

The symmetric tail candidates `5%, 7.5%, 10%, 12.5%, 15%, 20%` were inspected
using only source timestamps, source values, side counts, and month counts. No
BTC or funding outcome was joined. The frozen 12.5% tail is the sparsest member
meeting the preregistered density and balance floors.

| Window | Events | Long | Short | Max month share |
|---|---:|---:|---:|---:|
| 2021 | 59 | 36 | 23 | 27.12% |
| 2022 | 53 | 27 | 26 | 18.87% |
| Stage1 2021–2022 | 112 | 63 | 49 | 14.29% |
| 2023 H1 | 27 | 17 | 10 | 29.63% |
| 2023 H2 | 47 | 33 | 14 | 29.79% |
| Sealed 2023 | 74 | 50 | 24 | 18.92% |

Month concentration is gated on full Stage1 and full 2023, not on half-year
diagnostics. The final 2023 source event has no bounded next-operation exit and
is omitted, which accounts for one fewer event than a timestamp-only tail
count.

## Frozen execution

- entry: source availability + 5 minutes;
- exit: next normal ON RRP result availability + 5 minutes;
- exposure: fixed 0.5x BTCUSDT perpetual;
- base cost: 6 bp/notional/side;
- stress cost: 10 bp/notional/side;
- exact realized funding on `[entry, exit)`;
- full-calendar CAGR including idle cash;
- strict intratrade MDD including pre-entry high-water, costs, funding,
  favorable-before-adverse held OHLC, and hypothetical liquidation.

An event exits at or before the next selected event's entry, so clocks cannot
overlap. Weekend and holiday exposure is intentionally retained until the next
normal operation result.

## Controls

1. `one_day_delta_tail`: replace the five-operation-median residual with the
   one-operation log-amount change, retaining the same 104-observation rank and
   1/8 tails;
2. exact direction flip;
3. one complete ON RRP-release delay;
4. deterministic hash-random side on the exact primary entries.

The primary must beat the one-day-delta mechanism control by at least 0.25 in
CAGR/strict-MDD. A fully qualifying flipped, delayed, or random clock
falsifies interpretation.

## Sequential outcome protocol

### Stage1 — physically parse only 2021–2022

Required gates include:

- positive absolute return overall and in both calendar years;
- CAGR / strict MDD at least 3.0;
- strict MDD at most 15%;
- weekly-cluster sign-flip `p <= 0.10`;
- at least 100 trades, 45 per year, and 35 per side;
- mean gross underlying return at least 35 bp;
- positive absolute return at 10 bp/notional/side stress cost;
- mechanism-control margin at least 0.25.

### Stage2 — sealed 2023

It may open only after an exact, hash-bound unchanged Stage1 replay passes all
gates. It independently requires at least 60 trades, 20 trades in each half,
15 trades per side, positive return in each half, CAGR/strict-MDD at least 3,
strict MDD at most 15%, `p <= 0.10`, stress-cost profitability, and the same
mechanism-control margin. Only ORFR-1 can run; there is no fallback or repair.
2024 and later remain sealed. Any Stage1 failure retires ORFR-1 unchanged.

## Orthogonality gate after standalone pass

- exact entry Jaccard <= 0.02;
- fraction of candidate entries within six hours of an existing entry <= 0.25;
- position Jaccard <= 0.15;
- absolute daily PnL Pearson <= 0.30.

No threshold, direction, hold, calendar gate, or crypto feature may be changed
after an outcome is viewed.

## Frozen identity

- source commit: `f90cc3a`;
- manifest hash:
  `76db178758a33057ef396c9c268246e6cbf6c9ce8ab9ba2739eb422cf6458d4a`;
- preregistration JSON SHA-256:
  `5ff8ffbe448a76dfdf6dde6aefbd83194f90429e55c75f1c6fc4e9749d3f09dd`.
