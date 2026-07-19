# RQCI-24 preregistration — residual quote-curvature impulse

## Evidence boundary

RQCI-24 is frozen before inspecting its real 2023 event incidence or any
entry/later price. Support selection may read only the physically sealed,
checksum-audited 2023 Binance USD-M `BTCUSDT` cumulative average-quote panel.
It may not read OHLC, funding, returns, PnL, equity, CAGR, MDD, excursions, hit
rate, or any post-2023 row.

`notional/depth` is a cumulative depth-weighted average quote, not a pure
order-price centroid. The historical archive is next-day data; production use
requires a correct live local book and a separately demonstrated feature-parity
contract.

## Frozen geometry

For five-minute median directional skew at radial band `k`:

```text
skew(k,t) = log(ask_avg(k,t) / ask_avg(1,t))
            - log(bid_avg(1,t) / bid_avg(k,t))

curvature(t) = [skew(5,t) - skew(4,t)]
               - [skew(3,t) - skew(2,t)]

raw_impulse(t) = curvature(t) - curvature(t-6)
center(t) = sqrt(bid_avg(1,t) * ask_avg(1,t)), aggregated by 5m median
x(t) = log(center(t) / center(t-6))
```

All source bars from `t-6` through `t` must be complete. Net, path, efficiency,
future source completeness, interpolation, forward fill, and nearest joins are
forbidden.

Fit an intercept-bearing rolling OLS slope between `raw_impulse` and `x` from
at most the prior `8,640` valid observations. Current `t` is excluded and at
least `4,032` prior pairs are required. Then:

```text
residual(t) = raw_impulse(t) - beta(t) * x(t)
dominance(t) = abs(residual(t)) / abs(raw_impulse(t))
```

Eligibility requires:

- nonzero finite residual;
- `dominance(t) >= 0.25`;
- `abs(x(t))` no greater than its strictly-prior rolling median using the same
  `8,640`/`4,032` history contract.

The last condition suppresses mechanical discrete-ladder turnover while the
underlying absolute book is fixed.

## Frozen threshold, direction, and execution

Threshold quantiles are tried in this exact strictest-first order:

```text
0.995, 0.99, 0.985, 0.975, 0.95
```

Each threshold is the corresponding rolling quantile of `abs(residual)` from
the prior `8,640` observations, excluding current `t`, with minimum history
`4,032`. An event occurs only when eligibility holds, current absolute residual
meets the threshold, and the preceding bar's absolute residual was below its
own prior-only threshold.

- positive residual: long, interpreted as outer ask-side impact convexity
  expanding relative to bid-side convexity;
- negative residual: short;
- decision: close of completed five-minute source bar `t`;
- entry: next five-minute open `t+1`;
- exit: scheduled open after `24` completed bars (`t+25`, two hours);
- eventual exposure: `0.5x`.

## Frozen support stopping rule

For each quantile, build four quarter-contained schedules with independent
non-overlap state reset at each known UTC quarter boundary. Signal, entry, and
exit must stay inside that quarter. Future RQCI source gaps do not cancel an
event selected at `t`.

Stop at the first quantile that passes all gates:

- at least `180` non-overlapping events in 2023;
- at least `70` in each half;
- at least `30` in every quarter;
- long and short share each at least `35%`;
- no quarter contributes more than `40%`.

If none passes, reject without outcomes. After a first incidence pass, lower
quantiles remain unopened. Novelty failure does not permit fallback.

## Frozen mechanical and novelty gates

Before loading real RQCI source values, the complete pipeline must produce zero
raw and zero non-overlapping events at every quantile on all five deterministic
fixed-absolute-book controls inherited from the hash-pinned RNCM utility:

- smooth symmetric constant density;
- tick-rounded anchor;
- stepped asymmetric spread;
- deterministic missing rows;
- stationary asymmetric discrete tick ladder.

The first support-passing clock is compared by one-to-one entry matching within
`+/-12` five-minute bars against CCBVFR-72, PDF-10, and CRRC-72. Every tolerant
Jaccard must be at most `0.35`. Comparator artifacts and the shared causal
utility are SHA-256 pinned.

## Later strict outcome contract

Only a fully passing source-only event clock may receive a separately committed
evaluator. It must use one-position `0.5x` exposure, `6 bp` notional per side,
`10 bp` stress, exact conservative funding-boundary accounting, full-calendar
CAGR, and strict held-path MDD from the global pre-entry HWM through entry cost,
every five-minute high/low, virtual adverse exit fee, and actual exit.

The primary must have positive absolute return, `CAGR / strict MDD >= 3.0`,
strict MDD at most `15%`, positive contained halves and quarters, weekly
clustered sign-flip `p <= 0.10`, positive stress return, stress ratio at least
`2.5`, and ratio margin at least `0.25` over the strongest finite mechanism
control. Controls include exact reverse, always-long, always-short, same-clock
30-minute momentum/reversion, deterministic side permutation, and matched
price-only curvature clocks.

Opening is sequential: 2023 train, 2024 test, 2025 eval, then recent 2026 final.
The first failure retires the exact candidate without repair. The broader branch
is historically contaminated, so any pass is candidate-level frozen evidence,
not a globally pristine discovery.
