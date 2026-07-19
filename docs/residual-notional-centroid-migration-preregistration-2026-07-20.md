# RNCM-72 preregistration — residual notional-centroid migration

## Evidence boundary

This document and its implementation are frozen before inspecting RNCM event
incidence or any entry/later price. The only empirical input admitted here is
the physically sealed calendar-2023 Binance USD-M `BTCUSDT` `bookDepth` source
panel. No OHLC, funding, return, PnL, equity, CAGR, MDD, hit rate, excursion, or
post-2023 row is allowed in support selection.

The quantity is a **cumulative depth-weighted average-quote transform**, not a
pure order-price centroid. Historical daily archives are next-day publications;
their research timestamp is usable at bar close only conditional on equivalent
live order-book reconstruction. Live parity remains a production gate.

## Why raw RNCM was not admitted

The provisional raw rule used the common sign of 30-minute changes in radial
skews at 2%, 3%, 4%, and 5%. Before real event incidence was opened, a fixed
absolute, symmetric constant-density book was observed through smoothly moving
percentage bands. The raw rule generated many false events even though book
liquidity never changed. It was therefore rejected as a mechanical price-band
shadow, not tested for returns, and replaced by the residual rule below.

## Frozen feature

For completed five-minute bar `t` and radial band `k in {2,3,4,5}`:

```text
avg_quote(side,k,t) = cumulative notional / cumulative depth
skew(k,t) = log(ask_avg(k,t) / ask_avg(1,t))
            - log(bid_avg(1,t) / bid_avg(k,t))
center(t) = sqrt(bid_avg(1,t) * ask_avg(1,t)), aggregated by 5m median
d(k,t) = skew(k,t) - skew(k,t-6)
x(t) = log(center(t) / center(t-6))
```

Every source row from `t-6` through `t` must be complete. No interpolation,
forward fill, nearest join, partial snapshot salvage, path statistic, net
statistic, or efficiency statistic is permitted.

For each `k`, fit an intercept-bearing rolling OLS slope `beta(k,t)` between
`d(k)` and `x` using at most the prior `8,640` five-minute observations. The
current pair is excluded; activation requires at least `4,032` valid prior
pairs. Variance at or below `1e-24` is unavailable. Then:

```text
r(k,t) = d(k,t) - beta(k,t) * x(t)
I(t) = median(r(2,t), r(3,t), r(4,t), r(5,t))
dominance(t) = median(abs(r(k,t))) / median(abs(d(k,t)))
```

The signal is eligible only when all four residuals are strictly positive or
all four are strictly negative, `dominance(t) >= 0.25`, and `abs(x(t))` is no
larger than its strictly-prior rolling median over the same `8,640`/`4,032`
history contract. This quiet-center condition prevents discrete percentage-band
turnover from being mistaken for latent book migration.

## Frozen threshold and trigger

For each support quantile in this exact strictest-first order:

```text
0.995, 0.99, 0.985, 0.975
```

compute the rolling quantile of `abs(I)` from the prior `8,640` observations,
excluding current `t`, with minimum history `4,032`. A raw event occurs only
when all feature conditions hold, `abs(I(t))` meets the current prior-only
threshold, and `abs(I(t-1))` was below its own prior-only threshold. Missing or
equal-to-zero residual signs are not events.

- `I(t) > 0`: long
- `I(t) < 0`: short
- decision/availability: close of source bar `t`
- entry: next five-minute open `t+1`
- exit: scheduled open after `72` completed five-minute bars (`t+73`)
- exposure for later outcome evaluation: `0.5x`

## Frozen support stopping rule

For each quantile, form four independently reset, quarter-contained,
non-overlapping schedules. Signal, entry, and exit must all lie inside the same
known UTC quarter. A new entry can occur at or after the prior scheduled exit.
Future RNCM source gaps do not retroactively cancel an event selected at `t`.

Choose the first quantile in the frozen order that passes all incidence gates:

- at least `120` non-overlapping events in calendar 2023;
- at least `45` in each half;
- at least `20` in every quarter;
- long share and short share each at least `35%`;
- no quarter contributes more than `40%` of all events.

If no quantile passes, reject RNCM-72 without opening outcomes. The selected
quantile receives no fallback or repair after the novelty gate.

## Frozen mechanical and novelty gates

Before reading the real source, the identical pipeline must produce exactly
zero non-overlapping events at every threshold quantile across the deterministic
fixed-book/moving-band null suite encoded in the preregistration source:
smooth symmetric density, tick-rounded anchor, stepped asymmetric spread,
deterministic missing rows, and a stationary asymmetric discrete tick ladder.

The selected real clock is compared by one-to-one tolerant entry matching,
within `+/-12` five-minute bars, against these previously frozen depth clocks:

- CCBVFR-72;
- PDF-10, replayed from its outcome-blind source and checked against its frozen
  canonical clock hash;
- CRRC-72.

Each intersection-over-union must be at most `0.35`. Any failure rejects the
candidate; another support quantile is not tried. Comparator artifact hashes
are constants in the preregistration source.

## Later strict outcome evaluator — frozen contract

Only a fully passing source-only support artifact may open 2023 outcomes. The
strict evaluator must be committed before that opening and use:

- one position maximum, `0.5x` exposure;
- `6 bp` notional per side base cost and `10 bp` stress cost;
- exact funding timestamps: interior events symmetric; exact entry/exit credits
  dropped and debits retained;
- strict MDD from the global pre-entry high-water mark through entry fee, every
  held five-minute high/low path, conservative virtual adverse exit fee, and
  actual exit;
- full-calendar CAGR including warmup and idle time;
- reverse RNCM, always-long, always-short, same-clock 30-minute momentum,
  same-clock 30-minute reversion, deterministic side permutation, and matched
  price-only threshold clocks as controls;
- base absolute return positive, `CAGR / strict MDD >= 3.0`, strict MDD at most
  `15%`, every contained half and quarter positive, and clustered weekly
  sign-flip `p <= 0.10`;
- `10 bp` stress absolute return positive and stress ratio at least `2.5`;
- primary ratio at least `0.25` above the strongest finite mechanism control.

Opening is sequential and stops on the first failure:

1. calendar-2023 train;
2. calendar-2024 test;
3. calendar-2025 eval;
4. recent-2026 final holdout.

No failed stage may be repaired under the RNCM-72 name. Because the surrounding
research branch has seen many historical outcomes, any eventual claim is only
candidate-level frozen sequential evidence, not a globally pristine discovery.
