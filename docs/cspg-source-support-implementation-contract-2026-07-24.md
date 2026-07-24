# CSPG-288 source-support implementation contract

Status: frozen before CSPG source values, pressure coordinates, token
incidence, or market outcomes are decoded.

This contract implements the already frozen boundary, mechanism, and
preregistration for `CSPG-288`. It does not change any source, feature,
threshold, token, clock, reservation rule, temporal role, or support gate.

## Allowed inputs

The real support build may decode only the exact preregistered predictor
allowlists from:

- CBOE volatility term structure;
- CBOE tail risk; and
- CBOE option flow.

It may hash the frozen documents, manifests, and preregistration artifact.
It may not decode BTC market data, funding, comparator clocks, labels, actions,
rewards, returns, PnL, portfolio state, or any 2024-or-later source value.
There are no network calls.

## Causal feature construction

Each source is validated independently for exact path, full-file hash,
full-header hash, allowlist order, unique strictly increasing UTC-midnight
dates, finite strictly positive primitives, and a strict pre-2024 endpoint.

Term and tail primitives are computed exactly as preregistered. Option-flow
primitive deltas use the immediately preceding option-source observation,
including dates that later disappear at the exact-date intersection.

Every primitive is ranked against at most 252 immediately preceding finite
values from its own source and requires at least 126. The current value is
never inserted before its rank is fixed. Independent source histories are
joined only after their pressure coordinates are complete.

Term-panel and tail-panel `VIX_close` values must match exactly on intersecting
dates. Missing dates are never filled or substituted.

## Token and clock construction

Rank-complete pressure states are sorted by source date. The first common state
has no token row but is retained as the predecessor of the second. Every later
state emits the exact twelve-token grammar in canonical order.

For source date `D`:

- availability is calendar `D+1 09:30 America/New_York`;
- entry is calendar `D+1 09:35 America/New_York`;
- exit is entry plus exactly 288 five-minute bars.

Weekend and holiday entries remain valid. Future source-row existence is not
consulted. The complete pre-2024 clock is greedily reserved in entry order
using half-open intervals `[entry, exit)`. Reservation occurs before any split
or policy action; no action column exists in the source clock.

The deterministic gzip clock contains only:

- signal identity and causal timestamps; and
- the exact twelve categorical tokens.

It contains no raw value, rank, action, side, market, funding, return, label,
reward, PnL, CAGR, or MDD field.

## Support evaluation

Counts use token-ready, globally reserved, split-contained rows. A row belongs
to a temporal window only when source date, availability, entry, and exit are
all inside that half-open window.

The implementation applies every incidence, calendar, concentration, token
diversity, downstream-vocabulary, timing, schema, and reservation gate in the
frozen preregistration. The first failed check is reported in deterministic
order.

Synthetic or injected frames can exercise the implementation but can never
authorize a real support pass. The real build is authorized only when the
implementation source, its primary test file, and this contract are tracked
and byte-identical to `HEAD`.

## Atomic artifacts

The real command writes, once:

```text
data/cboe_cross_surface_pressure_grammar_clocks_2020_2023.csv.gz
results/cboe_cross_surface_pressure_grammar_support_2026-07-24.json
```

Gzip metadata are canonical (`mtime=0`, empty filename). Existing identical
artifacts are accepted; any byte drift fails closed.

The report records exact implementation, preregistration, document, source,
manifest, clock, and report-core hashes. Its outcome boundary must keep market,
funding, comparator, return, PnL, and post-2023 decoded-row counts at zero.

Any support failure retires `CSPG-288` unchanged before market outcomes. A
complete pass authorizes only the separately frozen cheap-baseline/economic
implementation; it does not authorize opening 2023 outcomes or GPU training.
