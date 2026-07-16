# AFCH v1 preregistration — 2026-07-17

## Evidence boundary

The six-alt source history and adjacent funding-score experiments have been
seen elsewhere, so historical results cannot by themselves promote AFCH. The
**exact payoff object**—four overlapping, factor-beta-neutral sleeves with
exact realized funding cashflow and portfolio-level strict MDD—has not been
opened. AFCH must pass sequential 2023/2024/2025 research gates and then at
least 90 forward-shadow days before promotion.

## Orthogonal economic object

Every Monday at 00:05 UTC, AFCH sums each alt's exact realized funding over the
prior 28 days. It goes long the lowest-funding symbol and short the highest-
funding symbol, with causal rolling beta-neutral weights. It has no BTC
position and uses no REX, OI, premium, Kimchi/FX, Markov, tree, LLM, or price-
direction gate. Profit is required to come from realized cross-sectional
funding transfer, not merely favorable price drift.

## Frozen policy AFCH01

- universe: ETH, SOL, BNB, XRP, ADA, DOGE USD-M perpetuals;
- trailing exact funding window: 28 days;
- beta: shifted 720-hour leave-one-out factor estimate, clipped `[0.25, 2.5]`;
- trade only if beta-weighted projected 28-day carry is at least `18 bp`;
- one new sleeve per qualifying Monday, gross `0.25`, hold exactly 28 days;
- at most four concurrent sleeves, portfolio gross at most `1.0`;
- signal `00:05`, next 5m-open entry `00:10`, scheduled exit after 28 days;
- base cost `6 bp/notional/side`, stress `10 bp`, exact funding settlements;
- aggregate favorable-before-adverse strict MDD and full-calendar CAGR.

The 18 bp hurdle is fixed by economics: 1.5 times the gross-one 12 bp
round-trip cost. It is not selected from post-entry returns.

## Qualification

Support must provide at least 110 sleeves across 2023–2025, 35 per year, and
12 per half-year. The single policy then needs positive 2023 and 2024 returns,
ratio >= 1.5 in each, combined CAGR/strict-MDD >= 3, strict MDD <= 15%, positive
10 bp stress, weekly cluster p <= 0.10, and realized funding cash at least equal
to all transaction costs. Only then may 2025 be opened under ratio >= 3 and the
same carry-attribution requirement.

## Anti-repair and production boundary

No sign, hold, funding lookback, hurdle, beta, pair, sleeve, or regime repair is
allowed after outcomes open. Even a historical pass remains research-only
until a multi-symbol live ledger and at least 90 forward-shadow days confirm it.

Protocol hash: `15a7d0adbace0255e1ea4359e4869154dfb34ad891a2125239340ff70c4e2a09`
