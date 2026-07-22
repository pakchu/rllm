# WTSL-168-SOURCE-SEEN support rejection — 2026-07-23

## Verdict

**Retire WTSL-168-SOURCE-SEEN without opening BTC outcomes.**

The exact frozen implementation failed both the disclosed source-count
reproduction and five prespecified structural support gates. No BTC OHLC,
funding, future return, PnL, absolute return, CAGR, strict MDD, or post-2023
contract-event value was opened. Profitability statistics are therefore `N/A`.

## Exact primary clock

| Split | Trades | LONG | SHORT | Largest month | Largest quarter | Longest side run |
|---|---:|---:|---:|---:|---:|---:|
| Train 2021-2022 | 152 | 128 | 24 | 18.42% | 32.89% | **71** |
| Selection 2023 | **15** | **4** | 11 | **26.67%** | **53.33%** | 10 |
| Total | 167 | 132 | 35 | — | — | — |

Year counts were 123 in 2021, 29 in 2022, and 15 in 2023. Selection had 7
candidates in H1 and 8 in H2. The actor breadth gates passed (18 train, 9
selection), but source incidence was too concentrated and one-sided to support
a reliable staged market test.

Failed structural gates:

1. selection total 15, required 20;
2. selection LONG count 4, required 8 per side;
3. selection month share 26.67%, maximum 20%;
4. selection quarter share 53.33%, maximum 40%;
5. train longest same-side run 71, maximum 20.

## Disclosed-count mismatch

The pre-freeze informal source probe had reported 236 candidates (189 LONG / 47
SHORT; yearly 168 / 43 / 25). The exact frozen implementation instead produced
167 (132 / 35; yearly 123 / 29 / 15), so all four reproduction invariants
failed.

The diagnostic `no_black_funds_veto` control produced 240 candidates with
exact 2022 and 2023 counts of 43 and 25 and an exact 2023 side split of 12 / 13.
That shows the informal probe was much closer to *ignoring* black-funds events
than to the now-frozen rule that vetoes an anchor whenever such an event is in
the seven-day window. Even that control was 4 candidates above the disclosed
236, so the old probe is not an executable specification and cannot be used to
repair this identity.

This discrepancy is precisely why the source-seen status was disclosed and an
exact reproduction gate was required. Changing the veto, start boundary,
actor cap, median inequality, or support floors now would create a new
candidate after seeing source incidence.

## Controls

| Control | Trades | Exact-entry Jaccard vs primary | Purpose |
|---|---:|---:|---|
| Primary | 167 | 1.000 | rejected clock |
| Direction flip | 167 | 1.000 | exact opposite side |
| Stablecoin only | 724 | 0.063 | directional component without WBTC gate |
| WBTC signed placebo | 256 | 0.449 | forbidden WBTC direction |
| Stale 24h | 167 | 0.621 | one-hold stale state |
| Stale 48h | 167 | 0.465 | two-hold stale state |
| Actor cap 0.60 | 72 | 0.177 | stricter custody breadth |
| No black-funds veto | 240 | 0.682 | isolates veto effect |
| USDC only | 241 | 0.652 | token attribution |
| USDT only | 77 | 0.386 | token attribution |
| Year amount permutation | 198 | 0.159 | timing/amount placebo |
| Random side | 167 | 1.000 | exact-clock outcome control |

The broad stablecoin-only clock and large veto effect confirm that WTSL did not
produce a stable, independently identifiable WBTC-conditioned interaction.

## Integrity evidence

- preregistration commit: `48ef81f`
- preregistration manifest:
  `81f41c68b526a2e22a4da769e973026255d44251f7df996e7cdbc5eb8a66ac4a`
- support implementation SHA-256:
  `c527e0d8b6e64657e9e6a49f0f13a53acd589c26677d1a790f8e69d2faf4e57e`
- support clock SHA-256:
  `df8cb085d439c9ee9e89334cb891b9e3b04f54c2a8e70bd4f552a90648ea8b6d`
- support report SHA-256:
  `1415b8e2a40f2aff908bfec1d1faa9621445c3fe87b41c43fd95a991725b23bd`
- support report manifest:
  `b53de47d743f7f61240e59ac3149c0a37467f6bb8ce580c9c3c2bc84341b7e9e`
- clock rows: 2,643 across primary and eleven controls.

Every state uses `available_at`, a six-hour operational lag, exact half-open
168-hour windows, 1,460 strictly prior six-hour WBTC baseline samples, a
10-minute entry delay, exact 24-hour hold, split containment, and global
non-overlap. The stablecoin stream stops at a timestamp-only 2024 boundary
sentinel before parsing event values.

## Next research implication

Do not create a WTSL threshold variant. The useful surviving information is
negative: a persistent administrative event such as black-funds destruction
should not veto an entire seven-day liquidity window, and a broad stablecoin
net-flow direction needs a genuinely event-transition-based interaction rather
than a level gate. The next candidate must use a new causal geometry and a new
identity before any BTC outcome access.
