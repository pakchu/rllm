# WSCF-72 source-support rejection — 2026-07-23

## Verdict

**Retire WSCF-72-SOURCE-FAMILY-SEEN without opening BTC outcomes.**

The exact frozen clock had ample incidence and broad calendar coverage, but it
failed the prespecified same-side-run gate and seven novelty gates. More
importantly, source-only controls show that the stablecoin confirmation was
nearly non-selective: USDC alone reproduced 190 of 193 accepted WBTC batch
identities, while amount permutation and 24/72-hour stale anchors retained
similar clock sizes.

No BTC OHLC, funding, future return, PnL, absolute return, CAGR, strict MDD, or
post-2023 contract-event value was opened. Profitability statistics are `N/A`.
The candidate cannot be repaired by changing the 12-hour confirmation window,
72-hour hold, same-side-run ceiling, novelty threshold, or token scope.

## Primary source support

| Split | Trades | LONG | SHORT | Largest month | Largest quarter | Longest side run | Max gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train 2021-2022 | 141 | 87 | 54 | 5.67% | 14.18% | **20** | 14.21d |
| Selection 2023 | 52 | 28 | 24 | 13.46% | 34.62% | 8 | 30.06d |

Year counts were 72 in 2021, 69 in 2022, and 52 in 2023. Every half-year,
side-count, month/quarter concentration, gap, actor-breadth, identity-uniqueness,
and split-containment gate passed. Train failed only the frozen maximum of 10
consecutive same-side trades.

## Confirmation selectivity

Across 193 accepted primary trades:

- median WBTC-to-confirmation delay: 750 seconds;
- 90th-percentile delay: 8,857 seconds;
- maximum delay: 35,856 seconds, below the frozen 12-hour ceiling;
- 53.37% confirmed on the first stablecoin availability batch;
- 70.47% confirmed within five stablecoin batches;
- `usdc_only_confirmation` produced 192 trades and retained 98.96% of primary
  accepted WBTC batch identities (Jaccard 98.45%);
- `stablecoin_year_amount_permutation` still produced 192 trades;
- stale WBTC +24h and +72h each produced 193 trades.

This is not evidence of an independently selective two-stage relay. Dense USDC
administrative flow usually supplied a matching sign shortly after the WBTC
anchor, so WBTC incidence dominated the clock.

## Frozen novelty failures

Exact-entry Jaccard was near zero against every comparator, and WSCF passed
novelty against WCDR, WTSL, UGCI, and all three live-portfolio members. It
failed the preregistered WSCF-to-comparator `±12h <= 0.30` containment gate for
seven source clocks:

| Comparator view | Entries | WSCF near share | Exact Jaccard |
|---|---:|---:|---:|
| AMTR cross-minter | 411 | 40.93% | 0.33% |
| SDDR primary | 78 | 36.36% | 0.00% |
| SQFD no-participation | 89 | 63.64% | 0.00% |
| SQFD no-USDT-lag | 114 | 72.73% | 0.00% |
| SQFD primary | 55 | 45.45% | 0.00% |
| UCBR primary | 28 | 36.36% | 0.00% |

These late-2023 comparisons have a small WSCF denominator, but that threshold
and minimum comparator support were frozen before WSCF incidence. Reinterpreting
them after seeing the clock would be a repair, not validation.

## Controls

| Control | Trades | Train LONG/SHORT | Selection LONG/SHORT |
|---|---:|---:|---:|
| Primary | 193 | 87 / 54 | 28 / 24 |
| WBTC only | 208 | 91 / 61 | 31 / 25 |
| Stablecoin 12h grid | 364 | 155 / 88 | 58 / 63 |
| Anchored first nonzero | 208 | 74 / 78 | 32 / 24 |
| Opposite confirmation | 196 | 85 / 60 | 25 / 26 |
| Lead-lag reverse | 153 | 81 / 33 | 19 / 20 |
| Stale WBTC +24h | 193 | 87 / 53 | 30 / 23 |
| Stale WBTC +72h | 193 | 91 / 52 | 26 / 24 |
| Stablecoin amount permutation | 192 | 87 / 51 | 28 / 26 |
| Black-funds causal veto | 193 | 87 / 54 | 29 / 23 |
| USDC only | 192 | 87 / 54 | 27 / 24 |
| USDT only | 10 | 9 / 0 | 1 / 0 |

The near identity of primary, USDC-only, amount-permuted, and stale clocks is a
stronger rejection reason than the single train run failure.

## Integrity evidence

- preregistration commit: `8729160`
- source-support implementation commit: `1b5e66f`
- implementation SHA-256:
  `4cb048d6cc70efb40f6b0a7a5cd728977e928f3dae35fba1b012ca09aa0c18ee`
- report manifest:
  `1a7ec88467779e461217af1430f79f21fdeb127ba7f29abd1a836a36c99b1faf`
- report SHA-256:
  `add1f54034953d1040fdf5b34d794865fde84d05675c8b7f7f8e4e8c7918f2bd`
- clock SHA-256:
  `86565774ae97a1024c5a66b4d59a1f5413bf4608398623359dd3ee24572f0ef3`
- clock rows: 2,681 across primary and thirteen controls;
- WBTC rows/batches: 993 / 992;
- pre-seal stablecoin rows/directional/veto: 266,360 / 265,718 / 642;
- post-2023 contract-event values loaded: 0;
- sealed non-timestamp fields decoded: 0;
- market/funding/future-return rows read: 0 / 0 / 0.

## Next research implication

Do not create a WSCF threshold or timing variant. The negative result is
specific and useful: with this event feed, a first same-sign stablecoin passage
is too easy to obtain and is not an independent confirmation layer. The next
candidate must leave the WBTC/stablecoin source family and use a new causal
geometry rather than another amount/window filter.
