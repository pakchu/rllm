# SQFD-6 preregistration — Stablecoin Quote Flow Diffusion

## Decision boundary

Freeze one new BTCUSDT perpetual candidate before opening any exact-candidate
post-entry outcome. `SQFD-6` asks whether aggressive spot demand or supply that
appears **in both active alternative stablecoin quote books** (`BTCUSDC` and
`BTCFDUSD`) propagates into the reference USDT market only after a delay.

The clock uses completed-hour spot base volume, trade count, and taker-buy/sell
BTC only. It contains no retained spot price, perpetual price, return, funding,
OI, liquidation, options, FX, Kimchi, REX, rank-7, portfolio, or future field.

The fixed quote basket reflects the three books required by the intended 2026
live implementation. Selecting USDC and FDUSD because they remain deployable in
2026 creates a disclosed current-survivor/source-selection bias relative to a
hypothetical 2023 research process. It does not expose a future price or return,
but later splits validate a present-day deployable feature rather than pristine
historical discovery.

This is not a pristine global clean room: the repository has inspected BTC
outcomes in every declared calendar for unrelated alpha families. The exact
stablecoin-quote feature family has not previously existed in the repository,
and no SQFD post-entry outcome has been opened. A source-only probe inspected
19 threshold/control shapes and did expose future event density, direction
balance, and calendar dispersion. An adversarial review correctly classified
that as future-support contamination. The binding repair therefore ignores all
post-2023 support when choosing the final strength threshold: from the fixed
descending grid `[1.25, 1.00, 0.75, 0.50]`, select the first value with at least
50 train events, at least 35% on each side, and at most 30% in one train month.
The train-only counts are `13 / 28 / 55 / 104`; `0.75` is the deterministic
selection. Future support remains non-pristine diagnostic evidence and cannot
change any rule. No return, price, funding, PnL, CAGR, MDD, or win rate has been
opened for any SQFD shape.

## Frozen causal feature

For each completed source hour `h` and book `j`:

```text
imbalance[j,h] = (taker_buy_base - taker_sell_base) / base_volume
```

The book is valid only when the checksum-verified row exists after its frozen
activation, `base_volume > 0`, and `trade_count > 0`.

For every book independently, calculate strictly-prior rolling quantiles from
the preceding 720 completed hours, excluding `h`, with at least 672 valid
observations:

```text
center[j,h] = prior q50(imbalance[j])
scale[j,h]  = (prior q75 - prior q25) / 1.349
z[j,h]      = (imbalance[j,h] - center[j,h]) / scale[j,h]
```

Zero or non-finite scale invalidates the hour; it is never replaced.

The exact implementation is pandas/NumPy float64:
`series.shift(1).rolling(window=720, min_periods=672).quantile(q,
interpolation="linear")`. The window is 720 positions on the exact hourly
source grid, not 720 surviving observations; invalid rows stay `NaN` and do not
count toward `min_periods`. No feature is rounded before comparison, and ties
use the displayed `>=` or `<` operator literally.

Alternative-quote participation is:

```text
alt_share[h] = (volume_USDC + volume_FDUSD)
               / (volume_USDT + volume_USDC + volume_FDUSD)
```

Its threshold is the strictly-prior 720-hour median with the same 672-hour
minimum.

## Frozen setup and action

At source hour `h`, all of the following are required:

1. `sign(z_USDC) == sign(z_FDUSD) != 0`;
2. `min(abs(z_USDC), abs(z_FDUSD)) >= 0.75`;
3. `side = sign((z_USDC + z_FDUSD) / 2)`;
4. `side * z_USDT < 0.50`, so reference-book flow is opposite or materially
   weaker than the alternative-book consensus;
5. `alt_share >= strictly-prior q50(alt_share)`;
6. the complete boolean setup changes from false to true.

Only a false-to-true transition can signal. At most one global position is
open. No threshold grid, direction search, model fit, or hold search is
authorized after this commit.

The historical source row `[h,h+1h)` has an exchange close timestamp immediately
before `h+1h`. SQFD conservatively waits a full five-minute latency bucket and
enters the BTCUSDT USD-M perpetual at the `h+1h+5m` open. This is a historical
clock assumption, not proof that a production collector can finalize all three
books by then. Live promotion additionally requires a forward latency/parity
audit that fails closed if any book is not finalized before the decision. The
position exits at the scheduled open exactly six elapsed hours later.
There is no stop, take-profit, dynamic exit, or sizing model. Notional is fixed
at `0.5x`.

Primary setup/onset rows are built over the complete three-year timeline.
Candidate entries are sorted once; an entry is accepted only when
`entry_time >= prior_accepted_exit_time`, with the exit boundary exclusive.
This global reservation happens before split containment. Independent-clock
controls repeat the same reservation on their own clock; same-clock controls
reuse primary reservations. Only afterward are rows sliced to splits requiring
signal, entry, and exit all contained.

## Source contract

- panel:
  `data/binance_stablecoin_quote_flow_btc_2023_2026/BTC_stablecoin_quote_flow_1h_2023-07-01_2026-06-30T23.csv.gz`
- panel SHA-256:
  `064d1c88d5a72efe43bb05b360b1e6b62d75366d52e8bd9fafe963a9e2f9862b`
- source manifest:
  `data/binance_stablecoin_quote_flow_btc_2023_2026/build_manifest.json`
- source manifest SHA-256:
  `9e6a82b9747df5c0ba1c9278e436551de03ef6136c0ad3aeb05f0a451ed12134`

The support builder may read only this panel, the source manifest, the
preregistration, and explicitly frozen comparator clocks. It may not read
execution OHLC, funding, returns, PnL, win rate, CAGR, or drawdown.

## Outcome-blind support gates

| Window | Minimum non-overlapping events |
|---|---:|
| 2023 H2 train | 50 |
| 2023 Q3 / Q4 | 10 / 25 |
| 2024 test | 100 |
| 2024 H1 / H2 | 40 / 40 |
| 2025 eval | 100 |
| 2025 H1 / H2 | 40 / 40 |
| 2026 H1 final | 50 |
| 2026 Q1 / Q2 | 20 / 20 |

Every parent window must have at least 30% on each side. Maximum one-month
share is 35% in train, 20% in each full OOS year, and 30% in final. Exact-entry
Jaccard against a frozen existing-alpha comparator must be at most 0.10 and
near-six-hour containment at most 0.35 wherever a common-coverage comparator
clock can be reproduced without opening outcomes.

The frozen comparator clocks are primary rows only from:

- `data/options_perpetual_demand_relay_clocks_2023_2026.csv.gz`, SHA-256
  `ceb79b206c3e1f6bf78b02cd2ace9a94f875ce930a704cc6e7a5a8b255021b99`;
- `data/premium_compression_breakout_relay_clocks_2020_2026.csv.gz`, SHA-256
  `659fc1b6b6e3a20e60031ed1d50f51c8c7d2836956f911f62ad13e4152740cda`;
- `data/premium_snapback_recenter_clocks_2020_2026.csv.gz`, SHA-256
  `cb209ed35f9baa08cc2fb3dd5bd60b8e747b1408c09507b774ca275e0b2b2db6`.
- `results/fiat_quote_participation_rotation_clocks_2026-07-17.csv`, SHA-256
  `54a70cce565d4f1727d095707471235f01345b94179a6c37df9f4c37d1a458a2`.

Near-six-hour containment is calculated in both directions on the exact common
calendar; the larger fraction must be at most 0.35. These failed research
clocks are representation checks, not profitability benchmarks. Rank-7 and the
funding/premium shadow candidate lack a standalone frozen clock artifact here;
their return correlation is tested only if SQFD survives sequential standalone
evaluation and cannot be used to select or repair SQFD.

FQPR is explicitly included because SQFD is not structurally orthogonal to its
general quote-participation/reference-suppression gate archetype. The claimed
novelty is narrower: intraday stablecoin-quote taker-flow diffusion on a new
source axis, not invention of a new gate shape.

Any support failure retires SQFD-6 before outcomes. Observed support may not be
repaired by changing activation, lookback, minimum history, z threshold,
participation rule, USDT lag rule, onset, latency, side, or hold.

## Sequential strict evaluation

The declared three-year calendar is exactly `[2023-07-01, 2026-07-01)`:

1. train: `[2023-07-01, 2024-01-01)`;
2. test: `[2024-01-01, 2025-01-01)`;
3. eval: `[2025-01-01, 2026-01-01)`;
4. final: `[2026-01-01, 2026-07-01)`.

Signal, entry, and exit must all be contained in the reported split. Open only
train first. Open the next split only if every unchanged gate passes. Stop on
the first failure and keep later outcomes sealed.

Every opened split must report absolute return, full-calendar CAGR, strict MDD,
CAGR/strict-MDD, trades, long/short counts, mean gross underlying move, and
weekly-cluster sign-flip probability, and must satisfy:

- positive absolute return;
- CAGR/strict-MDD at least 3.0;
- strict MDD at most 15%;
- weekly-cluster two-sided sign-flip `p <= 0.10`;
- mean gross underlying move at least 20 bp;
- both contained half-calendar returns positive;
- positive 10 bp/notional/side stress return;
- stress CAGR/strict-MDD at least 2.5;
- primary ratio at least 0.25 above every frozen mechanism control.

The weekly-cluster test is fully bound. Assign each trade to its UTC entry
timestamp's ISO year/week and sum **net account returns after base costs and
realized funding** within each week. The observed statistic is the absolute
mean net trade return. A null draw multiplies each weekly sum by an independent
Rademacher sign and divides the signed total by the number of trades. Count
`abs(null) >= observed - 1e-15`. With at most 20 non-empty weekly clusters,
enumerate all `2^K` sign vectors and use `exceed/2^K`. Otherwise use NumPy
`default_rng(20260719)` for 20,000 draws in deterministic batches and report
`(1+exceed)/(20,000+1)`. An empty trade set has `p=1`. No alternative weekly
boundary, return field, seed, draw count, smoothing, or small-sample method is
allowed.

Base cost is 6 bp per notional side. Strict MDD includes the global/pre-entry
HWM, entry cost, conservative funding-boundary marks, favorable-then-adverse
movement inside every held 5-minute bar, virtual adverse-mark exit cost, and
actual exit cost. CAGR includes the complete declared split, including warm-up
and idle cash.

## Frozen controls

- `no_alt_breadth`: use the volume-weighted alternative aggregate without
  requiring sign agreement, where
  `alt_z=(v_USDC*z_USDC+v_FDUSD*z_FDUSD)/(v_USDC+v_FDUSD)`, require
  `abs(alt_z)>=0.75`, and set `side=sign(alt_z)`; all other gates stay fixed;
- `no_usdt_lag`: remove only `side*z_USDT < 0.50`;
- `no_participation`: remove only the alternative-share median condition;
- `usdt_only`: use `sign(z_USDT)` and `abs(z_USDT)>=0.75` on its own onset;
- `direction_flip`: same primary entries, opposite side;
- `extra_latency_1h`: same signal and side, entry/exit delayed one hour;
- `deterministic_random_side`: hash ASCII
  `SQFD-6|YYYY-MM-DDTHH:MM:SSZ` at decision time; an even first hexadecimal
  nibble is long and an odd nibble is short.

Controls may falsify the mechanism. None may replace or repair the singleton.
For every metric, `ratio = CAGR_pct / strict_MDD_pct`. If strict MDD is zero,
the ratio is `+inf`, `0`, or `-inf` for positive, zero, or negative CAGR; an
infinite positive control therefore fails the primary-margin gate rather than
being silently discarded.

The machine manifest hash is SHA-256 of UTF-8 JSON after removing the
`manifest_hash` field and serializing with sorted keys, no insignificant
whitespace, and `ensure_ascii=false`. The separately reported file SHA-256 is
not expected to equal this canonical payload hash.

## RLLM boundary

SQFD-6 is a formulaic alpha admission test. If and only if it passes every
sequential gate, an RLLM may later reason over the frozen symbolic state to
abstain or scale down under a separately preregistered reward. It may not
create an entry, reverse the side, increase fixed leverage, or rehabilitate a
failed deterministic alpha.
