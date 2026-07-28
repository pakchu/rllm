# COGR-12 Gross9 marginal preregistration

**Frozen before any COGR BTC price, funding, return, PnL, Gross9 path, or
post-entry outcome is opened.**

## Target

COGR-12 asks whether the current QQQ/GLD US cash-opening geometry reaches BTC
with enough delay to add a genuinely useful 12-hour bidirectional sleeve to
Gross9. The only current-session external prices are QQQ and GLD opens. Every
other source feature ends at the preceding completed US cash session.

The research clock is fixed:

1. QQQ/GLD feature available at `09:35 America/New_York`;
2. Binance timestamps denote five-minute **bar opens**;
3. the Gross9 snapshot is position `T-1`, the 09:30 bar ending at 09:35;
4. position `T`, the complete BTC 09:35–09:40 latency bar, is forbidden to
   both model and gate;
5. enter at position `T+1`, the BTCUSDT 09:40 open;
6. exit at `T+145`, exactly 144 held five-minute bars (12 hours) later.

This is not the old cross-asset transfer family, which traded QQQ/KODEX
200/GLD using translated BTC rules, and it is not a crypto spot→perpetual
clock. The source, target market, and information-arrival mechanism differ.

## Frozen source

- safe feature artifact:
  `data/cash_open_cross_asset_gap_relay_pre2025/qqq_gld_cash_open_safe_features_pre2025.csv.gz`
- 5,063 sessions / 5,002 valid / 31 features / last session 2024-12-31;
- the frozen source artifact ends exclusively at 2025-01-01, while the
  physical selection reader stops exclusively at 2024-01-01;
- artifact SHA-256:
  `d0d04293cf05a7703b6970e5b97dc2e1e69ecb2d42de4b20c78958523e2e9c44`;
- manifest SHA-256:
  `5f61359d837106acd84c2dc509600cda7b5d10522acd644c768bfb769d2909f8`.

The machine contract is
`results/cash_open_cross_asset_gap_relay_marginal_preregistration_2026-07-28.json`.
It binds all input paths and hashes. Its frozen SHA-256 is
`d90ce7d5d43ce761d1d7078cf1abe0767200772fc85e6c435f8a4a9b69f10dce`.

The frozen evaluator is
`training/evaluate_cash_open_cross_asset_gap_relay_marginal.py`; its SHA-256
is `d7de5f20836ae18a20a48973cfa6804f0a200d7fa4beb93d229ad34d7732c817`
and is also recorded inside the machine contract. That self-binding prevents
changing the model, controls, accounting, gates, or phase isolation after
observing a selection or evaluation result.

## Learner

- `ExtraTreesRegressor`, sklearn `1.7.2`;
- 256 trees, depth 4, minimum leaf 32, `max_features=0.75`;
- seeds `7, 71, 715`, one thread each;
- direct finite float64 features, no imputation or scaling;
- 0.5x unit leverage, 6 bp/notional/side normal cost;
- 10 bp/notional/side replay stress;
- no TP/SL; exact funding and 12-hour time exit.

For long and short separately, the model predicts exact net return and strict
adverse excursion. Side score is:

`mean(return - 0.5*adverse) - 0.5*seed_std(return - 0.5*adverse)`.

The higher side is selected. Entry requires a score at least
`max(0, prior-fold q75)`. Calibration uses predicted scores only, never
calibration outcomes.

## Clean split

| role | period | fit boundary | use |
|---|---|---|---|
| calibration | 2023 H1 | labels fully before 2023-01-01 | q75 only |
| test/selection | 2023 H2 | labels fully before 2023-07-01 | choose one top cell |
| eval | 2024 | labels fully before 2024-01-01 | exact top1 pass/veto only |

Complete held paths are purged at every fit and metric boundary. H1 outcomes
cannot rank a candidate. H2 picks one cell among three coordination modes and
weights `0.25, 0.50, 0.75, 1.00`. 2024 cannot rerank, repair, lower a
threshold, or substitute rank 2.

Coordination modes are:

- unrestricted;
- Gross9 completely flat at the 09:35 information boundary;
- Gross9 at least 5% below its continuous running strict upper-envelope peak
  at that same boundary.

An underfilled mode fails closed.

## Same-gross portfolio test

Gross9 remains:

| sleeve | weight |
|---|---:|
| `cand_rex_veto_7` | 1.6 |
| `fresh_kimchi_fx` | 2.0 |
| `frozen_annual_rank7` | 3.0 |
| `markov_transition_long` | 2.0 |
| `rex_taker_low_range_position` | 0.4 |

For candidate weight `c`, the treatment keeps Gross9 unchanged and applies
multiplier `c` to the canonical COGR path, which already embeds 0.5x BTC
notional. The comparison unit is therefore a **configured sleeve multiplier**,
not instantaneous exchange notional. Treatment and comparator both have
`9+c` configured units; the comparator scales every Gross9 sleeve by
`(9+c)/9`. A pass needs at least `+0.05` CAGR/strict-MDD
ratio points versus that same-gross comparator, at least 97% of unscaled
Gross9 absolute return, lower strict MDD than unscaled Gross9, portfolio MDD
at most 20%, and exact-entry Jaccard at most 0.25 versus every Gross9 sleeve.

## Standalone and mechanism gates

2023 H2 must have:

- positive absolute return;
- CAGR/strict-MDD at least 1.5 and strict MDD at most 15%;
- at least 25 trades, with long and short each at least 20%;
- no month or weekday above 35% of trades;
- positive 10 bp stress return;
- at least +0.10 CAGR/MDD over the best eligible frozen control.

The exact 2024 top1 must additionally have:

- at least 50 trades;
- no month above 20% and no weekday above 30%;
- positive standalone return in both calendar halves;
- non-negative CAGR/MDD margin over the best eligible control;
- standalone weekly-cluster sign-flip `p <= 0.10`;
- paired same-configured-gross portfolio weekly sign-flip `p <= 0.10`;
- a strictly positive 90% bootstrap lower bound for mean paired weekly
  portfolio log excess;
- at least 26 active weeks in each confirmatory test;
- every standalone, stress, same-gross, return-retention, MDD, and overlap gate
  above.

Controls are QQQ-only, GLD-only, prior-only/no-current-open, one-session stale
opening gaps, weekday-only, exact side flip, constant long, deterministic
random side, and one-US-session delayed entry. Controls can only reject the
mechanism; they cannot replace it. All nine always enter the best-control
comparison. A zero-trade control receives ratio 0 rather than disappearing.
The delayed control maps to the next valid common cash session, preserves the
original side, re-applies chronological nonoverlap, and drops any path crossing
the measured boundary.

## Accounting

Every table must report:

`absolute return / full-calendar CAGR / strict MDD / CAGR-MDD / trades`.

Idle days remain in CAGR. Strict MDD includes global/pre-entry HWM, entry and
hypothetical adverse-exit costs, exact funding debit path, every held
five-minute favorable-then-adverse same-BTC envelope, and terminal exit.

## Stop rule

The frozen evaluator has separate `selection` and `eval` commands. Selection
builds no 2024 COGR feature, target, control, or metric; the legacy Gross9
context may parse only its checksum-bound pre-2025 sources but exposes only
the H2 metric slice. If no H2 cell passes, the eval command fails closed. If
the exact H2 top1 fails 2024,
COGR-12 is rejected and this QQQ/GLD cash-open shallow-tree family closes.
2025/2026 may later veto a passing frozen top1, but can never certify, rerank,
repair, or select another cell.
