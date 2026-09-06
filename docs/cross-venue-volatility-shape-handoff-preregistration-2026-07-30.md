# CVVH-432 preregistration — 2026-07-30

## Research boundary

This document freezes the complete `CVVH-432` protocol before Binance BTCBVOL
or Deribit DVOL values, candidate incidence, comparator rows, reconstructed
Gross9 clocks, BTC execution prices, funding, returns, PnL, CAGR, or drawdown
are opened.  The producer may verify only exact compressed-file hashes, byte
counts, one CSV header line, hash-bound JSON metadata, and read-only Git
identity.  Git identity includes one `ls-remote` attestation that the exact
canonical origin fetch/push URL contains the current branch SHA; it makes no
source or web request.

The candidate is the exact volatility-shape handoff defined in
`docs/cross-venue-volatility-shape-handoff-mechanism-decision-2026-07-30.md`.
No close-ratio, BTC price-follow, premium-path direction, 24/48-hour hold
repair, polarity inversion, grid, or rank-2 substitution is permitted.

## Sources and causal clock

The write-once JSON binds the committed Binance BTCBVOL hourly file and
manifest and the committed Deribit DVOL hourly file and summary by path,
SHA-256, exact UTF-8 header, and parser.  Hours join one-to-one on completed
UTC-hour availability: Binance `feature_available_time_utc` to Deribit
`close_time`.  No tolerance, nearest join, fill, or imputation exists.

OHLC tokens are exact bounded decimals converted to exact rationals.  Entry is
`ceil_to_5m(joint availability)+5m`; aligned hours still wait five minutes.
Exit is 432 five-minute bars later.  Reservation is global on `[entry,exit)`,
leverage is 0.5x, and base/stress costs are 6/10 bp per notional side.

## Source-support gate

The committed source-support evaluator must run before novelty or outcomes.
It must pass all of:

- selection `[2023-06-01,2025-01-01)`: at least 45 accepted events, at least
  12 in 2023H2, 12 in each 2024 half, at least 14 per side, and no month above
  20%;
- 2025: at least 30, at least 8 per side, no month above 25%;
- 2026 through May: at least 15, at least 4 per side, no month above 30%;
- full horizon: no accepted-entry gap above 90 elapsed days and no same-side
  run above 12; and
- exact future-append invariance of the pre-2025 validity, states, candidates,
  accepted ids, sides, and entry/exit clocks.

Each independent structural control must remain below strict 0.90 exact-entry
Jaccard and strict 0.95 deterministic one-to-one 24-hour matched share.  Any
failure retires the unchanged identity.

## Novelty

The protocol binds the prospective common-window policy by SHA-256.  It also
binds exact path/hash/header/filter/parser/window contracts for:

- OPDR-24;
- the old DVOL price-follow clock;
- PSR-30/6;
- PCBR-12; and
- CMSR-36.

Disclosed contamination: during prior source-only RMSR verification, a valid
prior comparator interval crossed the 2023/2024 comparison boundary.  That
timing fact motivated the prospective common-window policy.  It supplies no
CVVH source value, incidence, overlap statistic, or outcome, and the
full-containment eligibility rule is frozen here before CVVH incidence opens.

Raw comparator groups are validated before window filtering.  Only fully
contained intervals enter metrics; no clipping or partial exposure is allowed.
Each comparator and each positive-weight Gross9 sleeve needs at least ten
contained rows.

The six-hour match is deterministic one-to-one: maximize cardinality, minimize
total exact absolute lag, then choose the lexicographically smallest ordered
timestamp-pair list.  Every prior-volatility clock and every Gross9 sleeve must
pass:

- exact-entry Jaccard at most 0.10;
- maximum candidate/comparator one-to-one matched share at most 0.35;
- occupied five-minute-bar Jaccard at most 0.25; and
- absolute signed-exposure Pearson at most 0.35.

Undefined metrics fail.  A comparator cannot be removed after overlap is seen.

## Gross9 and same-gross selection

Gross9 is the exact five-sleeve gross-9 roster:

| Sleeve | Weight |
|---|---:|
| `cand_rex_veto_7` | 1.6 |
| `fresh_kimchi_fx` | 2.0 |
| `frozen_annual_rank7` | 3.0 |
| `markov_transition_long` | 2.0 |
| `rex_taker_low_range_position` | 0.4 |

The preregistration binds the full ESDI Gross9 authority closure, including all
reachable local imports and package initializers, runtime configs/manifests and
source hashes, `pyproject.toml`, `uv.lock`, ABI, and the exact 108-distribution
inventory.  Full-domain sleeve clocks may be reconstructed after source
support only for structural novelty.  Future rows may veto novelty but cannot
rank economic weights; no portfolio PnL is computed during novelty.

Candidate weights are exactly 0.25, 0.50, and 0.75.  For weight `w`, every
Gross9 sleeve is scaled by `(9-w)/9` and CVVH is added at `w`; configured gross
remains 9.  Ranking uses only 2023H2 and calendar 2024.  In both periods and at
both base/stress cost, treatment must improve CAGR/strict-MDD by at least 0.05,
retain at least 97% of unscaled Gross9 absolute return, stay positive, and stay
liquidation-safe.  Strict MDD must fall in at least one selection cost cell.
Rank by maximum minimum improvement, tie lower weight, and freeze top 1 only.

## Strict economics and future veto

The economics evaluator must be committed, tested, hash-bound, and reproduced
from committed-clean novelty before opening market/funding rows.  It uses the
approved ESDI fixed-quantity strict path: exit then entry at exact open, size
from post-exit pre-entry equity, exact funding
`-side*quantity*rate*settlement_mark`, and strict MDD ordered as global HWM,
favorable OHLC plus funding credits, adverse OHLC plus funding debits,
hypothetical liquidation cost, then side costs.

Every gated stage at base and stress requires positive return,
CAGR/strict-MDD at least 3, strict MDD at most 15%, mean gross underlying move
at least 20 bp, and exact UTC entry-month clustered sign-flip p-value at most
0.10.  All gated stages, including the 19-month combined selection stage, use
exhaustive `2^k` sign enumeration.

In both selection periods and at both cost levels, the primary
CAGR/strict-MDD must strictly exceed both `body_lead_only` and
`range_lead_only`.  `deribit_led`, `stale_deribit`, and
`one_bar_delayed_entry` remain diagnostics and cannot replace the primary.
Direction-flip, deterministic-random, constant-long, and constant-short
controls cannot completely qualify.

The frozen rank-1 weight alone proceeds to 2025 and then 2026 through May.
Each future window repeats exact standalone and same-gross gates and may only
veto.  There is no reranking, repair, alternate weight, or rank-2.  The final
exact three-year report is required only after both vetoes and is descriptive,
not another selection gate.

## Claims, replay, and terminal behavior

Each authoritative stage writes one atomic claim before its first protected
read.  The claim binds commit, preregistration, evaluator closure, dependencies,
and prior receipts.  There is no retry, resume, fallback, or repair after a
claim.

Only after successful authoritative bytes are committed may one separately
claimed clean-checkout replay run with `verification_only=true`.  It has no
canonical-write, ranking, or repair authority and must reproduce byte-identical
temporary artifacts and receipts.  An authoritative failure permanently
forbids replay.  A replay mismatch is a terminal reproducibility failure.

The machine-readable contract is produced once at
`results/cross_venue_volatility_shape_handoff_preregistration_2026-07-30.json`
after the producer and tests are committed and pushed.
