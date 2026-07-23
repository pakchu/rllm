# URCD-72 source-support implementation contract

## Scope

Implement the committed URCD-72 mechanism exactly through source support and,
only if every source gate passes, timestamp-only comparator novelty.  This
work unit may not open BTC OHLC, funding, future returns, labels, PnL,
absolute return, CAGR, strict MDD, or post-2023 USDC value fields.

The implementation and synthetic tests must be committed and clean before the
real promoted source CSV is parsed.  An injected/synthetic build is never
artifact-eligible and may not authorize comparator access.

## Required causal implementation

- pre-screen every row by `available_at`; rows at or after
  `2024-01-01T00:00:00Z` are timestamp-only sentinels and their other fields
  remain semantically undecoded;
- retain only valid pre-seal USDC `mint` rows and use `indexed_address_2` as an
  operational endpoint;
- compute exact integer amount-weighted HHI on `(D-24h,D]` at the six-hour UTC
  grid;
- use exactly 180 same-hour daily reference endpoints and at least 120 valid
  windows;
- calculate q20/q80 HHI and independently sorted q50 amount with nearest-rank
  order statistics and no binary float;
- emit only tail-entry transitions; schedule train and selection independently
  with crossing candidates removed before reservation;
- implement every frozen control and the exact SHA-256 permutations; and
- write deterministic gzip sorted by `(entry_time, signal_id, control)`.

## Source-support order

1. Verify preregistration plus the boundary, mechanism, source CSV, source
   manifest, source header, replay/finality assertions, and implementation
   bindings without decoding a source value row.
2. Parse the promoted pre-2024 source and build all primary/control clocks.
3. Evaluate every structural and permutation-selectivity gate.
4. If any gate fails, stop with zero comparator data rows decoded.
5. Only on a complete pass, verify the whole frozen comparator cohort's file
   and header checksums before reading the first comparator data row.  Then
   inspect each raw `entry_time` only as a lexical timestamp sentinel: rows
   outside the frozen overlap are skipped without parsing their timestamp,
   candidate, control, side, or duplicate identity.  Within the overlap,
   decode only `candidate`, `control`, `entry_time`, and `side`; hash-bound
   known-but-unselected controls are skipped, while any truly unknown control
   fails closed.
6. Evaluate the frozen timestamp-only novelty metrics.
7. Regardless of verdict, keep all economic outcome counters at zero.

## Required artifacts

- clocks:
  `data/usdc_recipient_concentration_dislocation_2021_2023/urcd72_support_clocks_2021_2023.csv.gz`;
- report:
  `results/usdc_recipient_concentration_dislocation_support_2026-07-23.json`.

Both artifacts are write-once.  The report must bind the committed evaluator,
tests, implementation contract, preregistration artifact, source artifact,
clock bytes, support checks, comparator-access count, verdict, and all zero
outcome counters.  A failed support or novelty gate retires URCD-72 unchanged.

## Bound inputs

- `docs/usdc-recipient-concentration-dislocation-boundary-2026-07-23.md`
- `docs/usdc-recipient-concentration-dislocation-mechanism-decision-2026-07-23.md`
- `results/usdc_recipient_concentration_dislocation_preregistration_2026-07-23.json`
- `results/ethereum_stablecoin_issuance_redemption_source_manifest_2026-07-21.json`
- `data/ethereum_stablecoin_issuance_redemption_2020_2023/ethereum_usdt_usdc_issuance_redemption_2020_2023.csv.gz`
