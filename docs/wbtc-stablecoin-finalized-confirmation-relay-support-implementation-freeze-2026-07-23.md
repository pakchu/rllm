# WSCF-72 source-support implementation freeze — 2026-07-23

## Status

The exact source-only implementation is frozen before WSCF source incidence or
BTC outcomes are computed. It is bound to preregistration commit `8729160`,
artifact SHA-256
`b105051e2b3bdf806c3abff30312889656534f49914ca4e4f584cb9723fb2fe0`,
and manifest
`1466ec5118df70985dda8692df1496d2d03285449ecadd5f8fcdec216b3f978f`.

Implementation:
`training/build_wbtc_stablecoin_finalized_confirmation_relay_support.py`
with SHA-256
`4cb048d6cc70efb40f6b0a7a5cd728977e928f3dae35fba1b012ca09aa0c18ee`.

## Executable interpretation

The implementation preserves the preregistered mechanism and makes four
previously prose-level details executable:

1. rows sharing exact `available_at` are one atomic batch; intra-batch
   transaction/log ordering cannot create a first passage;
2. raw candidates are globally sorted before split containment and 72-hour
   non-overlap are applied;
3. an accepted confirmation identity cannot be reused;
4. `black_funds_veto` remains causal: confiscation vetoes only when its own
   `available_at` is after the WBTC anchor and no later than the would-be
   confirmation batch. A same-time confiscation and directional batch are
   simultaneous and the veto wins. A confiscation that becomes available after
   an already knowable confirmation cannot retroactively cancel an entry.

The fourth rule is necessary to avoid consulting the remainder of the 12-hour
window before an earlier entry. It changes no primary candidate and cannot be
used to rescue a failed primary.

## Sealed-boundary handling

The stablecoin file contains finalized rows whose availability crosses the
`2024-01-01` seal. The loader extracts only the final physical CSV field
`available_at` from each raw line. At the first sealed timestamp it stops before
CSV-decoding or converting any other field. Audit counters require zero sealed
non-timestamp fields and zero post-2023 contract-event values decoded.

No BTC OHLC, funding, return, PnL, CAGR, MDD, network call, or subprocess is
available to this program.

## Frozen outputs

If executed, the implementation may create only:

- `data/wbtc_stablecoin_finalized_confirmation_relay_2021_2023/`
  `wscf72_support_clocks_2021_2023.csv.gz`;
- `results/wbtc_stablecoin_finalized_confirmation_relay_`
  `support_2026-07-23.json`.

Both are deterministic and write-once. Source failure retires the candidate
without changing the mechanism. Source pass only authorizes a separately
implemented and hash-frozen market evaluator.
