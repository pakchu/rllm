# IVPLH-72 source-support implementation contract

## Scope

Implement the sealed IVPLH-72 mechanism through source support only. This work
unit may decode the frozen 2020–2023 BTCUSDT source and the hash-bound IVFHR-72
predecessor clock, but it may not open comparator rows, post-entry prices,
funding, returns, labels, PnL, CAGR, or strict MDD.

The evaluator, tests, and this contract must be committed and clean before the
real source or predecessor rows are decoded. Synthetic builds are never
artifact-eligible.

## Required causal implementation

- validate the sealed preregistration and every frozen document/source/
  predecessor checksum before row decoding;
- reuse the tested exact-grid source validator and causal equal-notional anchor
  construction;
- compute reference readiness from at most 180 strictly prior eligible anchors,
  requiring at least 90, without deriving a value threshold;
- reset handoff state across a missing or invalid UTC source day;
- admit the primary only for a reference-ready consecutive side handoff whose
  contemporaneous directional anchor return is non-positive;
- decide at anchor-open plus five minutes, enter one complete bar later, and
  exit after exactly 72 five-minute bars;
- derive canonical SHA-256 signal identities before deterministic
  `(entry_time, signal_id)` reservation;
- remove split-crossing rows before independent calibration and selection
  reservation; and
- exactly reproduce the 66 frozen predecessor `any_handoff` identities, with
  candidate decision equal to predecessor entry and candidate entry/exit each
  shifted by five minutes.

## Controls and gates

Implement all nine controls in their frozen order. Year permutations must be
SHA-256 lexical donor/destination bijections and may not use RNG state.
`stale_24h`, `direction_flip`, and deterministic random side must rebuild
signal identity after their timing or side transformation.

Train statistics use the already-reserved 2021–2022 subset of the independent
2020–2022 calibration reservation. Selection uses its own 2023 reservation.
Month, quarter, gap, side-run, half-year, side-balance, and permutation
denominators must match the sealed definitions. Empty permutation denominators
evaluate to one and fail.

Any identity, integrity, support, or permutation failure retires IVPLH-72
unchanged. Comparator and economic evaluation remain closed.

## Required artifacts

- clocks:
  `data/intrinsic_volume_price_lag_handoff_clocks_2020_2023.csv.gz`;
- report:
  `results/intrinsic_volume_price_lag_handoff_support_2026-07-24.json`.

Both artifacts are deterministic and write-once. The clock contains only
control, signal identity, source day, decision, entry, exit, and side. The
report binds the committed evaluator, tests, this contract, preregistration,
source, predecessor, clock bytes, support verdict, and explicit zero outcome
and comparator counters.

## Bound inputs

- `docs/intrinsic-volume-price-lag-handoff-boundary-2026-07-23.md`
- `docs/intrinsic-volume-price-lag-handoff-mechanism-decision-2026-07-23.md`
- `docs/novelty-comparator-common-window-policy-2026-07-23.md`
- `results/intrinsic_volume_price_lag_handoff_preregistration_2026-07-23.json`
- `results/intrinsic_volume_flow_handoff_relay_preregistration_2026-07-23.json`
- `results/intrinsic_volume_flow_handoff_relay_support_2026-07-23.json`
- `data/intrinsic_volume_flow_handoff_relay_clocks_2020_2023.csv.gz`
- `data/binance_um_kline_reference_btc_2020_2023/build_manifest.json`
- `data/binance_um_kline_reference_btc_2020_2023/BTCUSDT_5m_2020-01-01_2023-12-31.csv.gz`
