# RQCI source/mechanism decision — 2026-07-20

## Decision

The next candidate is **RQCI-24 — Residual Quote-Curvature Impulse**, a
two-hour BTCUSDT USD-M policy built from the already checksum-audited 2023
cumulative average-quote panel.

No RQCI event incidence, entry/later price, return, PnL, or post-2023 row was
opened for this decision.

## Why this is not an RNCM repair

RNCM-72 tested coherent translation of all four radial skew levels over 30
minutes and was rejected because its deconfounded form produced only 39
non-overlapping events at the loosest frozen threshold. RQCI does not weaken
RNCM's threshold, dominance, quiet-center, coherence, or holding rule.

RQCI instead studies a different geometric derivative:

```text
curvature(t) = [skew_5(t) - skew_4(t)]
               - [skew_3(t) - skew_2(t)]
```

Its 30-minute impulse measures whether directional average-quote asymmetry is
moving into the outer radial shell relative to the inner shell. A positive
impulse means ask-side impact convexity is expanding relative to bid-side
convexity; the provisional economic orientation is long. Negative is short.
This is a cross-band curvature twist, not the common radial translation RNCM
required.

## Mechanical preflight

Before reading real RQCI incidence, a prototype curvature impulse was subjected
to the existing fixed-absolute-book null suite. Strictly-prior quote-center
regression, residual/raw dominance, and a quiet-center condition eliminated
events in all five deterministic scenarios, including the asymmetric discrete
tick ladder that exposed the raw RNCM defect.

The exact RQCI formula, threshold grid, scheduler, support limits, novelty gate,
and later strict evaluator contract must be committed separately before the
real 2023 source is inspected.

## Why a new external source was not selected

Several genuinely different live observables fail the three-year historical
contract:

- Bybit's public insurance-pool endpoint is useful live, but the documented
  interface is a current balance feed and its rollout/change-log history is too
  recent to reconstruct 2023-2026 minute observations:
  <https://bybit-exchange.github.io/docs/v5/market/insurance>;
- Coinbase exposes public order/trade and auction feeds live, but the auction
  channel does not provide a public three-year historical indicative-quote
  archive:
  <https://docs.cdp.coinbase.com/exchange/websocket-feed/channels>;
- Bitcoin Core exposes current-node mempool ancestry, fee, and replacement
  state, but those observations are node-local and cannot be reconstructed from
  the confirmed blockchain without contemporaneous snapshots:
  <https://bitcoincore.org/en/doc/23.0.0/rpc/blockchain/getmempoolentry/>.

Those sources remain future-forward-data candidates, not admissible historical
discovery axes for the current three-year target.

## Frozen sequence

1. Commit RQCI source-only preregistration and synthetic tests.
2. Run the strictest-first 2023 incidence and prior-depth-clock novelty gate.
3. Reject without returns if support fails; no threshold or hold repair.
4. Only a passing canonical clock may receive a separately committed strict
   outcome evaluator.
5. Open 2023, then 2024, then 2025, then recent 2026 sequentially, stopping on
   the first failed gate.
