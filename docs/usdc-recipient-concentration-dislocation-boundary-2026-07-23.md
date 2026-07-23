# URCD-72 candidate boundary — USDC recipient-concentration dislocation

## Decision

Select exactly one new source/mechanism axis for the next outcome-blind alpha
cycle: **URCD-72, USDC recipient-concentration dislocation**.

URCD-72 will test whether the cross-sectional concentration of causally
finalized USDC `Mint` amounts across recipient addresses changes the BTC
liquidity regime.  Diffuse routing is the prospective `LONG` state and
concentrated routing is the prospective `SHORT` state.  The complete clock,
history window, quantiles, materiality rule, latency, hold, controls, support
floors, and novelty contract must be frozen in a later mechanism document
before any real URCD feature value or incidence is calculated.

This decision does not assert that a recipient is a customer, exchange,
beneficial owner, or BTC buyer.  A USDC recipient is only an on-chain
operational endpoint.  The falsifiable claim is about changes in routing
geometry, not actor identity or fiat ownership.

## Why this is a distinct axis

The promoted Ethereum source has already supported or rejected the following
stablecoin mechanism families:

| Identity | Frozen mechanism | Relationship to URCD |
|---|---|---|
| IRH-36 | opposite-signed USDC/USDT large-event pair | cross-issuer event order; no recipient distribution |
| AMTR-48 | same-minter burn/mint turnaround | caller continuity; recipient used only as a concentration gate |
| UGCI-288 | six-hour USDC gross/net clearing imbalance | aggregate signed amount; ignores recipient partition |
| USDC role-topology audit | `mint.to == burn.burner` closure | temporal role relation; no cross-sectional amount geometry |
| WCDR/WTSL/WSCF | WBTC flow combined with stablecoin direction/confirmation | wrapped-collateral interaction; URCD uses no WBTC |
| SQFD/SDDR/UCBR | secondary-market quote/denominator/collateral clocks | exchange-market observables; URCD uses finalized primary-contract logs |
| stablecoin supply breadth | daily multi-issuer supply breadth | aggregate supply snapshot; no event-level recipient routing |

URCD is therefore not a threshold, window, direction, or hold repair of an
opened candidate.  It uses a field that no prior trading clock used as its
primary state: the **amount partition across `Mint` recipients within a causal
window**.  It will use neither USDC burn amount nor USDT/WBTC/BTC state to
create direction.

## Evidence already exposed before this boundary

The following source-only aggregate facts were already public in committed
audits and are disclosed as prior exposure:

- 2020–2023 promoted Ethereum panel: 266,362 rows;
- USDC `Mint`: 99,033 rows; USDC `Burn`: 166,552 rows;
- 6,311 distinct full-period mint recipient addresses;
- the source is Ethereum mainnet, independently replayed, and each event is
  available only at canonical block `N+64`;
- the role-topology audit found very high concentration in the special subset
  of recipients that later also appear as burn callers; and
- prior stablecoin mechanisms' aggregate source-support successes/failures are
  known from their committed rejection documents.

These facts make the routing observable semantically and structurally
available, but they are not URCD incidence or alpha evidence.  Full-period
recipient membership may not be used as a historical feature.

## Values that remain unopened

As of this boundary, the current work unit has opened none of the following:

- raw source CSV rows or individual mint amounts/recipient identities;
- per-window recipient count, amount share, HHI, entropy, novelty, transition,
  quantile, tail, or candidate-entry values;
- URCD train/selection event counts, side counts, calendar dispersion, or
  control incidence;
- comparator clock timestamps or overlap statistics;
- BTC OHLC, funding, future returns, labels, PnL, absolute return, CAGR, or
  strict MDD for URCD; or
- any post-2023 contract-event value.

The source manifest, source audit, source-feasibility document, builder schema,
and prior mechanism/rejection summaries were read.  No candidate outcome or
raw source-value artifact was read.

## Hard research boundary

Before the mechanism implementation is committed:

1. do not parse the promoted source CSV;
2. do not calculate a real recipient-concentration statistic or clock;
3. do not open comparator timestamps;
4. do not read BTC market, funding, future-return, or portfolio rows; and
5. do not inspect post-2023 source values.

The next work unit may use only schemas, synthetic fixtures, committed source
hashes, public contract semantics, and prior aggregate rejection evidence to
freeze URCD-72.  A source-support failure retires this identity without
changing the concentration estimator, direction map, window, threshold,
latency, hold, or support floors.  Outcome access requires a separate,
committed authorization after source support and novelty pass.

## Bound source references

- `results/ethereum_stablecoin_issuance_redemption_source_manifest_2026-07-21.json`
- `docs/ethereum-stablecoin-issuance-redemption-source-audit-2026-07-21.md`
- `docs/ethereum-stablecoin-issuance-redemption-source-feasibility-2026-07-21.md`
- `training/build_ethereum_stablecoin_issuance_redemption.py`
- `docs/authorized-minter-turnaround-relay-support-rejection-2026-07-21.md`
- `docs/usdc-gross-clearing-imbalance-support-result-2026-07-22.md`
- `docs/usdc-role-topology-audit-2026-07-21.md`
- `docs/issuer-rotation-handoff-source-support-rejection-2026-07-21.md`
- `docs/wbtc-stablecoin-finalized-confirmation-relay-support-rejection-2026-07-23.md`
- `docs/wbtc-turnover-stablecoin-liquidity-support-rejection-2026-07-23.md`
