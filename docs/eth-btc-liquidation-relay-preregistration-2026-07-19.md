# EBLR-60/30 — ETH→BTC liquidation relay preregistration

Date frozen: 2026-07-19  
Candidate count: **one**  
Market outcomes opened for this candidate: **no**

## Hypothesis

A directional, extreme inverse-collateral liquidation wave in ETH may reach BTC
with a short delay when BTC's own inverse-collateral forced-flow market is
still quiet. The trade follows the ETH forced-flow direction; BTC is only a
same-source "not yet relayed" filter.

This is not a repair of the rejected ICLA-60 absorption rule:

- ETH COIN-M snapshots are the sole trigger;
- BTC COIN-M snapshots supply only a quietness gate;
- USD-M taker flow, BTC price action, funding, open interest, regime, tree,
  LLM, and prior alpha signals are excluded.

## Frozen sources

| source | role | SHA-256 |
|---|---|---|
| `BTCUSD_PERP` COIN-M liquidation 5m | quietness filter | `a23b93d8567a589e9f045ae4a56393e493a8da2748c5a051804c9bdf9388ccc3` |
| BTC source manifest | source contract | `5d78686e7c40d69261f09bc77e27ff734f682abba4abb95c2291e8282380053e` |
| `ETHUSD_PERP` COIN-M liquidation 5m | trigger | `8d17ab3d5f9592f5254fef2e649065233be1777b8976983b4af38c77a8cc5bff` |
| ETH source manifest | source contract | `c515731a9029d1786c8650f5106923d4cfbe8c35ed7a947f5420a16154601f5d` |

Both feeds are censored liquidation snapshots, not complete fill tapes. Missing
archive days are invalid and may never be treated as zero-liquidation days.

## Frozen source-only rule

For each completed five-minute bar `t` and each symbol `s`:

1. Sum the last **12 bars / 60 minutes** of:
   - total accumulated liquidation contracts;
   - signed contracts = forced-buy contracts minus forced-sell contracts;
   - snapshot event count.
2. Require all 12 bars for both ETH and BTC to be source-valid.
3. Define within-symbol imbalance:

   `imbalance_s(t) = signed_60m_s(t) / total_60m_s(t)`.

4. Define `Q95_s(t)` as the 95th percentile of **positive** 60-minute total
   contract waves in the strictly prior 28 calendar days. The current wave is
   excluded by shifting one completed bar. Require at least 300 positive prior
   windows for each symbol.
5. Define within-symbol severity:

   `severity_s(t) = total_60m_s(t) / Q95_s(t)`.

   Raw ETH and BTC quantities may not be compared. A constant contract-size
   multiplier cancels inside this ratio.
6. ETH trigger:
   - `severity_ETH >= 1.00`;
   - `abs(imbalance_ETH) >= 0.70`;
   - ETH 60-minute event count `>= 3`.
7. BTC not-yet-relayed gate:
   - `severity_BTC <= 0.50`.
8. Direction follows the ETH forced flow:
   - `imbalance_ETH >= +0.70` → long BTC (`+1`), corresponding to dominant
     forced purchases closing ETH shorts;
   - `imbalance_ETH <= -0.70` → short BTC (`-1`), corresponding to dominant
     forced sales closing ETH longs.
9. The joint feature becomes available at the later of the two source
   availability timestamps. Enter at the next five-minute open, normally the
   last source bar open plus ten minutes.
10. Hold exactly **6 bars / 30 minutes**. There is no stop, take profit, or
    overlapping position.

Frozen constants: 12-bar wave, 28-day prior reference, q95, 300 positive prior
windows, three ETH events, 0.70 ETH imbalance, 1.00 ETH severity, 0.50 BTC
severity, 30-minute hold. They may not be repaired after any execution outcome
is opened.

## Sequential split

| stage | start inclusive | end exclusive |
|---|---|---|
| train | 2023-06-25 | 2023-10-15 |
| test | 2023-10-15 | 2024-04-15 |
| eval | 2024-04-15 | 2024-10-15 |

All stage clocks may be built from source-only data before outcomes. Market
outcomes remain physically sealed and must later be opened train → test → eval.

## Outcome-blind support and novelty gates

After non-overlap and split-boundary enforcement:

| gate | train | test | eval |
|---|---:|---:|---:|
| minimum clocks | 20 | 50 | 50 |
| minimum clocks per side | 6 | 12 | 12 |
| maximum single-month share | 40% | 30% | 30% |

Exact entry-clock Jaccard must be at most 0.10 against each frozen comparator:

- BTC-only CLBR-24;
- rejected BTC COIN-M + USD-M ICLA-60.

Reject before outcomes if any source, support, side, concentration, latency,
non-overlap, or novelty gate fails.

## Frozen causal controls for a later evaluator

1. **Direction flip:** identical clocks, opposite side.
2. **BTC-only direct shock:** BTC's own q95/imbalance trigger without ETH.
3. **Quiet-gate removal:** identical ETH trigger without the BTC `<=0.50`
   gate. If this dominates, the proposed relay mechanism is unsupported.
4. **Future-ETH placebo:** a deliberately noncausal ETH shock after entry; it
   is a falsification diagnostic only and can never be promoted.
5. **Additional +5m delay:** checks dependence on an unrealistically narrow
   execution window.
6. **Deterministic random clocks:** preserve split, month, side, and count.

Control definitions and clocks must be frozen before the primary train outcome
is opened. Future-ETH placebo outcomes must never enter model or threshold
selection.

## Later strict evaluation contract

If support passes, freeze a separate evaluator before outcomes with:

- one-times notional exposure;
- 6bp per side base cost and 10bp per side stress cost;
- exact funding with conservative entry/exit boundary treatment;
- full-calendar CAGR including idle time;
- strict MDD from the global pre-entry high-water mark, entry cost, favorable
  then adverse held-bar OHLC ordering, exact funding, virtual adverse-mark exit
  cost, and actual exit cost;
- circular stationary trade-block bootstrap with a fixed seed;
- train promotion before test opens, and test promotion before eval opens.

The archive ends in 2024 and prior repository research has seen adjacent market
history. Even a complete pass is a retrospective candidate requiring forward
live-shadow evidence, not production proof.
