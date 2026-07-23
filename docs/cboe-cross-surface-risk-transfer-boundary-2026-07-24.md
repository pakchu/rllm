# CXRT-288 candidate boundary — CBOE cross-surface risk-transfer relay

## Selection

Select one new weak-state composition candidate:
**CXRT-288 — CBOE Cross-Surface Risk-Transfer Relay**.

CXRT will combine three independently observed CBOE source surfaces:

1. VIX9D/VIX/VIX3M term pressure;
2. SKEW/VVIX/VIX tail pressure; and
3. U.S. option-volume hedge-flow pressure.

Each surface contributes one causally normalized weak state on a common CBOE
source date. A deterministic cross-surface relation fixes LONG or SHORT. A
later RLLM may only execute that fixed side or abstain.

This file selects the observable axis only. It does not freeze the exact
normalization, relation algebra, state tokens, entry/exit clock, support gates,
controls, model, reward, or economics. Those must be committed separately
before the exact CXRT composite clock or any new post-entry BTC outcome is
computed.

## Why this follows the CVICR failure

CVICR was rejected with only nine train events and zero 2023 selection events.
Its exact same-day conjunction collapsed several individually common
observations into a nearly nonexistent event.

CXRT changes both source and representation:

- it leaves crypto exchange microstructure and intrinsic-volume clocks;
- it uses dense daily exogenous source states;
- weak evidence is composed relationally instead of requiring an exact
  four-condition transition; and
- direction is deterministic before the RLLM, preserving the repository's
  `TRADE_FIXED_SIDE` / `ABSTAIN` architecture.

No CVICR threshold, condition, direction, or hold is repaired.

## Frozen source identities

### Volatility term structure

```text
data/cboe_volatility_term_structure_2018_2023/
cboe_vix_term_structure_2018-01-01_2023-12-31.csv.gz
SHA256 6f1b2f7f3a5b1e4d5001d673e6ff54374791879c278248ce27b3d610e4f75dc7
```

Manifest:

```text
data/cboe_volatility_term_structure_2018_2023/build_manifest.json
SHA256 42b2a35ad131bd63574d2adcf684e28766dc3060fa645fc749df10dd3fb27f27
```

### Tail-risk surface

```text
data/cboe_tail_risk_2018_2023/
cboe_tail_risk_2018-01-01_2023-12-31.csv.gz
SHA256 cdde3f8d4bb1e23d00b192f5f9ef759aefba9087be5fd60653e9c02479dfa41a
```

Manifest:

```text
data/cboe_tail_risk_2018_2023/build_manifest.json
SHA256 9ef80ef3034c93d97c5b2a8160b2502527287d570d15f9d7166d631d9866c7bd
```

### Option-flow surface

```text
data/cboe_option_flow_2020_2023/
cboe_option_flow_2020-01-01_2023-12-31.csv.gz
SHA256 35ef106ef01e3abadbcb4a6227187dd1d7cf2722191bd146bac06d08d1684a78
```

Manifest:

```text
data/cboe_option_flow_2020_2023/build_manifest.json
SHA256 0a513b146ad5857d9ab7311e978152c308de64db8ef29c4d463eb07ea503089e
```

The already-audited panels contain:

- 1,509 term-structure dates over 2018–2023;
- 1,507 tail-risk dates over 2018–2023; and
- 1,006 option-flow dates over 2020–2023.

The later mechanism must use the exact date intersection, never forward-fill a
missing CBOE date, and delay use of a completed source close until a later CBOE
source date. These are frozen current historical vintages, not a proof of
point-in-time revision history. Live promotion requires forward-vintage hash
monitoring.

## Why this is not a repair of the three rejected CBOE policies

Repository outcomes already show:

- **CVTR-1** term-tail events had only +13.18 bp mean gross movement and failed
  Stage 1;
- **CTHD-1** hidden tail-pressure shorts had negative mean gross movement and
  failed Stage 1; and
- **CIHM-1** option-flow hedge-migration tails failed its sealed Stage 1.

Those exact policies remain retired. CXRT may not:

- reuse any selected event-tail threshold from CVTR, CTHD, or CIHM;
- relabel a prior candidate clock;
- choose a source weight, sign, threshold, hold, or token after looking at the
  exact CXRT composite outcome;
- claim that a CBOE aggregate identifies institutions, opening trades, or
  causal BTC demand; or
- treat a single surface as sufficient evidence.

The new falsifiable object is the **cross-surface relation among three dense
weak states**, not another extreme observation inside one surface.

## Rejected alternatives for this turn

### DCLB-864 — reserve

Official dollar-liquidity and collateral panels are highly orthogonal, but
their mixed daily/weekly release calendars, vintage semantics, and multi-day
hold require a larger availability proof. Retain as the next backup if CXRT
fails unchanged.

### FCSP-288 — reserve

Finalized stablecoin and WBTC settlement pressure is crypto-native and dense
after rolling aggregation, but it is adjacent to several already-rejected
stablecoin/WBTC mechanisms and has greater finality/live-index complexity.

### Dense cross-venue RLLM — not selected

Turning the just-opened CVICR source into a dense same-source model immediately
would be too close to a post-failure gate removal. The next test uses a
different exogenous source family.

## Evidence boundary

During this selection unit, only:

- committed source audits and manifests;
- immutable file hashes and exact headers;
- prior documented CBOE candidate outcomes; and
- repository architecture constraints

were inspected.

The following were not computed for CXRT:

- exact three-panel common dates;
- normalized surface states;
- cross-surface votes, majority sides, transitions, or candidate timestamps;
- annual/monthly side or event counts;
- comparator overlap;
- any CXRT post-entry return, funding, PnL, CAGR, strict MDD, or hit rate;
- any 2024-or-later source or outcome.

This branch has broad prior CBOE outcome exposure. CXRT can establish a frozen
candidate-level test, not a pristine global market holdout.

## Mandatory sequence

1. commit this boundary;
2. freeze one exact source-only state algebra, side, clock, support gates,
   controls, novelty cohort, failure action, and RLLM token boundary;
3. commit a write-once preregistration without computing CXRT incidence;
4. commit and test an outcome-blind source-support/novelty evaluator;
5. retire CXRT unchanged on any source, support, selectivity, or novelty
   failure;
6. only a complete pass may freeze an economic/RLLM evaluator;
7. open train, validation, selection, test, and eval strictly in the order
   preregistered by that evaluator.

## RLLM boundary

The deterministic composer owns opportunity timing and side. The RLLM may
receive only compact causal relation tokens, prior position state, and source
validity. Its action set is:

```text
TRADE_FIXED_SIDE
ABSTAIN
```

It may not create a timestamp, reverse the side, change the hold, use raw
identifiers/calendar memorization, or inspect evaluated-split outcomes during
training or threshold selection.
