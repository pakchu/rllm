# Post-CEFS-D2 alpha mechanism audit

Date: 2026-07-25

## Outcome

Select one new outcome-blind source-composition candidate:

**CLOR-D1 — Collateral Liquidity Ordering-Relation target policy.**

CLOR-D1 will combine three independently published U.S. Treasury-collateral
surfaces:

1. original-issue nominal Treasury auction results;
2. New York Fed SOMA securities-lending operations and CUSIP details; and
3. OFR preliminary U.S. repo-market releases.

The intended object is the ordered evolution of primitive collateral-demand,
allocation, and financing relations as the three publications arrive on
different clocks. The eventual policy is a dense target-position sequence,
not another sparse fixed-side event rule. A separately frozen RLLM may choose
only `TARGET_LONG`, `TARGET_FLAT`, or `TARGET_SHORT` from compact categorical
state cards plus current position.

This selection is permission to freeze one exact source-only mechanism. It is
not source support, alpha, profitability, or live-deployment evidence.

## Why CEFS cannot be repaired

CEFS-D2 failed its frozen source-only primitive gate because the 2023 EVAL
`TERM_BACK_LEVEL` state was `LOWER` on all 250 rows. Its retirement contract
forbids removing or redefining the edge, changing a split, relaxing support
limits, or creating CEFS-D3 from the observed incidence:

```text
docs/cefs-d2-source-support-retirement-2026-07-25.md
results/cboe_edge_flip_sequence_policy_d2_source_rejection_2026-07-25.json
```

CLOR-D1 uses no Cboe source, relation, threshold, sequence, clock, control, or
result.

## Candidate screen

| candidate | useful property | disqualifying risk | decision |
|---|---|---|---|
| **CLOR-D1** | Three official 2019–2023 collateral surfaces; SOMA and OFR prior candidates reached only source-support failures; asynchronous relation ordering is unevaluated | Current historical APIs are not original point-in-time captures; exact joined live parity is not yet proven; Treasury-family outcomes have prior exposure | **Select for source-only freeze** |
| MARS-W8: CFTC + EIA + CPI weak-signal sequence | Directly matches the weak-signal-combination thesis and has causal official releases | All three singleton families already opened 2020–2022 outcomes. Selecting their combination after seeing those results would be explicit development-outcome mining | Reject for this turn |
| SOMA-only dense sequence | Daily, balanced, operationally rich source | SCAF already revealed component dominance and selection density. Redesigning the same source language now would condition on opened source incidence | Reject for this turn |
| SQRB-H1 stablecoin reservoir | Crypto-native relation language | Common denominator, collateral-breadth, and BTCDOM history is concentrated in late 2023 and component outcomes are already exposed | Reject |
| VPRS session policy | Volatility assimilation is logically distinct | BTCBVOL begins in June 2023 and the design reuses Cboe, so it cannot support the required pre-2023 chronology | Reject |
| DeFi governance text/payload | Strong deductive-LLM fit | No locally frozen historical source and no proven two-provider/live archive parity | Source-axis reserve |
| Forward disclosure revisions | True point-in-time latency evidence | No historical holdout can be manufactured retrospectively | Forward research only |

## Frozen source identities available to the next boundary

### Treasury auction results

```text
data/us_treasury_auction_demand_2016_2023/
us_treasury_nominal_original_auctions_2016_2023.csv.gz
SHA256 34a19163630c015a4f9d2671c95ca7cf7cc8a8ada024b3ef985405704fe0e4c1

data/us_treasury_auction_demand_2016_2023/build_manifest.json
SHA256 6da6a3848e89c3418efcbf0d836fda34b537a2da87a8777b74670f3912ad94f2
```

The panel contains 445 original nominal coupon auctions from 2016–2023, with
five later-updated rows quarantined. Historical research uses the conservative
22:00 UTC result clock.

### SOMA securities lending

```text
data/new_york_fed_securities_lending_2019_2023/
new_york_fed_securities_lending_operations_2019_2023.csv.gz
SHA256 99eb8c37c05417789dfad7452c7b2ddc5b6b640078b87451f1c945158af77906

data/new_york_fed_securities_lending_2019_2023/
new_york_fed_securities_lending_details_2019_2023.csv.gz
SHA256 27178d8738cb50c4e6c13f1e5940fcfdf4009e6979b006c42fb86fb399d0716d

data/new_york_fed_securities_lending_2019_2023/build_manifest.json
SHA256 58b9eb56728065d919978b8969e9bbb4bcb291f723a290d22045fe2ca3da2019
```

The source audit contains 1,259 operations and 182,616 unique
operation/CUSIP rows. Every operation is unavailable until the later of the
next UTC midnight and the recorded New York `lastUpdated`, followed by one
complete computation bar.

### OFR preliminary repo

```text
data/ofr_repo_preliminary_2019_2023/
ofr_repo_preliminary_observations_2019_2023.csv.gz
SHA256 6bb24b9a3d6ed6d266b0c885a9d9888bcccb3045df66d37d83031af66c6dc02a

data/ofr_repo_preliminary_2019_2023/
ofr_repo_preliminary_metadata_2019_2023.json.gz
SHA256 19a04e82eb5d8ddc6c3cb8dc64694438abd6b1987951470bb317659d9c53ef4f

data/ofr_repo_preliminary_2019_2023/build_manifest.json
SHA256 f937f567e1789ecb39a2b84d6288b2cbab931da4e9f1f4e51addea4b3423b705
```

The normalized panel contains 77,369 unique preliminary series/date rows.
Only disclosure-edit-zero, finite preliminary rows may be used, under the
existing `max(observation date + 8 elapsed days, 2020-09-10 UTC)` clock.
Final-vintage values and sparse unsupported series remain forbidden.

## Contamination and no-repair boundary

The source families are not globally pristine:

- Treasury auction demand has prior economic results;
- SOMA aggregate scarcity and allocation-fracture incidence is known, but no
  SOMA BTC/funding outcome was opened;
- several OFR source compositions were rejected before outcomes; and
- repository-wide BTC calendar outcomes are heavily inspected.

CLOR-D1 may claim only that its exact three-source primitive relation
sequence, action-independent update schedule, and target-position policy have
not been evaluated.

It may not reuse or repair:

- TADI bid-to-cover/indirect-share tail ranks, direction, or hold;
- TASCC issue/settlement collision packaging;
- SLCS aggregate scarcity ranks or vote;
- SCAF's four divergence components, three-of-four consensus, direction, or
  48-hour clock;
- any RVFC, RMSR, RCRE, DMSH, or other OFR threshold, product, handoff,
  direction, or event clock;
- a source-owned LONG/SHORT side, scalar pressure score, equal vote, fitted
  threshold, percentile-selected event, or post-result gate; or
- any CEFS source, edge, split, token, or control.

Known component failures may define exclusions only. They may not choose a
CLOR feature, vocabulary, sequence length, action, reward, hold, or checkpoint.

## Required next freeze

Before decoding one CLOR joint state, the next boundary must fix:

1. exact column allowlists and source hashes;
2. primitive relation formulas that do not recreate a retired component;
3. causal equal-time batching and latest-as-of joins;
4. a source-update schedule independent of future rows and market outcomes;
5. source freshness, invalidity, revision, and fail-flat behavior;
6. categorical vocabulary, sequence length, and current-position context;
7. `TARGET_LONG|TARGET_FLAT|TARGET_SHORT` only;
8. TRAIN 2020–2021, TEST 2022, EVAL 2023, with 2024+ sealed;
9. source-only diversity, stability, append, permutation, staleness, and
   retired-clock novelty gates; and
10. first-failure retirement with zero economic or model access.

Only a complete source-support pass may authorize a separately committed
cheap-baseline and RLLM evaluator.

## Evidence boundary for this audit

This unit inspected committed source-audit prose, headers, manifests, hashes,
and prior terminal aggregate reports. It decoded no Treasury, SOMA, or OFR
source value and built no CLOR feature, relation, sequence, action, or
incidence. It opened no new BTC bar, funding row, future return, reward, model
output, trade, PnL, absolute return, CAGR, or strict MDD.
