# BCRT-72 source-support retirement

Date: 2026-07-24

## Decision

`BCRT-72` is retired unchanged before market, funding, comparator, return,
PnL, label, or model-training data was opened.

The exact frozen decision is:

```text
retire_BCRT_72_unchanged_before_market_outcomes
```

The first and only failing development source check was:

```text
max_entry_gap_days_2020_2022
```

The preregistered maximum was three calendar days. The observed development
maximum was five calendar days.

## What passed

The source itself was usable:

- 213,095 source and 213,095 reference rows validated;
- 2,918 causal twelve-hour buckets formed;
- all 2,918 prefix replays passed;
- no later backdated member entered a closed bucket;
- 2,792 rank-complete states and 2,791 token-ready states formed;
- 2,787 intervals were globally reserved;
- four overlaps and 32 split-crossing intervals were suppressed; and
- 2,755 source-only opportunities were emitted.

Development incidence otherwise passed:

| Window | Opportunities |
|---|---:|
| 2020-2022 | 2,035 |
| 2020-2021 train | 1,314 |
| 2020 | 595 |
| 2021 | 719 |
| 2022 | 721 |
| 2023 report-only | 720 |

Every frozen train and 2022 token-support check passed. This includes pair
value shares, leader diversity and concentration, breadth/occupancy shares,
transition shares, exact-signature concentration, token validity, and
train-vocabulary coverage.

The relational grammar was therefore expressive and well populated. It was
not rejected for token collapse.

## Why the clock failed

The causal clock combines:

- first median-time anchor at the half-day boundary;
- exact 288-block prefix closure;
- a further 48-hour historical embargo;
- one complete five-minute latency bar; and
- strict split containment with `exit < split_end`.

At each year boundary, states whose source, confirmation, or execution path
crossed the split were suppressed but still consumed their global reservation.
The resulting development gaps were:

| Previous entry | Next entry | Gap days |
|---|---|---:|
| 2020-12-31 11:10 UTC | 2021-01-05 12:05 UTC | 5.0382 |
| 2021-12-31 14:30 UTC | 2022-01-05 09:10 UTC | 4.7778 |
| 2022-12-31 12:45 UTC | 2023-01-05 10:50 UTC | 4.9201 |

The gap is a deterministic consequence of the conservative clock and split
rules, not missing Bitcoin blocks.

## Why the rule is not repaired

Changing the gap threshold, ignoring year boundaries, releasing reservations,
weakening split containment, reducing the embargo, or changing the
confirmation horizon after observing these counts would be post-selection
repair. The preregistration explicitly required retirement on any source
support failure.

No cheap learner, economic evaluator, BTC price, funding, comparator, LLM
label, or model checkpoint was opened. BCRT has no performance statistic and
must not be described as profitable or unprofitable.

## Frozen artifacts

| Artifact | SHA-256 |
|---|---|
| `results/block_clearing_relational_topology_support_2026-07-24.json` | `9ccccf7a3176fcf86baddacb65c11bbde78ea73ed7ab18d3594b0e6327567055` |
| `data/block_clearing_relational_topology_clocks_2020_2023.csv.gz` | `c0420c7175410a822455a0d68bf877cba94a2ec17b31f6d9a588244cb893c909` |

Support manifest hash:

```text
e2b2d7301d204043f2df33f4453da82112fb5db7bfb9aed66a74bee6ec76932b
```

## Design lesson for the next candidate

The next candidate must preregister a boundary-aware incidence rule before
source decoding if its causal observation delay is multiple days. It must not
pretend that deterministic train/test boundary blackouts are ordinary missing
data.

This lesson may influence a new mechanism prospectively. It cannot revive or
repair `BCRT-72`.
