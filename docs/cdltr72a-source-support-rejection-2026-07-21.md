# CDLTR-72A source-support rejection — 2026-07-21

## Verdict

**Reject CDLTR-72A before opening any BTC market outcome.**

The single hash-frozen v3 execution produced 74 non-overlapping primary relay
events, but failed the immutable support, control-support, and novelty stages.
The preregistration forbids repairing thresholds or timing under the same
candidate identity, so strict economic training is not authorized.

No BTC price, funding, future return, label, PnL, equity, absolute return, CAGR,
or MDD data was read or calculated.

## Primary support

| Statistic | Result | Gate | Pass |
|---|---:|---:|:---:|
| Train events | **42** | at least 60 | **no** |
| Selection events | 32 | at least 30 | yes |
| 2021 / 2022 train events | **17 / 25** | each at least 25 | **no** |
| 2021H1 / 2021H2 | **5 / 12** | each at least 12 | **no** |
| 2022H1 / 2022H2 | **10 / 15** | each at least 12 | **no** |
| 2023H1 / 2023H2 | 17 / 15 | each at least 12 | yes |
| Train LONG / SHORT | **33 / 9** | each at least 18 | **no** |
| Selection LONG / SHORT | **30 / 2** | each at least 8 | **no** |
| Largest month share | 9.52% train / 12.50% selection | at most 20% | yes |
| Largest weekday share | 33.33% train / **43.75% selection** | at most 35% | **no** |

The mechanism is not merely a little below one count threshold. It is sparse in
the early train halves, strongly LONG-skewed, and selection entries are too
concentrated by weekday. Those are independent preregistered rejection reasons.

## Control support

All seven emitted clocks passed the mechanical 72-hour hold, interval-order,
non-overlap, side-domain, split-containment, and pre-2024 checks. Only the
`network_only` control passed the complete support gate.

| Clock | Rows | Full support and containment |
|---|---:|:---:|
| `primary` | 74 | no |
| `macro_only` | 142 | no |
| `network_only` | 191 | yes |
| `reverse_order` | 48 | no |
| `one_network_report_delay` | 74 | no |
| `direction_flip` | 74 | no |
| `deterministic_random_side` | 74 | no |

The passing network-only clock does not rescue CDLTR-72A. Selecting it after
seeing this result would define a different candidate after incidence was
opened.

## Novelty result

Every exact-date Jaccard check passed the 0.30 limit, and every available signed
5-minute exposure correlation passed the absolute 0.40 limit. Four comparators
nevertheless exceeded the preregistered maximum of 50% of CDLTR dates within
plus or minus one UTC day:

| Comparator | CDLTR dates within ±1 day | Limit | Pass |
|---|---:|---:|:---:|
| `CVTR-1` | **44 / 74 (59.46%)** | at most 50% | **no** |
| `prior_microstructure:mfic_fast` | **57 / 74 (77.03%)** | at most 50% | **no** |
| `prior_microstructure:mfic_slow` | **58 / 74 (78.38%)** | at most 50% | **no** |
| `prior_microstructure:mfic_union` | **63 / 74 (85.14%)** | at most 50% | **no** |

The low exact overlap and low signed exposure correlations do not override the
near-date containment gate. CDLTR-72A is temporally too close to already-known
event families to qualify as a new orthogonal alpha clock.

## Source and outcome boundary

- bound source rows read: 4,468;
- derived source-vote rows: 4,315;
- bound comparator rows read: 9,985;
- primary incidence rows derived: 74;
- BTC market rows read: 0;
- funding rows read: 0;
- return/PnL fields or rows read: 0;
- post-2023 rows read: 0;
- network or subprocess calls: 0.

## Integrity anchors

- evaluator commit: `b42774eee3e602ab04590107e72999c4b170f973`;
- evaluator SHA-256:
  `30b2c85e406fdd7ef54fb97035390ccad58b97dbb23e4347271ce7037c7e3bdc`;
- v3 freeze commit: `29e040f`;
- report SHA-256:
  `ae56177d73836f9d232842ef72d05f385b066f741371defdbc15e909a5775e93`;
- report manifest hash:
  `cacc812a248263d688766d6a366a96a9aeb8531375638685399ea57a7c5adcfb`;
- clock SHA-256:
  `aa2bcafd0f62ebe585f93cbd357d29c37ae526a95a90b8a6c0bd7c068cd6e5a1`.

CDLTR-72A is retired. Any future use of a network-only clock, a wider relay
deadline, a different side balance, or a comparator exclusion must be proposed
and frozen as a new identity before its event incidence or outcomes are opened.
