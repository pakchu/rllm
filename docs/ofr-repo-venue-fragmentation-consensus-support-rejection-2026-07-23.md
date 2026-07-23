# RVFC-72 source-support rejection — 2026-07-23

## Decision

**Reject `RVFC-72-NEW-SOURCE` before outcomes. Do not repair this identity and
do not open its BTC or funding returns.**

The committed source-only builder reproduced byte-identical artifacts:

- report: `results/ofr_repo_venue_fragmentation_consensus_support_2026-07-23.json`;
- report SHA-256:
  `c5918606c958fc8f966e8bd1884e75a91a6cec44074e2edbe86675fa7f978402`;
- report manifest:
  `88275871e76ac1af6c5124466a4cb63426f2bbca2001bbbb0c5aa91426593f52`;
- source/control clock:
  `results/ofr_repo_venue_fragmentation_consensus_clocks_2026-07-23.csv.gz`;
- clock SHA-256:
  `b2d251b147feecc98def94610834c7b11f078f35e580f6e7a950fc55bfe0724e`.

## Source-support failure

The source audit read 77,369 normalized OFR observations and 17,462 required
rows over 1,249 dates. It formed complete features on 1,245 dates and strict
252-day prior-history ranks on 993 dates. Equal-availability batches suppressed
417 rows from one another's history. No post-2023 source row was opened.

The frozen primary clock was too sparse, strongly one-sided, and unstable:

| Split | Events | Long | Short | Active months | Maximum entry gap |
|---|---:|---:|---:|---:|---:|
| train 2021–2022 | 30 | 3 | 27 | 15 | **187 days** |
| selection 2023 | 10 | 10 | 0 | 4 | **147 days** |

It failed the preregistered total-count, per-year, per-half, both-side,
every-quarter, selection concentration, and maximum-gap gates. Selection
incidence was concentrated in four months, with one month containing 50% of
all entries. Only primary-clock validity and train month concentration passed.

## Structural diagnosis

The primary clock is exactly identical to the `mean_without_consensus` control.
The nominal three-of-four vote therefore adds no event specificity beyond the
frozen mean-magnitude threshold. The mechanism also changes polarity across
the boundary: train is 90% short while selection is 100% long.

Venue dominance confirms that this is not a stable four-way consensus:

- transaction-volume concentration is TRIV1-dominant for 100% of train events
  and 90% of selection events;
- collateral-spread disagreement is GCF-dominant for 93.3% of train events;
- rate dispersion is GCF-dominant for 100% of selection events.

Independent component controls are substantially denser and often more
balanced. That does not validate an alpha; it shows why simultaneous
same-direction extremes discard most information and collapse into a sparse
regime marker. Threshold relaxation or component deletion would be an
outcome-adjacent repair and is prohibited under the RVFC identity.

## Closed boundaries and retained lesson

The failed source gate short-circuited novelty comparison. The run read zero
comparator rows, BTC bars, funding rows, and future-return rows and opened no
PnL, CAGR, or MDD. RVFC therefore ends as a clean source-only negative result.

A successor must be preregistered under a new identity and encode a different
causal object. The justified direction is an ordered cross-component handoff
or lead/lag transition—not another simultaneous high/low fragmentation
consensus—and it must independently pass source-support and novelty gates
before any market outcome is accessed.
