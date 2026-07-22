# BIRB-120 source-support builder freeze — 2026-07-23

## Frozen boundary

The source-only BIRB support builder is committed before its first read of the
exact reactivation/breadth incidence.

- candidate: `BIRB-120-SOURCE-FAMILY-SEEN`;
- preregistration SHA-256:
  `acdd79007901427657a360acc15613fdacaf34c2238033eb9c77bda694527023`;
- preregistration manifest:
  `2e0c7dcf55963a5e5f9ea87b3e3b6f551b34b4016cb1a292603e31b41e0be5a4`;
- preregistration policy:
  `424a904a8512635275ad276666ee66155ffa251721c24a08c4a9a057fe8b15a4`;
- builder:
  `training/build_sec_bitcoin_issuer_reactivation_breadth_support.py`;
- builder SHA-256:
  `02e4199484163403afd33c706a173aeae3902bacf4d799d23bb1d9663bcfa12c`;
- builder tests SHA-256:
  `7be9a9bff7d791301fc9096e2b3316ee536cf6dd22b1058032542dc4a9fa09a1`.

Seventeen preregistration and synthetic support tests passed. They cover
accession deduplication, same-ready issuer batching, exactly-365-day
reactivation, first-ever/repeat controls, open-left seven-day breadth,
whole-batch first passage, fixed one-bar execution delay, 120-hour nonoverlap,
split crossing, component-control orientation, comparator overlap arithmetic,
and output-field exclusion.

## Authorized read

The first real run may read only:

1. the frozen 2018–2023 SEC metadata source;
2. the frozen preregistration and source-audit metadata;
3. timestamps from the three hash-bound comparator clock artifacts.

It may not fetch SEC bodies, call a model, access the network, read BTC or
funding rows, derive future returns, or calculate PnL/CAGR/MDD. Its only valid
terminal decisions are `PASS_SOURCE` and `REJECT_SOURCE`. A rejection retires
the candidate without threshold, gap, window, side, or hold repair.

At this freeze point exact BIRB incidence remains unopened and calendar 2024+
source values remain sealed.
