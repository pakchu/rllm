# SMCC-144 support rejection — 2026-07-20

## Verdict

**REJECT_NO_REPAIR.**  SMCC-144 produced `1,021` chronological,
non-overlapping 12-hour events over 2020–2023, exceeding the preregistered
maximum of `900`.  The candidate is retired without changing its q99.5
threshold, collision floor, coherence gate, delay, hold, side, or source
quarantine.

No post-entry market row or funding row was read.  Novelty comparison and all
economic stages were skipped at the first failed gate.

## Blinded source evidence

- complete UTC 5m grid: `420,768` rows;
- source-observed rows: `420,732`;
- verified zero-volume empty rows: `26`;
- source-gap-day rows: `1,728` across six quarantined UTC days;
- post-gap quarantine rows: `1,866`;
- source-complete rows after quarantine: `418,896`;
- raw eligible signal bars: `2,200`;
- scheduled non-overlapping events: `1,021`.

The source build failed closed twice before incidence access.  Positive holes
between underlying trade-ID ranges were found to be compatible with continuous
aggregate IDs and the frozen aggregate-source audit.  One exact duplicated
underlying event in the hash-bound `2020-01-15` archive was instead frozen as a
full-day SMCC-specific quarantine.  This repair changed no signal or support
parameter and was independently reviewed before the source-access seal.

## Support gates

| Gate | Frozen requirement | Observed | Result |
|---|---:|---:|---|
| total events, minimum | 150 | 1,021 | pass |
| total events, maximum | 900 | 1,021 | **fail** |
| each calendar year | >=30 | 247–268 | pass |
| each 2023 half | >=15 | 116 / 152 | pass |
| long share | 25%–75% | 45.94% | pass |
| short share | 25%–75% | 54.06% | pass |
| largest month share | <=15% | 3.53% | pass |

Annual event counts were `257`, `247`, `249`, and `268` for 2020 through
2023.  The clock is balanced and broadly distributed, but it is not sparse
enough to satisfy the frozen rare-cascade thesis.

## Why there is no repair

Increasing the quantile, collision count, coherence threshold, or hold period
after seeing `1,021` events would select directly on observed source incidence.
The preregistration explicitly forbids that repair.  A future candidate may
test a materially different mechanism under a new name and a new pre-incidence
protocol; it may not reuse SMCC-144 outcomes, because none were opened.

## Frozen artifacts

- source SHA-256:
  `8fa03b0d7f58db9d0ba6c889e99ce87ba668f55a3c7f0ab5638a374c4584bfd1`;
- source manifest SHA-256:
  `e6ba3fbf74bc9bc1a7c1b35873e9ff430e5bc0a7b7edcc7e082f3f397362c805`;
- support clock SHA-256:
  `3b255b224ab510afc30edb265d62428db9fdf07d90610499df62efff9ffa410d`;
- support result SHA-256:
  `a52f1e5582c24bc14896dd985aeed02768434d73c6f7904f9a016b4f4ba19240`.

Machine-readable decision:
`results/same_millisecond_cascade_support_2026-07-20.json`.
