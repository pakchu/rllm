# Cross-asset transfer source recovery — 2026-07-19

The first frozen evaluator invocation stopped before writing a result because the
Yahoo `069500.KS` payload contains 549 consecutive null quote rows from
2007-02-08 through 2009-04-16. After invalid rows were dropped, the apparent
valid-price gap was 800 calendar days.

The process had evaluated QQQ in memory before reaching the KODEX source error,
but it emitted no summary and wrote no result or documentation artifact. No
policy, sign, threshold quantile, hold, cost, split, or admission gate was
changed after that invocation.

The recovery is availability-only and fail-closed:

1. A null run of at least 20 provider rows marks an unusable listing prefix.
2. The evaluator discards through that run and starts at the first subsequent
   valid provider row.
3. Later invalid provider rows remain explicit `source_valid=false` gaps.
4. A REX signal requires 111 consecutive valid source rows.
5. A barrier signal requires its complete frozen lookback to be valid.
6. A trade is skipped if signal, entry, any held row, or exit crosses a source
   gap.
7. Source hashes, discarded-prefix count, quarantined-row count, and an invalid
   date hash are written to the final result.

This repair cannot improve a policy by inspecting returns; it only removes or
invalidates unsupported observations.
