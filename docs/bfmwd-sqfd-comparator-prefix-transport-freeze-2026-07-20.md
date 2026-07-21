# BFMWD SQFD comparator-prefix transport freeze — 2026-07-20

The BFMWD novelty registry was frozen against the immutable SQFD clock artifact
`data/stablecoin_quote_flow_diffusion_clocks_2023_2026.csv.gz`, SHA-256
`a81e144eea1e80ae5439fc66db1fad5bbd00cd9ac177e25142b5cfb5a07bcc5b`.
That artifact also contains 2024–2026 comparator incidence, which the BFMWD
source-support evaluator is forbidden to parse.

Before any SQFD row or BFMWD source feature is opened, this transport freezes a
single deterministic projection:

- stream the hash-bound original artifact once;
- parse only the `control` and `entry_time` clock fields;
- retain `control == primary` and
  `2023-01-01T00:00:00Z <= entry_time < 2024-01-01T00:00:00Z`;
- deduplicate exact timestamps and sort ascending;
- write only `control,entry_time` to deterministic gzip; and
- report all discarded post-2023 row counts in a hash-bound manifest.

This does not replace the frozen SQFD comparator or choose another control. It
materializes the exact pre-2024 primary projection already specified by the
comparator freeze. The BFMWD evaluator may read only the resulting prefix and
must validate both the original binding and prefix manifest before use.

The transport may stream post-2023 comparator-clock bytes solely to discard
them. No BTC market price, future return, position funding, label, PnL, BFMWD
candidate incidence, or performance value may be opened. The later BFMWD
source-support report therefore counts zero post-2023 comparator rows read by
the evaluator itself and separately binds this audited transport manifest.
