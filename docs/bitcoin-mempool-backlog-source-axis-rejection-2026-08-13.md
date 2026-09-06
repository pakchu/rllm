# Bitcoin mempool backlog source axis — source-blind rejection

## Decision

Reject historical Bitcoin mempool backlog, transaction-count, vsize, total-fee,
and fee-histogram state as the next alpha source before preregistration or any
new response body, source incidence, market outcome, or PnL is opened.

## Reason

Bitcoin Core exposes the mempool of the particular running node through current
RPC state such as `getrawmempool`, `getmempoolinfo`, and `getmempoolentry`.
Those values depend on that node's receipt history, policy, connectivity,
replacement handling, restarts, and eviction. They are not consensus objects
and cannot be reconstructed from the confirmed blockchain after the fact.

The official Mempool project exposes current mempool and fee-estimation
surfaces, but the already audited repository evidence identifies projections
and `feeRange` as node-local current state. Its reproducible multi-year mining
endpoints describe **confirmed mined blocks**, not snapshots of the transaction
backlog that existed before those blocks. No complete, version-pinned,
causally timestamped 2023-01-01 through 2026-08-01 backlog archive is bound in
this workspace.

An owned node started now could support a forward shadow study, but it cannot
backfill the frozen train, test, evaluation, and final windows. Inferring the
missing queue from later mined transactions would introduce survivorship,
unknown propagation delays, dropped/replaced transactions, and future block
information.

## Evidence boundary

This decision relies on the official Bitcoin Core current-mempool RPC contract,
the official Mempool project/API contract, and the existing source decisions in
`docs/rqci-source-mechanism-decision-2026-07-20.md` and
`docs/utxo-fee-clearing-polarity-mechanism-decision-2026-07-20.md`. No new
mempool response body or candidate value was opened.

No snapshot provider, inferred backlog, confirmed-transaction proxy, source
delay, threshold, side, or clock substitution is authorized. The axis is
terminal for the current historical alpha search.
