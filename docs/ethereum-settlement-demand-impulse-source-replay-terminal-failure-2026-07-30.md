# ESDI-288 source replay terminal failure — 2026-07-30

## Decision

`ESDI-288` is **retired unchanged before source support, novelty, or economic
outcomes**. The committed one-shot dual-transport replay created its durable
claim and then received HTTP `429 Too Many Requests` from the frozen
`https://eth.merkle.io` transport during the first boundary-header validation.
The source contract forbids retry, backoff, transport substitution, fallback,
resume, or stage recovery.

This is an execution-source failure, not evidence for or against the economic
mechanism. It nevertheless terminates this exact policy because rerunning after
observing the failure would violate the preregistered one-shot rule.

## Frozen sequence

The policy, preregistration, evaluator, tests, and independent reviews were
committed and pushed before the first RPC request:

- mechanism commit: `dc9885c5`;
- hardened preregistration commit: `aadf2a43`;
- write-once preregistration producer commit: `e21c9e81`;
- canonical preregistration artifact commit: `1533a719`;
- complete source/novelty/economics protocol commit: `f3de120a`;
- source replay claim commit: `44bd37a3`;
- preregistration artifact SHA-256:
  `2a481fc60044d3d468340457d50f92a91f2a52184a464e1a91badfb418bbcaba`;
- preregistration `manifest_hash`:
  `d5279f95cc7b92757aa77ecbbc5835d8b1cc4ce34f5a81d6f279abdcf2fcfe8a`;
- pre-replay protocol `seal_hash`:
  `f98352414aee6b757c3f70056ba1a36fe53811d535dcbda6e71ec7bf856a2089`.

Before replay, the exact four-file test battery reported `135 passed`; all
four production modules compiled, all three independent reviewers approved,
the 37-path protocol seal validated from committed-clean Git blobs, and the
branch HEAD exactly matched its upstream.

## Terminal event

The exact frozen command was executed once:

```bash
env PYTHONPATH=. uv run --frozen python \
  -m training.build_ethereum_settlement_demand_impulse_source
```

The process exited nonzero after the right-hand transport returned HTTP `429`
for `eth_getBlockByNumber` while validating the first frozen boundary pair.
The builder converted the transport exception to `TerminalSourceFailure`, as
required. It did not retry.

The replay claim records:

- status: `claimed_before_first_rpc`;
- `one_shot: true`;
- `retry_backoff_fallback_or_resume: false`;
- claim SHA-256:
  `fd93620613c8dc42e4614c8c89263ac101d6a3e58f6d5b223cf0797982228d65`;
- claim hash:
  `6a6a9783135d49efcbfb1c0e9f1d0f85f21942b354180e6ade9e423cfd7f167e`.

The exact stderr/time record is preserved at
`results/ethereum_settlement_demand_impulse_source_replay_failure_2026-07-30.log`
with SHA-256
`bc39b8e8f0b4034a24a334974369f092df761f89e894f75a90ad8fecaebbbb38`.

## Publication and evidence boundary

After failure:

- the immutable replay claim exists;
- the canonical source generation directory does not exist;
- no durable stage directory exists;
- raw fee-history artifact published: `false`;
- normalized epoch artifact published: `false`;
- source manifest published: `false`;
- source-support rows opened: `0`;
- comparator rows opened: `0`;
- Gross9 reconstruction rows opened: `0`;
- BTC market rows opened: `0`;
- funding rows opened: `0`;
- return or PnL rows opened: `0`;
- CAGR or strict-MDD metrics computed: `false`.

No source-support, novelty, Gross9, or economic command is authorized because
there is no complete atomically published source generation. Repairing an
endpoint, waiting for a rate limit, or rerunning the same policy is forbidden.
Any future Ethereum-derived mechanism must be a separately named and
preregistered policy, not a continuation of `ESDI-288`.
