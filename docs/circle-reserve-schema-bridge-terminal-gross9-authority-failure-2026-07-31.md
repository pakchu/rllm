# CRSB-336 terminal Gross9-authority failure — 2026-07-31

## Decision

`CRSB-336` is **retired unchanged before source access, source support,
novelty, comparator access, Gross9 clock access, or economic outcomes**.

The terminal action is:

```text
REJECT_GROSS9_AUTHORITY_UNAVAILABLE_PREPRODUCTION
```

The sealed CRSB preregistration imports the ESDI Gross9 authority unchanged.
That authority requires Gross9 clock reconstruction:

```text
after ESDI source-support pass and before ESDI economics
```

All five required positive-weight sleeves also declare
`clock_artifact_preexists=false`. ESDI was permanently retired during its
one-shot source replay before source support, and its terminal record expressly
forbids retry, recovery, substitution, or downstream Gross9 execution.
Consequently neither of the exact prerequisites exists:

```text
results/ethereum_settlement_demand_impulse_source_support_2026-07-30.json
results/ethereum_settlement_demand_impulse_gross9_clocks_2026-07-30.json
```

The paths are absent from the worktree, all 22 refs, and the readable Git object
inventory. A full `git rev-list --all --objects` lookup returned zero matching
objects.

## Frozen authorities

- CRSB preregistration:
  `results/circle_reserve_schema_bridge_preregistration_2026-07-30.json`
- CRSB preregistration SHA-256:
  `1f0eb234d4f8f12ab3f28568636fa4e9550857a17ebd69533c77521e7106aa23`
- CRSB preregistration manifest hash:
  `cb7f255e00697796ce48bd4f16f686855fdc30bc83f6f813b845731acbab8d2a`
- ESDI terminal decision:
  `docs/ethereum-settlement-demand-impulse-source-replay-terminal-failure-2026-07-30.md`
- ESDI terminal-decision SHA-256:
  `929c8df5c7f496143fd9449665532030840b2dd5c789e709d1dcb4004465fa7b`
- ESDI one-shot failure log:
  `results/ethereum_settlement_demand_impulse_source_replay_failure_2026-07-30.log`
- ESDI failure-log SHA-256:
  `bc39b8e8f0b4034a24a334974369f092df761f89e894f75a90ad8fecaebbbb38`

The ESDI terminal authority states that no source-support, novelty, Gross9, or
economic command is authorized because no complete source generation exists.
Using CRSB source support in place of ESDI source support would change the
canonical authority and stage. Reconstructing the five sleeves now would both
violate ESDI's terminal no-repair rule and open Gross9 runtime market/feature
and outcome-dependent path rows, contrary to CRSB's clock-only novelty
boundary.

An independent read-only critic returned
`REJECT_TERMINAL_PREPRODUCTION`: no contract-compliant path exists without
altering the frozen CRSB mechanism/preregistration or opening prohibited rows.

## Evidence boundary

At retirement:

- CRSB source-access claim exists: `false`;
- CRSB durable request-attempt ledger exists: `false`;
- CRSB source manifest exists: `false`;
- SEC production requests: `0`;
- CRSB source incidence opened: `false`;
- liquidity, WAM, or WAL values opened: `false`;
- candidate clock rows opened: `0`;
- comparator rows opened: `0`;
- Gross9 clock rows opened: `0`;
- BTC market rows opened: `0`;
- funding rows opened: `0`;
- return or PnL values opened: `0`;
- CAGR or strict-MDD metrics computed: `false`;
- economic outcomes opened: `false`.

The eight future protocol files remained untracked and were never committed,
pushed, sealed, or authorized for production. Their final local SHA-256 values
were:

```text
49eee4c0dfb33ea29946fc542dff19228fef8a5e78b905a9c4138ecdc33d2405  training/build_circle_reserve_schema_bridge_source.py
501a731028e9e8916f849307cd677b69ce70057a0498fc1e75349b8549e0372c  tests/test_build_circle_reserve_schema_bridge_source.py
e80fd16a2df4a7dd216a9c03d201e809b453dcf3c865208278abbecd615c6970  training/evaluate_circle_reserve_schema_bridge_source_support.py
ca0bcfaaeb01f7aba122bf234106796e7a36f0e6fdeb51fe139a96004835be15  tests/test_evaluate_circle_reserve_schema_bridge_source_support.py
63c3b14f13dd39b1edabd8df5c01a9c24a55320790cdee48e756afc5b6cff504  training/evaluate_circle_reserve_schema_bridge_novelty.py
7814aeac232542d8b1b08412f6233fd41ada6ca1fc39d1b7e07aa77f1b06c940  tests/test_evaluate_circle_reserve_schema_bridge_novelty.py
eff7efd9fb8e6ef9860627bd410af1d631d98582bfc5476e0d448f3398122bc9  training/evaluate_circle_reserve_schema_bridge_economics.py
52777be15cac02d4c92d0e996c47f4287f16fa17de91f660be3e4c42352481b8  tests/test_evaluate_circle_reserve_schema_bridge_economics.py
```

The uncommitted files were preserved outside the repository only as an audit
archive; they are not production authority.

## No repair

CRSB-336 may not be repaired by:

- generating the missing ESDI artifacts after ESDI's terminal failure;
- treating CRSB source support as ESDI source support;
- reconstructing Gross9 sleeves during CRSB novelty;
- weakening the clock-only novelty boundary;
- dropping Gross9 novelty comparisons; or
- changing the imported Gross9 authority after preregistration.

Any successor must be a separately named and preregistered candidate. Before
that successor is frozen, its novelty contract must bind a **preexisting,
committed, independently authenticated Gross9 clock bundle** whose provenance
does not depend on a terminally retired policy.
