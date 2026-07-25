# CEFS-D1 source-support retirement

Date: 2026-07-25

## Decision

**RETIRE CEFS-D1 unchanged before source decoding, market-outcome access,
reward construction, or model training.**

The official invocation used:

```text
runner/tests commit
  d7213f647128fc6160672bc61f080b3dcf7d1f42
runner SHA256
  2069084d65146540488672115ee09f292cd31e6611bf92a569d534ab8a74c688
tests SHA256
  01de1671cdf3c7fb4acf1b6e9ec8cb06d94f1c971381ab4252a4140c01132937
execution-seal SHA256
  c1d6aa251108afa520d1279c3fd0f2795a1c92ee229a593c1effa28f2445f331
execution-seal manifest_hash
  05c36438d6f82499c9fce6fff55e993bbdb05d09fd1c7ce92aa9ff7ca4a00f96
```

Machine-readable terminal evidence:

```text
results/cboe_edge_flip_sequence_policy_source_rejection_2026-07-25.json
SHA256 4c2839e5ac59738367d5116ff05ed50c900d16f06de8c4d4cc724fd25978c169
result_hash 9963981f6d56fcff65f1367fc7c3c1fc006b60b821894b3e0ff59c6b9aa35d7b
```

## Exact failure

The runner stopped at Gate 1:

```text
1  authority_forbidden_access  fail
```

Exact exception:

```text
FileNotFoundError: [Errno 2] No such file or directory: 'git'
```

This was an operator-shell failure, not a source-support result. The zsh
preflight loop used the reserved variable name `path`; zsh maps `path` to
`PATH`, so the loop replaced the executable search path before Python
started. The runner therefore could not execute `git` during frozen-authority
validation.

## Evidence boundary

The failure occurred before the source loader:

- no Cboe source value row was decoded;
- no source relation, sequence, prompt, or control was built;
- no BTC market or funding row was opened;
- no future return, reward, model row, action, or trade was built;
- no PnL, CAGR, or MDD was computed; and
- no pass source/control artifact was written.

Every forbidden counter is zero. CEFS-D1 therefore has **no source-support,
alpha, profitability, or deployability result**.

## No retry under CEFS-D1

The frozen contract makes any terminal report idempotent and requires
retirement at the first failed gate. Do not delete the rejection, rerun the
same identity, or reinterpret Gate 1 as a pass.

A successor may reuse the still-unobserved mechanism only under a new
candidate ID, boundary, preregistration, implementation contract, runner,
tests, and execution seal. Its official launcher must preserve `PATH` and
must not use zsh's reserved `path` variable.
