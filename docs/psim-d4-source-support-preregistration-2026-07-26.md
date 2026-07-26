# PSIM-D4 source-support preregistration

Date: 2026-07-26 KST

## Decision

PSIM-D4 is preregistered as a newly named, source-only successor to the
terminally rejected PSIM-D3 candidate.

The only semantic change is the historical EIP front-matter parser rule
selected by the committed synthetic probe: after the unchanged D1 byte
normalization and unchanged header resource bounds, normalized-empty lines are
ignored as non-semantic separators. Every nonempty EIP line is still parsed by
the D1 state machine. The BIP parser is unchanged.

PSIM-D3 is not repaired, resumed, or rerun. Its source roots are forensic
residue and may not be read, reused, copied, alternated, hard-linked, repaired,
continued, or cached by D4.

This preregistration is outcome-blind; official source execution is not authorized
yet. No market row, funding row, future return, model, reward, trade, PnL, CAGR,
strict MDD, leverage, portfolio, or live-trading operation is authorized.
The next unit may only implement, test, independently review, and seal a
synthetic-only D4 source-support evaluator.

## Bound selection authority

The D4 selection decision is:

```text
docs/post-psim-d3-alpha-mechanism-audit-2026-07-26.md
commit 131009359c60bc5b28b76d22a63abf698011fbcb
SHA-256 2615736ba063c2b8e35811d3d01ab3517b345d74a2f7b70d248899aa393d7b99
```

The D3 terminal authority remains:

```text
results/protocol_specification_intent_maturity_d3_source_rejection_2026-07-25.json
commit f9089a300d4ba97722ecc1b59f8f8260eff8851b
SHA-256 a9be5b5990ad79b7da7d72a22968f4f62a2700877b198606565cc70206fe9802
result_hash b00b54b70720d42d213b315e82e7ff3ad0df03909b92aaa514299e750fa1ba2c
first failure Gate 4 historical_blob_preamble_dependency_integrity
```

Its terminal action is unchanged:

```text
REJECT_PSIM_D3_UNCHANGED_BEFORE_MARKET_MODEL_OR_OUTCOMES
```

D3 opened 5,206 explicitly hydrated proposal blobs and 17 unique proposal
texts, then rejected the eighteenth text before any card, model, market,
reward, trade, or outcome stage.

## Synthetic parser authority

The selected parser behavior is bound to:

```text
results/protocol_specification_intent_maturity_d4_parser_probe_2026-07-26.json
commit 131009359c60bc5b28b76d22a63abf698011fbcb
SHA-256 fbb97d65ef93b307c47055ed1883d6416e510a70b38083ae17ced2c78e4745ee
result_hash 4a3cca52755716dbf6e9b4cd801e46b72bab841cea5609f6bb42519487e5f5e6
protocol psim_d4_historical_eip_preamble_probe_v1
parser PSIM_PREAMBLE_STATE_MACHINE_V2_EIP_EMPTY_SEPARATORS
```

The probe used synthetic bytes only. It read the canonical D3 rejection JSON
for its terminal binding but did not read the D3 forensic root, the historical
proposal blob, any official proposal repository, market data, a model, or
outcomes.

The probe proved:

- five D1-accepted EIP outputs remained unchanged;
- thirteen nonblank D1 EIP rejections remained rejected;
- six normalized-empty/control pairs produced identical field mappings;
- normalized-empty lines did not bypass header line or byte limits;
- a separator did not rescue an empty field value;
- the D3 failure-shaped synthetic fixture was accepted;
- two accepted BIP outputs remained unchanged;
- six BIP rejections remained rejected; and
- the D4 BIP function was the identical D1 function object.

## Machine-readable contract

The canonical preregistration is:

```text
results/protocol_specification_intent_maturity_d4_preregistration_2026-07-26.json
SHA-256 52d77eafef0e9e79f1d7a47b9c262aad148765a34ac1928b26992cfafce4d515
manifest_hash b37fe58cf7a043d2164f2e3b08856a75fefad87aef85c02083873e7f3cffb1c8
authorized_delta_hash dd27b354bbe4c44052af2fab7b576198930487053947a93ef89a2977887b4eb1
parser_delta_contract_hash 6cc28c808e36b15470423bf6d728bb8033bff65d3dcf7dc50987f6ae2e779b3c
batch_hydration_contract_hash e07466131aba3aa0f5e39f73fbd95a070d39aa956e5b76c1778db8da8c78d3d2
```

The builder and its tests are:

```text
training/preregister_protocol_specification_intent_maturity_d4.py
SHA-256 978e51f7bcb2b5fef71558dc6b68611f93680ee3dbb9f8eaa10a58c2cb1fd3bc

tests/test_preregister_protocol_specification_intent_maturity_d4.py
SHA-256 3845f348468e9f8939c4d4f26df3cc31e91fd9c3a7e1a0aaac72616650b0b702
```

The builder validates and canonical-hash binds:

1. the exact D3 preregistration;
2. the exact terminal D3 Gate 4 rejection;
3. the committed D4 parser decision; and
4. the exact synthetic D4 parser probe.

It removes only the D3 outer manifest and inheritance-proof envelopes, applies
the explicit D4 successor changes, recursively compares every leaf, and
rejects unless the exact changed-path roster and complete value delta match
their frozen hashes.

## Frozen parser delta

The reference parser version becomes:

```text
PSIM_PREAMBLE_STATE_MACHINE_V2_EIP_EMPTY_SEPARATORS
```

The EIP function name becomes:

```text
parse_eip_preamble_d4
```

Its frozen sequence is:

1. reject blobs over the unchanged D1 maximum;
2. decode strict UTF-8, reject NUL, normalize CRLF/CR to LF, normalize NFC,
   and strip trailing ASCII horizontal whitespace exactly as D1;
3. require the unchanged normalized opening and first closing `---` fences;
4. count all normalized header lines and bytes against the unchanged D1
   limits;
5. remove only lines whose normalized value is empty;
6. pass every remaining line to the unchanged D1 header state machine;
7. require a positive ASCII-decimal `eip` value; and
8. preserve the unchanged dependency parser.

Because D1 strips trailing spaces and tabs, physically empty lines and
ASCII-horizontal-whitespace-only lines both normalize to empty. D4 does not
add a raw-byte side channel to distinguish them.

D4 does not adopt a general YAML parser and makes no current-`eipw`
compatibility claim. Quotes, collections, implicit types, anchors, aliases,
directives, multiline scalar semantics, duplicate-key behavior, comments,
continuations, malformed lines, empty values, and all BIP rules remain under
the inherited D1 state machine.

Official format evidence:

- <https://eips.ethereum.org/EIPS/eip-1>
- <https://yaml.org/spec/1.2.2/#66-comments>
- <https://yaml.org/spec/1.2.2/#67-separation-lines>
- <https://github.com/ethereum/eipw/blob/5d3cfc2585aadd5f3c8c2c223582e2f889c82bfa/eipw-preamble/src/lib.rs#L103-L155>

The version distinction is explicit: current `eipw-preamble 0.4.0` rejects an
empty line because every extracted line must contain a colon. D4 is a bounded
historical compatibility shim, not an upstream validator replacement.

## D3 transport is mechanically inherited

The D3 one-fetch targeted batch-hydration mechanism is unchanged:

```text
/usr/bin/git -C <fresh-bare-root> \
  -c fetch.negotiationAlgorithm=noop \
  fetch origin \
  --no-tags \
  --no-write-fetch-head \
  --recurse-submodules=no \
  --filter=blob:none \
  --no-auto-maintenance \
  --stdin
```

The complete sorted OID manifest remains the only stdin. There is still one
explicit fetch per replica, concurrent stdout/stderr drainage, exact new-object
inventory, no loose objects, no ref or `FETCH_HEAD` mutation, no maintenance
child, and local-only post-hydration decoding with `GIT_NO_LAZY_FETCH=1`.

Only the failure namespace changes from D3 to D4, and the forbidden
predecessor-reuse string now includes D3. Replacing those D4 strings with their
D3 equivalents makes the complete batch-hydration contract byte-equal to the
D3 contract.

The exact local Git binding remains:

```text
/usr/bin/git
git version 2.43.0
SHA-256 2a8c18fbf43da9f692d75474c72bea9dfd796c260b0f3dfe456376abc3bbd668
```

## Fresh roots, refs, and artifacts

Any later official one-shot must start from four absent, fresh roots:

```text
/tmp/psim-d4-source/ethereum-a.git
/tmp/psim-d4-source/ethereum-b.git
/tmp/psim-d4-source/bitcoin-a.git
/tmp/psim-d4-source/bitcoin-b.git
```

The sealed traversal namespace is:

```text
refs/psim-d4/sealed-tip
```

Shared objects, alternates, copied predecessor caches, hard links, symlinks,
checkout, index, linked worktrees, tags, remote-tracking refs, retries, fallback
lazy fetch, full clone, `--refetch`, and D1/D2/D3 source-object reuse remain
forbidden. The 300 GiB WSL disk guard remains unchanged.

All event, card, control, source-pass, source-rejection, and lock paths must be
D4-namespaced. D1/D2/D3 artifacts cannot become D4 outputs.

## Mechanically inherited source contract

Outside the frozen D4 delta, the D3 contract remains byte-equal:

- official Ethereum and Bitcoin remotes, branches, and sealed tips;
- source interval `[2020-01-01, 2024-01-01)`;
- card interval through `2024-04-01`;
- complete root-to-tip first-parent traversal;
- SHA-1 commit/blob recomputation;
- running-maximum UTC committer-day causal clock;
- path, event, pre-window, dependency, section, bucket, and card grammars;
- strict parser success fraction `1.0`, no repair, and no row dropping;
- `ARCHIVE_D2`, `D7`, `D30`, and primary `D90`;
- deterministic daily relation cards at `12:05Z`;
- famous-proposal quarantine;
- seven exact relation controls and control-sensitivity floor;
- train/test/eval support floors and split boundaries; and
- zero model, market, funding, outcome, reward, trade, PnL, CAGR, and strict
  MDD access before source support passes.

The thirteen source gates and their order remain exactly D1/D2/D3's. The first
failed D4 gate retires D4 unchanged:

```text
REJECT_PSIM_D4_UNCHANGED_BEFORE_MARKET_MODEL_OR_OUTCOMES
```

No source repair, parser change, row drop, threshold change, schedule change,
control change, split change, quarantine change, retry, or model/economic stage
is permitted after an official D4 attempt begins.

## Verification and next boundary

The D4 preregistration tests prove:

- exact D3 preregistration, D3 terminal, decision, and parser-probe bindings;
- recursive path- and value-level D3 inheritance;
- exact parser delta and independent parser-delta hash;
- BIP identity and rejection preservation through the bound probe;
- exact D3 transport mechanics after failure-namespace rebasing;
- fresh D4 roots, refs, artifacts, and predecessor-reuse prohibition;
- unchanged gates, controls, splits, clone shape, and Git binary;
- canonical, deterministic, atomic, conflict-safe output;
- fail-closed authority, delta, parser, transport, and split mutations; and
- no network/Git execution, source repository, market, model, or trading
  imports/calls during preregistration.

The next allowed work unit is implementation plus synthetic adversarial testing
of the D4 evaluator. Official source remains closed until that implementation
is independently reviewed and an exact direct-child execution seal is
committed.
