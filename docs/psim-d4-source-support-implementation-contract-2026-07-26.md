# PSIM-D4 source-support implementation contract

Date: 2026-07-26

## Scope

This document binds:

```text
training/build_protocol_specification_intent_maturity_d4_source_support.py
tests/test_build_protocol_specification_intent_maturity_d4_source_support.py
```

PSIM-D4 is source-support-only. It makes no model, alpha, profitability,
portfolio, leverage, or live-trading claim. Market data, funding, models,
rewards, trades, PnL, CAGR, strict MDD, and all economic outcomes remain
forbidden until every source gate passes.

PSIM-D3 is terminally rejected and is never repaired, rerun, or reused as a
source root. PSIM-D4 mechanically inherits the committed D3 evaluator and
changes one source semantic only: an EIP front-matter line that is empty after
the frozen D1 normalization is ignored as a non-semantic separator.

No official EIP/BIP source may be opened until this implementation unit is
committed, independently reviewed, and followed by an exact direct-child
execution-seal commit.

## Frozen authority

| authority | commit | SHA-256 / result |
|---|---|---|
| D4 decision and parser selection | `131009359c60bc5b28b76d22a63abf698011fbcb` | `2615736ba063c2b8e35811d3d01ab3517b345d74a2f7b70d248899aa393d7b99` |
| D4 synthetic parser probe | `131009359c60bc5b28b76d22a63abf698011fbcb` | JSON `fbb97d65ef93b307c47055ed1883d6416e510a70b38083ae17ced2c78e4745ee`; result `4a3cca52755716dbf6e9b4cd801e46b72bab841cea5609f6bb42519487e5f5e6` |
| D4 preregistration unit | `7731f8322b1700550ff1aa46d8a6c6898c31eef0` | JSON `52d77eafef0e9e79f1d7a47b9c262aad148765a34ac1928b26992cfafce4d515`; manifest `b37fe58cf7a043d2164f2e3b08856a75fefad87aef85c02083873e7f3cffb1c8` |
| D3 terminal rejection | `f9089a300d4ba97722ecc1b59f8f8260eff8851b` | JSON `a9be5b5990ad79b7da7d72a22968f4f62a2700877b198606565cc70206fe9802`; result `b00b54b70720d42d213b315e82e7ff3ad0df03909b92aaa514299e750fa1ba2c` |
| sealed D3 evaluator | `cf85aedaad0a0e2b15a440362d03702aad10175f` | runner `a32f6fa3354a9765469985bcc78dc35fc67ac4d07b5216dc212c81b8e20d72dd`; tests `a0e5dad8cb78d462828a63ab5b1a20fae9101cef4588dce40b8e3dcc78e9dc17` |
| D3 synthetic transport probe | `126f7f1354eff90f30d5a6b3d60bd6641268b03b` | JSON `4a815145a1f2ab9c6c61d599cf0aaf2218172e9f71251e95ce7178c1f3be13b7`; result `0df158cddd9b663b2daca14e01bcaa5c2e64b7f5d976720282120585bc41c63a` |
| sealed D1 core | `80b656994f17548a7a599a548e23e9f1cd01302d` | runner `414e83256b3ea489a9e1cd0995f6061e5fab550cd12c795ef7e88eff8998d9fb`; tests `343aa1a72cfbca23d9756988ced042b5c61a6e8fc5a21a0b6d18e45870e906e9` |

Frozen D3→D4 contract hashes:

```text
authorized delta
dd27b354bbe4c44052af2fab7b576198930487053947a93ef89a2977887b4eb1

parser delta
6cc28c808e36b15470423bf6d728bb8033bff65d3dcf7dc50987f6ae2e779b3c

batch hydration
e07466131aba3aa0f5e39f73fbd95a070d39aa956e5b76c1778db8da8c78d3d2
```

## Exact parser delta

The D4 EIP parser executes the following fixed order:

1. run the exact D1 strict UTF-8 and line normalization;
2. locate the exact opening and closing `---` fences;
3. apply the unchanged 256-line and 131,072-byte header limits to all
   normalized front-matter lines, including empty lines;
4. remove only lines whose normalized value is exactly empty; and
5. parse every remaining line with the unchanged D1 header state machine.

Both a physically empty line and a line containing only ASCII horizontal
whitespace normalize to the same empty value under D1. The exception applies
only inside EIP front matter. It does not:

- adopt a general YAML parser;
- change strict UTF-8, fence, key, value, duplicate, continuation, comment,
  proposal-number, dependency, or path-number rules;
- rescue an empty header value;
- bypass header line or byte bounds; or
- claim compatibility with the current `eipw` validator.

The BIP parser remains the exact D1 function object.

The implementation replaces only the EIP preamble call inside
`parse_blob_features`. Blob SHA-1 verification, path/preamble number equality,
dependency extraction, normalized source lines, section classification, and
the complete `BlobFeatures` shape remain D1-identical.

## D3 evaluator inheritance

The D4 evaluator was mechanically copied from the sealed D3 evaluator. An AST
regression test normalizes the D4 namespace and the nested D3 authority path,
then requires every shared function to be identical except:

```text
build_self_check_manifest
_run_self_check_subprocess
static_authority
```

The only D4-only functions are:

```text
parse_blob_features
_load_parser_probe
```

The three changed shared functions add parser-probe and predecessor bindings;
they do not alter source traversal, hydration, event, card, gate, control, or
publication behavior. The same regression test freezes the complete top-level
assignment delta to D4 preregistration hashes, predecessor/parser bindings,
the parser function selection, and the expanded verification roster.

The D3 targeted batch-hydration contract remains unchanged after the
preregistered failure-namespace rebase. Each of four fresh replicas performs
one explicit sorted-OID batch fetch, exact object-store delta validation, and
local-only length-delimited `cat-file --batch` decoding with
`GIT_NO_LAZY_FETCH=1`. Retries, lazy fallback, full clone, predecessor object
reuse, checkout, shared objects, and source repair remain forbidden.

Every Git subprocess remains bound to:

```text
/usr/bin/git
git version 2.43.0
SHA-256 2a8c18fbf43da9f692d75474c72bea9dfd796c260b0f3dfe456376abc3bbd668
```

## Synthetic self-check

The D4 self-check loads only committed control artifacts and synthetic bytes.
It proves:

- D1 rejects the synthetic shape representing the published D3 blank-line
  failure;
- D4 parses that shape to the same fields as D1 parses the no-blank control;
- the BIP parser remains the D1 function object;
- the complete D1 synthetic core remains byte- and hash-bound; and
- the D3 one-fetch/no-lazy transport probe remains hash-bound.

Observed before the implementation commit:

```text
stdout SHA-256
ee03f9747602ac7910ee35b391560ccb0553c60a22f29e4dd80712885d4699ea

manifest hash
fa216eab3422ad86fdb3a9dc676199ed031927ef59722f5480f46b9ee0c1b93e

failed []
network_calls 0
git_commands 0
source_event_rows_opened 0
official_source_opened false
outcomes_opened false
```

## Synthetic and local-only verification

The combined D1/D2/D3/D4 battery passed:

```text
289 passed
```

The D4 tests add:

- exact D4 preregistration, parser-probe, D3 terminal, D3 evaluator, D3
  transport, and D1 core bindings;
- synthetic D1-reject/D4-accept parser differentiation;
- equal parsed headers, dependency edges, and section presence against the D1
  no-blank control;
- BIP function identity;
- AST proof that all unmodified D3 evaluator functions remain identical;
- real local synthetic partial-clone hydration and D1-equivalent event rows;
- exact-binary/no-lazy Git invocation;
- fresh-root, sealed-ref, object-store, Trace2, and publication fail-closed
  behavior; and
- zero model, market, reward, trade, PnL, CAGR, strict-MDD, and outcome access.

No official EIP/BIP source root was created or opened while producing this
implementation unit. `/tmp/psim-d4-source` must remain absent until the
execution seal is committed.

## Execution-seal topology

The runner, evaluator tests, and this contract must share one clean
implementation commit. Seal creation reruns the exact verification roster:

1. D1 preregistration and evaluator tests;
2. D2 preregistration and evaluator tests;
3. D3 transport-probe, preregistration, and evaluator tests;
4. D4 parser-probe and preregistration tests; and
5. D4 evaluator tests.

The seal JSON and its dedicated seal test must be the only paths in the exact
direct-child commit. The official one-shot may run only when that seal commit
is current `HEAD`, every authority binding is unchanged, the worktree is
clean, and all four D4 source roots are absent.

## Fresh roots and publication paths

```text
/tmp/psim-d4-source/ethereum-a.git
/tmp/psim-d4-source/ethereum-b.git
/tmp/psim-d4-source/bitcoin-a.git
/tmp/psim-d4-source/bitcoin-b.git

refs/psim-d4/sealed-tip

results/protocol_specification_intent_maturity_d4_source_support_2026-07-26.json
results/protocol_specification_intent_maturity_d4_source_rejection_2026-07-26.json
data/protocol_specification_intent_maturity_d4_events_2020_2023.jsonl.gz
data/protocol_specification_intent_maturity_d4_cards_2020_2024q1.jsonl.gz
results/protocol_specification_intent_maturity_d4_source_controls_2026-07-26.json
```

The first failed source gate terminates D4 unchanged:

```text
REJECT_PSIM_D4_UNCHANGED_BEFORE_MARKET_MODEL_OR_OUTCOMES
```

There is no repair, row drop, parser amendment, retry, threshold change,
source substitution, model access, or economic-stage access after the one
official attempt begins.

## Reference version boundary

- [EIP-1](https://eips.ethereum.org/EIPS/eip-1)
- [YAML 1.2.2 comments and separation](https://yaml.org/spec/1.2.2/#66-comments)
- [`eipw-preamble` 0.4.0 source at the reviewed commit](https://github.com/ethereum/eipw/blob/5d3cfc2585aadd5f3c8c2c223582e2f889c82bfa/eipw-preamble/src/lib.rs#L103-L155)
- [Git partial clone 2.43.0](https://git-scm.com/docs/partial-clone/2.43.0.html)
- [git-fetch 2.43.0](https://git-scm.com/docs/git-fetch/2.43.0.html)
- [git-cat-file 2.43.0](https://git-scm.com/docs/git-cat-file/2.43.0.html)

Current `eipw-preamble` 0.4.0 rejects an empty extracted line because it parses
each line as a colon-delimited field. PSIM-D4 is a bounded historical
compatibility shim, not a replacement for or compatibility claim about that
current validator.
