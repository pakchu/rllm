# Post-PSIM-D3 audit and PSIM-D4 historical-parser decision

Date: 2026-07-26 KST

## Scope

PSIM-D3 is terminally rejected. It is not repaired, resumed, or rerun.
Its retained `/tmp/psim-d3-source` roots remain forensic residue and are not
read or reused by this decision.

This document answers one outcome-blind compatibility question:

> May a newly named candidate treat lines that are empty after the frozen D1
> normalization as non-semantic separators inside historical EIP front matter,
> while preserving every other D1 parser, source, support, model, and economic
> rule?

The answer is **yes for a synthetic probe and a later preregistration only**.
This decision does not authorize official source execution, model loading,
market access, outcome access, or any alpha or profitability claim.

## Terminal PSIM-D3 boundary

The D3 terminal commit is:

```text
f9089a300d4ba97722ecc1b59f8f8260eff8851b
```

The immutable rejection is:

| artifact | identity |
|---|---|
| `results/protocol_specification_intent_maturity_d3_source_rejection_2026-07-25.json` | SHA-256 `a9be5b5990ad79b7da7d72a22968f4f62a2700877b198606565cc70206fe9802` |
| terminal result | `b00b54b70720d42d213b315e82e7ff3ad0df03909b92aaa514299e750fa1ba2c` |
| first failed gate | Gate 4, `historical_blob_preamble_dependency_integrity` |

D3's targeted batch hydration succeeded, but the unchanged D1 parser rejected
the eighteenth unique Ethereum text with:

```text
PSIM blank line inside header
```

The post-terminal, local-only forensic receipt identified the already
published failure shape:

```text
proposal       2378
path           EIPS/eip-2378.md
commit         b788f38a216ca4cfea9d9de8ccfcf4cf658c8950
blob OID       ac34c07b91d6dffa14922951473f50dd587eb900
effective day  2020-01-29
raw SHA-256    a2fd3d87db7861f2b50739bf6c9015b968abc6fb6ffee7629492626034f41bb1
```

Its front matter had an empty physical line between the last field and the
closing fence. D3 correctly rejected because its preregistration prohibited
parser relaxation. The D3 artifact remains the only D3 terminal result.

## Official format evidence and version notes

### EIP-1

The canonical EIP-1 page says an EIP begins with an RFC-822-style header
preamble preceded and followed by `---`, and also names that region Jekyll
front matter:

- <https://eips.ethereum.org/EIPS/eip-1>

EIP-1 specifies fields and delimiters but does not explicitly state whether an
empty line inside the fenced region is accepted or rejected. The simultaneous
RFC-822 and Jekyll terminology therefore does not, by itself, settle historical
blank-line compatibility.

### Current EIP validator

The current EIPs repository README says EIP-1 rules are enforced with `eipw`:

- <https://github.com/ethereum/EIPs>

The current `eipw-preamble 0.4.0` source extracts the text between the first two
fences and then parses each line as a colon-delimited field. An empty line has
no colon and is consequently rejected:

- <https://github.com/ethereum/eipw/blob/5d3cfc2585aadd5f3c8c2c223582e2f889c82bfa/eipw-preamble/src/lib.rs#L103-L155>
- <https://docs.rs/eipw-preamble/0.4.0/src/eipw_preamble/lib.rs.html#85-164>

The published 0.4.0 source is bound here to upstream VCS commit
`5d3cfc2585aadd5f3c8c2c223582e2f889c82bfa`.

This is a material version/scope distinction.
PSIM-D4 is a historical compatibility parser.
It handles immutable 2020-2023 repository bytes, is not a replacement for
`eipw`, and makes no current-validator compatibility claim. A file accepted by
D4 may still fail the current repository validator.

### YAML presentation semantics

YAML 1.2.2 states that a line containing only whitespace outside scalar
content is treated as a comment line, and that tokens may be separated by
multi-line, possibly empty, comments:

- <https://yaml.org/spec/1.2.2/#66-comments>
- <https://yaml.org/spec/1.2.2/#67-separation-lines>

This is supporting evidence for treating normalized-empty separator lines as
non-semantic. D4 does **not** adopt YAML generally. Quoting, collection syntax,
implicit typing, anchors, aliases, directives, and other YAML behavior remain
outside the parser. The unchanged D1 line state machine remains authoritative
for every nonempty line.

## Minimal D4 grammar delta

The selected successor is:

```text
PSIM-D4 — Protocol Specification Intent-Maturity relation RLLM,
historical EIP normalized-empty separator grammar
```

Its parser version is:

```text
PSIM_PREAMBLE_STATE_MACHINE_V2_EIP_EMPTY_SEPARATORS
```

The exact delta is:

1. run the unchanged D1 `normalize_blob_bytes`;
2. locate the EIP opening and first closing normalized `---` fence exactly as
   D1 does;
3. enforce the unchanged normalized header byte and line limits **before**
   filtering any line;
4. remove only lines whose normalized value is the empty string;
5. parse every remaining line with the unchanged D1 header state machine; and
6. retain the unchanged positive `eip` number requirement.

D1 removes trailing ASCII spaces and tabs before parsing. Consequently, both a
physically empty line and a line containing only ASCII horizontal whitespace
become normalized-empty and receive the same D4 treatment. Rejecting one but
not the other would require a new raw-byte side channel or a normalizer fork,
which is a larger and less auditable change. The probe records this behavior
explicitly rather than calling the rule “exact physical blank line.”

The following remain unchanged:

- strict UTF-8, NUL rejection, NFC normalization, and blob/line limits;
- header byte and line limits, including normalized-empty lines;
- field syntax, key normalization, comments, and continuations;
- duplicate-key, malformed-line, orphan-continuation, and empty-value
  rejection;
- positive proposal number and dependency grammar;
- opening/closing fence selection;
- the complete BIP parser, represented by the identical D1 function object;
- source interval, sealed tips, causal clocks, support floors, archive
  schedules, controls, splits, and failure actions;
- D3's proven targeted batch-hydration transport; and
- every market, model, reward, execution, and outcome boundary.

## Synthetic-only probe

The executable probe is:

```text
training/probe_protocol_specification_intent_maturity_d4_parser.py
```

Its canonical artifact is:

```text
results/protocol_specification_intent_maturity_d4_parser_probe_2026-07-26.json
result_hash 4a3cca52755716dbf6e9b4cd801e46b72bab841cea5609f6bb42519487e5f5e6
SHA-256 fbb97d65ef93b307c47055ed1883d6416e510a70b38083ae17ced2c78e4745ee
```

The probe reads only the already published D3 terminal JSON for its rejection
binding. It does not read the historical proposal blob, any official proposal
repository, or any D1/D2/D3 source root. Its EIP-2378-shaped input is explicitly
a synthetic structural fixture and does not reproduce the historical bytes.

Observed synthetic battery:

| assertion | count/result |
|---|---:|
| D1-accepted EIP outputs unchanged | 5 |
| D1 nonblank EIP rejections preserved | 13 |
| normalized-empty/control output pairs equal | 6 |
| synthetic D3 failure shape accepted | true |
| BIP accepted outputs unchanged | 2 |
| BIP rejections preserved | 6 |
| BIP parser is the identical D1 function object | true |
| normalized-empty lines count against header line and byte limits | true |
| separator cannot rescue a field with no value | true |

The six newly accepted cases cover one trailing empty line, a middle
separator, multiple separators, an ASCII-horizontal-whitespace-only separator,
CRLF normalization, and a separator before a valid continuation. Each result
is byte-for-byte equal at the parsed field mapping level to the corresponding
D1-accepted control with those separators removed.

## Rejected alternatives

The following were rejected:

1. **Repair or rerun D3.** D3 is terminal.
2. **Drop EIP-2378 or lower parser success below 1.0.** This selects source rows
   after observing failure and damages completeness.
3. **Adopt a general YAML parser.** This changes quoting, typing, collections,
   and duplicate-key behavior far beyond the observed incompatibility.
4. **Modify the shared D1/BIP parser.** The failure is EIP-fenced-front-matter
   specific; changing BIP grammar has no evidence.
5. **Ignore blank lines after applying header limits.** This permits empty-line
   padding to bypass the frozen resource guard.
6. **Distinguish physical empty from whitespace-only after D1 normalization.**
   This requires a second raw-byte grammar and is not the minimal successor.
7. **Open another official source root to discover more parser cases before
   preregistration.** That would adapt the grammar to observed source bytes.

## Authorization and stop condition

The successful synthetic probe authorizes only:

1. independent review of this parser delta and evidence boundary; and
2. a PSIM-D4 preregistration that binds the new parser version, the unchanged
   D3 transport, four fresh `/tmp/psim-d4-source` roots, and a larger sealed
   synthetic malformed-header corpus.

It does not yet authorize cloning, fetching, or opening official EIP/BIP source
for D4. Before any such execution, D4 requires a committed preregistration,
implementation, independent review, execution seal, and fresh-root
one-shot contract.

No market rows, funding rows, future returns, models, rewards, trades, PnL,
CAGR, or strict MDD may be accessed until every source-only gate passes.
