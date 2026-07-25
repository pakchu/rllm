# GIPR-D1 source-support preregistration

Date: 2026-07-25

## Decision

GIPR-D1 is preregistered for a source-only Ethereum governance replay. No
governance event log, description, payload, lifecycle incidence, BTC market
row, funding row, future return, reward, model output, action, trade, PnL,
CAGR, or strict MDD was opened while producing this artifact.

The candidate-selection boundary is:

```text
docs/post-clor-d1-alpha-mechanism-audit-2026-07-25.md
SHA256 fd52594fcdfbd85d9a8385ae57513f98ebca169da43f31cafbe058b7519e7c43
```

The machine-readable preregistration is:

```text
results/governance_intent_payload_relation_preregistration_2026-07-25.json
SHA256 319ac26108e936331f95d047a69f739ffecfd2bdc777573100a1ce83c771c197
manifest_hash daccd86ac2b218552f6312189fe4cd0ce6c775fe3b91f37ace31e8d12e1b1645
```

The preregistration implementation and tests are:

```text
training/preregister_governance_intent_payload_relation.py
SHA256 cb9273a98aebb21c6a11803eac106663235440c4cb7dcc1d7232b37859d1c24f

tests/test_preregister_governance_intent_payload_relation.py
SHA256 03f24896d7dcf03bbe5849e3e0b4d4caa948b071c05671470e395e7678472fa6
```

## Frozen source identity

The source roster contains exactly five Ethereum mainnet governor contracts:

- Compound GovernorAlpha;
- Compound GovernorBravo;
- Uniswap GovernorAlpha v0;
- Uniswap GovernorAlpha v2; and
- Uniswap GovernorBravo.

The source interval is block `9,193,266` inclusive through block
`18,908,894` inclusive. Block `18,908,895` is the exact 2024 end boundary and
is not a source block. Events become available only at the canonical
timestamp of block `N+64`.

The exact proposal/lifecycle events are:

```text
ProposalCreated(uint256,address,address[],uint256[],string[],bytes[],uint256,uint256,string)
ProposalCanceled(uint256)
ProposalQueued(uint256,uint256)
ProposalExecuted(uint256)
```

All topic hashes, contract addresses, first-code blocks, boundary block
hashes, runtime-code hashes, parser limits, splits, and finality rules are
bound by the JSON manifest.

## Frozen representation

The source evaluator may validate and derive only structural, categorical,
and relational governance states. It may not assign a market side. A later
separately frozen single model may emit only:

```text
TARGET_LONG
TARGET_FLAT
TARGET_SHORT
```

Raw prices, returns, ranks, PnL, block numbers, timestamps, unknown addresses,
and raw numeric calldata values are forbidden model inputs. There is no
analyzer/trader pair.

## Frozen split and support minima

- TRAIN: 2020–2021
- TEST: 2022
- EVAL: 2023
- 2024+: sealed

The source stage must establish proposal, action, month, protocol, target,
selector, lifecycle, daily-decision, and vocabulary support under the exact
per-split minima in the manifest. No minimum may be changed after source
incidence.

## Frozen controls

All six controls are mandatory:

1. protocol-label swap;
2. within-day event-order reversal;
3. text/payload-pair permutation;
4. ordered-action permutation;
5. lifecycle-event rotation; and
6. availability shifted by seven days.

Future append must leave every pre-2024 normalized event, daily state, and
relation card unchanged.

## Terminal rule

The twelve source gates execute in the manifest order. The first failure
retires GIPR-D1 unchanged before market, funding, model, action, or outcome
access:

```text
REJECT_GIPR_D1_UNCHANGED_BEFORE_MARKET_MODEL_OR_OUTCOMES
```

Dropping a protocol, governor generation, event, field, split, control, or
support gate is not an authorized repair.

## Verification

The preregistration test battery passed:

```text
21 passed
```

The next and only authorized unit is implementation and sealing of a
synthetic-only source-support evaluator. Full Ethereum event incidence remains
closed until that evaluator, its tests, and its execution seal are committed.
