# Post-CLOR-D1 alpha mechanism audit

Date: 2026-07-25

## Outcome

Select one new source-blind mechanism for an exact source-only freeze:

**GIPR-D1 — Governance Intent–Payload Relation daily target-position RLLM.**

GIPR-D1 will use immutable Ethereum governance events from Compound and
Uniswap. Each proposal exposes both a human-readable description and the
ordered executable action bundle that may later be queued and executed. The
candidate asks one narrow question that is materially better matched to an
LLM than raw numerical forecasting:

> Does the stated governance intent agree with the contracts, function
> signatures, calldata shapes, and lifecycle transitions that can actually
> execute, and how does that relation evolve across protocols?

The eventual policy is a single-model daily `TARGET_LONG`, `TARGET_FLAT`, or
`TARGET_SHORT` decision over a finite causal history plus current position.
There is no analyzer/trader pair. Deterministic code owns Ethereum
chronology, ABI validation, payload normalization, numeric bucketing,
finality, market execution, costs, funding, reward, and strict drawdown.

This selection is not source support, alpha, profitability, or live evidence.
No GIPR proposal event, description, payload, lifecycle incidence, BTC row,
funding row, future return, reward, model output, action, trade, PnL, CAGR, or
MDD was opened in this audit.

## Why the prior source-pass roster is not the answer

A repository-wide metadata audit initially surfaced many earlier
source-support passes. Full-history cross-checking shows that the highest
ranked identities are not unopened alpha candidates:

- TBASR reached an economic TRAIN result;
- BATE-288 reached a terminal TRAIN rejection;
- DLPD-12 reached a TRAIN result;
- FLNSR-2016 reached a terminal Stage-1 rejection;
- QLCD-288 reached a TRAIN result;
- TSDR-72 reached a terminal novelty rejection; and
- BCTP reached a terminal 2021 cheap-policy transfer rejection.

Other available dense sources are concentrated in the same Binance
microstructure cluster whose singleton and composition outcomes have already
been heavily inspected. Recombining those fields after observing their
failures would be outcome-conditioned repair, not independent alpha research.

CLOR-D1 is also terminal. Its frozen decimal grammar rejected signed zero
before outcomes. Relaxing that grammar or retaining the same
Treasury/SOMA/OFR identity is forbidden.

## Why GIPR-D1 is materially different

GIPR-D1 uses no CEFS, Cboe, CLOR, Treasury, SOMA, OFR, Binance trade, kline,
funding, premium, open-interest, quantity-lattice, cross-venue, Trollbox, or
prior portfolio-alpha source.

Its primitive object is not a market measurement. It is a cryptographically
committed relation between:

1. the prose intent published in `ProposalCreated`;
2. the exact ordered target contracts;
3. the exact ordered function signatures and calldata byte shapes;
4. the proposal lifecycle events; and
5. the corresponding relation state in the other protocol.

Compound documents the exact `ProposalCreated`, `ProposalCanceled`,
`ProposalQueued`, and `ProposalExecuted` event contracts and states that
proposal actions are ordered target/value/signature/calldata arrays with a
human-readable description:

- <https://docs.compound.finance/v2/governance/>
- <https://github.com/compound-finance/compound-protocol/blob/a3214f67b73310d547e00fc578e8355911c9d376/contracts/Governance/GovernorAlpha.sol>
- <https://github.com/compound-finance/compound-protocol/blob/a3214f67b73310d547e00fc578e8355911c9d376/contracts/Governance/GovernorBravoInterfaces.sol>

Uniswap's official technical reference identifies its historical and active
Ethereum governance contracts, including GovernorBravo:

- <https://developers.uniswap.org/docs/ecosystem/governance/technical-reference>
- <https://github.com/Uniswap/governance/blob/eabd8c71ad01f61fb54ed6945162021ee419998e/contracts/GovernorAlpha.sol>

OpenZeppelin independently documents the same Governor proposal abstraction:
a proposal contains executable targets/calldatas plus a human-readable
description, and the complete proposal parameters are recoverable from
events:

- <https://docs.openzeppelin.com/contracts/4.x/governance>
- <https://docs.openzeppelin.com/contracts/4.x/api/governance>

These references establish the mechanism and ABI semantics only. They do not
establish historical incidence or economic value.

## Frozen source-axis candidates

The next preregistration must bind exactly these Ethereum mainnet governor
addresses:

| protocol | contract | address | first-code block |
|---|---|---:|---:|
| Compound | GovernorAlpha | `0xc0dA01a04C3f3E0be433606045bB7017A7323E38` | 9,601,447 |
| Compound | GovernorBravo | `0xc0da02939e1441f497fd74f78ce7decb17b66529` | 12,006,099 |
| Uniswap | GovernorAlpha v0 | `0x5e4be8Bc9637f0EAA1A755019e06A68ce081D58F` | 10,861,678 |
| Uniswap | GovernorAlpha v2 | `0xC4e172459f1E7939D522503B81AFAaC1014CE6F6` | 12,543,659 |
| Uniswap | GovernorBravo | `0x408ED6354d4973f66138C91495F2f2FCbd8724C3` | 13,059,157 |

The Compound addresses and deployment blocks are present in the official
`networks/mainnet.json` artifact:

<https://github.com/compound-finance/compound-protocol/blob/a3214f67b73310d547e00fc578e8355911c9d376/networks/mainnet.json>

The three Uniswap addresses are published by the official Uniswap technical
reference. The first-code blocks above were found without reading event logs
by binary-searching `eth_getCode`. Two Ethereum transports independently
returned identical first-code block hashes and bytecode hashes for all three
addresses. This bounded transport probe is feasibility evidence only and must
be hash-bound by the preregistration.

## Historical envelope and causal clock

The exact source envelope is:

- start inclusive: `2020-01-01T00:00:00Z`;
- end exclusive: `2024-01-01T00:00:00Z`;
- start block: `9,193,266`;
- end-exclusive block: `18,908,895`;
- last permitted source block: `18,908,894`; and
- availability: canonical event block `N` becomes available only at the
  canonical timestamp of block `N+64`.

Two transports independently reproduced the exact year boundaries:

| boundary | first block at/after UTC boundary | canonical block hash |
|---|---:|---|
| 2020-01-01 | 9,193,266 | `0xfa39cb98792d28b79138fc0a2ab7c08e59c00b43bd0dcd0b7e4bd49684f7216c` |
| 2021-01-01 | 11,565,019 | `0x12620d72a9306bc7c5ed1e2ba1ac75115197020d5101925305c714e8e1d174d7` |
| 2022-01-01 | 13,916,166 | `0x2f03b3220805ee951f2d4c250621e9c2175439cef524e976aed707a2d595729e` |
| 2023-01-01 | 16,308,190 | `0x53dd35d982c984441b3b613919d64dbbf131063d0f85804d77f93f190fa5e106` |
| 2024-01-01 | 18,908,895 | `0x08f760c77e7a5843464404bf59d1f042a8b2ec18ae9345a16940546216069eec` |

The final source must be replayed independently through two transports and
compared by canonical log identity and normalized payload hash. Provider host
names are execution details, not source identity. Any disagreement fails
closed.

## Permitted representation

The source-only freeze may expose structural and categorical relations, not
source-owned market direction:

- protocol and governor generation;
- proposal/lifecycle event type;
- proposal age and lifecycle age bucket;
- number-of-actions bucket;
- exact target-address role labels from a frozen registry, otherwise
  `UNKNOWN_TARGET`;
- exact function signature or four-byte selector token, otherwise
  `UNKNOWN_SELECTOR`;
- calldata validity and ABI-shape class;
- native-value presence bucket, never the raw amount;
- description structural tokens and bounded text;
- intent/payload `AGREE`, `PARTIAL`, `CONFLICT`, or `UNRESOLVED` relation
  produced only by the later frozen single model;
- cross-protocol relation persistence/flip; and
- current target position.

Raw token amounts, addresses not converted to a frozen role label, block
numbers, timestamps, prices, returns, ranks, PnL, and hidden model reasoning
are forbidden model inputs. Numeric calldata values remain deterministic-code
owned and may only enter through preregistered non-directional buckets.

The model may emit only:

```text
TARGET_LONG
TARGET_FLAT
TARGET_SHORT
```

It may not choose leverage, order size, stop, fee, funding treatment,
execution timing, or reward.

## Required source-only gates

Before any market, funding, reward, or model access, GIPR-D1 must pass all of
the following under one sealed evaluator:

1. exact contract-address, block-envelope, boundary-hash, topic, and bytecode
   validation;
2. byte-identical dual replay after canonical normalization;
3. strict dynamic-ABI validation for every `ProposalCreated` array and string;
4. unique canonical log identities and canonical block-header agreement;
5. valid lifecycle ordering with no queue/execute/cancel before creation;
6. proposal and action support in TRAIN 2020–2021, TEST 2022, and EVAL 2023;
7. both Compound and Uniswap support in every split;
8. non-collapsed description, target-role, selector, action-count, and
   lifecycle vocabularies;
9. daily schedule coverage after warm-up with explicit stale/no-proposal
   states rather than silent forward fill;
10. future-append invariance;
11. source-order, text/payload-pairing, protocol-label, and lifecycle-order
    controls that materially alter the relational language; and
12. zero access to post-2023 source rows, BTC, funding, returns, rewards,
    models, actions, trades, or portfolio statistics.

The first failed gate retires GIPR-D1 unchanged before outcomes. A source
failure may not remove one protocol, governor generation, event type, target
role, selector, text field, lifecycle transition, split, or control.

## Later economic/RLLM falsification

Only a complete source-support pass may authorize a separately committed
economic and single-RLLM evaluator. That later evaluator must be frozen before
opening BTC outcomes and must compare:

- deterministic no-text payload-only policy;
- deterministic no-payload text-only policy;
- source-order-shuffled control;
- text/payload-pair-shuffled control;
- one-protocol ablations;
- lifecycle-only baseline;
- frozen cheap categorical policies; and
- the single GIPR RLLM.

GIPR fails if the model does not add transfer value beyond cheap baselines, if
either text or payload can be removed without material loss, or if shuffled
intent/payload pairs preserve performance.

The historical split is fixed:

- TRAIN: 2020–2021;
- TEST/checkpoint selection: 2022;
- EVAL: 2023;
- 2024+ remains sealed until a separately frozen extension.

Repository-wide BTC outcomes are already heavily inspected, so even a 2023
pass is candidate-specific historical transfer evidence, not a globally
pristine discovery. Live profitability still requires exact source parity
and prospective shadow/live evidence.

## Stop condition

The next unit may preregister and implement only the source-support evaluator.
It must not read one governance event before the source contract, grammar,
gates, controls, hashes, and terminal action are committed and sealed.
