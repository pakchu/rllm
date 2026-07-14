# LURI-48 support freeze — 2026-07-14

## Decision

**PASS support and novelty; LURI outcomes remain unopened.** No LURI future
return, held high/low path, win rate, CAGR, MDD, or funding-adjusted return was
read during this selection.

- preregistration commit:
  `4a237096dd42c585deeb28cbdc7d7017563a4af3`
- preregistration source SHA-256:
  `733d3bde49d0be37eef4dd282dee7464bdcb5af359bace7777f49636b16003ba`
- frozen feature source SHA-256:
  `00ab6a55fc7bfeb3012584db5bc97a7d7b98dd995491acfd3f865c6bd41f92cc`
- support artifact:
  `results/leveraged_um_inventory_release_handoff_support_2026-07-14.json`
- support artifact SHA-256:
  `58a8c3dfd1727b48fe7548ec3b59290060f690f7d0cce8bdb5b13a0bdfd4e3a9`
- selected strictly-prior positive-basis quantile: **0.40**
- fixed hold: **48 five-minute bars (four hours)**

The selected row is the highest frozen grid row that passed every count,
balance, ablation, temporal-placebo, and prior-clock novelty gate. The `0.25`
row had more support but failed two preregistered novelty limits; `0.55` and
`0.70` failed count support. No threshold was repaired after this run.

## Selected support

| Period | Non-overlapping events | Long / short |
|---|---:|---:|
| 2020 | 113 | 41 / 72 |
| 2021 | 113 | 60 / 53 |
| 2022 | 108 | 42 / 66 |
| 2023 | 98 | 37 / 61 |
| **Total** | **432** | **180 / 252** |

- raw candidates before global 48-bar reservation: **486**
- 2023 H1/H2: **45 / 53**
- 2023 Q1/Q2/Q3/Q4: **24 / 21 / 26 / 27**
- overall long/short share: **41.67% / 58.33%**
- active months with at least five scheduled events: **47**
- every frozen count and side-balance floor passed

## Structural necessity and novelty

The four mandatory component ablations stayed below their frozen maximums on
both raw incidence and each control's own non-overlapping schedule:

| Ablation | Raw retention | Scheduled retention | Maximum |
|---|---:|---:|---:|
| no inferred inventory | 0.0209 | 0.0675 | 0.10 |
| no basis history | 0.4386 | 0.4789 | 0.55 |
| no cash refusal | 0.1119 | 0.1545 | 0.20 |
| basis only | 0.1412 | 0.2291 | 0.30 |

The temporal and venue controls also passed. Values are `raw Jaccard /
scheduled Jaccard`; scheduled primary containment is shown separately.

| Control | Raw / scheduled Jaccard | Scheduled containment |
|---|---:|---:|
| Spot-confirmed | `0 / 0` | 0 |
| reverse-time | `0 / 0` | 0 |
| simultaneous-only | `0.1307 / 0.1222` | 0.5231 |
| Spot-inventory swap | `0 / 0` | 0 |
| stale 24h | `0.0031 / 0` | 0 |
| one-bar delay | `0.0021 / 0` | 0 |

Scheduled overlap with earlier primary clocks remained negligible:

- CSPR-12: intersection `0`, Jaccard `0`, containment `0`;
- RIFT-96: intersection `1`, Jaccard `0.00112`, containment `0.00231`;
- CATCH-12: intersection `0`, Jaccard `0`, containment `0`.

The structurally closest frozen reference, CATCH's `0.975` venue-swap
control, also passed the tighter LURI comparison:

- raw: intersection `88`, Jaccard `0.01830`, containment `0.18107`;
- scheduled: intersection `69`, Jaccard `0.01638`, containment `0.15972`.

Thus more than 80% of selected LURI events lie outside that nearest control.

## Grid decision audit

| Basis quantile | Scheduled events | Count pass | Novelty pass | Final |
|---:|---:|:---:|:---:|:---:|
| 0.25 | 498 | yes | no | reject |
| **0.40** | **432** | **yes** | **yes** | **select** |
| 0.55 | 334 | no | yes | reject |
| 0.70 | 226 | no | no | reject |

At `0.25`, no-basis-history scheduled retention was `0.55211` against a
`0.55` maximum, and raw Jaccard against CATCH venue-swap was `0.02029`
against a `0.02` maximum. These small misses are retained as failures rather
than rounded into passes.

## Interpretation and boundary

This is evidence that the proposed causal episode is sufficiently frequent,
bidirectional, structurally nontrivial, and distinct from prior clocks. It is
not evidence of profitability. “Inventory” remains an inference from signed
aggressive quote flow, not observed account positioning.

The next step may serialize the exact 432-event outcome-free clock. Realized
funding data and the evaluator must then be frozen in separate commits before
any LURI return is opened. Calendar 2024 and later remains sealed.
