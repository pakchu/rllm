# FLNSR-2016 source-support and novelty pass

## Decision

**Advance FLNSR-2016 only to a separately frozen strict evaluator.** The
outcome-blind source build produced a balanced, dispersed weekly clock and
passed the preregistered FLCC overlap limits. This is support for testing, not
profitability evidence.

No BTC candle, post-entry price, funding value, future return, PnL, absolute
return, CAGR, strict MDD, hit rate, model label, or 2024+ source row was loaded.

## Primary source clock

| Window | Events | LONG | SHORT | Longest run |
|---|---:|---:|---:|---:|
| Train 2020–2022 | 67 | 35 | 32 | 6 |
| 2020 | 25 | 16 | 9 | 6 |
| 2021 | 23 | 11 | 12 | 6 |
| 2022 | 18 | 8 | 10 | 3 |
| Selection 2023 | 22 | 15 | 7 | 5 |
| 2023 H1 | 12 | 8 | 4 | 5 |
| 2023 H2 | 10 | 7 | 3 | 3 |
| All | **89** | **50** | **39** | **6** |

Across train plus selection, entries span 40 active months. Maximum month and
quarter shares are 4.49% and 10.11%, and the largest gap is 63.04 days. Every
frozen density, side, dispersion, concentration, gap, run, source, and schema
gate passed.

The causal funnel contained 205 release features, 170 non-neutral H.4.1
liquidity tails, 104 positive and 101 negative narrative rotations, 93 raw
same-side agreements, and 77 disagreements. Chronological seven-day
non-overlap left 89 primary entries.

## Source-only controls

| Clock | Events | LONG | SHORT | Max gap days | Max side run |
|---|---:|---:|---:|---:|---:|
| primary | 89 | 50 | 39 | 63.04 | 6 |
| liquidity only | 160 | 81 | 79 | 29.04 | 10 |
| narrative only | 193 | 102 | 91 | 14.00 | 11 |
| disagreement | 74 | 32 | 42 | 74.04 | 7 |
| stale narrative | 80 | 40 | 40 | 77.04 | 7 |
| exact side flip | 89 | 39 | 50 | 63.04 | 6 |
| deterministic random side | 89 | 48 | 41 | 63.04 | 9 |

Controls remain report-only. Their outcomes are unopened and they cannot
replace FLNSR if its deterministic economics fail.

## FLCC novelty audit

Deterministic one-to-one matching used only timestamps and sides, with ±15
minutes tolerance.

| FLCC comparator | Matched | Jaccard | FLNSR containment | Same-side containment |
|---|---:|---:|---:|---:|
| H4-Q60 | 62 | 0.3804 | **0.6966** | 0.5056 |
| H4-Q65 | 54 | 0.3439 | 0.6067 | 0.4494 |
| H8-Q60 | 48 | 0.2892 | 0.5393 | 0.3371 |
| H8-Q65 | 42 | 0.2577 | 0.4719 | 0.3034 |

All frozen limits passed: Jaccard at most 0.50, FLNSR containment at most
0.70, and same-side containment at most 0.75. H4-Q60 containment passed by a
narrow 0.34 percentage-point margin. That closeness must be carried as a risk;
the threshold cannot be widened or the comparator removed after incidence.

## Authorized next step

Before opening train outcomes, commit a strict evaluator that binds:

- this exact 89-event primary clock and all six controls;
- physical parsing limited first to 2020–2022;
- 0.5x, 6/10 bp costs, exact funding, full-calendar CAGR, and strict MDD;
- positive train and each-year return, CAGR/strict-MDD at least 3, MDD at most
  15%, at least 30 bp mean gross move, monthly-cluster `p <= 0.05`, component
  superiority, and one-extra-bar-delay survival; and
- an unreachable 2023 stage unless the complete train battery passes.

The LLM remains unauthorized. It may be preregistered only after deterministic
train and selection pass.

## Integrity

- preregistration commit: `dc1e21b`;
- preregistration SHA-256:
  `252952438eb2a87dc5f85fbe887a4f99a5f3a7a8a7e764feac414fac2929fd6d`;
- preregistration manifest:
  `4277c6f7fb5a9c075492dd901f55d7b2d4e8b39dbab1e0c40010f996d80dc00c`;
- support manifest:
  `377fd7afd68e99e9c3cd93340505b80fc1ab7263ae2cf8ef117daa304227d895`;
- clock SHA-256:
  `3096143d397fc6d8dac639841c96538979772734dcf2fd8157df580f5b297c6c`;
- rows across primary and controls: 774.
