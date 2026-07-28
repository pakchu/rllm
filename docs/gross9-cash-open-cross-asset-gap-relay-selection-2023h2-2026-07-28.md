# COGR-12 selection report

Metric format: absolute return / full-calendar CAGR / strict MDD / CAGR-MDD / trades.

- as of: `2026-07-28`
- decision: **reject_no_passing_2023h2_cell**
- tested cells: `12`

## Integrity

- committed evaluator: `b99bb754427e0ad161a9322e13e451f92069b684d131da3a62682e2c6348568f`
- preregistration SHA-256: `a989b3be122a99a94531c05735128ba64030b637cca59cad4af75c2713fe649a`
- result hash: `38b3a98115fb06b169b79c387692a540bbd30c6b30125d8b178c2abd088769d9`
- result-file SHA-256:
  `fd0e8b61da4633b57a732a54041fc9cbf17d4f984a96484da754f4a8dd344639`
- two independent committed-code runs were byte-identical;
- wall times were `4:38.92` and `4:40.38`, with about 19.8 GB peak RSS.

## Selection result

All three coordination modes and all four configured candidate weights had the
same terminal defect: **zero COGR trades**.

| scope | absolute return | full-calendar CAGR | strict MDD | CAGR/MDD | trades |
|---|---:|---:|---:|---:|---:|
| COGR standalone, every cell | 0.0000% | 0.0000% | 0.0000% | 0.0000 | 0 |
| unscaled Gross9, 2023 H2 | 1.4549% | 2.9087% | 19.9781% | 0.1456 | 93 |
| Gross9 + inactive COGR | 1.4549% | 2.9087% | 19.9781% | 0.1456 | 93 |

The largest same-configured-gross ratio difference was only `+0.01920`, below
the frozen `+0.05` gate. It cannot be attributed to COGR PnL because COGR
never entered; the candidate path was identically flat.

Every cell failed:

- positive standalone return;
- minimum 25 trades;
- both long and short share;
- standalone CAGR/MDD;
- stressed standalone return;
- MDD reduction;
- same-gross ratio improvement;
- best-control margin.

## Why no trade survived

The H1 score calibration was uniformly negative. The preregistered threshold
was `max(0, prior-fold q75)`, so every variant used threshold zero:

| variant | raw H1 q75 | applied threshold |
|---|---:|---:|
| primary | -0.011329 | 0.000000 |
| QQQ only | -0.011106 | 0.000000 |
| GLD only | -0.011071 | 0.000000 |
| prior only | -0.011325 | 0.000000 |
| one-session stale open | -0.011211 | 0.000000 |
| weekday only | -0.010974 | 0.000000 |

The unrestricted primary schedule also had zero entries, proving this was not
caused by a Gross9 coordination gate. Lowering the threshold after seeing the
result would be post-selection repair and is forbidden.

## Isolation and decision

The 2024 eval command failed closed with
`selection artifact did not freeze a passing top1`; it created no eval
artifact. Therefore no 2024 COGR feature, target, control, or metric was
opened by the formal run.

An independent reviewer had executed an earlier pre-commit selection smoke
and observed the same rejection. Only phase-isolation and artifact-integrity
bugs—not learner, features, thresholds, costs, weights, or performance
gates—were changed afterward. The conservative conclusion is still:

**COGR-12 is rejected, is not promoted into the live Gross9 portfolio, and
the QQQ/GLD cash-open shallow-tree family is closed.**
