# IVLIR-72 source-support rejection

## Verdict

**Retire IVLIR-72 without opening a post-entry outcome.** The exact frozen
intrinsic-volume first-passage implementation passed event-count, calendar
dispersion, and aggregate side-balance checks, but failed its maximum
same-direction-run guard.

No post-entry return, funding cash flow, PnL, absolute return, CAGR, strict MDD,
hit rate, or 2024+ source row was calculated. The strict evaluator and LLM
abstention stage are not authorized.

## Frozen source-only result

| Window | Events | LONG | SHORT | Longest same-side run |
|---|---:|---:|---:|---:|
| Train 2020–2022 | 180 | 62 | 118 | **26** |
| 2020 | 42 | 11 | 31 | 15 |
| 2021 | 67 | 10 | 57 | 22 |
| 2022 | 71 | 41 | 30 | 9 |
| Selection 2023 | 53 | 20 | 33 | 9 |
| 2023 H1 | 23 | 9 | 14 | 5 |
| 2023 H2 | 30 | 11 | 19 | 9 |
| All | 233 | 82 | 151 | **26** |

The frozen maximum was 15. The longest run was 26 consecutive SHORT events,
from the 2020-10-22 entry through the 2021-05-09 entry. Aggregate all/train/
selection side shares remained inside 25–75%, but that aggregate check hid a
long regime-dependent directional episode. This is exactly the failure the
run-length guard was intended to catch.

All other support checks passed:

- 180 train events versus the minimum 120;
- train-year counts 42 / 67 / 71 versus 30 each;
- 53 selection events, split 23 / 30 versus 15 each half;
- 44 active months;
- largest overall month share 4.29%;
- largest overall quarter share 9.87%; and
- clock schema contains timestamps and side only.

## Source funnel and controls

The exact causal build produced 1,197 valid first-passage anchors, 1,107 with
enough strictly-prior event history, and 233 primary events after flow,
alignment, impact-under-response, and rolling-range headroom gates.

| Clock | Events | LONG | SHORT | Longest run |
|---|---:|---:|---:|---:|
| primary | 233 | 82 | 151 | 26 |
| flow only | 415 | 176 | 239 | 25 |
| no under-response | 248 | 90 | 158 | 26 |
| no headroom | 372 | 157 | 215 | 25 |
| fixed noon | 71 | 25 | 46 | 15 |
| exact side flip | 233 | 151 | 82 | 26 |
| stale previous-anchor side | 233 | 101 | 132 | 8 |
| deterministic random side | 233 | 112 | 121 | 12 |

The same directional persistence remains after removing under-response or
headroom, so it is not an isolated implementation artifact. The stale-side and
random controls have shorter runs, but controls cannot repair a failed primary
identity.

## Rejection boundary

Do not change the flow q60, impact q70, 50% volume target, seven-day headroom,
side mapping, UTC-day origin, anchor cutoff, or six-hour hold and call it
IVLIR-72. Any anti-persistence filter or side-balancing rule would use the now
opened source incidence and create a new candidate.

The useful surviving result is structural: an equal-notional daily clock is
dense and calendar-balanced, but cumulative taker-flow sign can remain one-
sided for months. A successor must make a **state transition or sign handoff**
the event itself rather than filtering a persistent flow level.

## Integrity

- preregistration commit: `2edfc63`;
- preregistration manifest:
  `4aef96a9516d2e2506778cc7adb43eaaf2751f2ab87aad47dc69854c55ee9cda`;
- support report manifest:
  `2286c1368eb4a2cd8062d2ade34c88c7ba1185bf5e218cd8a80216af7c16bf42`;
- clock SHA-256:
  `523f24a0d955fe99cfb86c62078532c5fc9091234e6669ab9acff2a8f3367788`;
- clock rows across primary and controls: 2,038.
