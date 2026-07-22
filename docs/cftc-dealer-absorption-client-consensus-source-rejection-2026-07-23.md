# DAIC-168 source-stage rejection — 2026-07-23

## Decision

Retire **DAIC-168 — Dealer Absorption / Client Consensus** before opening any
DAIC BTC return, funding, PnL, CAGR, or drawdown value. No nearby threshold,
hold, magnitude, or participant-sign repair is authorized.

## Frozen idea

The candidate idea was stated before its exact incidence was computed:

- asset-manager and leveraged-money published weekly net-position changes must
  have the same nonzero sign;
- dealer published weekly net-position change must have the opposite sign;
- follow the asset-manager/leveraged-money consensus;
- use the already audited conservative CFTC availability timestamp, including
  the seven 2023 ION publication overrides;
- enter five elapsed minutes later, hold 168 elapsed hours at 0.5x, and skip
  releases whose trade would overlap the prior accepted trade.

This rule is mechanically distinct from CITA-1. CITA requires asset-manager and
leveraged-money changes to have opposite signs; therefore the raw DAIC and CITA
report-date sets are disjoint.

## Research-integrity boundary

A delegated read-only review computed exact DAIC source incidence before a
repository preregistration artifact had been committed. Although the candidate
identity already existed and the reviewer opened no market outcome, treating a
later support threshold as pristine would be false. The conservative response
is to record the incidence, apply only the already committed CITA statistical
support reference, and retire DAIC unchanged before economic evaluation.

## Reproducible source result

The audit reads only the hash-bound official CFTC TFF Bitcoin panel columns for
report date, conservative availability, dealer/asset-manager/leveraged-money
published changes, source completeness, and ION override status.

| Source-only quantity | Result |
|---|---:|
| Complete CFTC rows | 298 |
| Raw DAIC events, 2018–2023 | 63 |
| Accepted non-overlapping DAIC events | 61 |
| Overlap-suppressed ION events | 2 |
| Train 2020–2022 | 30 (15 long / 15 short) |
| 2020 split-contained | 7 |
| 2021 split-contained | 6 |
| 2022 split-contained | 16 |
| Selection 2023 | 25 (10 long / 15 short) |
| 2023 H1 / H2 split-contained | 10 / 14 |
| Raw report overlap with CITA | 0 |
| Accepted-entry Jaccard with CITA | 0.0 |

The unchanged CITA-1 preregistration required at least 75 Stage1 trades and 20
in every calendar year. DAIC has 30 Stage1 trades and only 7, 6, and 16 in the
three split-contained years. It is too sparse for the user's statistical-alpha
standard even before accounting for the formal preregistration boundary breach.

## Causal and execution checks

- source panel, build manifest, and source manifest SHA-256 values replay;
- exactly one first-report row remains quarantined;
- signal uses `available_time_utc`, never report date;
- entry waits exactly five minutes and exit is exactly 168 hours later;
- compressed ION releases are skipped under global non-overlap;
- DAIC and CITA raw reports and accepted entries are disjoint;
- BTC market rows read: **0**;
- funding rows read: **0**;
- return/PnL/CAGR/MDD values opened: **false**.

## Interpretation

The dealer-opposition condition may also be an accounting residual rather than
independent dealer information: when two client groups move together, another
participant group must absorb part of the change. The rule is novel relative to
CITA's event set, but not source-orthogonal and not sufficiently populated.

## Artifacts

- `training/audit_cftc_dealer_absorption_client_consensus_source.py`
- `tests/test_audit_cftc_dealer_absorption_client_consensus_source.py`
- `results/cftc_dealer_absorption_client_consensus_source_rejection_2026-07-23.json`

The next candidate must leave this neighboring CFTC sign-combination family and
must be committed before its exact source incidence is derived.
