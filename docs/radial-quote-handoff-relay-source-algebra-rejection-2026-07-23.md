# RQHR-72 source-algebra rejection — 2026-07-23

## Decision

Reject **RQHR-72 — Radial Quote Handoff Relay** unchanged before feature,
arm, event, comparator-overlap, or market-outcome incidence.

The frozen support builder stopped at the first real-source algebra check. The
stored `skew_k_efficiency` did not equal `abs(skew_k_net) / skew_k_path` within
the preregistered absolute tolerance `5e-12`. The tolerance, source values,
race, thresholds, and gates were not changed after the failure.

The machine-readable rejection audit is:

- `results/radial_quote_handoff_relay_source_algebra_rejection_2026-07-23.json`
- file SHA-256:
  `92aa49128906e007beae1e7f65120741bc7942ee49371cbb264aee6313a63167`
- manifest hash:
  `bb20b4fb60f0f9f166eedbf00d4102d8f5f9787d8c844842114967d96d83cc55`

## Execution order and closed boundaries

The support builder and its full synthetic-null artifact were committed before
the real source was opened:

- support builder commit: `0a05f3f`;
- synthetic-null commit: `77d9436`;
- all five synthetic scenarios: zero raw confirmations and zero accepted
  events;
- real-source columns read during the synthetic stage: zero;
- comparator and market/outcome rows read during the synthetic stage: zero.

The first real-source run then failed in `validate_algebra()` while loading the
panel. A separate outcome-blind rejection audit scanned only the exact frozen
RQHR projection to characterize that failure. It did not derive RQHR features,
arms, confirmations, events, comparator overlap, returns, funding, PnL, CAGR,
or MDD.

The contamination boundary is now explicit:

- all 2023 `skew_2..5_{net,path,efficiency}` source values have been opened for
  algebra diagnostics;
- RQHR feature/event incidence remains unopened;
- comparator overlap remains unopened;
- every market and performance outcome remains unopened.

## Frozen failure evidence

The exact panel contained 105,120 five-minute rows:

- complete rows: 101,956;
- incomplete rows: 3,164;
- complete radius-level algebra observations: 407,824;
- failed observations: 307, all `efficiency_ratio` failures;
- radius 2 / 3 / 4 / 5 failures: 41 / 79 / 80 / 107;
- maximum absolute efficiency discrepancy:
  `8.6537675544275273e-12`.

The first failure was at source position 256
(`2023-01-01 21:20:00`, radius 3):

- stored net: `0.000104385510952`;
- stored path: `0.000132558095267`;
- stored efficiency: `0.787469907005`;
- recomputed efficiency: `0.7874699069999877022297528001`;
- absolute discrepancy: `5.0122977702471999e-12`.

## Root cause

The source builder computed net, path, and efficiency in binary floating point,
then reused `_write_gzip_csv()` from
`training/build_binance_aggtrade_microstructure.py`. That writer serializes
each float independently with `%.12g`. Recomputing the ratio from independently
rounded decimal text can differ from the independently rounded stored
efficiency by more than `5e-12`, even when the pre-serialization float algebra
was internally consistent.

This is a protocol-design failure, not evidence for or against the economic
handoff hypothesis. The preregistration should have bound either a source
serialization-aware tolerance, a higher-precision source artifact, or one
canonical primitive representation from which efficiency is derived. Because
none of those repairs was frozen before source values were opened, applying
one now would be a post-observation candidate amendment.

## Consequence for subsequent alpha research

RQHR-72 is retired and cannot be rescued by relaxing `5e-12`, recomputing the
stored efficiency, dropping the 307 rows, or rebuilding the source in place.
Any successor must have a new identity and disclose that this source family's
raw net/path/efficiency values and serialization failure statistics are now
known. Prefer a new source axis; if this source family is ever revisited, freeze
serialization-aware algebra from source-code analysis before opening event
incidence and do not claim a pristine discovery.
