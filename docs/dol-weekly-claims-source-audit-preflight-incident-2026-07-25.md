# DOL Weekly Claims Source Audit Preflight Incident

## Scope

This note records a non-authoritative preflight abort before the single
authorized `DOL-WCRV-D1` source attempt.

## Incident

- Time: `2026-07-25T02:13:22+09:00`
- Invoked commit:
  `4f5551bc8bef2489f03a5318944380d46e01fabe`
- Command:
  `.venv/bin/python training/audit_dol_weekly_claims_release_vintage_source.py`
- Exit stage: `_disk_guard()`
- Exception: `DiskGuardError`
- Network requests: none
- Sentinel created: no
- Manifest created: no
- Raw source directory created: no
- Aggregate report created: no

The boundary consumes the production attempt only when the exclusive sentinel
is created. The abort occurred before `reserve_attempt()` and before transport
construction, so it did not consume the authorized attempt or expose any
production source row.

## Root cause

The guard calculated used space as `total - free`. On this filesystem that
includes blocks unavailable to the current process and reported 342.42 GiB,
while `shutil.disk_usage(...).used` reported the actual filesystem use as
291.20 GiB. The frozen boundary requires use below 300 GiB and at least 8 GiB
free; the measured free space was 664.44 GiB.

## Repair

The guard now uses the standard-library `disk_usage().used` field directly.
A regression test supplies a filesystem result where `total - free` exceeds
300 GiB but `used` remains below the boundary, and requires the preflight to
pass. No source, parser, support threshold, inventory, or market/model boundary
was changed.
