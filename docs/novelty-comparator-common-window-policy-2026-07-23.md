# Prospective novelty comparator common-window policy — 2026-07-23

## Scope

This policy applies prospectively to candidate identities preregistered after
this document is committed. It does not amend, repair, or reinterpret any
already-frozen candidate.

The repository previously treated any comparator interval extending past a
candidate's comparison window as a malformed comparator. During source-only
RMSR verification, a valid prior comparator trade entered in late 2023 and
exited in early 2024, exposing that this rule confounded artifact validity with
common-window eligibility. No RMSR or successor market outcome was opened.

This timing fact is disclosed contamination. The policy below is universal and
candidate-independent: it is fixed before any successor candidate clock or
overlap metric is computed.

## Frozen raw-artifact validation

For every hash-bound comparator artifact:

1. verify exact file SHA-256 before parsing;
2. require the exact preregistered header and group parser;
3. parse every row in the artifact, including rows outside the comparison
   window;
4. require timezone-aware entry and exit times and side in `{LONG, SHORT}` or
   exact numeric equivalents;
5. require `exit_time > entry_time` for every included group row;
6. require unique entry times and chronological non-overlap inside each raw
   comparator group over the entire artifact; and
7. fail closed on missing files, hash/header/parser drift, invalid fields,
   duplicate entries, overlap, or an empty required artifact/group.

Rows cannot be silently dropped before these validations.

## Frozen common-window eligibility

For a preregistered comparison window `[W0, W1)`, an interval is eligible for
novelty metrics only when:

```text
entry_time >= W0 and exit_time <= W1
```

An interval ending before `W0`, starting at or after `W1`, or crossing either
boundary is excluded whole. It is never clipped, shifted, shortened, split, or
assigned a partial side/exposure. The report must record, for every comparator
group:

- total raw rows parsed;
- fully contained rows used;
- rows before the window;
- rows after the window; and
- rows crossing a boundary.

A preregistration may require a minimum in-window group count. A required
group with zero fully contained rows fails closed. Merely having valid rows
outside the comparison window is not a failure.

## Frozen metric rules

Candidate events are subjected to the same full-containment rule. Novelty
metrics use only fully contained candidate and comparator intervals and the
exact preregistered common window:

- exact-entry Jaccard on contained entry timestamps;
- deterministic one-to-one entry matching under the preregistered elapsed-time
  tolerance; and
- signed occupied-exposure correlation on a fixed grid over `[W0, W1)`.

The grid outside every contained interval is zero. A boundary-crossing interval
cannot contribute even one grid cell. Undefined correlation fails its check
unless the candidate preregistration explicitly froze a stricter behavior.

## Immutability and contamination disclosure

Every future candidate using this policy must:

- bind this document's SHA-256 in its preregistration;
- enumerate exact comparator paths, hashes, parsers, groups, window, and
  minimum counts;
- disclose that a prior cross-boundary comparator timing row motivated this
  prospective policy;
- read zero candidate outcome rows while preregistering; and
- forbid changing window eligibility after candidate incidence or overlap is
  opened.

This policy standardizes time support; it does not make previously viewed
comparator timing pristine and does not validate any candidate alpha.
