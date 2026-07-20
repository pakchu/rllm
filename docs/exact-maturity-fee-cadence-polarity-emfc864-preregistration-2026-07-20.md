# EMFC-864 source-support preregistration — 2026-07-20

## Status

**Frozen before EMFC event incidence or any BTC market/funding outcome was
opened.** The generator validated only hashes, schema metadata, source audit
counters, the mechanism decision, and existing source-clock identities. It
did not decompress or parse the frozen block CSV.

The broader design process was not source-diagnostic pristine. A read-only
architecture review had already observed that daily matured fees can shadow
same-day fees and that daily cadence can shadow block count. It did not inspect
EMFC event counts, sides, market prices, future returns, funding, PnL, CAGR, or
drawdown. The frozen source-independence gates are therefore conservative
falsification checks, not a pristine source-discovery claim.

Artifacts:

- preregistration JSON:
  `results/exact_maturity_fee_cadence_polarity_preregistration_2026-07-20.json`;
- JSON SHA-256:
  `43f1505786ad5ddd8a076afebccc26bff65387d8ef9b7a443035136606157ff6`;
- canonical manifest hash:
  `487a4c0dd3aa501605274f0afaacb6714c668078e6fac0506798afa4f9b0d743`;
- policy hash:
  `a264e58f834f2a58dda9ddcf3dcf5035ef941cd2087124b3ea1c8c306559b92f`;
- generator:
  `training/preregister_exact_maturity_fee_cadence_polarity.py`;
- generator SHA-256:
  `5e8f1e9b857b0cb2a248738e69d97d8ec1c5e4c80410f0362b1dd89656333c1b`;
- mechanism decision SHA-256:
  `a640d13f02b23b0c76d5acb73427be0ad6fb87a3d08f9cd392a47b22f2918a39`.

Two consecutive generation runs reproduced the preregistration JSON
byte-for-byte.

## Singleton block-level policy

For each canonical maturity height `h`:

```text
origin_height = h - 100
maturity_height = h
confirmation_height = h + 6
matured_fee_component = total_fees[h - 100]
fee_pressure = log1p(matured_fee_component)
maturity_elapsed_seconds = mediantime[h] - mediantime[h - 100]
cadence_compression = -log(maturity_elapsed_seconds / 60000)
```

A height is valid only when every required height exists, the matured fee is
non-negative, and maturity elapsed time is positive. The exact frozen prefix
permits 212,989 candidate heights before any normalization or event rule.

Each channel receives a strict-prior empirical midrank over the last **26,208
valid maturity heights** strictly below `h`, nominally 182 days at 144 blocks
per day. The current height, confirmation heights, invalid rows, future rows,
and full-sample statistics never enter its reference.

Frozen states:

- `fee_rank >= 0.90` and `cadence_rank >= 0.90`:
  high-pressure/compressed state, **short**;
- `fee_rank <= 0.10` and `cadence_rank <= 0.10`:
  low-pressure/expanded state, **long**; and
- otherwise neutral.

A signal is only the onset of an extreme state relative to the immediately
preceding valid state. Invalid rows cannot manufacture a transition. Candidate
onsets are processed by increasing height; the earliest non-overlapping onset
is accepted and all onsets before its scheduled exit are suppressed without
price- or PnL-based replacement.

There is one reference length, one pair of tail thresholds, one direction, one
confirmation rule, and one hold. No grid is authorized.

## Availability and execution

For maturity height `h`:

```text
raw_available = max(timestamp[h:h+6]) + 7200 seconds
decision_boundary = ceil raw_available to a 5-minute UTC boundary
entry_time = decision_boundary + 5 minutes
exit_time = entry_time + 864 five-minute bars
```

This waits through six successors, a two-hour historical header-time embargo,
and one complete latency bar. The support calendar belongs to `entry_time`,
not `h-100`, its day, or header time `h`.

Any later strict evaluator is fixed to 0.5x notional, next-open five-minute
execution, exact entry-inclusive/exit-exclusive funding at fixed entry
quantity, 6 bp/notional/side base cost, 10 bp/notional/side stress cost,
full-calendar CAGR, and global/pre-entry-high-water-mark strict MDD.

## Source-only support gates

Before any price or funding value may be opened, all of these must pass:

- exact 212,989 candidate maturity heights;
- at least 99.95% positive maturity elapsed spans;
- no invalid elapsed run longer than 12 heights;
- complete six-successor containment;
- contiguous, hash-linked, unique, pre-2024 source rows;
- non-negative total fees; and
- zero market, funding, return/PnL, or post-2023 source rows.

The non-overlapping primary clock must also satisfy:

- 2021-2022: 60-200 events, at least 24 per year, and at least seven of each
  side in each year;
- 2023: 24-105 events, at least ten per half, three per quarter, and at least
  three of each side in each half;
- each side is 25%-75% of train and selection separately;
- no month exceeds 20% of either window;
- no more than half of accepted gaps equal the exact 72-hour hold boundary;
  and
- median accepted-entry gap is at least 84 hours.

The upper counts and gap checks reject an effectively always-on state machine;
the lower counts reject an untestably sparse clock.

## Mandatory source falsification and novelty

Frozen internal shadows are fee-only, cadence-only, same-height fee,
99-block pseudo-maturity, 101-block pseudo-maturity, completed-day aggregate,
seven-day stale feature, direction flip, constant side, matched random, and
one-bar delayed clocks. Origin-height/day assignment exists only as a leakage
sentinel and can never become primary.

The candidate is rejected before outcomes when:

- absolute Spearman association exceeds 0.90 for a frozen source-feature or
  pseudo-maturity comparison;
- absolute five-minute exposure correlation exceeds 0.80 against any internal
  shadow; or
- absolute five-minute exposure correlation exceeds 0.35 against any frozen
  network-source comparator.

Frozen network-source comparator clocks and SHA-256 values:

- BATE-288:
  `cd4fbd01c104bd969ca1c12a53b8da82dd0e9376990e233c286ff009a5115c02`;
- UFCP-1:
  `8338c290d63b522531c8d55c8a79ba73cc13915c936733ec03ffcf6ab0e86c1b`;
- MCR-7:
  `2535244889b046ff00c369ee854973a91c23429dff82a6dd3c1a293a01352b0b`;
- NTB-7:
  `6b1bd7c7458cffa062e40872c3ad1730007c01426790b1ba8e52c6eb853de42f`;
- BFC-3:
  `edda7bb8ae8a1de4e51a3b86e98d533748e73d203125a3ded1a487e9a0e93632`.

Exposure comparisons use a flat-zero, five-minute UTC grid over 2021-2023.
Failure permits no lag, threshold, side, onset, latency, or hold repair.

## Sealed evaluation order

1. Run source integrity, event support, shadow, and novelty only.
2. Reject without market data if any source gate fails.
3. After an exact source pass only, commit and hash-freeze one strict outcome
   evaluator.
4. Open 2021-2022 train; stop on failure.
5. Open 2023 only after an exact train pass.
6. Keep 2024+ sealed and sequential.

Both train and 2023 would need positive absolute return, CAGR/strict-MDD at
least 3.0, strict MDD at most 15%, positive stress return, mean gross move at
least 30 bp, `p <= 0.10` weekly-cluster sign-flip evidence, positive long and
short contributions, positive required subperiods, and positive delayed-entry
performance. A failed singleton is rejected, not repaired.

