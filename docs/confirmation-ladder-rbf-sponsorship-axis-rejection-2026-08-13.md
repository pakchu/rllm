# Confirmation-ladder RBF sponsorship axis rejection — 2026-08-13

## Decision

Reject before preregistration and before opening block transactions, source
incidence, BTC outcomes, Gross9 rows, execution prices, or funding values.

The screened object classified a confirmed non-coinbase transaction as
explicitly signaling BIP125 replaceability when any input had
`nSequence < 0xfffffffe`, compared transaction-weighted signaling shares in the
first and last three blocks of a six-block ladder, and proposed following the
completed BTC trend when the share migrated upward during high variation.

## Terminal confirmation-ladder collision

The source primitive is new to the repository, but the candidate mechanism is
not an independent research unit.  Both CLTDR and CLWMSR already freeze:

- a deterministic height-modulo confirmation ladder;
- six individual canonical block intervals;
- a first-three versus last-three composition comparison;
- source completion only after the terminal block is contained;
- price direction supplied by the late completed intervals; and
- a multi-hour continuation hold.

CLWMSR has already reached a terminal train-economic failure.  Replacing its
witness-share coordinate with an RBF-signaling share after that result, while
retaining the same ladder geometry and price-sponsorship story, would be a
post-failure source-feature substitution.  Changing the height modulus,
confirmation count, trend window, rank threshold, or hold cannot make that
repair admissible.

## Observable-to-mechanism mismatch

The proposed raw predicate identifies explicit opt-in signaling, not an
observed replacement or fee bump.  A low sequence can coexist with locktime
semantics, while inherited replaceability and actual mempool replacement
history cannot be reconstructed from the confirmed child transaction alone.
Calling the confirmed-sequence share “fee-rebid urgency” would therefore add a
behavioral interpretation not proven by the frozen observable.

Recovering actual replacement history would require contemporaneous mempool
state.  That historical transport is already unavailable under the repository
source boundary and cannot be substituted after this audit.

## Boundary record

- Repository code and prior documentation only were inspected.
- No block hash, transaction body, input sequence, source count, or timestamp
  was opened.
- No BTC price row, return, event incidence, Gross9 value, or PnL was computed.
- No formula, threshold, clock, side, hold, or universe was selected after
  seeing source incidence or outcomes.
- This exact confirmation-ladder RBF sponsorship concept is terminal and will
  not be repaired, inverted, or re-anchored.

