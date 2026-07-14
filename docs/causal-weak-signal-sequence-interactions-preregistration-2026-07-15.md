# Causal weak-signal sequence interactions — preregistration (2026-07-15)

## Research question

Can two weak but causally available microstructure mechanisms become tradable when
they occur in a specific order, even though neither mechanism clears costs as a
standalone sleeve?

This experiment does **not** allocate among sleeves and does **not** globally vote
their directions. A trade remains the later trigger policy's single trade. An
earlier, economically distinct primary signal only decides whether that trade is
admitted.

## Data boundaries

- Fit and rank candidates on 2020-01-01 through 2022-12-31 only.
- Report 2023, 2023 H1, and 2023 H2 only after the train champion is fixed.
- Earlier experiments have already exposed 2023 aggregate outcomes. Therefore
  2023 is a reusable development confirmation split, **not** pristine OOS.
- Calendar 2024 and later remain sealed. The first genuinely untouched OOS split
  for this hypothesis family is 2024.
- Every signal is computed from a completed five-minute bar. Entry is the next
  five-minute open. Exit and hold are inherited unchanged from the trigger's
  frozen primary schedule.

## Promotable primitives

Only the six frozen **primary** causal clocks are promotable:

- CSPR: cash-sponsored perpetual rejection
- CATCH: cash auction transfer/catch-up handoff
- CLASP: cash late-arrival spillover propagation
- LURI: leveraged USD-M inventory release handoff
- RIFT: refill-inference flow topology
- UMFR: USD-M forced-flow reversion

Direction flips, delayed signals, stale signals, component-removal policies, and
other controls are falsification controls. They may not become live primitives.

## Frozen interaction operators

The antecedent may occur on the trigger bar or during the preceding causal
lookback. Lookbacks are fixed to 12 and 36 five-minute bars.

1. **O1 — cash-confirmed derivative transition**
   - Trigger: UMFR or LURI primary.
   - Antecedent: CSPR, CATCH, or CLASP primary.
   - Require the antecedent trade direction to equal the trigger direction.

2. **O2 — ordered cash propagation chain**
   - Trigger and antecedent are distinct members of CSPR, CATCH, and CLASP.
   - Require equal direction and preserve temporal order: antecedent then trigger.

3. **O3 — cash-confirmed refill continuation**
   - Trigger: RIFT primary, long side only.
   - Antecedent: CSPR, CATCH, or CLASP primary, same direction.
   - Veto the trade if UMFR or LURI emitted the opposite direction during the
     prior 12 bars.

4. **O4 — refill-confirmed derivative transition**
   - Trigger: UMFR or LURI primary, long side only.
   - Antecedent: RIFT primary, same direction.

No side-specific rule other than the structurally declared long-only O3/O4 rules,
no threshold optimization, no dynamic exit, and no additional regime gate is
allowed in this hypothesis family.

## Execution and risk contract

- Leverage: 0.5x.
- Fee: 5 bp per notional side.
- Slippage: 1 bp per notional side.
- Realized USD-M funding is applied over the actual holding interval.
- Same-trigger schedules remain non-overlapping; filtered schedules may not add or
  duplicate a trade.
- Strict MDD uses the conservative held-bar ordering of favorable extreme before
  adverse extreme, includes entry cost and funding debits, and excludes the exit
  bar high/low because the exit occurs at its open.
- CAGR uses the complete wall-clock split, including inactive periods.

## Train-only selection

Candidates are ranked using 2020-2022 only. The primary champion must satisfy:

- positive absolute return;
- CAGR / strict MDD at least 3.0;
- strict MDD no greater than 15%;
- mean gross underlying move greater than 12 bp;
- at least 80 effective trades total and 15 in each calendar year;
- weekly-cluster one-sided sign-flip p-value below 0.10.

Among passing candidates, rank by the minimum of the three calendar-year net
mean trade returns, then full-train CAGR/strict-MDD, then trade count. Freeze one
champion only. Shadow candidates are diagnostic and cannot replace a champion
after seeing 2023.

## 2023 development confirmation

The frozen champion passes development confirmation only if:

- full-2023 absolute return is positive;
- both H1 and H2 absolute returns are positive;
- full-2023 CAGR / strict MDD is at least 3.0;
- full-2023 strict MDD is no greater than 15%;
- at least 40 full-year trades and 12 trades in each half;
- full-2023 mean gross underlying move is greater than 12 bp;
- the result remains positive under 8 bp fee-plus-slippage per notional side.

Failure rejects this operator family before any 2024+ outcome is opened.

## Falsification controls

For the frozen champion, evaluate matched controls on the same trigger clock:

- flip the executed side;
- reverse antecedent/trigger order where defined;
- delay the trigger one five-minute bar while preserving hold length;
- use antecedent age 13-24 hours instead of the causal short lookback.

The primary is invalid if a logically destructive control matches or exceeds its
minimum train/2023 CAGR-to-strict-MDD score.

## Stop condition

Do not open 2024+ unless one train-only champion passes every train and 2023 gate
and beats its matched controls. A train winner that fails 2023 is evidence that
this interaction family is regime-unstable, not permission to select a shadow
candidate.
