# DCLB-864 pre-incidence amendment — control-only macro balance

## Status

This is a prospective, outcome-blind amendment to the DCLB-864 mechanism.
It is committed before any DCLB source value row, joint-state incidence,
comparator row, BTC row, funding row, return, or PnL is decoded.

The original mechanism and write-once preregistration v1 remain immutable.
This amendment and a new write-once preregistration v2 become the complete
effective protocol for the first source-support execution.

## Discovered implementation gap

The frozen component-only controls permit a causally valid H.4.1 rank, ON RRP
rank, and H.8 state while choosing direction from only one component or H.8.
At such a control-only timestamp, the two macro terms can oppose and cancel
exactly:

```text
13*h41_center_num - 104*rrp_center_num == 0
```

The primary is already ineligible at zero and is unchanged. However, the
original symbolic relation vocabulary did not name this state for a
component-only control clock. Silently dropping the control row would add an
unregistered control eligibility threshold; assigning either component as
dominant would be false.

## Exact amendment

Add one diagnostic relation token:

```text
MACRO_BALANCED_OPPOSITION
```

It is emitted if and only if:

- H.4.1 and ON RRP relief signs are both non-neutral and opposite; and
- `macro_integer == 0`.

The token is allowed only on `h41_only`, `rrp_interval_only`, or `h8_only`
control rows whose own frozen direction remains nonzero. It may also appear on
the exact NSA control only if that control is otherwise valid, though the NSA
control still requires the nonzero primary macro side and therefore cannot
produce it under the current mechanism.

The token:

- does not make a primary row eligible;
- does not change any rank, side, hold, reservation, support gate, comparator,
  economic gate, or outcome boundary;
- is forbidden from the primary RLLM prompt and action path; and
- cannot be selected, promoted, or removed after incidence.

All original primary relation tokens and definitions remain unchanged.

## Evidence boundary

This amendment was motivated by static algebra and adversarial implementation
review only. No source/comparator/market row or aggregate DCLB incidence was
opened. No threshold or direction was chosen from an outcome.
