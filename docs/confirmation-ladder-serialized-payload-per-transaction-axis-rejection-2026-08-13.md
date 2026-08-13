# Confirmation-ladder serialized payload per transaction — source-blind rejection

## Decision

Reject `size / tx_count` as the next confirmation-ladder alpha object before
candidate preregistration, incidence, execution prices, funding, or post-entry
outcomes are opened. It is not an independent source axis; it is an algebraic
re-expression of two already opened block-composition axes.

For every valid block, let serialized size be `S`, block weight be `W`, and
transaction count be `N`. The proposed average serialized payload per
transaction is exactly

```text
S / N = (W / N) * (S / W).
```

`N / W` is the transaction-density primitive already frozen and economically
rejected in CLTDR-6. `S / W` is a monotone transform of the BIP141 witness-share
primitive already frozen and economically rejected in CLWMSR-6:

```text
Q = (4*S - W) / (3*S)
S / W = 1 / (4 - 3*Q).
```

Consequently, an early-versus-late migration in `S/N` would combine the inverse
CLTDR packing object with the already opened witness-composition object. BCRT
also freezes aggregate `log((sum(tx_count)+1)/(sum(weight)+1))` as its PACKING
primitive. Changing aggregation order, taking the inverse, or coupling the
result to the same late-three unanimous BTC-return condition does not create a
new economic observable.

## Evidence boundary

This audit read only committed mechanism definitions and terminal decisions.
It did not compute `S/N`, inspect its distribution or event incidence, build a
clock, read a new market row, or open any future return or PnL. No threshold,
side, hold, anchor, control, or alternative payload formula is authorized from
this rejected proposal.

The next candidate must use a genuinely independent immutable source object,
not another deterministic composition of block size, weight, and transaction
count.
