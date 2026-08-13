# HVMDPAC-6 source-support rejection

All six preregistered 6h multi-domain intersections were materialized twice from
hash-bound clocks without opening prices, returns, funding, PnL, or combination
Gross9 rows. The 12-test preregistration/support suite passed twice and outputs
were byte-identical.

No pair passed all-stage source support. Train event counts were respectively
0, 0, 7, 4, 7, and 5 against the minimum of 8. The closest pairs,
`HVAFC-6__AND__RIVSCR-6` and `HVCBR-6__AND__RIVSCR-6`, each had seven train
events; the former also had only five test events, while the latter failed final
side balance and month concentration. Under the frozen no-repair rule, neither
may be relaxed or promoted.

HVMDPAC-6 is terminal before combination Gross9 and economics. Deterministic
support SHA-256:
`48bb1f78a7ab8532714d4fea4f687fa7625cacd28fef8900b2748b4a0c2dd8a1`.
