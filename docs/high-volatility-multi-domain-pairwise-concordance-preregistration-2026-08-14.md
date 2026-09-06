# HVMDPAC-6 multi-domain pairwise battery preregistration

A second, independent combination battery is frozen after HVMCPAC-8's terminal
train rejection. It does not reuse or repair that battery's components.

The fixed components are HVAFC-6 (perpetual aggressive flow), HVCBR-6
(cash/perpetual basis reversion), HVELR-6 (ETH directional leadership), and
RIVSCR-6 (realized-versus-implied volatility shock continuation). Each component
already has frozen source and Gross9 passes, exact 00:00/08:00/16:00 UTC
entry-at-+5m clocks, and a six-hour hold.

The family is exactly all six unordered pairwise AND rules. A pair requires exact
entry-time equality and exact strict-side equality; no tolerance, threshold
change, side reconciliation, clock change, or higher-order rule is permitted.
All six hypotheses count under Bonferroni (`raw weekly p<=0.10/6`). Source and
Gross9 precede train-only raw-rank-one selection; any rank-one or later-stage
failure terminates without substitution.

Standalone component outcomes are known, so this is exploratory discovery rather
than fresh confirmatory evidence. Pair incidence and pair outcomes were unopened
when this artifact was locked.

The preregistration reproduced byte-for-byte twice with six targeted tests passing
twice. SHA-256:
`ba21fd9cb7d89de2391497d7db8642d15eafe0a5973606be9c4528774938957b`.
