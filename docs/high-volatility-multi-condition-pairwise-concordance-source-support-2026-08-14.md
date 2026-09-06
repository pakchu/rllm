# HVMCPAC-8 source-support result

The six preregistered pairwise intersections were materialized from hash-bound
component clocks without opening entry/exit prices, funding, post-entry returns,
PnL, or Gross9 comparator rows. The build and its ten targeted tests reproduced
twice byte-for-byte.

Only `CARSC-8__AND__HVTFR-8` passed every all-stage source gate. Its event counts
were train 22, test 29, eval 23, final 11; minority-side shares were 40.9%, 37.9%,
47.8%, and 27.3%; maximum month shares were 31.8%, 17.2%, 21.7%, and 36.4%.

The other five frozen pairs are terminal source-support rejections. Four fail the
final-period month-concentration ceiling (46.2% or 50.0%); the remaining sparse
pairs also fail minimum-event gates. They cannot be repaired or substituted into
the eligible set.

The deterministic support result SHA-256 is
`dd2e185fd924ce60eda3bd9c0c4fb1813ec79ff696cf3e6011182ff5c09293c6`.
The sole eligible pair advances to unchanged Gross9 novelty; economic outcomes
remain sealed.
