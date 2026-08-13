# HVMCPAC-8 Gross9 novelty result

The sole source-supported pair `CARSC-8__AND__HVTFR-8` passed the unchanged
Gross9 structural novelty battery before any execution price, return, funding, or
PnL row was opened.

Worst observed comparisons across the complete Gross9 roster were:

- exact-entry Jaccard: 0.01266 (limit 0.10),
- one-to-one ±6h matched share: 0.11765 (limit 0.35),
- occupied 5m-bar Jaccard: 0.03291 (limit 0.25),
- absolute signed-exposure Pearson: 0.05546 (limit 0.35).

The evaluator and four targeted tests reproduced twice. The deterministic result
SHA-256 is
`de26f154dfa803b570f5b5d865b3e74e345aa3462a7f2236eaa6b79c8ffc9bda`.
The pair may now open train economics only; test/eval/final remain sealed until
the frozen train gate succeeds.
