# FMGRR-6 terminal source-contract rejection

FMGRR-6 was rejected before candidate incidence, Gross9 rows, execution prices,
or post-entry returns were opened.

The preregistration required every selected 08:00 UTC Binance settlement mark
to be finite and strictly positive with no imputation. The frozen source query
returned 1,401 target-hour rows, of which 303 had a nonpositive `mark_price`.
Those invalid target rows span 2023-01-01 through 2023-10-30 and therefore
intersect the causal history and train window.

Replacing the historical zero marks, falling back to another mark source,
starting later, or changing the settlement clock would alter the frozen source
contract. The candidate is therefore terminally rejected unchanged. No support
clock, novelty evaluation, or economic stage is authorized.
