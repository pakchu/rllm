# HVCKIHR-8 terminal source failure — 2026-08-10

HVCKIHR-8 stopped at its first scientific gate. The PostgreSQL perpetual table
covered the requested history, but the exact aligned `bars_binance_spot`
BTCUSDT one-minute source produced only 81 complete eight-hour boundaries, all
from 2026-07-05 through 2026-07-31. Consequently neither the 270/180 causal
impact-handoff rank nor the 270/180 variation rank was available at any row.

The frozen primary clock therefore contained zero events in train, test, eval,
and final versus required counts 8/12/12/8. This is a source-coverage failure,
not an economic result. Post-entry prices, funding cash, returns, PnL, Gross9
rows, CAGR, and MDD remained sealed.

An exact rerun was byte-identical for the source panel, source manifest, empty
primary clock, and support report. Per the preregistered stopping rule, no
history minimum, venue, formula, rank, clock, side, hold, subset, or diagnostic
control is changed. Gross9 novelty and economics remain unopened.
