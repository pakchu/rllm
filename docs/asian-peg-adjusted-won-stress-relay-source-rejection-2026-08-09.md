# APWSR-12 terminal source-support rejection

APWSR-12 produced `12/29/24/13` train/test/eval/final events with every
minimum-count and side-balance gate satisfied. Test, eval, and final also
passed month concentration. Train did not: six of twelve events occurred in
one month, so its `0.50` maximum month share exceeded the frozen `0.45` limit.

Two executions reproduced identical terminal artifacts:

- source-session SHA-256: `56aa4a416b5a61aacc7a161942794700e99d0e38005796598a0175012b01a980`
- clock SHA-256: `ee0d0d0f92fc322ab7b070605cc395485557785958941ce69de6d46b54b9f9c1`
- result SHA-256: `472fa25ba0585ff7a5f83c46eda1ab1a5b1368c656c4c4c55083073bd6f981cc`

APWSR-12 is rejected unchanged. Gross9 rows, execution prices, funding,
post-entry outcomes, economics, and RV20 remain unopened. No pair, spread,
rank, side, session, clock, hold, subset, or control may be repaired.
