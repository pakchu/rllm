# CLCS-6 terminal source-support rejection

CLCS-6 was preregistered before exact ladder incidence was computed. The
source evaluator found zero accepted events in all four stages. This is a
source-contract failure rather than a threshold or directional result:
historical `bars_binance_spot` contains finite `quote_asset_volume` and
`taker_buy_quote` in only `39,501` of `1,710,713` queried BTCUSDT minutes.
Consequently only `25` of `4,630` height-modulo confirmation ladders had all
six complete aligned spot/perpetual intervals, and none reached the immutable
112-row prior-rank minimum.

Two executions reproduced the same artifacts:

- clock SHA-256: `516e8a74621c5bdb7569fca5ca9155ed29b39b87ef64667a852e654168b92377`
- result SHA-256: `9b46356b5adc0311bac927b8fa46ba5f6e899608ce63ad85e7a42d33991325bf`

CLCS-6 is rejected unchanged. Backfilling or substituting spot volume fields,
loosening completeness, shortening rank history, changing height anchors, or
removing ladder intervals would alter the frozen source contract. Gross9,
execution prices, funding, outcomes, economics, and RV20 remain unopened.
