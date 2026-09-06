# HVQPLR-6 source rejection

The frozen `bars_binance_spot` source contract contains `BTCUSDT` one-minute rows but no `BTCUSDC` or `BTCFDUSD` rows. No candidate clock, Gross9 row, execution price, funding value, or economic outcome was opened.

Changing to a prior cache or another table would substitute the preregistered source. `HVQPLR-6` is therefore rejected unchanged at its first source-contract failure.
