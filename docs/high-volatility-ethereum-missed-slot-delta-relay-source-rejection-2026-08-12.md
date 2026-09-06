# HVEMSD-24 source rejection

The first frozen source execution failed while collecting historical Ethereum
boundary headers from `eth-mainnet.public.blastapi.io`.  Four attempts to call
`eth_getBlockByNumber` ended with HTTP `429 Too Many Requests`.

This is terminal under the preregistered first-source-failure rule.  Changing
the RPC host set, cadence, retry policy, batching, source method, date range, or
resuming from partial in-memory headers would repair the source after observing
its failure.  None is permitted.

Only a partial header prefix existed in memory.  No complete daily boundary
panel, missed-slot count, change, rank, candidate event, BTC preentry variation,
Gross9 row, execution price, funding value, return, or PnL was opened or
published.  The search must move to an independent source-native mechanism.
