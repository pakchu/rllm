# BitMEX zero-tick frustration source-feasibility rejection — 2026-07-20

## Decision: reject before implementation

The proposed **zero-tick frustration** axis is rejected at the production
source-entitlement gate. It would classify aggressive XBTUSD trades whose
reported `tickDirection` failed to advance through the opposite quote, but the
project's target is profit-seeking live trading and the repository has no
written commercial-data consent from BitMEX.

BitMEX's official Terms of Service, clause 18.6(a), limit the Services and
Trading Platform to personal use and require the Company's prior written
consent for non-personal, public, or commercial use. The same clause states
that the restrictions apply to data made available through APIs. The public
archive and unauthenticated REST surface therefore do not establish a
production-use entitlement.

Official references reviewed on 2026-07-20:

- [BitMEX Terms of Service — June 2025, clause 18.6](https://static.bitmex.com/documents/Terms_of_Service__June_2025.pdf)
- [BitMEX API overview](https://www.bitmex.com/app/apiOverview)
- [BitMEX Get Trades API](https://docs.bitmex.com/api-explorer/get-trade)

This is an entitlement failure, not evidence for or against the proposed
market mechanism. The axis may not be used in research that is intended to
graduate silently into this live portfolio.

## Bounded source-only probe

Before the terms gate was closed, a bounded schema/coverage probe established
that historical XBTUSD trade rows are available through the REST endpoint and
daily public archives. A 2020 response exposed the fields `timestamp`,
`symbol`, `side`, `size`, `price`, `tickDirection`, `grossValue`,
`homeNotional`, `foreignNotional`, `pool`, and `trdMatchID`. Archive HEAD
requests also found daily compressed trade files in 2020, 2023, 2025, and
2026. BitMEX's API overview recommends those daily historical extracts instead
of deep REST pagination.

The separately probed liquidation endpoint returned no historical rows for
bounded 2020 and 2023 requests and no rows in the unfiltered probe. It is not a
viable three-year historical liquidation source through that surface.

No complete source prefix was downloaded. No feature incidence, threshold,
event clock, BTC outcome, funding row, return, PnL, absolute return, CAGR, or
strict MDD was opened or calculated.

## Unopened mechanism sketch

The rejected hypothesis would have contrasted:

```text
buy_frustration  = aggressive buy volume with ZeroMinusTick
sell_frustration = aggressive sell volume with ZeroPlusTick
```

Persistent buy frustration could indicate exhausted upside impact and
tentatively map short; persistent sell frustration could indicate absorbed
downside impact and tentatively map long. This interpretation was never
implemented or tested, and `tickDirection` alone is not proof of aggressor
intent, queue depletion, or hidden liquidity.

## Reopening condition

This axis can be reconsidered only after BitMEX grants prior written consent
that explicitly permits the historical research, derived signals, and
profit-seeking live-trading use. Reopening would then require a fresh
source/mechanism decision and an outcome-blind support preregistration. The
bounded probe cannot be reused as an implicit commercial licence.
