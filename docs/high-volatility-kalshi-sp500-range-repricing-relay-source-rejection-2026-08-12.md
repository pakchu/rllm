# HVKSRR-24 source-contract rejection — 2026-08-12

## Verdict

`HVKSRR-24` is terminally rejected without repair at its first historical
source-contract gate. The official Kalshi `KXINX` series query returned 1,094
event identities spanning `INX-22APR28` through the post-final
`KXINX-26AUG14H1600`, but the preregistered official event endpoint returned
`HTTP 404` for the listed in-window event `INX-23AUG01`:

```text
https://external-api.kalshi.com/trade-api/v2/events/INX-23AUG01?with_nested_markets=true
urllib.error.HTTPError: HTTP Error 404: Not Found
```

The failure occurred before a historical nested-market or candlestick payload
was returned. No historical candidate repricing, BTC outcome, funding row,
Gross9 row, clock, return, CAGR, or MDD was opened.

## Boundary

The frozen protocol requires each series-listed event to resolve through the
official individual-event endpoint before its complete mutually exclusive
ladder can be verified. A listed identity that cannot be replayed fails that
contract. Substituting `/historical/markets`, deriving market identities from
ticker grammar, changing the legacy lineage, shortening the interval, or
switching Kalshi series would change the preregistered source.

Therefore source support, novelty, economics, and RV20 q90 remain unopened.
No endpoint, alias, interval, ladder, anchor, side, hold, threshold, subset, or
control repair is authorized for this candidate.
