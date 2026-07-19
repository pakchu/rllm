# BitMEX XBt insurance-fund source freeze — 2026-07-20

The IFAR source was downloaded only after the mechanism, downloader, exact
rule, support gate, embargo, and hold were committed in `02944eb`.

## Frozen source audit

- endpoint: `GET https://www.bitmex.com/api/v1/insurance`;
- currency: `XBt` only;
- interval: `[2018-01-01, 2023-01-01)`;
- pagination: 500, 500, 500, and 326 rows;
- selected rows: 1,826 of 1,826 expected calendar days;
- first timestamp: `2018-01-01 12:00:00 UTC`;
- last timestamp: `2022-12-31 12:00:00 UTC`;
- duplicates, missing days, off-noon rows, other currencies, and non-positive
  balances: none;
- raw gzip SHA-256:
  `523d179d4a4ac51e3ebf5ce24f188f23cda02f31d8f879e0d256361af333c6dc`;
- source manifest hash:
  `4c751b96a4d877bc558bf37e693396fc529326feb050862ea5ddb9100cde8612`;
- source manifest file SHA-256:
  `c9b8df43a07a5f6887cc43dea300698af7d455c70bfc582899504fa3eb6dda6e`.

The raw 18 KB source remains ignored and local under the data-use boundary.
Only this audit and the hash-bound manifest are committed. No balance change,
candidate incidence, BTC post-decision return, funding, PnL, CAGR, or strict
MDD was inspected while freezing the source.

Official references:

- [BitMEX insurance history endpoint](https://docs.bitmex.com/api-explorer/get-insurances)
- [BitMEX API overview](https://docs.bitmex.com/api-explorer/bitmex-api)
- [BitMEX Exchange Rules](https://www.bitmex.com/legal/exchange-rules)
- [BitMEX Terms of Service](https://www.bitmex.com/terms)
