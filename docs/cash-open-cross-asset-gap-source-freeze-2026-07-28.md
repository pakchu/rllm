# COGR-12 outcome-blind source freeze

This work unit freezes a QQQ/GLD feature source **before any COGR BTC
post-entry outcome is opened**.

## Economic object

COGR-12 tests whether information arriving in the US cash opening auction
propagates into the continuously traded BTC market. At each common QQQ/GLD
session:

- the only current-session prices admitted are the raw QQQ and GLD opens;
- the feature becomes available at `09:35 America/New_York`;
- a later evaluator may enter BTC no earlier than `09:40 America/New_York`;
- every other price, range, volatility, correlation, beta, and volume feature
  ends at the preceding completed cash session.

This differs from the rejected cross-asset transfer batteries: those traded
QQQ/KODEX 200/GLD with translated BTC rules. COGR trades BTC from a new
cash-open information-arrival axis. It is also unrelated to the crypto
Spot→USD-M CATCH clock despite the shared word “cash.”

## Frozen source

- raw research caches:
  - QQQ SHA-256 `e9d0cbb6bbe41345f8897071198322f14f82f065c2f8ba0b9896be1ad434f162`
  - GLD SHA-256 `f564a4f7f4fb582dafc40a06a02b12bedd599f0300bf1874ce20bf9507ccd928`
- safe feature file:
  - `data/cash_open_cross_asset_gap_relay_pre2025/qqq_gld_cash_open_safe_features_pre2025.csv.gz`
  - SHA-256 `d0d04293cf05a7703b6970e5b97dc2e1e69ecb2d42de4b20c78958523e2e9c44`
- manifest:
  - `data/cash_open_cross_asset_gap_relay_pre2025/build_manifest.json`
  - SHA-256 `5f61359d837106acd84c2dc509600cda7b5d10522acd644c768bfb769d2909f8`
- feature-frame hash:
  - `2373443785dcdafb8466a53cbd296180d01dde2d923e193413d32631681b89bb`

The output has 5,063 common sessions, 5,002 valid rows, 31 features, and ends
at `2024-12-31`. Approximate frozen evaluation support is:

| role | sessions |
|---|---:|
| fit 2020-10-15 through 2022 | 557 |
| H1 2023 calibration | 124 |
| H2 2023 selection | 126 |
| 2024 untouched veto | 252 |

## Causal and corporate-action boundary

The builder truncates both normalized sources before `2025-01-01` **before**
rolling features are calculated. Tests mutate every future source value and
prove the emitted prefix is unchanged.

Provider OHLC is used on the split-normalized price basis present in the frozen
payload. Split events are audited but are not multiplied into prices again;
doing so would double-apply Yahoo's historical split normalization. Current
opening gaps and daily close returns add only effective-date cash dividends,
known by the cash open. The builder never reads adjusted close. A corporate
action occurring after a feature session cannot enter that session's formula.
A regression test also applies a uniform historical price-unit factor and
verifies all 31 ratio features are unchanged.

The evaluator will receive no current-session high, low, close, or volume
column. Tests mutate those forbidden values and prove the same-session feature
row cannot change. Mutating the current open changes only the declared opening
gap geometry.

## Production limitation

The frozen Yahoo payloads are suitable only for research reproducibility. A
live promotion is prohibited until an entitled QQQ/GLD cash-open feed
(for example an appropriately subscribed broker/exchange feed) reproduces the
signal clock and feature values under a frozen parity audit.

No BTC price, funding, return, PnL, Gross9 state, or post-entry outcome was read
while selecting or building this source.
