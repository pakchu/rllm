# HVEFPR-24 source-support pass — 2026-08-12

The unchanged preregistered `HVEFPR-24` Ethereum implementation-flaw-pressure
clock passes every frozen source gate without opening execution prices,
post-entry returns, funding, PnL, or Gross9 rows.

| split | events | long / short | minority share | max month share |
|---|---:|---:|---:|---:|
| train 2023H2 | 29 | 13 / 16 | 0.4483 | 0.3793 |
| test 2024 | 83 | 41 / 42 | 0.4940 | 0.1807 |
| eval 2025 | 53 | 26 / 27 | 0.4906 | 0.2264 |
| final 2026 through 2026-08-01 | 40 | 22 / 18 | 0.4500 | 0.3250 |

All minimum-incidence, 20% minority-side, and 45% maximum-month-share checks
pass. The exact full build was replayed twice with identical result, clock, and
source-manifest hashes.

- support artifact SHA-256:
  `fed76db422e146c704bdc5fc912efd01a158fe154f03835185322a28445be5b8`
- support manifest hash:
  `897a0658a43b84add4c0e4792d3fe9a29f36336c4435230b565fad6c5770c4fa`
- primary clock SHA-256:
  `7681e52ab8d73a8e1a35215848a1cbbfafb696f0491493d699fe56d5a9cbc34b`
- source manifest SHA-256:
  `20b7fb3f103741a41d761007eb46ed7799cc2c96c57a19816f9d70a2d6115893`

This pass authorizes only the frozen Gross9 novelty comparison. Diagnostic
controls cannot be promoted and economic outcomes remain sealed.
