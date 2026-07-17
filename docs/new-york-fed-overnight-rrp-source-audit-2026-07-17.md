# New York Fed overnight RRP source audit — 2026-07-17

## Decision

**PASS for outcome-blind preregistration.** The frozen panel contains 1,498
normal afternoon overnight reverse-repo operations from 2018 through 2023.
It reads zero BTC, funding, portfolio, or existing-alpha rows.

This source is structurally distinct from the repository's price, order-flow,
funding/premium, OI, Kimchi/FX, and participant-position families. It is also
different from the rejected weekly H.4.1 family: this panel records the actual
daily ON RRP operation result and its own operation clock rather than a weekly
balance-sheet snapshot.

## Official source contract

- New York Fed Markets Data API documentation:
  <https://markets.newyorkfed.org/static/docs/markets-api.html>
- Official API endpoint used by the builder:
  <https://markets.newyorkfed.org/api/rp/results/search.json>
- New York Fed reverse-repo FAQ and economic mechanics:
  <https://www.newyorkfed.org/markets/rrp_faq.html>

The official FAQ states that normal ON RRP operations run every business day
from 12:45 to 13:15 Eastern unless otherwise announced, and that the Desk
publishes total submitted, total accepted, and the award rate after the
operation. It also explains that ON RRP settlement moves cash from reserves to
the Fed's reverse-repo liability, reducing reserve balances by the same amount.

## Causal availability and quarantine

The API also contains occasional morning operational-readiness exercises.
Those are not the normal daily facility and are excluded mechanically by
requiring the published close time to be at or after noon. This leaves exactly
one normal afternoon operation per date.

Every result becomes usable at `closeTime + 15 minutes` in
`America/New_York`. A downstream strategy must wait one additional complete
five-minute bucket before entry. The 15-minute publication allowance is more
conservative than treating the operation close itself as tradable.

The historical API is not a formal point-in-time revision archive. Therefore:

- exact annual JSON responses are frozen and hash-bound;
- a row whose API `lastUpdated` date is later than `operationDate` is retained
  only as a clock row;
- all amount and counterparty values on that row are blanked;
- a later feature builder must fail closed and must not bridge a rolling
  baseline across that row.

This quarantines nine rows. It removes known later-update risk but cannot prove
that every same-day metadata update left every initially published field
unchanged. That residual archive-vintage limitation must remain disclosed.

## Frozen coverage

| Year | Normal rows | Complete | Quarantined | First | Last |
|---:|---:|---:|---:|---|---|
| 2018 | 249 | 247 | 2 | 2018-01-02 | 2018-12-31 |
| 2019 | 250 | 249 | 1 | 2019-01-02 | 2019-12-31 |
| 2020 | 251 | 249 | 2 | 2020-01-02 | 2020-12-31 |
| 2021 | 250 | 247 | 3 | 2021-01-04 | 2021-12-31 |
| 2022 | 249 | 249 | 0 | 2022-01-03 | 2022-12-30 |
| 2023 | 249 | 248 | 1 | 2023-01-03 | 2023-12-29 |
| **Total** | **1,498** | **1,489** | **9** | 2018-01-02 | 2023-12-29 |

All retained operations are fixed-rate, overnight, same-day-settling reverse
repos backed by Treasury collateral. Operation totals reconcile exactly to the
single Treasury detail row. Duplicate normal operation dates are rejected.

## Artifacts

- Builder: `training/build_new_york_fed_overnight_rrp.py`
- Tests: `tests/test_build_new_york_fed_overnight_rrp.py`
- Panel:
  `data/new_york_fed_overnight_rrp_2018_2023/new_york_fed_overnight_rrp_2018-01-01_2023-12-31.csv.gz`
- Build manifest:
  `data/new_york_fed_overnight_rrp_2018_2023/build_manifest.json`
- Panel SHA-256:
  `49f67ed44b7eb81fd35c17a8209cf14d6a8019d7e9f77fce8c343d1a7fb66b27`
- Manifest hash:
  `de6708a85fd7626e19adb48bf89a27cf2e50cbc09f8caddb9a6f67c03ca7140a`

An offline `--from-snapshot` rebuild must reproduce the panel and manifest
byte for byte. Any live annual response hash change fails closed.

## Authorized next step

One source-only daily liquidity-flow candidate may be preregistered before any
BTC outcome is opened. The candidate must freeze its strictly-prior
normalizer, tail rule, direction, entry and exit clocks, controls, costs,
strict-MDD implementation, and sequential Stage1/Stage2 gates. Source
orthogonality is not profitability evidence; return overlap is measured only
after standalone economic gates pass.
