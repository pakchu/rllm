# BLS CPI release-breadth source audit — 2026-07-17

## Decision

**PASS for outcome-blind alpha preregistration.** The frozen panel contains 60
monthly CPI releases from 2019 through 2023 and reads zero BTC, funding, or
existing-alpha rows.

## Official source contract

- BLS identifies `CUUR0000SA0` as the regular, U.S.-city-average,
  not-seasonally-adjusted all-items CPI-U series:
  <https://www.bls.gov/cpi/factsheets/cpi-series-ids.htm>.
- BLS states that CPI-U and CPI-W are final when issued except for rare explicit
  corrections:
  <https://www.bls.gov/cpi/questions-and-answers.htm>.
- Historical publication dates and the fixed 08:30 Eastern clock come from the
  official yearly BLS calendars, for example:
  <https://www.bls.gov/schedule/2022/home.htm>.
- Point-in-time headline and core 12-month values come from each official BLS
  archived CPI release, for example:
  <https://www.bls.gov/news.release/archives/cpi_09132022.htm>.
- The independent final-index checks use the Federal Reserve Bank of St. Louis
  FRED mirrors `CPIAUCNS` and `CPILFENS`:
  <https://fred.stlouisfed.org/series/CPIAUCNS> and
  <https://fred.stlouisfed.org/series/CPILFENS>.

## Retrieval boundary

The BLS CDN returned HTTP 403 to the local automated client. The builder used
`r.jina.ai` only as a text-rendering transport. Each payload must identify the
expected official `bls.gov` URL; publication dates must match the official
calendar; archived release dates must match their URL; and the two published
12-month values must agree within rounding tolerance with independently
downloaded FRED index ratios. A transport payload alone cannot pass the build.

## Audit results

| Check | Result |
|---|---:|
| Rows | 60 |
| Reference months | 2018-12 through 2023-11 |
| Release clock | 2019-01-11 13:30 UTC through 2023-12-12 13:30 UTC |
| Expected releases per calendar year | 12 / 12 / 12 / 12 / 12 |
| Headline maximum FRED absolute error | 0.049780 percentage points |
| Core maximum FRED absolute error | 0.049245 percentage points |
| Failed FRED cross-checks | 0 |
| Incomplete source rows | 0 |
| Market/funding rows read | 0 |

Eastern daylight-saving conversion is explicit: 08:30 Eastern is 12:30 UTC
during daylight time and 13:30 UTC during standard time.

## Frozen artifacts

- Builder: `training/build_bls_cpi_release_breadth.py`
- Tests: `tests/test_build_bls_cpi_release_breadth.py`
- Panel:
  `data/bls_cpi_release_breadth_2019_2023/bls_cpi_release_breadth_2019_2023.csv.gz`
- Panel SHA-256:
  `d199f409952d8cb83218864d0a96573bed82b59e649067b22fc97580a06d1059`
- Source manifest SHA-256:
  `7f889310707e4c490124ac2ce6817add7a227d6b0fa6d495c00405aba456aadc`

Raw schedule pages, all 60 archived releases, and both FRED CSVs are retained
compressed under `data/bls_cpi_release_breadth_2019_2023/raw/` (2.4 MiB total).

## Allowed next step

The source may now be used to freeze a source-only event clock. The 2020–2022
and 2023 BTC outcome windows remain unopened for this family.
