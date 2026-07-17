# Cleveland Fed CPI-surprise source audit — 2026-07-18

## Decision

Freeze the Federal Reserve Bank of Cleveland's historical inflation-nowcast
chart as a **source-only event panel**.  The panel retains the last headline
and core CPI month-over-month nowcasts strictly before each official CPI
release, then compares them with the chart's first available actual values.

No BTC price, funding, return, trade, portfolio, label, or alpha-overlap row was
read while selecting or validating this data source.  Its trading value is
therefore still unknown at this stage.

## Official sources

- [Cleveland Fed Inflation Nowcasting](https://www.clevelandfed.org/indicators-and-data/inflation-nowcasting)
- [Official historical monthly-chart JSON](https://www.clevelandfed.org/-/media/files/webcharts/inflationnowcasting/nowcast_month.json?sc_lang=en)
- [Inflation Nowcasting User's Guide](https://www.clevelandfed.org/-/media/project/clevelandfedtenant/clevelandfedsite/indicators-and-data/inflation-nowcasting/nowcasting_users_guide.pdf)
- Frozen BLS release clock:
  `data/bls_cpi_release_breadth_2019_2023/bls_cpi_release_breadth_2019_2023.csv.gz`

The Cleveland Fed describes these figures as daily inflation nowcasts.  The
historical chart exposes the nowcast path and the first available actual CPI
release.  The normalized panel binds the chart's actual date to the separately
frozen official BLS release clock rather than inferring a timestamp from a
market bar.

## Frozen artifact

| Item | Value |
|---|---:|
| Release horizon | 2019-01-11 through 2023-12-12 |
| CPI releases | 60 |
| Concordant headline/core surprise signs | 47 |
| Raw response bytes SHA-256 | `b2e1f0fb174be417eb417488c93bf9dbcb619c4ebcaef06ed35b18b704968cd9` |
| Deterministic raw gzip SHA-256 | `c53ccc1a64aca61e3bcfe309d91a564f4c257f2e81a91140d64bba9dc3247709` |
| Normalized panel | `data/cleveland_fed_cpi_surprise_2019_2023/cleveland_fed_cpi_surprise_2019_2023.csv.gz` |
| Normalized panel SHA-256 | `e8755bfd15ec135b2a85cedada8880bf5d4518ed07f4eef43b4b3820211d508e` |
| Build manifest SHA-256 | `33f6719bae4d0b9e6c1edb8e93adc3f0cdd60891c92ec05ab92e77287bd946e6` |
| Manifest canonical hash | `fa2efe8d08f0368e44a0583b6b39caf818defdec60db0b352c4307f3f4ebfba2` |

Each row contains the reference month, official UTC release time, latest
pre-release nowcast date, headline/core nowcasts, first-release actuals, their
two surprises, their equal-weight composite, and sign concordance.

## Validation performed

1. Exactly one chart object must exist for every frozen BLS reference month.
2. Exact Cleveland series names and equal date/value lengths are required.
3. Headline and core actuals must appear at one shared chart index.
4. The retained headline and core nowcasts must share the latest populated
   chart date strictly before that actual index.
5. The chart actual date must equal the frozen BLS CPI release date.
6. Every retained numeric value must be finite.
7. Coverage must remain exactly 60 releases from 2019-01-11 through
   2023-12-12.
8. Deterministic snapshot replay must reproduce the normalized panel
   byte-for-byte.
9. The source builder records zero market or funding rows read.

## Interpretation boundary

The later candidate may treat actual-minus-nowcast as a public inflation shock
proxy: hotter-than-nowcast inflation is a possible risk-off input and
cooler-than-nowcast inflation a possible risk-on input.  That direction is a
research hypothesis, not a Cleveland Fed claim, and the nowcast is not a
consensus-economist forecast.  Headline/core concordance can reduce ambiguous
releases but cannot establish causality.

## Vintage and live boundary

The downloaded chart is a **current historical chart vintage**, not a proven
immutable point-in-time archive.  Freezing its raw bytes prevents later silent
research changes, but does not prove that every displayed historical nowcast
was exactly the value visible in real time.  Before live promotion, the system
must collect the Cleveland Fed response with retrieval timestamps and hashes,
retain the last value observed before each release, and demonstrate forward
parity with this parser.  It must never substitute a post-release chart update
for a missing pre-release observation.
