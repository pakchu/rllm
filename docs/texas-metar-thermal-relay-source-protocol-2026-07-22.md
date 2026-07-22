# TMTR source protocol — 2026-07-22

## Status

**Frozen before full 2020–2023 MADIS incidence and before every BTC outcome.**

- source-axis decision:
  `docs/texas-metar-thermal-relay-source-axis-decision-2026-07-22.md`
- executable protocol:
  `training/preregister_texas_metar_thermal_relay.py`
- machine manifest:
  `results/texas_metar_thermal_relay_source_protocol_2026-07-22.json`
- manifest hash:
  `e24d1a8c30efa5142b9f81ddd78f47a1673c5df2e6dc610072a06cca87c5e4eb`
- manifest file SHA-256:
  `16d01d0d1e8f617b78c054e81e9407a9986625071ec77ff3584398fb39525f7d`
- outcomes opened: `false`
- historical source incidence opened: `false`
- thermal parser opened: `false`

## Frozen source

TMTR streams NOAA/NWS MADIS public METAR archive objects at exactly 00, 06,
12, and 18 UTC from 2020-01-01 through 2023-12-31. The envelope contains
exactly 5,844 expected gzip-compressed netCDF objects.

The station panel is immutable:

- `KMAF` — Midland International;
- `KABI` — Abilene Regional;
- `KLBB` — Lubbock Preston Smith; and
- `KACT` — Waco Regional.

Only `stationName`, `timeObs`, `timeReceived`, `reportType`, `correction`, and
`rawMETAR` may enter normalized source rows. `correction` is the integer MADIS
correction flag, not text. Every decoded MADIS weather field,
QC summary, coordinate, and elevation is excluded. A later parser may derive
weather tokens only from the frozen raw METAR string.

## Causal membership

A retained row must:

- belong to one frozen station;
- have `reportType=METAR` and `correction == 0`;
- full-match the frozen visible-ASCII station/time report grammar;
- reject raw `COR`, `NIL`, and `SPECI` tokens while permitting routine `AUTO`;
- have observation time in `(archive label - 15 minutes, archive label]`;
- have receipt time in `[timeObs, archive label + 15 minutes]`; and
- have its raw `DDHHMMZ` resolve, including month/year rollover, exactly to
  `timeObs`.

These strict bounds imply an observation-to-receipt delay below 30 minutes.

Exact duplicate rows are deduplicated before cardinality checking. Two
different eligible reports for the same station and archive object are fatal.
Historical archive membership alone does not prove point-in-time publication.
The live route is frozen as
`https://madis-data.ncep.noaa.gov/madisPublic/data/point/metar/netcdf/YYYYMMDD_HH00.gz`,
and row availability is conservatively
`max(timeReceived, archive label + 60 minutes)`. The archive label and HTTP
modification time are not availability claims. The 60-minute floor follows
MADIS's documented five-minute current/previous-hour processing and nominal
file window ending 44 minutes after the label. The tighter membership window
excludes the post-label portion and late data-recovery reports.

## Exact netCDF extraction contract

Only classic CDF1 magic (`43 44 46 01`) is accepted. Required dimensions are
the unlimited record dimension `recNum`, `maxStaNamLen=5`, `maxRepLen=6`, and
`maxMETARLen=256`. The six required variables are exact and aliases are
forbidden:

| variable | dimensions | typecode | fill |
|---|---|---:|---:|
| `stationName` | `recNum,maxStaNamLen` | `c` | none |
| `reportType` | `recNum,maxRepLen` | `c` | none |
| `rawMETAR` | `recNum,maxMETARLen` | `c` | none |
| `timeObs` | `recNum` | `d` | `1.7976931348623157e+308` |
| `timeReceived` | `recNum` | `d` | `1.7976931348623157e+308` |
| `correction` | `recNum` | `i` | `-2147483647` |

The NOAA object may contain additional dimensions and variables, as the bounded
probe did, but the extractor may read only these six required variables.
Character rows are C-order bytes with trailing NUL/space trimmed; empty values,
embedded NUL, control bytes, and non-ASCII bytes are fatal. Epochs must be
finite integer seconds in UTC and fills are rejected. `correction` must be
integer zero. The canonical row tuple is archive label plus the exact six
decoded values; exact tuples are deduplicated, while multiple distinct eligible
tuples for a station/object are fatal. Every retained tuple receives a SHA-256
identity.

## Frozen source gates

Transport must satisfy:

- at least 99.5% of expected objects in every year;
- at least 98% in every month;
- valid gzip CRC and classic-netCDF magic;
- SHA-256 coverage for every fetched object; and
- two identical in-memory parse passes before any output write.

The panel must satisfy:

- at least 95% eligible coverage for every station-year;
- all four stations in at least 90% of anchors each year and 85% each month;
- no more than four consecutive anchors without a complete panel;
- 100% retained raw-syntax and causal-time validity;
- zero conflicting target reports; and
- no imputation or forward fill.

Any failed gate is `REJECT_NO_REPAIR`. Stations, hours, interval, fields, and
thresholds cannot change after incidence.

Every ratio denominator is formed from expected archive-label opportunities,
not fetched objects or successful rows. A missing/unparseable object is missing
for all four station opportunities and is an incomplete panel. A fetched
gzip/netCDF/schema-corrupt object is fatal rather than merely missing. The
consecutive-panel scan walks all 5,844 expected labels in chronological order.

## Deferred mechanism and RLLM boundary

A source pass authorizes only a separately committed raw-METAR parser and
thermal-mechanism protocol. Before market data opens, it must freeze weather
token parsing, thermal algebra, event clock, direction, hold, execution delay,
leverage, controls, and sparse-clock novelty comparisons.

An LLM or RLLM may later consume a frozen state or make train-only abstention
and sizing decisions. It may not create, delete, retime, relabel, impute, or
repair source reports, and eval outcomes may not select direction or threshold.
