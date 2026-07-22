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
  `e53c8de3424d0e804645622ae4e526f0c3a217b85576d5534cd2a9a77f9e884d`
- manifest file SHA-256:
  `f246936105cd6e713fd627c2800a1c1c4d6220ec7cc2a67bd431b37205e9378a`
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
`rawMETAR` may enter normalized source rows. Every decoded MADIS weather field,
QC summary, coordinate, and elevation is excluded. A later parser may derive
weather tokens only from the frozen raw METAR string.

## Causal membership

A retained row must:

- belong to one frozen station;
- have `reportType=METAR` and an empty correction field;
- full-match the frozen visible-ASCII station/time report grammar;
- have observation time in `(archive label - 75 minutes, archive label]`;
- have receipt time in `[timeObs, archive label + 15 minutes]`; and
- have observation-to-receipt delay no greater than 30 minutes.

Exact duplicate rows are deduplicated before cardinality checking. Two
different eligible reports for the same station and archive object are fatal.
The causal timestamp is `timeReceived`; the archive label and HTTP modification
time are not availability claims.

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

## Deferred mechanism and RLLM boundary

A source pass authorizes only a separately committed raw-METAR parser and
thermal-mechanism protocol. Before market data opens, it must freeze weather
token parsing, thermal algebra, event clock, direction, hold, execution delay,
leverage, controls, and sparse-clock novelty comparisons.

An LLM or RLLM may later consume a frozen state or make train-only abstention
and sizing decisions. It may not create, delete, retime, relabel, impute, or
repair source reports, and eval outcomes may not select direction or threshold.
