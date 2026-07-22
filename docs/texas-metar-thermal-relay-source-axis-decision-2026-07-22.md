# TMTR source-axis decision — receipt-time Texas METAR panel

## Decision

Advance **TMTR (Texas METAR Thermal Relay)** to a source-only
preregistration. TMTR is a new exogenous source axis: raw airport weather
reports received by NOAA MADIS at four fixed Texas stations. It is not a
price, volume, funding, order-flow, on-chain, news, search, or calendar
observable.

This decision is frozen before reading full 2020–2023 source incidence and
before opening any BTC clock, price, return, PnL, CAGR, MDD, prior-alpha clock,
or portfolio result.

## Why the mechanism is plausible

The intended causal chain is physical rather than correlational:

1. unusually broad Texas heat or cold changes ERCOT system demand;
2. large flexible loads, including Bitcoin mining facilities, can curtail
   rapidly during stressed conditions; and
3. curtailment and later restoration change mining operating state and may
   create a delayed BTC inventory/liquidity response.

The mechanism is supported externally, but no claim is made yet that it is
economically material. ERCOT states that its voluntary program lets large
flexible customers such as Bitcoin mining facilities reduce power use during
high demand. The U.S. Energy Information Administration reports that crypto
miners are major participants in ERCOT's large-flexible-load program, cites up
to 1,530 MW of enrolled load, and notes large Rockdale mining facilities. EIA
also documents that Texas heat waves and cold snaps raise ERCOT electricity
demand.

Official references:

- ERCOT voluntary-curtailment release:
  https://www.ercot.com/news/release/2022-12-06-ercot-creates-voluntary
- EIA cryptocurrency-mining electricity analysis:
  https://www.eia.gov/todayinenergy/detail.php?id=61364
- EIA Texas large-flexible-load analysis:
  https://www.eia.gov/todayinenergy/detail.php?id=63344
- EIA Texas heat-wave demand analysis:
  https://www.eia.gov/todayinenergy/detail.php?id=57240
- EIA Texas winter-weather analysis:
  https://www.eia.gov/todayinenergy/detail.php?id=46836

These sources motivate only the source axis. They do not choose a signal
direction, threshold, hold, leverage, or regime.

## Official source and point-in-time field boundary

Use the NOAA/NWS Meteorological Assimilation Data Ingest System (MADIS) public
METAR archive:

```text
https://madis-data.ncep.noaa.gov/madisPublic/data/archive/
  YYYY/MM/DD/point/metar/netcdf/YYYYMMDD_HH00.gz
```

NOAA describes MADIS as an operational observational database with uniform
timestamps and quality flags, available from July 2001 to the present. NOAA's
historical-data documentation identifies the date-partitioned archive and its
compressed netCDF files.

- MADIS overview: https://madis.ncep.noaa.gov/
- MADIS historical archive documentation:
  https://madis.ncep.noaa.gov/faq_historicaldata.shtml
- MADIS METAR description:
  https://madis.ncep.noaa.gov/madis_metar.shtml

TMTR must retain only fields embedded in the archived netCDF object:

- `stationName`;
- `timeObs`;
- `timeReceived`;
- `reportType`;
- `correction`; and
- `rawMETAR`.

Decoded MADIS temperature, dew point, wind, precipitation, pressure, QC
summary, latitude, longitude, and elevation fields are forbidden from the
frozen source artifact. A later separately committed parser may derive weather
tokens only from `rawMETAR`. This prevents later archive reprocessing or a
changed decoder from silently entering the signal.

The causal timestamp is the archived `timeReceived`, not the observation time,
file modification time, or nominal archive-hour label. A source row may not be
used before `timeReceived` plus a later frozen execution delay.

## Fixed station panel

The source panel is exactly:

| station | airport | geographic role |
|---|---|---|
| `KMAF` | Midland International | west Texas |
| `KABI` | Abilene Regional | west-central Texas |
| `KLBB` | Lubbock Preston Smith | northwest Texas |
| `KACT` | Waco Regional | central Texas / Rockdale corridor proxy |

The panel was selected for geographic breadth across west-to-central Texas,
stable airport identifiers, and direct routine METAR reporting. No station may
be added, removed, substituted, or weighted after full incidence is opened.

## Fixed archive sampling envelope

- physical source interval: `[2020-01-01, 2024-01-01)` UTC;
- archive hours: exactly `00`, `06`, `12`, and `18` UTC each day;
- expected archive objects: 5,844;
- target station IDs: exactly the four above;
- source construction: stream one archive object, hash it, parse twice in
  memory, retain target raw reports, then delete the compressed object;
- disk guard: abort before a download at 300 GiB used;
- no retry may switch to GHCNh, ISD CSV, an unofficial METAR mirror, forecast
  data, or a different MADIS dataset.

The source stage may inspect object availability, bytes, hashes, target-report
coverage, receipt delays, raw-report syntax, duplicate identities, and
cross-pass equality. It may not parse thermal state, calculate an event clock,
or load any market/outcome data.

## Bounded source-only probe already opened

One archive object was inspected solely to establish schema and transport:

```text
2023/01/01/point/metar/netcdf/20230101_0000.gz
```

The object was about 1.2 MiB compressed and 12 MiB decompressed, used classic
netCDF readable by SciPy, and contained all four target stations once. The
observations were received at `2022-12-31T23:58:00Z`; their raw reports were
ordinary `KACT`, `KMAF`, `KABI`, and `KLBB` METAR strings. No other archive
hour, source incidence, thermal value, BTC series, or outcome was counted.

Separately, bounded NOAA GHCNh metadata probes confirmed the corresponding
airport identities. GHCNh is not an authorized TMTR source.

## Novelty boundary

Repository-wide search found no prior MADIS, METAR, ERCOT, weather, or thermal-
breadth alpha implementation. TMTR is therefore source-distinct from the
existing exchange microstructure, derivatives, on-chain, macro-FX, calendar,
document, narrative, and price-action families.

Clock-level novelty is not asserted here. If source support passes, TMTR must
freeze its own clock and compare it with the canonical sparse-alpha clocks
before any economic result can promote it.

## Stop and anti-repair rule

The next commit must preregister exact source parsing, identity, coverage,
receipt-delay, duplication, and deterministic-replay gates before downloading
the 5,844-object envelope. Failure retires TMTR without changing station panel,
archive hours, source interval, accepted fields, or thresholds after incidence.

Only a source-support pass may authorize a separately committed raw-METAR
thermal parser and mechanism protocol. Signal direction, hold, leverage, and
economic gates remain unopened.
