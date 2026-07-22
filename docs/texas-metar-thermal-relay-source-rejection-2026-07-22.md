# TMTR source rejection — 2026-07-22

## Terminal decision

**TMTR is rejected with `REJECT_NO_REPAIR`.** No BTC market clock, return,
PnL, prior-alpha clock, thermal parser, or outcome was opened.

The terminal machine result is:

- `results/texas_metar_thermal_relay_source_support_2026-07-22.json`
- file SHA-256:
  `74ecdc2fb4e10ff78648d1560db9b96e8eba16cedf1de61d8f097a7424c55480`
- manifest hash:
  `0c5f4974ba821e7f3882afae8c28522fdbe4335669ef4dc5d48fa10eec9b70c8`
- frozen protocol hash:
  `e24d1a8c30efa5142b9f81ddd78f47a1673c5df2e6dc610072a06cca87c5e4eb`
- committed builder SHA-256:
  `951e4c92023c18c4bfc938928d4c22c3bb850c9720dca6b4e62e70a3405903ae`

## Failure point

The first 429 expected archive labels were checkpointed without a fatal source
failure. The next label, `2020-04-17T06:00:00Z`, failed the frozen rule that a
station/object may contain at most one distinct eligible report after exact
duplicate deduplication.

The compressed object was independently fetched during the builder's fatal
confirmation and again during the rejection audit with the same identity:

- URL:
  `https://madis-data.ncep.noaa.gov/madisPublic/data/archive/2020/04/17/point/metar/netcdf/20200417_0600.gz`
- compressed bytes: `1,147,277`
- compressed SHA-256:
  `2f0466a9be4983a3ca1be0e263ba779a579495fec24f5235aa2a6f17f538ee7e`

## Exact conflict

`KMAF`, `KABI`, and `KACT` each had one eligible row. `KLBB` had two different
eligible rows:

1. observation `2020-04-17T05:53:00Z`, receipt
   `2020-04-17T05:57:00Z`, identity
   `e17c2a31af974783c528b946742ccb03716070701cb12879fd6bd405f86bb701`;
2. observation `2020-04-17T05:57:00Z`, receipt
   `2020-04-17T06:02:00Z`, identity
   `0043a2b26018bfd07907eac05170d8aee35ba5a4e2d39d076d9952ca48e4034a`.

Both rows had archived `reportType=METAR`, integer `correction=0`, exact raw
`DDHHMMZ == timeObs`, observation inside `(label-15m,label]`, and receipt inside
`[timeObs,label+15m]`. The second raw report included `WSHFT` and `FROPA`, which
is consistent with a genuine weather-driven extra report rather than an exact
duplicate or parser alias.

## Why the run cannot resume

Choosing the first row, last row, shortest delay, or a preferred raw token now
would be a post-incidence membership repair. The v3 protocol explicitly made
distinct multiple eligible rows fatal and prohibited changing stations, hours,
interval, accepted fields, or membership rules after source incidence opened.

Therefore the partial SQLite checkpoint is not an admissible basis for another
TMTR run. TMTR is retired before mechanism parsing and before all economic
evaluation. The next research unit must start from a different predeclared
source axis rather than relaxing this failure.
