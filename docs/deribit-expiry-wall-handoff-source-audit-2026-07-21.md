# DEWH-144 strike-wall source audit — 2026-07-21

## Decision

The DEWH-144 source is sufficient to proceed to the already frozen
source-support and pure-clock novelty gates. This is **not alpha evidence**.
No candidate incidence, BTC future path, funding, return, PnL, CAGR, or MDD was
computed.

## Frozen artifacts

Source panel:

```text
data/deribit_btc_expiry_wall_2019_2023/
  BTC_deribit_expiry_wall_2019-01-01_2023-12-31.csv.gz
SHA-256 53e8c829d8dd49eb669218067409a1b5175900c88fd75652c0ad420f6b6167f5
bytes 91,455
rows 1,484
```

Manifest:

```text
data/deribit_btc_expiry_wall_2019_2023/build_manifest.json
SHA-256 dde10a20d6efc3026be253daefe88bff0bae4ba379deaa90b4d08431dd741c36
manifest hash dbecf89849c356e4b5900600e2727ad5f972c9a62487b8d897407a0e22da104e
```

Implementation bindings:

```text
training/build_deribit_expiry_wall_handoff_source.py
SHA-256 708a5315476c56286ab0b195d839aab521ec0d5d47c206e7c43976342af75b20

training/download_deribit_btc_option_deliveries.py
SHA-256 aa925828cf8350ed522c0ac559c64faed90fc049b99228d60b349d2771b1cd4c

docs/deribit-expiry-wall-handoff-mechanism-decision-2026-07-21.md
SHA-256 f5b378b75e3d32c18e32b245f62674a7a7b25f90ec7761d865ddb6c627a93ce8
```

## Official-source result

The builder followed 45 continuation pages from Deribit's public BTC
`delivery` endpoint and crossed the frozen 2019 lower bound.

| Audit | Result |
|---|---:|
| received settlement rows in interval | 44,036 |
| selected BTC option rows | 43,877 |
| excluded futures rows | 159 |
| source expiry events | 1,484 |
| wall-valid expiry events | **1,484** |
| fewer-than-three-strike exclusions | 0 |
| tied-dominant-wall exclusions | 0 |
| invalid-spacing exclusions | 0 |
| first expiry | 2019-01-04 08:00 UTC |
| last expiry | 2023-12-31 08:00 UTC |
| maximum delivery delay | 4,707.676 seconds |
| delayed events | 1 |
| maximum within-expiry timestamp span | 0 seconds |

Annual wall-valid expiry counts were 52 in 2019, 337 in 2020, and 365 in each
of 2021, 2022, and 2023. Every source expiry had at least three strikes, one
unique maximum combined call-plus-put strike position, and positive local
strike spacing. The source therefore represents the frozen wall geometry over
the full train and selection calendar.

## Retained fields

Only expiry-level source values were retained:

- scheduled expiry, actual delivery event, and conservative observation time;
- delivery index price;
- distinct strike count and total reported position;
- dominant strike and its combined call-plus-put position;
- wall share, strike-position HHI, and largest individual instrument share;
- local log strike spacing and normalized signed index-to-wall distance; and
- wall/timing source diagnostics.

Instrument names, option type, ITM/OTM state, mark price, settlement PnL,
session PnL, DEHR release side, and instrument-level rows are absent from the
panel. Raw responses were not persisted.

## Reproducibility correction

The first real-network reproducibility check found that hashing the complete
JSON-RPC envelope was unstable even though the derived panel was byte
identical. Deribit changes `usIn`, `usOut`, `usDiff`, and continuation tokens
between otherwise identical requests.

Before accepting or committing the source artifact, the downloader was fixed
to support a source-specific semantic commitment over the **ordered settlement
rows only**. Query configuration, page lengths, pagination monotonicity,
continuation uniqueness, boundary crossing, and every derived aggregate remain
separately validated. Dynamic request metadata and opaque continuation tokens
are explicitly excluded from page hashes.

After that correction, two complete real-network builds produced byte-
identical panel and manifest files with the hashes above. Tests also vary the
dynamic envelope fields and continuation token while requiring an identical
manifest.

## Outcome boundary

```text
Binance market rows loaded       = 0
funding rows loaded              = 0
future-return rows loaded        = 0
performance artifacts parsed     = 0
return/PnL fields retained       = 0
economic outcomes computed       = false
raw Deribit rows persisted       = false
candidate incidence computed     = false
parameter search performed       = false
```

## Next sealed work unit

Freeze the comparator cohort, including a deterministic reconstruction of the
unchanged DEHR-72 candidate clock, before computing DEWH incidence. Then apply
the singleton 365-day strict-prior ranks, `0.50/0.70` gates, normalized wall
distance band, support floors, and novelty limits exactly as committed. A
support or novelty failure retires DEWH before any economic evaluator exists.
