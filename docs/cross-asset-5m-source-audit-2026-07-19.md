# Cross-asset five-minute source audit — 2026-07-19

Preregistration: `5cacbead33f2b66c5961e22666708cde72f9844233821002b421a17a658b6775`

The strategy universe is QQQ, KODEX 200, and GLD only. No KOSPI price or signal is used.

| Asset | 5m rows | Sessions | Range (UTC) | Yahoo matches | Median / p95 close diff |
|---|---:|---:|---|---:|---:|
| QQQ | 37,848 | 488 | 2024-08-06T13:30:00+00:00 → 2026-07-17T19:55:00+00:00 | 3,006 | 0.139 / 1.420 bp |
| 069500 | 43,686 | 576 | 2024-03-05T00:00:00+00:00 → 2026-07-16T06:15:00+00:00 | 2,784 | 1.127 / 6.954 bp |
| GLD | 37,770 | 487 | 2024-08-06T13:30:00+00:00 → 2026-07-17T19:55:00+00:00 | 3,006 | 0.003 / 1.369 bp |

## Limitation

Investing.com TVC is an unofficial source and is not suitable as a production market-data contract.
The committed artifact records canonical chunk hashes; extracted raw payloads remain local.
Production replication should use an entitled broker feed such as IBKR or KIS and re-run parity checks.
