# Live DB Data Field Request: Binance Premium Index & Funding Rate

## Target Database

```text
candledb
```

## New Table 1: `bars_binance_premium`

### Purpose

Store Binance USDⓈ-M Futures Premium Index Klines for `BTCUSDT`.

### Required Source

```http
GET https://fapi.binance.com/fapi/v1/premiumIndexKlines
```

### Required Parameters

```text
symbol=BTCUSDT
interval=1m
limit=1500
```

### Required Fields

| DB Field | Type | Source Field | Required |
|---|---|---|---|
| `symbol` | `TEXT` | request symbol | yes |
| `interval` | `TEXT` | request interval | yes |
| `ts` | `TIMESTAMPTZ` | response `[0]` open time ms | yes |
| `open` | `NUMERIC` | response `[1]` | yes |
| `high` | `NUMERIC` | response `[2]` | yes |
| `low` | `NUMERIC` | response `[3]` | yes |
| `close` | `NUMERIC` | response `[4]` | yes |
| `close_time` | `TIMESTAMPTZ` | response `[6]` close time ms | yes |
| `volume` | `NUMERIC` | constant `0` | yes |
| `created_at` | `TIMESTAMPTZ` | insert time | yes |
| `updated_at` | `TIMESTAMPTZ` | upsert time | yes |

### Primary Key

```sql
PRIMARY KEY (symbol, interval, ts)
```

### Required Indexes

```sql
CREATE INDEX IF NOT EXISTS idx_bars_binance_premium_symbol_ts
ON bars_binance_premium (symbol, ts DESC);

CREATE INDEX IF NOT EXISTS idx_bars_binance_premium_symbol_interval_ts_desc
ON bars_binance_premium (symbol, interval, ts DESC);
```

### Required Freshness

```text
BTCUSDT 1m latest ts stale <= 3 minutes
```

---

## New Table 2: `funding_rates_binance`

### Purpose

Store Binance USDⓈ-M Futures funding rate history for `BTCUSDT`.

### Required Source

```http
GET https://fapi.binance.com/fapi/v1/fundingRate
```

### Required Parameters

```text
symbol=BTCUSDT
limit=1000
```

### Required Fields

| DB Field | Type | Source Field | Required |
|---|---|---|---|
| `symbol` | `TEXT` | `symbol` | yes |
| `funding_time` | `TIMESTAMPTZ` | `fundingTime` ms | yes |
| `funding_rate` | `NUMERIC` | `fundingRate` | yes |
| `mark_price` | `NUMERIC` | `markPrice` | yes |
| `created_at` | `TIMESTAMPTZ` | insert time | yes |
| `updated_at` | `TIMESTAMPTZ` | upsert time | yes |

### Primary Key

```sql
PRIMARY KEY (symbol, funding_time)
```

### Required Indexes

```sql
CREATE INDEX IF NOT EXISTS idx_funding_rates_binance_symbol_time_desc
ON funding_rates_binance (symbol, funding_time DESC);
```

### Required Freshness

```text
BTCUSDT latest funding_time stale <= 9 hours
```

---

## Required Validation Queries

### Premium Latest Rows

```sql
SELECT *
FROM bars_binance_premium
WHERE symbol = 'BTCUSDT'
  AND interval = '1m'
ORDER BY ts DESC
LIMIT 5;
```

### Premium Freshness

```sql
SELECT
    symbol,
    interval,
    MAX(ts) AS latest_ts,
    NOW() - MAX(ts) AS stale
FROM bars_binance_premium
WHERE symbol = 'BTCUSDT'
  AND interval = '1m'
GROUP BY symbol, interval;
```

### Funding Latest Rows

```sql
SELECT *
FROM funding_rates_binance
WHERE symbol = 'BTCUSDT'
ORDER BY funding_time DESC
LIMIT 5;
```

### Funding Freshness

```sql
SELECT
    symbol,
    MAX(funding_time) AS latest_funding_time,
    NOW() - MAX(funding_time) AS stale
FROM funding_rates_binance
WHERE symbol = 'BTCUSDT'
GROUP BY symbol;
```

### Premium Query Performance

```sql
EXPLAIN ANALYZE
SELECT *
FROM bars_binance_premium
WHERE symbol = 'BTCUSDT'
  AND interval = '1m'
ORDER BY ts DESC
LIMIT 1;
```

### Funding Query Performance

```sql
EXPLAIN ANALYZE
SELECT *
FROM funding_rates_binance
WHERE symbol = 'BTCUSDT'
ORDER BY funding_time DESC
LIMIT 1;
```

Target:

```text
Execution Time < 5 ms
```
