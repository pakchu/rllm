# RFXS-576 source-integrity rejection — 2026-07-20

## Decision

**RFXS-576 is retired before source-support or outcome evaluation.** The first
production source build failed closed because the frozen common calendar did
not exist at its preregistered start.

The failure is not an alpha result. It says only that this exact source contract
cannot be instantiated without changing a frozen boundary.

## Frozen inputs

- mechanism commit:
  `1d5805397ed72c98bc83597544b949b07d425f32`
- mechanism document SHA-256:
  `c3f7bcfd12c4412be0ad8696b2fa339c709fa94f1a5e61a22cf33c45e4d3ae89`
- source-builder commit:
  `11b8ec87c42125a128365d7aec1406b11f01e3e3`
- source-builder SHA-256:
  `80b7a54643df84c6da2e545cd8180a43fa71767e4f21b115653af52adb847749`
- requested source horizon:
  `[2020-10-01, 2024-01-01)` UTC
- required symbols:
  `BTCUSDT`, `BTCEUR`, `BTCTRY`, `BTCBRL`

## Failure evidence

The command was:

```bash
uv run python -m training.build_binance_regional_fiat_cross_rate
```

The official checksum-verified `BTCBRL-1d-2020-10.zip` contained no daily rows
for 2020-10-01 through 2020-10-12. Its observed rows began on 2020-10-13. The
builder therefore raised:

```text
ValueError: BTCBRL 2020-10 does not match the exact UTC daily grid;
missing=['2020-10-01', ..., '2020-10-12'], extra=[]
```

This contradicts the mechanism document's assumption that 2020-10-01 was the
first complete common monthly boundary. Merely observing that an October ZIP
and companion checksum existed had not established full-month coverage.

## Non-observation audit

The builder downloaded and parsed source archives in memory to validate them,
but it failed before the four-symbol panel was composed or written. Fresh
filesystem verification after the failure found
`data/binance_regional_fiat_cross_rate_btc_2020_2023/` absent.

Consequently, this run produced:

- no source CSV;
- no build manifest;
- no residual, median, MAD, z-score, state, event, or support count;
- no comparator statistic;
- no USD-M execution or funding read;
- no return, PnL, CAGR, MDD, trade statistic, or stage result.

The process did validate that some later monthly source archives were
well-formed, but no close value or source-derived summary was displayed or
persisted. This remains a source-feasibility observation, not an outcome
observation.

## No-repair boundary

RFXS-576 cannot silently shift its start date, tolerate a partial month, fill
the missing dates, shorten the 180-day baseline, or drop BTCBRL. Any such
change would violate the frozen complete-calendar contract.

A successor beginning on the next exact full-month boundary may be considered
only under a new candidate identifier and a new preregistration committed
before residuals, event incidence, comparator statistics, or outcomes are
computed. It must disclose that the new source boundary was chosen after this
availability failure and therefore is not a pristine source-feasibility clean
room.
