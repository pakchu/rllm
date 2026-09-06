# TUSI-168 terminal source-replay failure — 2026-07-30

## Terminal decision

`TUSI-168` is permanently retired before source publication, source support,
novelty, or economics. The single claimed production replay failed during the
primary TRON transport's raw-log phase. The frozen contract prohibits retry,
resume, provider substitution, range reduction, throttle changes, or parser
repair.

- frozen protocol commit:
  `883239651cf164973091f12050487910c8f11cb6`
- claim-only commit:
  `2eb44d7da2a4e63e25e1a648b23c79773e1caaa9`
- replay-claim file SHA-256:
  `3ffe6bc79a9a7fdd34d5fa3040eb668283b738a6bcc2e915631051820b209f1a`
- replay-claim internal hash:
  `d5aa9e4371ba1dff8382f145285cdcdd49b134afb13f8cb1c895f3ca5955bd76`
- protocol-seal hash:
  `2b6f9eeb13f6f43e6a7a0b40934d6e9748629ff2008562af7940658ce26a9f39`
- elapsed replay time before failure:
  `1,069.917` seconds

The sanitized terminal exception was:

```text
TerminalSourceFailure: single RPC attempt failed for primary TRON RPC
```

It arose in the frozen category-log replay path while executing a primary
transport batch. No credential, provider path, request response, event row, or
partial incidence is recorded in this closure.

## Publication audit

After process termination, every post-claim production path remained absent:

```text
data/tron_usdt_supply_events_2023_2026/
  tron_usdt_supply_events_2023_2026.csv.gz
results/tron_usdt_supply_events_source_manifest_2026-07-30.json
results/.tron_usdt_supply_events_source_generation_v1.stage
results/tron_usdt_supply_impulse_primary_clock_2026-07-30.csv.gz
results/tron_usdt_supply_impulse_control_clocks_2026-07-30.csv.gz
results/tron_usdt_supply_impulse_source_support_2026-07-30.json
results/tron_usdt_supply_impulse_novelty_2026-07-30.json
```

The manifest-last generation commit point was therefore never reached, and no
partial CSV, manifest, support clock, or report became a canonical artifact.
The committed claim remained byte-identical after failure.

## Evidence boundary

The builder may have received pre-cutoff raw-log responses in process memory
before the later primary batch failed. Those partial responses were not
inspected, retained, serialized, or used to alter a source rule. Because any
source incidence may nevertheless have been exposed inside the failed
one-shot process, this exact source identity is contaminated as well as
operationally failed.

No source CSV row, source-support metric, comparator clock, Gross9 clock, BTC
market row, funding row, return, PnL, CAGR, or strict MDD was opened by a later
stage. Source support, novelty, and economics are prohibited for `TUSI-168`.

## No-repair rule

This failure is not a request to retry TronGrid, switch the primary provider,
reuse the verification provider as primary, split batches differently, or
continue from a checkpoint. Any such change would be a new source identity
chosen after partial source exposure. The TRON USDt supply-event axis is closed
for the current alpha search.
