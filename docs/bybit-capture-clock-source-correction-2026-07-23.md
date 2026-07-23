# Bybit capture clock-source correction — 2026-07-23

## Decision

Do **not** open the second Bybit REST/WebSocket capture with the WSL process
clocks used by capture protocol v2.  The implementation at reviewed commit
`ed2f962b6865766df0da3694f76f5d4b7bd4adb6` is fail-closed, but this WSL2
instance cannot satisfy its local-clock gate.

This is an execution-environment rejection, not a Bybit source rejection and
not permission to weaken the no-reversal gate.  No Bybit live connection or
market outcome was opened during this preflight.

Immutable preflight evidence:

- result file SHA-256:
  `9868e49ad722b5cf5d557efe88fcf1d6a24fc318117a71dc01d7a1680a28614e`
- result manifest hash:
  `40cce365242abade2aa79802f57167488f48b20319244a43af53826ecf1e28be`

## Failure reproduced before network access

A 60-second bracketed probe of `CLOCK_REALTIME` against
`CLOCK_MONOTONIC` recorded:

| Metric | Value |
|---|---:|
| samples | 2,894 |
| UTC reversals | 2 |
| monotonic elapsed | 59.995337624 s |
| UTC elapsed | 56.595713473 s |
| elapsed disagreement | -3.399624151 s |
| maximum adjacent disagreement | 1.702357955 s |

The kernel journal independently contained repeated `Time jumped backwards`
events.  `timedatectl` reported synchronization active, so a nominally
synchronized status is not sufficient evidence for this capture.

A simultaneous comparison showed that `CLOCK_TAI` shared the same two
reversals.  `CLOCK_MONOTONIC_RAW` and `CLOCK_BOOTTIME` did not reverse, but the
current `CLOCK_MONOTONIC` rate differed materially from raw/host elapsed time.
Consequently, changing only the UTC API or ignoring the reversal would not be
a valid repair.

## Feasible clock pair

A persistent Windows PowerShell process served host
`DateTime.UtcNow.Ticks`; each read was bracketed by
`CLOCK_MONOTONIC_RAW`.  Over a separate 120-second raw-monotonic probe:

| Metric | Value |
|---|---:|
| samples | 1,246 |
| host UTC reversals | 0 |
| raw-monotonic reversals | 0 |
| raw elapsed | 119.842784666 s |
| host UTC elapsed | 119.758375100 s |
| absolute elapsed disagreement | 0.084409566 s |
| median RPC uncertainty | 0.391656 ms |
| p99 RPC uncertainty | 0.456136 ms |

This is only a feasibility result.  It does not authorize live capture until
the following provider is implemented, tested, committed, and independently
reviewed.

## Frozen provider correction

The next capture protocol must:

1. launch exactly one fixed, non-shell PowerShell host-clock process before
   the capture boundary and warm it up before retaining any sample;
2. read Windows host UTC ticks through that process and bracket every retained
   read with `CLOCK_MONOTONIC_RAW`;
3. retain the raw-monotonic midpoint and the full request/response round-trip
   as sampling uncertainty;
4. use the same raw-monotonic clock for the 600-second deadline, REST cadence,
   heartbeat cadence, WebSocket receipt order, and REST boundary eligibility;
5. run a 60-second no-network provider preflight and require zero UTC
   reversals, zero nonincreasing raw samples, a complete ledger, and one UTC
   day;
6. reject process exit, malformed host output, write/read failure, reversal,
   UTC-day crossing, or provider fallback;
7. expose provider identity, fixed script hash, warm-up result, preflight
   metrics, and per-sample uncertainty in the manifest; and
8. forbid fallback to WSL `CLOCK_REALTIME`, `CLOCK_TAI`, synthetic UTC, an
   exchange timestamp, or a relaxed reversal threshold.

The PowerShell process is a local clock transport only.  It may not read a
network resource, account state, order, position, comparator venue, return, or
PnL.

## Outcome boundary

No BSEA event clock, candidate incidence, Binance comparator, direction,
market outcome, return, PnL, CAGR, or strict MDD was opened.  The invalid v1
capture remains invalid and cannot be relabeled.  The next network run must be
wholly fresh after the corrected provider passes review.
