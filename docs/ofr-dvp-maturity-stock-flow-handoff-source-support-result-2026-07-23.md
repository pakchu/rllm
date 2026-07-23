# DMSH-168 source-support result — 2026-07-23

## Decision

**`REJECT_NO_REPAIR`**

DMSH-168 (DVP Maturity Stock-Flow Handoff) failed its frozen, outcome-blind
source-support battery. The mechanism is retired unchanged. Comparator clocks,
BTC prices, funding, future returns, CAGR, MDD, and PnL were not opened.

## Frozen evaluation boundary

- preregistration manifest: `8c958e00649db244aff147c82c9ed1b9631ca4d015bdf8ce8085383e0619c678`
- committed source-support builder: `dd20d46`
- normalized OFR rows read: 77,369
- required DVP rows read: 11,241
- feature dates: 1,185
- rank-ready causal decision rows: 777
- comparator files/rows opened: 0 / 0
- BTC/funding/future-return rows opened: 0 / 0 / 0
- performance values opened: false
- network calls: 0

The only subprocesses were two fixed Git checks proving that the builder and
its regression tests were committed and identical to `HEAD` before source
incidence was derived.

## Observed source support

| Split | Events | Required | Direction | Maximum gap | Main concentration failure |
|---|---:|---:|---|---:|---|
| Train 2021–2022 | 12 | 40 | 9 LONG / 3 SHORT | 211 days | missing quarters; only 8 events in 2021 and 4 in 2022 |
| Selection 2023 | 6 | 18 | 6 LONG / 0 SHORT | 67 days | one polarity and one rate bucket only |

Additional failures:

- train confirmation ages had no `7–10` event;
- train maximum month share was `1/4`, above the frozen `1/5` cap;
- selection H2 had only two events, below the frozen minimum of six;
- selection confirmations were 100% `LE30` and 100% one precursor polarity;
- train/selection long-short support floors were not met.

The clock itself passed exact timing, state chronology, split containment, and
global non-overlap checks. The rejection is therefore caused by insufficient,
concentrated, and one-sided source incidence—not an implementation failure.

## Why no repair is allowed

Lowering the ±0.50 state threshold, expanding the ten-row confirmation window,
shortening the 168-hour hold, dropping side/balance floors, or selecting only
active quarters after seeing incidence would convert a preregistered mechanism
test into post-hoc optimization. None is authorized. No market outcome was
opened to justify such a change.

## Immutable artifacts

- clocks: `results/ofr_dvp_maturity_stock_flow_handoff_clocks_2026-07-23.csv.gz`
  - SHA-256: `0cfb881b4e3a0123111eeab904eba7bee074767b9c1315f74e7bddf54e3371c3`
- report: `results/ofr_dvp_maturity_stock_flow_handoff_support_2026-07-23.json`
  - SHA-256: `1e5205d7560e33f1a432f1828a573a04480fe92bc8b2493b06e198d638bb4d05`
  - manifest: `cd4b0eaef23c828c1045f9129702f534f07a401fcd03c422a38397a148879ffa`

## Next research constraint

The next candidate must use a genuinely different observation/mechanism and
must establish adequate two-sided train/selection incidence before novelty or
outcomes. DMSH thresholds and clocks will not be recycled under a new name.
