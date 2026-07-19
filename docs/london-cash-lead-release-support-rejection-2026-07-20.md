# LCLR-24 support rejection — 2026-07-20

Decision: **rejected before any post-window return, funding mark, execution
bar, PnL, CAGR, or strict MDD was loaded**.

LCLR-24 combines a mandatory Coinbase cash-lead condition with at least two of
four weak London-window votes. Its exact DST clock, source hashes, rule,
latency, two-hour hold, and support limits were frozen in commit `88aca2d`.

## Outcome-blind support result

| Window | Events | Frozen minimum | Result |
|---|---:|---:|---|
| 2020–2022 total | 325 | 180 | pass |
| 2020–2021 train | 209 | 110 | pass |
| 2020 | 81 | 45 | pass |
| 2021 | 128 | 45 | pass |
| 2022 test | 116 | 55 | pass |
| 2022 H1 | 60 | 22 | pass |
| 2022 H2 | 56 | 22 | pass |

Side balance also passed:

- all: 50.46% long / 49.54% short;
- train: 52.15% long / 47.85% short;
- test: 47.41% long / 52.59% short.

No quarter exceeded the frozen 18% concentration cap; the maximum was 11.69%.
However, the first 63 weekday windows are reserved for strictly prior
normalization, leaving only **one** eligible event in `2020Q1`. The frozen gate
requires at least eight in every calendar quarter. All other quarters contain
24–38 events.

The support gate therefore fails on exactly one check:
`each_quarter=false`. No event clock was written.

## Physical source audit

- Coinbase source-window rows parsed: 9,396;
- Binance source-window rows parsed: 9,396;
- complete weekday London windows: 783;
- source values outside 15:00–16:00 London parsed: 0;
- funding rows loaded: 0;
- post-window execution/outcome rows loaded: 0;
- rows at or after `2023-01-01` loaded: 0.

The candidate incidence was 325 after the 63-window warm-up, but no return
label or held price path was constructed.

## Freeze-process disclosure

A verification subagent ran the exact source-only support function with
temporary output paths after the code, documentation, and 13 targeted tests
were finalized but immediately before the freeze commit. It exposed the same
`2020Q1=1` support failure. No file or parameter was changed after that
incidence was observed; the working-tree bytes were committed unchanged as
`88aca2d`, and the canonical run above was then replayed from that commit.

This preserves the outcome-blind rejection, but it is not claimed that the
first incidence read happened after a Git commit.

## Integrity anchors

- support result hash:
  `8673c93acc8a3624895069298dc1073046ffbc85b909344b96f14c0d443f466b`;
- support JSON SHA-256:
  `9445daa781a6b798242f3166ffe5cbcb03e8d93d0bba35d9740294a93bba3ea1`;
- protocol hash:
  `931d59762eb27c9cdfee1237e791d6b2bb550cdd38b74718ab5801124232b37c`;
- preregistration source SHA-256:
  `d8b25131fdb375b0c327498238c8c8921eaf623d04cf9c363f0c2ffc78f025ac`;
- preregistration document SHA-256:
  `fd996475dba37953b1abc0ec29cfe9edbe7d33b91d61d7880f4e0c7ea9330c65`.

## Decision boundary

The failure is caused by the intentionally causal warm-up, not insufficient
overall incidence. Nevertheless, removing `2020Q1`, lowering the quarterly
minimum, shortening the reference history, or adding pre-2020 data after
seeing this result would repair the support gate post hoc. The frozen contract
forbids that repair.

LCLR-24 is therefore not eligible for a train/test evaluator, a 2023+ source
download, alpha registration, or portfolio promotion. A future London-session
candidate must use a genuinely different mechanism, not this rule with a
warm-up exemption.
