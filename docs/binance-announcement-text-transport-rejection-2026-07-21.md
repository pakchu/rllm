# Binance announcement text transport — rejected for historical alpha research

## Decision

Do not preregister or backtest a Binance-announcement text alpha from the
currently observable historical pages. The official real-time WebSocket is
good enough for **forward shadow capture**, but the historical transport cannot
replay the exact English text that was visible at the original publication
time. No candidate, model, prompt, BTC row, funding row, return, PnL, absolute
return, CAGR, or MDD was opened.

- audit artifact:
  `results/binance_announcement_text_transport_rejection_2026-07-21.json`;
- artifact SHA-256:
  `7526f2c3d3897c6d2a2ecb393e8c3b90011fe1db7a1186fa9fb6c5ca55c000f5`;
- manifest hash:
  `7ea1fab48a1f42320be69b6543b31f8d8f6f59f165467071eb145da2f5d692e3`;
- auditor SHA-256:
  `eb0e3ee82455e570ff131b4cf241ab2a783166d38358437b2425efc07672e579`.

## What passed

Binance officially documents an authenticated announcement WebSocket. The
English topic pushes `publishDate`, `title`, and `body`, so a collector started
today can bind the first-seen payload and use its receipt timestamp causally.
The announcement product changelog dates that feature to 2025-07-21.

- [WebSocket basic information](https://developers.binance.com/en/docs/products/announcements/general-info)
- [Announcement topic and payload](https://developers.binance.com/en/docs/products/announcements/announcement)
- [Announcement changelog](https://developers.binance.com/en/docs/products/announcements/cms-log)

The current website's internal JSON transport also enumerated 117 bounded
sample rows across six pages, spanning 2017-07-02 through 2026-07-21. This proves
that a current historical index is accessible; it does **not** prove that the
payload is the original historical vintage.

## What failed

The official developer catalog contains only the WebSocket general information,
changelog, and push-payload pages. It documents no historical REST replay and no
original-revision endpoint. Binance also states that undocumented interfaces
should not be relied on as supported production interfaces in its
[developer documentation](https://developers.binance.com/en/docs/introduction).

The bounded detail probe makes the vintage problem concrete:

- a known article declares in its body that it was updated after publication;
- the current detail payload returns the current body;
- it exposes no revision list or original body;
- its `lastUpdateTime` is `0`, so that field cannot reconstruct the change.

The official Telegram archive is not a repair. Across 140 bounded messages, the
audit observed seven messages marked `edited` and five unreplayed message-ID
gaps (`5986`, `6994`, `7989`, `8743`, `8744`). The archive exposes neither the
pre-edit text nor the missing payloads. Selecting only messages that still
survive unedited would use future survival information and would not match the
signals a live collector received at the time.

## Boundary

The committed audit made 20 source-only calls. Earlier interactive diagnosis
made at least 40 source calls and printed current titles and body metadata; this
is disclosed rather than represented as a clean room.

```text
BTC market rows    = 0
funding rows       = 0
future-return rows = 0
return/PnL fields  = 0
candidate signals = 0
```

The Binance stream may be collected prospectively as a shadow dataset, with
the raw first-seen payload, receipt time, article code, and SHA-256 stored before
any model inference. It is not authorized as the historical train/test/eval
source for the next alpha. The next candidate must use an immutable official
archive with original publication artifacts and causal timestamps.
