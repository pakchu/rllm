# BitMEX Trollbox attention source freeze — 2026-07-20

The complete English-channel prefix was downloaded only after the TBASR-24
mechanism, causal private loader, singleton attention rule, and support gate
were committed. Operational request throttling and streaming-finalizer changes
were separately disclosed and committed before completion; neither changed
source membership or a research threshold. No attention incidence, message
semantics, BTC price, post-event return, funding, PnL, CAGR, or strict MDD was
opened while freezing this source.

## Frozen source audit

- official endpoint: `GET https://www.bitmex.com/api/v1/chat`;
- channel: global English `channelID=1`;
- ID pagination: repeat the last known existing ID and drop the one-row overlap;
- request size: 500; request pause: 0.25 seconds;
- private pages: 13,610;
- selected private messages: 6,791,328;
- first selected ID/time: `48009301`, `2020-03-13 08:49:12.370 UTC`;
- last selected ID/time: `68729729`, `2022-12-31 23:59:41.712 UTC`;
- last fetched confirmation ID/watermark: `68730020`,
  `2023-01-01 02:17:48.039 UTC`;
- increasing IDs: verified;
- causal availability clock: cumulative maximum raw date in increasing-ID
  order, verified monotonic;
- maximum observed raw-date regression: `14,825.184` seconds; affected rows
  were delayed to the current availability watermark, never moved earlier;
- canonical private-stream SHA-256:
  `4b45cb6bb401aa5028d2e946da26a1ad550ce05c2b286600559732feca093ef3`;
- private page-container SHA-256:
  `011eeed3c3c95b588b7d85621deec20567f994b9009b2a5a8dc3af3a47e1f3bc`;
- private compressed bytes: `292,116,971`;
- complete aggregate grid: 294,807 five-minute rows from
  `2020-03-13 08:45 UTC` through `2022-12-31 23:55 UTC`;
- private aggregate SHA-256:
  `cb0bea6301826739b348c62e8926df7acb2391184d74b4f68c09db10f6a357b3`;
- source manifest hash:
  `ef20dd88c0755d81b95156410a217834db1c69dda1c2ca9bd3b5a1e1e4fbd892`;
- source manifest file SHA-256:
  `39396b980b7376101e1d515d709f8554a2ce85e3586f5fa48ecd3ad21eefe54d`.

## Privacy and repository boundary

Sender fields were replaced locally by stable study pseudonyms before page
persistence. This is pseudonymization, not anonymization; private message text
can still contain user-authored identifiers. The 306 MB private page spool and
2.4 MB aggregate remain ignored and local. No username, message, or aggregate
row is committed. Only the hash-bound source manifest and this audit are
committed. The repository does not grant redistribution rights.

The next step may read only aggregate message count, unique-participant count,
and maximum-participant share under the already committed support gate.
Character count and private text remain unopened.

Official references:

- [BitMEX Chat endpoint](https://docs.bitmex.com/api-explorer/chat-get)
- [BitMEX API changelog](https://www.bitmex.com/static/md/en-US/apiChangelog)
- [BitMEX WebSocket API](https://www.bitmex.com/app/wsAPI)
- [BitMEX Terms of Service](https://www.bitmex.com/terms)
- [BitMEX Privacy Notice](https://static.bitmex.com/documents/Bitmex_Privacy_Notice_2025.pdf)
