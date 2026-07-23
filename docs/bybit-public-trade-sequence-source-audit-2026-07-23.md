# Bybit public-trade sequence source audit — 2026-07-23

## Verdict

The corrected v2 source-only replay passed.  The official Bybit `BTCUSDT`
linear-perpetual trade archive exposes the timestamp, symbol, taker side,
size, price, and execution identifier needed to continue the provisional
**BSEA-24** investigation.  This pass authorizes only the next source/parity
stage; it is not evidence that BSEA exists or is profitable.

The v1 pass remains invalid and non-authoritative.  V2 was committed before
network access, independently reviewed, and approved with zero remaining
actionable findings before the real replay was run.

## Frozen replay result

The probe opened the official archive directory and exactly 16,384 compressed
bytes from each frozen boundary file:

| Day | Prefix SHA-256 | Header |
|---|---|---|
| 2020-03-25 | `dfde89d406b7d179c03fef8116d2668ab52999199e39e07cdd55175a6e749821` | frozen 10-field base |
| 2023-01-01 | `e91c088cdf75a06fcaecb42be75d59178706798029cdec098ef67e49cc2e7455` | frozen 10-field base |
| 2026-07-22 | `ad70af6e771252b3c5331234882d7a60defece0813e9ca189c91b64bc93e6039` | base plus recent-only `RPI` |

Every prefix matched the value already disclosed by invalid v1 before any CSV
record was decompressed.  Each prefix yielded exactly the header and first
logical data record for type validation.  No bytes after that first record
were decompressed, and no first-record value was retained in the artifact.

The directory contained each of the 365 exact daily 2023 archive names once,
with no missing, unexpected, or duplicate `href`.

## Schema decision

The six required canonical fields map consistently at all three boundaries:

- `timestamp` -> `timestamp`
- `symbol` -> `symbol`
- `side` -> `side`
- `size` -> `size`
- `price` -> `price`
- `execution_id` -> `trdMatchID`

The 2020 and 2023 headers are the exact frozen ten-field base.  The 2026
header adds only the explicitly classified `RPI` suffix.  `RPI` remains
forbidden from the primary BSEA feature because it is not present through the
full historical window.  The v2 implementation rejects the same suffix on an
older frozen day and rejects any other addition, removal, or reordering.

## Outcome and storage boundary

The probe did not open:

- a BSEA candidate clock or event count;
- any Binance comparator value;
- any post-entry market outcome;
- any return, PnL, CAGR, or strict MDD; or
- any raw daily archive beyond the three bounded prefixes.

Repository-filesystem use was 287 GiB before the probe, below the frozen
300 GiB abort threshold.  No raw archive was persisted.

## Immutable bindings

- V2 result file SHA-256:
  `916a55f7cd957eff39e84b2ac383c2b49cb342e2012a0f8bc15c3af98b3b3cb0`
- V2 manifest hash (canonical JSON excluding only the self field):
  `c36f46c8399692b62d202a7331c9215fc3a5684cc3b2d57ca04d7fc7c83a5f84`
- V2 probe SHA-256:
  `a808ff71ddbdce447764b8a5ed173a7816a10be2dc0add51628c5b08655f9bee`
- Source-axis decision SHA-256:
  `fb12c54b8a4a89cb446baa9014f89546bf6c99e46687be2471b51a2bf1989a21`
- V2 correction decision SHA-256:
  `73996f714ce74e1bc81268b545be04177375d61554ba33a219980b3b04ca0bda`
- Invalid v1 file SHA-256 retained for audit:
  `3e2872467acebfd07f91ff8b9ff0079eb9dc518f6f37ad79f83a6a47cf413536`

Artifacts:

- `training/probe_bybit_public_trade_sequence_source.py`
- `results/bybit_public_trade_sequence_source_feasibility_v2_2026-07-23.json`
- `tests/test_probe_bybit_public_trade_sequence_source.py`
- `tests/test_bybit_public_trade_sequence_source_artifact.py`

## Next authorized step

Before any BSEA alpha clock is built, a prospective recent archive/REST/
WebSocket overlap must reconcile timestamp, side, price, size, execution ID,
ordering, duplicates, and omissions.  After parity is sealed, one exact
venue-relative sequence-disagreement mechanism and its null battery may be
preregistered.  Generic Bybit volume, imbalance, or lead/lag repair remains
forbidden.
