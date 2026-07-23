# OFR preliminary repo source audit — 2026-07-23

## Boundary

This audit covers only the frozen Office of Financial Research U.S. Repo
Markets source. It did not read BTC prices, funding, returns, portfolio trades,
PnL, CAGR, MDD, model output, or candidate incidence. It computed no spread,
ratio, rank, z-score, change, state, side, or hold.

Official API references:

- <https://www.financialresearch.gov/short-term-funding-monitor/api/>;
- <https://www.financialresearch.gov/short-term-funding-monitor/api-specs/api-info-mnemonics/>;
- <https://www.financialresearch.gov/short-term-funding-monitor/api-specs/api-full-dataset/>.

The governing contract is
[`ofr-repo-segmentation-source-axis-decision-2026-07-23.md`](ofr-repo-segmentation-source-axis-decision-2026-07-23.md).

## Frozen retrieval

The source was retrieved once from the two committed official URLs and is now
offline-only.

| Object | Retrieved UTC | HTTP bytes | HTTP-body SHA-256 |
|---|---:|---:|---|
| repo mnemonics | 2026-07-23 00:34:25.680945 | 19,888 | `7c6bbf5a476787040999e387fc870cdc1868d497fe23d6d0bf5a0ba9d6bf11c5` |
| preliminary 2019–2023 data set | 2026-07-23 00:34:30.118030 | 544,729 | `5a7f1244849b7856d7c888c1225ac2a7c9f1bdb26e925cd2e64ae5940f5386e8` |

Both responses were HTTP 200 JSON from the exact requested HTTPS URL with no
redirect. OFR transport-compressed the full data-set JSON; the cache preserves
those exact compressed HTTP bytes inside deterministic gzip and decodes the
transport layer exactly once during replay.

## Metadata audit

- 164 catalog definitions: 82 preliminary and 82 final.
- Every preliminary mnemonic has one final mnemonic and the same series name
  after removing only its terminal vintage label.
- Preliminary definitions by segment: DVP 18, GCF 24, TRI 20, TRIV1 20.
- Preliminary definitions by measure: average rate 34, outstanding volume 14,
  transaction volume 34.
- All retained definitions identify the OFR U.S. Repo Markets daily release and
  a `Rate` or `Volume` unit.
- 62 definitions have at least one 2019–2023 aggregation row; 20 definitions
  were introduced outside the frozen window and remain metadata-only.

`TRI` includes Federal Reserve transactions; `TRIV1` is the explicitly named
version excluding Federal Reserve transactions. This is source semantics only,
not a selected trading mechanism.

## Coverage and missingness

The normalized panel contains 77,369 unique `(mnemonic, observation_date)`
rows from 2019-01-02 through 2023-12-29.

| Year | Rows |
|---:|---:|
| 2019 | 15,484 |
| 2020 | 15,546 |
| 2021 | 15,468 |
| 2022 | 15,433 |
| 2023 | 15,438 |

Observed definitions by segment are DVP 12, GCF 18, TRI 16, and TRIV1 16.
Observed definitions by measure are average rate 26, outstanding volume 10,
and transaction volume 26.

- 7,368 normalized aggregation rows are null.
- 7,374 in-window dates carry an OFR disclosure-edit marker.
- 12 disclosure-marked rows retain a non-null aggregate; 6 null rows have no
  disclosure marker, consistent with OFR's separate no-trade/missing channel.
- The raw response also contains 954 pre-2019 and 4,464 post-2023 disclosure
  markers because that subseries ignores the requested aggregation window.
  They remain hash-bound audit evidence and create no normalized row.
- GCF `G30`, `LE30`, and `OO` rate/transaction-volume buckets each have 1,165
  null rows out of 1,249. A later candidate may not rely on those sparse buckets
  without failing a preregistered source-support gate.
- DVP totals, GCF totals/collateral totals, and TRI/TRIV1 rows provide stable
  multi-segment daily coverage through the frozen window.

## Causal clock

Every normalized row uses:

```text
max(observation_date + 8 elapsed calendar days,
    2020-09-10 00:00:00 UTC)
```

The floor prevents 2019 history from being treated as known before OFR began
publishing preliminary data. Missing values are never filled, interpolated,
converted to zero, or borrowed from the final vintage.

## Deterministic replay

Two consecutive offline builds produced identical files:

| Artifact | SHA-256 |
|---|---|
| normalized metadata gzip | `19a04e82eb5d8ddc6c3cb8dc64694438abd6b1987951470bb317659d9c53ef4f` |
| normalized observations gzip | `6bb24b9a3d6ed6d266b0c885a9d9888bcccb3045df66d37d83031af66c6dc02a` |
| build manifest file | `f937f567e1789ecb39a2b84d6288b2cbab931da4e9f1f4e51addea4b3423b705` |
| canonical manifest | `802b83a9478711cd29d5b606d9e12eb1e90890e37f5908d4de64d7dd71f6d449` |

## Verdict

The source axis passes the source-only gate with restrictions:

1. use only the 62 definitions that have frozen-window observations;
2. exclude the three severely sparse GCF tenor buckets from required support;
3. retain the conservative publication floor and eight-day lag;
4. do not substitute final values or out-of-window disclosure markers; and
5. preregister exactly one mechanism before computing any candidate value.

No claim of alpha or market predictiveness has been made. The next work unit is
an outcome-blind mechanism decision and source-support/novelty gate.
