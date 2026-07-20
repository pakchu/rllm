# BDRC-864 source-only support rejection — 2026-07-20

## Verdict

**REJECT.** The frozen Bank-Deposit / Secured-Repo Concordance policy does not
produce enough independent events to justify opening BTC outcomes:

- 2020-2022 train support is `37`, below the frozen floor of `45`;
- 2021 contributes only `3` events, below the per-year floor of `10`; and
- train has `11` long events, below the per-side floor of `15`.

The 2023 selection clock is balanced enough (`17` events, `9` long and `8`
short), but it cannot repair the train deficiency. The mechanism is therefore
stopped without lowering support floors, adding Monday releases, changing the
SOFR lag, thresholding the observed SOFR magnitude, flipping direction, or
changing the 72-hour hold.

No BTC price, OHLC, funding, premium, OI, liquidation, future return, PnL,
equity, CAGR, or MDD value was loaded or calculated. There is deliberately no
performance table for BDRC-864.

## Frozen artifacts

- mechanism decision:
  `docs/bank-deposit-secured-repo-concordance-mechanism-decision-2026-07-20.md`;
- mechanism decision commit:
  `a0b7c319b874c662e1daad9cb5533c0984445dd6`;
- mechanism decision SHA-256:
  `142db103f211800fde8233e4e39e6907bb02ad7646c150f97d6c0adb8b92d09c`;
- source-support result:
  `results/bank_deposit_secured_repo_concordance_support_2026-07-20.json`;
- support-result file SHA-256:
  `74fea33f6b65eed01710824d57e339fbbec9245686ef13a58510a98cbdd1217c`;
- canonical manifest hash:
  `5cfe9968c579a64e04879e3b47d811e6797da93eaed073255c9ab7d2b5f66f5a`;
- combined source clocks:
  `results/bank_deposit_secured_repo_concordance_clocks_2026-07-20.csv.gz`;
- combined-clock SHA-256:
  `1ff3a6075e3ceff928e1dd19d05880dbe9dbab0e07d79b853146d7b4c8f6cabc`;
- source-support evaluator SHA-256:
  `bb7042b6ab621bc8cda05c05129c0cc80380cb974eceaf9abbd55ca664f9267b`.

Two complete source-only runs reproduced the result and gzip clock files
byte-for-byte. The measured successful run took `0.84` seconds wall time and
`126,112` KiB maximum RSS.

## Frozen mechanism

The candidate joins two independently timestamped weak signals at the archived
H.8 release clock:

1. H.8 bank stress is the equal-weight sign of three seasonally adjusted,
   prior-104-release robust states: large-minus-small other-deposit growth,
   small-bank borrowings growth, and negative small-bank cash growth. At least
   two component signs must agree.
2. Secured-repo tightening is the exact integer-basis-point SOFR change from
   the latest published observation to the observation five publications
   earlier. The latest observation must be no more than 36 hours old.
3. Concordant positive states enter short; concordant negative states enter
   long. Discordant or zero states do not trade.
4. Only Thursday and Friday H.8 releases are eligible. The decision is fixed at
   17:00 America/New_York, entry at 17:05 after one complete five-minute
   latency bar, and exit 864 five-minute bars later. Overlapping events are
   skipped.

The mechanism file and its hash were committed before the source incidence
review. All robust statistics are computed from observations preceding the
current release; the SOFR state uses only values whose publication timestamp is
at or before the decision timestamp.

## Source integrity

| Source | Rows read | Frozen SHA-256 |
|---|---:|---|
| Federal Reserve H.8 panel | 365 | `c8d1bfb0bbd13ef6d35f09ad7367ef8d2d5bb28981376223b735746ade68a572` |
| New York Fed SOFR panel | 1,436 | `4993eda2b659e346b4d7b6e3aa0e2ff31cacf868f0e1fe2e1a5a76a03d1b5852` |

The evaluator also hash-verifies the two build manifests and the frozen H8DM,
SFRD, and FLCC parent clocks before constructing any event. Source rows after
2023 are structurally unavailable to this evaluator. Every market/funding/
return counter in the artifact is zero and `outcomes_opened` is `false`.

## Event support

| Window | Total | Long | Short | `|SOFR Δ5| <= 3 bp` |
|---|---:|---:|---:|---:|
| 2020 | 15 | 7 | 8 | 13 |
| 2021 | 3 | 1 | 2 | 2 |
| 2022 | 19 | 3 | 16 | 11 |
| 2020-2022 train | 37 | 11 | 26 | 26 |
| 2023 H1 | 9 | 5 | 4 | 7 |
| 2023 H2 | 8 | 4 | 4 | 8 |
| 2023 selection | 17 | 9 | 8 | 15 |

Train maximum month share is `8.1081%` and selection maximum month share is
`11.7647%`, so calendar concentration is not the failure. The clock is sparse
because the exact-sign concordance removes most parent events, especially in
2021. Separately, `26/37` train events have an absolute five-observation SOFR
move of three basis points or less. That observation is diagnostic only; adding
a magnitude threshold after seeing it is prohibited.

The source-only controls reinforce the support diagnosis:

- the H.8-only clock has `125` train events;
- SOFR-only on the H.8 schedule has `85` train events; and
- exact concordance retains only `37` train events.

Thus neither raw source is intrinsically too sparse. The preregistered
conjunction is.

## Parent-clock diagnostics

The complete source clock has 75 primary events: 21 in 2019 and 54 in the
2020-2023 diagnostic horizon. Parent overlap and exposure statistics use that
same 2020-2023 horizon. Signed five-minute occupied-exposure correlations are:

| Comparator | Correlation | Entry Jaccard within ±6h |
|---|---:|---:|
| H8DM-1 | +0.3874 | 0.2966 |
| SFRD-1 | +0.1956 | 0.0559 |
| FLCC-H4-Q60 | +0.0761 | 0.0053 |
| FLCC-H4-Q65 | +0.0526 | 0.0057 |
| FLCC-H8-Q60 | +0.1140 | 0.0056 |
| FLCC-H8-Q65 | +0.1316 | 0.0059 |

These are diagnostics, not a route around the failed support gate. A distinct
clock with too few train observations is not statistically adequate.

Every frozen control also reports exact entry overlap and signed occupied-
exposure correlation against primary. As expected, direction-flip and random-
side controls retain exactly the same entries; the direction flip has `-1.0`
signed exposure correlation. H.8-only and SOFR-only retain every primary entry
but add many others, producing exact-entry Jaccards of `0.3214` and `0.4576`.
These diagnostics are descriptive and do not override the failed source gate.

## Stopping decision

The frozen source gate authorizes a strict BTC evaluator only if every support
condition passes. Three conditions failed, so no outcome evaluator will be
created for BDRC-864. Any variant that changes the event calendar, SOFR lag,
direction, agreement rule, support floors, or hold must begin as a genuinely
new preregistered mechanism, not as a repair of this rejected singleton.
