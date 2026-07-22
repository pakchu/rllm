# QLCD-288 source-only support pass — 2026-07-20

## Verdict

**PASS_SUPPORT.** QLCD-288 produced `489` chronological, non-overlapping
24-hour events over 2020–2023 and passed every preregistered incidence,
calendar-dispersion, side-balance, and sparse-clock novelty gate. This permits
freezing the strict economic evaluator; it is not evidence that QLCD is
profitable.

No post-entry market row, funding row, return, label, PnL, equity, CAGR, or MDD
was read. The source-only evaluator was run twice and verified byte-identical
write-once artifacts on the second execution.

## Blinded source evidence

- complete UTC 5m grid: `420,768` rows;
- source-observed rows: `420,732`;
- verified zero-volume empty rows: `26`;
- source-gap-day rows: `1,728` across six quarantined UTC days;
- post-gap quarantine rows: `1,866`;
- source-complete rows after quarantine: `418,896`;
- raw eligible signal bars: `1,204`;
- scheduled non-overlapping events: `489`.

The source signal contains exact aggregate-event base-quantity denomination,
maker-side, and source-integrity fields only. It stores no price or notional
feature. The support evaluator entered only after the completed decision bar,
with the preregistered two-bar delay and fixed 288-bar hold.

## Support gates

| Gate | Frozen requirement | Observed | Result |
|---|---:|---:|---|
| total events | 200–800 | 489 | pass |
| each calendar year | >=35 | 111–139 | pass |
| each 2023 half | >=15 | 55 / 56 | pass |
| long share | 25%–75% | 48.67% | pass |
| short share | 25%–75% | 51.33% | pass |
| largest month share | <=15% | 3.68% | pass |

Annual counts were `113`, `126`, `139`, and `111` for 2020 through 2023.
The clock is therefore sparse, directionally balanced, and distributed across
all four source years without a dominant month.

## Frozen novelty result

All ten registered sparse comparator members passed. Across MFIC, AFCS, TAAR,
RIFT, PCP, and SMCC, the worst observed metrics were:

| Metric | Frozen maximum | Observed maximum | Comparator | Result |
|---|---:|---:|---|---|
| exact-entry Jaccard | 5% | 0.266% | SMCC-144 | pass |
| one-hour one-to-one Jaccard | 15% | 3.142% | SMCC-144 | pass |
| QLCD one-hour containment | 30% | 11.247% | MFIC fast | pass |

The dense BAFR clock was report-only as preregistered: exact Jaccard was
`0.111%` and QLCD exact containment was `2.658%`. No comparator error occurred.
This evidence supports clock novelty only; outcome independence and portfolio
value remain untested.

## Frozen artifacts

- preregistration manifest:
  `9fd76b3dd9fd0d900689684c9d6b1d2c57ede9877eec73979b3ff11d29f59a16`;
- source SHA-256:
  `3ca945f134115fc7b58086405fd881db3e3b70087bd9da54ffc293f6b658072e`;
- source manifest SHA-256:
  `bcdf89924f54a5b97d4219749c2094d2a4c08d8473a37bc5367d9b8e5791284f`;
- source-access seal SHA-256:
  `cade903a3d15349903c3e16853a23a092b36a293cb46ceb7b0c5514737aca834`;
- support clock SHA-256:
  `ed882ac8a28f1f0b2b7ad7bf3d2de1f37b175cde63b20d4d1c7a290f3eb89bec`;
- support result SHA-256:
  `d5b5f2e59fe2f8d8df775a9ee7a05da0bab2898af210d6e724669d9781efe640`.

Machine-readable decision:
`results/quantity_lattice_cohort_disagreement_support_2026-07-20.json`.

## Next boundary

Before any BTC price or funding outcome is opened, the economic evaluator must
be committed with the already frozen 2020–2022 train, 2023 selection, 2024
test, 2025 eval, and 2026 recent-report boundaries; 0.5x exposure; 6 bp per-side
base and 10 bp per-side stress costs; full-calendar CAGR; strict held-path MDD;
and the preregistered profitability and weekly-cluster sign-flip gates. No
signal threshold, side, delay, hold, cost, split, or gate may be repaired after
outcomes open.
