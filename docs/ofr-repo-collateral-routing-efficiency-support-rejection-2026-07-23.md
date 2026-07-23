# RCRE-72 source-support rejection — 2026-07-23

## Decision

Retire `RCRE-72-SOURCE-REUSE` unchanged before comparator novelty or any BTC
market outcome is opened.

The signed routing interaction produced sufficient event frequency and passed
its causal, timing, side, product-sign, and venue-swap checks. It failed the
preregistered quadrant-diversity requirement, however. The accepted clock is
mostly generated with a positive GCF-minus-TRIV1 agency-share gap and therefore
does not demonstrate a venue-label-invariant routing mechanism across all four
signed quantity/rate configurations.

No threshold, side, lookback, hold, source field, support floor, or quadrant
gate is changed after observing this incidence.

## Frozen provenance

- mechanism decision SHA-256:
  `0b772a63093b39407e022cc7687cf8d49b0d476d465c3d0ee8177abf25b90629`;
- common-window policy SHA-256:
  `928bce6e04fb34001478b4b4ea84156580b661c88a0f0338065a891c009bd580`;
- preregistration artifact SHA-256:
  `1cd5c773e22101ae19d8d0753ca53e3248b10de3ecc947fc2e4e353961ee69e2`;
- preregistration manifest:
  `a8a7831c773666b42b98d673165b3e7844111c5d5ed280468307306c5843bf4b`;
- frozen support builder SHA-256:
  `be032fd7b2f17c9aa3cc5c42fef9e7da045586580c723ea9deac40756f29fd70`;
- source-support report SHA-256:
  `cd0ce324dfd5661898cee30603500eaf3e76f33604097392c765d7d1386e6451`;
- source-support report manifest:
  `d84ff0313b3d2dc0762799d90a959e75f6a0a57ed8b9186b7155c3567f872e9b`;
- source-only clock SHA-256:
  `cbe4e5f6fc52b66062abbf931e46ea4aa0d1f3c0157ffd365d0638aa573c2826`.

The support builder was committed before the exact signed quantity gap, signed
rate gap, product, states, or event incidence was computed.

## Source result

Source and causal checks:

- 77,369 normalized rows read;
- 9,976 required rows read;
- 1,249 source dates seen;
- 1,245 complete material feature dates;
- four missing/null or disclosure-edit-invalid dates;
- zero materiality-invalid dates;
- 417 equal-availability rows suppressed from decisions;
- exact venue-swap identity passed on all 1,245 complete dates;
- 827 strict-prior rank-ready decision rows;
- zero post-2023 source rows.

Accepted primary clock:

| Split | Events | LONG | SHORT | Max gap | Max month share |
|---|---:|---:|---:|---:|---:|
| Train 2021–2022 | 75 | 19 | 56 | 40 days | 8.00% |
| Selection 2023 | 39 | 19 | 20 | 26 days | 12.82% |

Both product signs passed their frozen minimum shares:

- train: 25.33% negative / 74.67% positive;
- selection: 48.72% negative / 51.28% positive.

## Exact failed gates

Only these three frozen checks failed:

1. `train_each_quadrant` — minimum required 10%:
   - `q+r+`: 48 / 75 = 64.00%;
   - `q-r-`: 8 / 75 = 10.67%;
   - `q+r-`: 18 / 75 = 24.00%;
   - `q-r+`: 1 / 75 = 1.33%.
2. `selection_each_quadrant` — minimum required 5%:
   - `q+r+`: 20 / 39 = 51.28%;
   - `q+r-`: 19 / 39 = 48.72%;
   - `q-r-`: 0 / 39 = 0%;
   - `q-r+`: 0 / 39 = 0%.
3. `train_quadrant_concentration` — `q+r+` is 64.00%, above the frozen 50%
   maximum.

All other preregistered source-support checks passed. In particular, event
floors, year/half/quarter activity, LONG/SHORT floors, month concentration,
maximum entry gap, product-sign shares, exact rational interaction, no-overlap
execution, label-pair identities, economic side controls, and source
provenance passed.

## Closed outcome boundary

Because source support failed:

- comparator rows read: **0**;
- novelty evaluated: **false**;
- BTC market rows read: **0**;
- funding rows read: **0**;
- future-return rows read: **0**;
- PnL/CAGR/MDD opened: **false**.

RCRE therefore has no profitability result. Reinterpreting the strong
`q+r+` concentration as an alpha, dropping the sparse quadrants, or lowering a
support floor would be an incidence-driven repair. Any successor must use a
new mechanism, ID, preregistration, and explicit contamination disclosure.
