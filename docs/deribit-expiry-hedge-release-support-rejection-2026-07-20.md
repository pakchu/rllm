# DEHR-72 — source-support rejection

## Verdict

**Reject DEHR-72 before opening any BTC market outcome.** The frozen singleton
source and candidate-support contract failed. No event clock was written, and
no Binance bar, funding row, post-entry path, return, PnL, CAGR, or MDD was
loaded or calculated.

The near misses are not repaired. Lowering the 2022-half or quarterly minimums
after seeing incidence would invalidate the candidate-level freeze.

## Source audit

Deribit source loader v3 completed its pre-2023 continuation scan and passed
all source-quality checks:

- 31 response pages;
- 30,357 BTC option rows and 107 futures rows excluded;
- 1,119 unique scheduled expiries: 52 in 2019, 337 in 2020, 365 in 2021,
  and 365 in 2022;
- first expiry `2019-01-04 08:00 UTC` and last expiry
  `2022-12-31 08:00 UTC`;
- 914 daily expiries in the eligible 2020-07 through 2022-12 interval;
- every eligible month complete, with maximum source gap one day; and
- one delayed delivery event, handled causally using actual reported delivery
  time plus the frozen 65-minute observation latency.

The aggregate is 61,160 bytes with SHA-256
`a59953eb0efddbab7a28af9fdd0f61f204fa98d2de330cf1a4090293378b0fda`.
It remains local/ignored. Raw responses were not persisted.

## Frozen candidate incidence

The q0.50 total-position plus q0.70 release-share singleton selected 159
events:

| Window | Events |
| --- | ---: |
| 2020H2–2022 total | 159 |
| train: 2020H2–2021 | 105 |
| 2020H2 | 55 |
| 2021 | 50 |
| test-support calendar 2022 | 54 |
| 2022H1 | **18** |
| 2022H2 | 36 |

Directional support was adequate: long/short shares were 38.1%/61.9% in
train, 40.7%/59.3% in 2022, and 39.0%/61.0% overall. Events covered 28 of 30
eligible months, and the largest month held 9.43% of events.

Quarterly counts were:

```text
2020Q3 21   2020Q4 34
2021Q1 25   2021Q2 10   2021Q3 9   2021Q4 6
2022Q1 7    2022Q2 11   2022Q3 12  2022Q4 24
```

## Failed immutable checks

Exactly two logical support checks failed:

1. `2022H1 >= 20`: observed **18**; and
2. every eligible quarter `>= 8`: observed **6** in 2021Q4 and **7** in
   2022Q1.

All source checks, overall/train/2022 totals, 2020H2 and 2021 totals, 2022H2,
active-month, side-balance, and month-concentration checks passed.

## Integrity boundary

- Successor preregistration artifact hash:
  `d1797ea2eae04a85f9d917e27412bc8456878bd4b1d350b461b4bd64208e4c1e`.
- Source manifest hash:
  `44b54dcd895a127dc89dc9c45f40f65c845814badeaf6c35bdabbc37e4e1b852`.
- Source manifest file SHA-256:
  `b1a2ed3a39b8e71adc0a46a5411d4f568eda3bdaa910cef64d9746fa6f5ea3e5`.
- Support result hash:
  `b118f24ac6e5796477865d3d9b95a3c0448b057f9b1fd30179588113e522a0f7`.
- Support result file SHA-256:
  `fcc33c324263bc10709041504b0b78fc055d83e18224a6c6622fa3c9f47c9231`.
- Event-clock artifact: absent by contract.
- Outcomes opened: false.

DEHR-72 must not be reopened under a looser support gate. A later Deribit idea
would require a new mechanism, new identifier, and a newly frozen protocol;
it cannot be presented as a repair of this candidate.
