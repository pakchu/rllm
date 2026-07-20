# UFCP-1 source-only support rejection — 2026-07-20

## Verdict

`UFCP-1` was rejected before any BTC execution price, funding cash flow,
forward return, PnL, equity, CAGR, or drawdown value was opened. The frozen
ledger-only clock passed the minimum event-count requirements, but failed the
preregistered direction-balance and month-concentration requirements.

The support builder was frozen in commit `dba8c6a` before the real source
incidence was inspected. No threshold, side rule, rank window, holding period,
or support floor may be repaired using the observations below. The train and
2023 outcome windows remain sealed and the strict evaluator must not be run for
this policy.

## Outcome-blind support result

| Window | Events | Long | Short | Largest month | Largest-month share | Gate |
|---|---:|---:|---:|---|---:|---|
| Train, 2021–2022 | 92 | 49 | 43 | 2022-11 (18) | **19.57%** | **fail** |
| Selection, 2023 | 120 | 117 | **3** | 2023-05 (31) | **25.83%** | **fail** |

The frozen maximum month share was 15% separately in train and selection. The
frozen side floor was 25% separately in each window. Train direction balance
passed, while selection was 97.5% long and 2.5% short.

All count floors passed:

- 2021: 32 events, against a minimum of 24;
- 2022: 60 events, against a minimum of 24;
- 2021–2022 total: 92 events, against a minimum of 60;
- 2023 H1: 78 events, against a minimum of 10;
- 2023 H2: 42 events, against a minimum of 10;
- 2023 total: 120 events, against a minimum of 24.

The source-integrity checks also passed: 213,095 exact contiguous blocks were
read, every usable UTC day had at least 72 blocks, fees and edge counts were
positive, no UTC source day was missing, the block hash chain was contiguous,
and the UTXO identity held.

## Why this rejects the mechanism

The 2023 polarity tail almost disappeared on the short side, while the admitted
events clustered in a few months in both declared windows. That is a source
support failure, not an unfavorable return result. Relaxing the 25/75 side
floor, raising the 15% month cap, changing the 0.75/0.25 ranks, or introducing a
calendar balancing rule after seeing this incidence would create a new
candidate and is prohibited under the singleton `UFCP-1` protocol.

## Reproducibility and sealing evidence

- support artifact:
  `results/utxo_fee_clearing_polarity_support_2026-07-20.json`
  - file SHA-256: `f1a9d9711b17ddd06f3f7c940cea109f63b618eab68ca2612d19e2b7b9b39031`
  - manifest hash: `fd77dd20f5568b705ffc0189af9390ec9cc8db7139824c1220e457b9f0c1dace`
- primary clock:
  `results/utxo_fee_clearing_polarity_primary_clock_2026-07-20.csv`
  - rows: 273
  - SHA-256: `8338c290d63b522531c8d55c8a79ba73cc13915c936733ec03ffcf6ab0e86c1b`
  - canonical frame hash:
    `1d3a87b54ac587c757fecd772d5584d8cf60dfc66fc5baee048fc15c96ada2ac`
- eight control clocks:
  `results/utxo_fee_clearing_polarity_control_clocks_2026-07-20.csv.gz`
  - rows: 2,556
  - SHA-256: `90434b0decc6dcfd6d950d1b2a235488b84ed5b348064faac536d6a5a64faa5f`
  - canonical frame hash:
    `8e9e1e149bc50c1f41a111fc2d1f9f00eb82d526fc52e82baf59c9600b3ed085`
- frozen preregistration SHA-256:
  `160efdd2eb857c47a80ec0ed4a976a659a1ee3dd3c930093d197798e619d65c9`
- frozen source SHA-256:
  `8d5160b59bd104819adb34752f2bd5b01e9c07d7aa108188abcd9e6d323d102f`
- frozen support-builder SHA-256:
  `ed1682ee65f189d63a79ffb6111b25525b0285c0cc97592b465b6fcbbb552505`
- runtime: 2.18 seconds; peak RSS: 283,592 KiB;
- market rows loaded: 0;
- funding rows loaded: 0;
- return rows loaded: 0;
- market/funding/return values read: 0;
- profit/loss fields read: 0.

`UFCP-1` is terminally rejected. Any follow-up must use a separately named,
outcome-blind preregistration on a genuinely distinct mechanism axis.
