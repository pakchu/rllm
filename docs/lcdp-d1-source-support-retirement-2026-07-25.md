# LCDP-D1 source-support retirement

Date: 2026-07-25

## Decision

**RETIRE LCDP-D1 unchanged before token-sequence materialization, reward
construction, model training, or market-outcome access.**

The official source-only run used:

```text
runner commit
  92f9fe11cd1047340c042c2b1ec3796add6523bf
runner SHA256
  d1fa16c8b57154e8102902f17bf7032e65a8f4cfc5cc5098b561d390cb285bda
tests SHA256
  51c015292c683ef55e090e9e7d5bf32f21fb4828102b3d8b69fe6c9f0445dbcf
execution-seal SHA256
  3a1aac307a06eaa22c08651957d07bff62ff542e4aadb4feb263a67d52cd7047
execution-seal manifest_hash
  55aa5c0081b23c2e0789e57085db8bf095729ff627581c5775d1fc5a028904d1
```

Machine-readable evidence:

```text
results/lcdp_d1_source_support_rejection_2026-07-25.json
SHA256 eb93e2a2dc1c7660a230dfc098a44ece96af3a2c94c18f391e8f664271841b9b
result_hash 7986295a8b7aceb96a827bcb14ab4bffb2b46a3121d6d290a68f01c454544c8e
```

## Exact failure

Gate 1 and Gate 2 passed:

```text
1  protocol_source_integrity  pass
2  calendar_dst_integrity     pass
```

Gate 3 failed both frozen source-validity checks:

```text
3  source_validity  fail
```

Annual validity:

| Year | Valid days | Calendar days | Valid share | Minimum |
|---|---:|---:|---:|---:|
| 2020 | 355 | 366 | 96.9945% | 97.00% |
| 2021 | 363 | 365 | 99.4521% | 97.00% |
| 2022 | 364 | 365 | 99.7260% | 97.00% |

The 2020 gate needed 356 valid days. It missed by exactly one day.

Quarterly validity:

| Quarter | Valid days | Calendar days | Valid share | Minimum |
|---|---:|---:|---:|---:|
| 2020Q1 | 89 | 91 | 97.8022% | 95.00% |
| 2020Q2 | 90 | 91 | 98.9011% | 95.00% |
| **2020Q3** | **87** | **92** | **94.5652%** | **95.00%** |
| 2020Q4 | 89 | 92 | 96.7391% | 95.00% |
| 2021Q1 | 90 | 90 | 100.0000% | 95.00% |
| 2021Q2 | 90 | 91 | 98.9011% | 95.00% |
| 2021Q3 | 92 | 92 | 100.0000% | 95.00% |
| 2021Q4 | 91 | 92 | 98.9130% | 95.00% |
| 2022Q1 | 90 | 90 | 100.0000% | 95.00% |
| 2022Q2 | 91 | 91 | 100.0000% | 95.00% |
| 2022Q3 | 91 | 92 | 98.9130% | 95.00% |
| 2022Q4 | 92 | 92 | 100.0000% | 95.00% |

The 2020Q3 gate needed 88 valid days. It also missed by exactly one day.

## First-stop evidence

The runner stopped at Gate 3. It did not calculate or report:

- model-eligible readiness;
- token-field diversity;
- source-control distinctness;
- append replay;
- a token-support row hash;
- funding or execution paths;
- rewards or model rows;
- actions or trades; or
- PnL, CAGR, or MDD.

It did not write:

```text
data/lcdp_d1_source_support/token_support.csv.gz
results/lcdp_d1_source_support_2026-07-25.json
```

Every forbidden counter is zero:

```text
funding rows opened                         0
execution/post-boundary rows opened         0
future-return rows built                    0
reward/model/action/trade rows built        0
PnL/CAGR/MDD values computed                0
at-or-after-2023 non-date rows parsed       0
```

LCDP-D1 therefore has no alpha, action, profitability, CAGR, MDD, or
deployability result.

## No repair under this identity

Do not:

- lower 97% to the observed 96.9945%;
- lower 95% to the observed 94.5652%;
- round either observed share up;
- drop `SOURCE_INVALID_START` from the denominator;
- omit or relabel an invalid 2020 day;
- fill Coinbase gaps from Binance or another provider;
- add pre-2020 source rows to repair the first line; or
- inspect later token/economic gates to justify a source exception.

Each change uses observed gate incidence and changes the frozen identity.
Readiness, token controls, rewards, RLLM training, and every economic stage are
permanently unauthorized for `LCDP-D1`.

A successor may use a separately justified source-validity policy or a source
whose point-in-time coverage is independently stronger, but it requires a new
candidate ID, boundary, preregistration, implementation contract, and sealed
source-support run.
