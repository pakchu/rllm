# MCR-7 miner cadence recovery preregistration

Status: **frozen before any exact-policy post-entry BTC return was loaded**.

MCR-7 uses only Coin Metrics `HashRate`, `BlkCnt`, and recorded daily
availability. A seven-day hash-rate change must have been at least one prior-only
standard deviation below normal during the prior 14 observations, then cross
back above its prior-only mean while three-day block cadence is no worse than the
strictly earlier 30-day reference. The side is long only.

- entry: first 5m open after availability, plus one complete 5m latency bar
- hold: 2016 five-minute bars / seven days
- exposure: 0.5x
- cost: 6 bp/notional/side base; 10 bp/notional/side stress
- 2021-2022: development train
- 2023: one frozen-policy selection year
- 2024+: sealed until every earlier gate passes

Support thresholds were shaped from source timestamps and feature counts only;
no market, funding, post-entry return, CAGR, or drawdown was loaded. Broader repo
research has seen old BTC returns, so 2023 is not described as pristine OOS.

Any failed gate retires this exact policy without threshold, side, hold, or
latency repair. Live promotion additionally requires forward-vintage parity and
90 shadow days.

Protocol hash: `20c5aa201e36169c775d64c2882361e3e21bb30c9b0ea88b2888d1b7281d14a1`
