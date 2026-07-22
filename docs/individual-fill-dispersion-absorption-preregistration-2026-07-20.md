# IFDA-72 preregistration — 2026-07-20

## Status

**Frozen before raw-trade incidence and before market outcomes.**  IFDA-72 is a
single six-hour policy, not a threshold, direction, or hold grid.

- outcomes opened: `false`
- source incidence opened: `false`
- policy manifest:
  `results/individual_fill_dispersion_absorption_preregistration_2026-07-20.json`
- manifest hash:
  `757401202eacd3dcf0540ae54f4c121ba595d2495d0f4ae69fe248bfa360e02c`
- manifest file SHA-256:
  `abecfddfaad7a7640d8d16fc04ff3938dad79cfa94d4b97095d01baf9f7c9b70`

The source-axis decision and TCDA source-feasibility rejection are recorded in
`docs/individual-fill-dispersion-absorption-source-axis-decision-2026-07-20.md`.

## Frozen mechanism

For every completed five-minute bar, each official USD-M individual trade has
notional `q_i = quoteQty` and aggressive side `+1` when
`isBuyerMaker=false`, otherwise `-1`.

Let `B` and `S` be aggressive-buy and aggressive-sell notional.  The dominant
side and flow coherence are:

```text
d = sign(B - S)
C = abs(B - S) / (B + S)
```

Within each aggressive side:

```text
HHI = sum(q_i^2) / sum(q_i)^2
E   = (1 / HHI) / fill_count
```

`E` is one when all reported fills on that side have equal notional and tends
toward `1/fill_count` as one fill dominates.  With `E_d` for the dominant side
and `E_o` for the opposing side:

```text
G     = max(E_d - E_o, 0)
score = C * E_d * G
side  = -d
```

The fixed hypothesis is that a coherent aggressor wave split across unusually
equal-sized individual matches, relative to the other side, met broad passive
absorption and subsequently reverts.  This interpretation does not identify a
participant, maker, parent order, or resting order-book owner.

## Frozen scheduler

- Stable bar: source complete, at least 128 total fills and 32 fills on each
  aggressive side, nonzero `d`, positive `G`, and positive score.
- Baseline: previous 8,640 five-minute bars, minimum 2,016 observations.
- Threshold: strictly prior `q99.5`; the current row is excluded with one-row
  shift.
- Signal: inactive-to-active crossing only.
- Entry: `t+2` open after completed decision bar `t`.
- Exit: open 72 bars after entry, exactly six hours.
- Global non-overlap; re-entry at the scheduled exit timestamp is allowed.
- Exposure: 0.5x.
- No grid, branch selection, direction inversion, regime gate, stop, or
  threshold repair is authorized.

## Outcome-blind support gates

The 2020–2023 source/clock must satisfy all of:

- 250–1,100 non-overlapping events;
- at least 50 events in every calendar year;
- at least 20 events in each 2023 half;
- each side between 25% and 75%; and
- no month above 15% of all events.

Every own-clock mechanism control must have at least 125 events and at least 20
per year.  Sparse prior comparator artifacts fail closed.  Novelty limits are:

- exact-entry Jaccard at most 5%;
- one-hour one-to-one Jaccard at most 15%; and
- primary containment at most 30%.

## Frozen falsification controls

1. exact side flip on primary clocks;
2. the same equalization formula on aggregate-event notionals;
3. flow-coherence-only fade;
4. remove cross-side equalization asymmetry;
5. all-fill equalization without side-specific dispersion; and
6. primary clock shifted by one hour and by 24 hours.

The aggregate-event control is mandatory.  IFDA can only survive economics if
individual-fill granularity beats that control in both absolute return and
CAGR/strict-MDD, with at least a 0.50 ratio margin independently in train,
selection, test, and eval.  A ratio is finite `CAGR_pct / strict_MDD_pct` with a
strictly positive denominator; zero/nonfinite MDD fails closed.  Every own-clock
control is written to one frozen control-clock bundle and its schema/hash is
mandatory in the support result.

## Sequential economic contract

Only a support and novelty pass may authorize a separately committed strict
evaluator.

1. Train: 2020–2022 only.  Require positive absolute return and stress return,
   CAGR/strict-MDD at least 1.5, MDD at most 20%, every year positive, mean gross
   underlying move at least 24 bp, and weekly-cluster `p <= 0.10`.
2. Selection: 2023 only.  Require positive absolute and stress return,
   CAGR/strict-MDD at least 3, MDD at most 15%, both halves positive, at least
   40 trades, and weekly-cluster `p <= 0.10`.
3. Only a frozen selection pass may open 2024 test, then 2025 eval, under the
   same ratio/MDD/statistical gates.  2026 remains report-only.

Accounting uses 6 bp/notional/side base cost, 10 bp stress cost, exact realized
funding, full-calendar CAGR including idle cash, and strict held-path MDD.

## Source and live constraints

- Historical source: official Binance USD-M daily `trades` archives with every
  `.CHECKSUM` verified.
- Raw ZIPs are parsed one day at a time and never persisted.
- The build aborts before a download once used filesystem space reaches
  300 GiB.
- Missing archives, non-contiguous trade IDs, unknown maker flags, malformed
  rows, or checksum revisions fail closed.
- No future OHLC, funding PnL, return, label, or prior alpha outcome may be read
  by the source/support stage.

USD-M provides individual trades through REST rather than a documented raw
futures trade WebSocket.  Archive `trade Id` and REST `id` normalize to the same
`trade_id`; the other shared fields normalize directly.  Current REST responses
also include `isRPITrade`: it must be a valid boolean and is counted for audit,
but is excluded from the frozen IFDA feature because the 2020–2023 archive does
not expose it.

The collector polls at most one second apart, uses the 1,000-row recent endpoint,
and repairs gaps by inclusive ID through the API-key-gated 500-row historical
endpoint.  That endpoint is limited to one month.  Any unresolved gap suppresses
new orders immediately; a gap still unresolved after 15 minutes or beyond the
one-month lookback halts IFDA and requires checksum-archive rebuild.  Live
promotion requires exact historical/live feature equivalence.  Failure leaves
IFDA research-only even if historical economics pass.

## Rejection rule

Any source, disk, support, control-support, novelty, economic, incremental-
granularity, or live-parity failure retires **IFDA-72** without repair.
