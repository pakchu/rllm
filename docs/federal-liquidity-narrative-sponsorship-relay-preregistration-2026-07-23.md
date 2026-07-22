# FLNSR-2016 preregistration — Federal liquidity narrative sponsorship relay

## Decision

The next standalone BTC candidate is **FLNSR-2016**. It combines two public,
causally available, individually weak axes: a one-release Federal Reserve
H.4.1 net-liquidity impulse and a contemporaneous change in Bitcoin narrative
composition. A trade exists only when both point in the same direction.

The event clock is the audited H.4.1 release availability, not a BTC market
event. A positive liquidity tail plus adoption-over-stress narrative rotation
creates a LONG; the symmetric negative conjunction creates a SHORT. Entry is
ten minutes after H.4.1 availability and hold is exactly seven calendar days.

## Honest history boundary

This is not pristine discovery evidence. FLCC's 2020–2022 H.4.1 outcomes and
GNRC's 2021–2023 GDELT outcomes are already known and both ancestors were
rejected. FLCC's 2023 window remained unopened. The exact FLNSR clock,
incidence, timestamps, and outcomes have not been computed at this commit.

FLNSR is not an FLCC threshold repair: it uses a one-release impulse rather
than four/eight releases, removes component breadth entirely, and requires a
different exogenous source to sponsor direction. It is not a GNRC repair: it
uses one threshold-free 7-versus-21-day rotation sign on an H.4.1 release
clock, not GNRC's 24-member score/threshold/hold family. These differences make
the event identity new, but prior ancestor results still weaken the discovery
claim and are disclosed explicitly.

## Frozen causal rule

### Federal liquidity

For H.4.1 release `t`:

```text
impulse[t] = net_liquidity[t] - net_liquidity[t-1]
rank_num   = 2*count(prior_104 < impulse[t])
             + count(prior_104 == impulse[t])
```

- LONG when `rank_num >= 125`;
- SHORT when `rank_num <= 83`;
- neutral otherwise.

The current impulse is excluded from the exact 104-release reference.

### Narrative sponsorship

At H.4.1 availability, select the latest GDELT source day whose audited
`available_at` is not later than the release. Require 28 exact consecutive
source days and define:

```text
q[d] = log((adoption[d] + 0.5) /
           (failure[d] + constraint[d] + 1.0))

rotation = mean(q over latest 7 days)
           - mean(q over immediately preceding 21 days)
```

- LONG when `rotation > 0`;
- SHORT when `rotation < 0`;
- neutral when exactly zero.

The two audited GDELT global-outage dates remain exact all-zero rows; the
frozen pseudocount keeps them finite. No fill or future source row is allowed.

### Candidate and execution

- candidate only when liquidity and narrative sides are equal and non-neutral;
- signal at H.4.1 `available_at_utc`;
- decision and entry at `available_at_utc + 10 minutes`;
- exit at `entry + 2,016 five-minute bars` / seven calendar days;
- 0.5x, 6 bp/notional/side base cost, 10 bp stress cost;
- exact funding on `[entry, exit)`;
- chronological non-overlap, allowing entry exactly at prior exit;
- complete split containment; no stop, TP, dynamic exit, price gate, or model.

## Frozen source support

Before any BTC outcome is read, the clock must have:

- at least 24 train events and six in each of 2020/2021/2022;
- at least eight 2023 events and three in each half;
- both sides between 25% and 75% in all/train/selection;
- at least 20 active months;
- no month above 15%, quarter above 30%, gap above 120 days, or same-side run
  above six.

Controls are liquidity-only, narrative-only, disagreement, exact side flip,
one-release stale narrative, and hash-seeded random side. They share execution
and clock schema, are report-only, and cannot rescue or reject primary.

The source-only novelty audit compares accepted FLNSR entries with every
frozen FLCC primary candidate using deterministic one-to-one matching within
±15 minutes. For every comparator, Jaccard must be at most 0.50, FLNSR entry
containment at most 0.70, and same-side FLNSR containment at most 0.75. The
hash-bound FLCC clock ledger is read as timestamps/sides only; no comparator
outcome may be loaded. Failure retires FLNSR as an H.4.1 re-expression.

Failure retires FLNSR-2016 without computing a BTC return. The q60 integer
tails, narrative formula, entry, hold, side, cost, and gates may not be repaired
after incidence is opened.

## Frozen economic sequence and RLLM boundary

After a support pass, a separately committed hash-bound evaluator opens train
2020–2022, then 2023 only after train passes, then 2024/2025/2026 sequentially.
Every opened stage requires positive absolute return, CAGR/strict-MDD at least
3, strict MDD at most 15%, base/stress and one-bar-delay profitability, at
least 30 bp mean gross underlying movement, and monthly-cluster sign-flip
`p <= 0.05`. Train years and 2023 halves must be positive. Primary mean gross
must exceed each liquidity-only, narrative-only, and disagreement component by
at least 5 bp.

FLNSR is one frozen singleton, not another tunable family. The known FLCC and
GNRC ancestor outcomes remain disclosed and prevent a global first-discovery
claim even if the singleton later passes.

Only after deterministic train and selection pass may a separately
preregistered compact LLM choose `TRADE_FIXED_SIDE` or `ABSTAIN`. Its useful
input is symbolic relation—liquidity shock bucket, asset/TGA/RRP contribution
signs, adoption-versus-stress rotation, evidence recency, and current
position—not raw-number forecasting. It cannot create, reverse, resize, or
retime a candidate.

## Frozen artifact

Run:

```bash
.venv/bin/python -m training.preregister_federal_liquidity_narrative_sponsorship_relay
```

The canonical checked-in artifact is
`results/federal_liquidity_narrative_sponsorship_relay_preregistration_2026-07-23.json`.
