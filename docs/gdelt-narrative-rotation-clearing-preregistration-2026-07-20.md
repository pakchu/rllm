# GNRC narrative rotation/clearing preregistration

## Hypothesis

Bitcoin news categories are individually weak signals. Their **composition and
rotation** may contain information that raw attention alone does not:

- adoption/approval narratives rotating above failure and regulatory stress are
  constructive;
- failure/constraint narratives clearing after a concentrated scare are
  constructive;
- failure narratives replacing adoption under elevated broad attention are
  destructive;
- regulatory attention is constructive only when adoption rises with it and
  failure does not dominate.

This document freezes the finite source-only family before the GDELT daily count
artifact is opened. No price, funding, return, PnL, CAGR, or MDD is available to
this decision.

## Causal source

The source contract is frozen in
`docs/gdelt-bitcoin-narrative-source-protocol-2026-07-20.md` and transport commit
`22af253848939855aa456b2e2a5dda02e01a84c5`.

For source day `t`, let `G/B/F/C/A` be global, broad Bitcoin/crypto, failure,
constraint, and adoption article counts. Each row becomes available at source
midnight + 48 hours + 15 minutes. A signal decision occurs exactly then and the
BTC entry is ten minutes later. The LLM may never control this clock.

Base features use a fixed `0.5` pseudocount:

```text
xB = log((B + 0.5) / (G + 0.5))
xF = log((F + 0.5) / (B + 0.5))
xC = log((C + 0.5) / (B + 0.5))
xA = log((A + 0.5) / (B + 0.5))
```

For `(fast, slow)` in `{(7,28), (14,56)}`, current-fast is `[t-f+1,t]`,
previous-fast is `[t-2f+1,t-f]`, and slow is `[t-s+1,t]`, all inclusive. `Zx`
is current-fast mean minus slow mean. `CLRx` is previous-fast mean minus
current-fast mean. Both are divided by `max(1.4826*MAD(slow), 1e-6)`. Mean is
arithmetic; for an even sample the median averages the middle two observations.
At least `slow` consecutive daily rows are required. Every row must satisfy
`available_at = source date UTC midnight + 48h15m`.

```text
Risk          = 0.5 * (ZF + ZC)
Quality       = ZA - Risk
ClearRisk     = 0.5 * (CLRF + CLRC)
ClearAdoption = CLRA

rotation long  = Quality - 0.25*max(ZB, 0); short = -long
clearing long  = ClearRisk - ClearAdoption
                 + 0.25*Quality - 0.25*max(ZB, 0); short = -long
rule long      = min(ZA, ZC) - ZF - 0.25*max(ZB, 0)
rule short     = ZF + 0.5*max(ZC, 0) - ZA + 0.25*max(ZB, 0)
```

## Frozen family

The family has exactly 24 variants:

```text
3 scores × 2 window pairs × 2 absolute thresholds × 2 holds
threshold ∈ {0.5, 1.0}; hold ∈ {3, 7} calendar days
```

- long when `long_score >= threshold`
- short when `short_score >= threshold`
- flat when both sides trigger on the same decision
- otherwise flat
- daily decisions, first qualifying event after flat
- derive every source date whose `source_date + 48h25m` entry and full hold are
  contained in the split; require exact equality to that complete UTC daily
  grid and process unique rows in ascending `(available_at, source_date)` order
- reset flat independently at each split
- admit only when `entry > previous admitted exit`; same-time roll is prohibited
- entry at `available_at + 10 minutes`, fixed 1.0x notional
- exit exactly 3 or 7 calendar days after entry
- require `split_start <= entry` and `exit < split_end_exclusive`

## Source evidence and support gates

Before a decision can trigger, its slow window must contain at least `10*slow`
broad articles, at least three articles in every category, and at least two
nonzero days in every category.

An **eligible decision** has full feature history, passes that evidence gate,
and has split-contained entry/exit before score or non-overlap is considered. A
raw directional trigger is an eligible decision with either side at threshold.
An admitted event remains after conflict and non-overlap. Active share is
admitted events divided by eligible decisions. Year, half, side, and month
counts use admitted events attributed by UTC entry timestamp; maximum month
share is the largest UTC entry-month count divided by admitted events.

Each variant must independently have:

- train 2021–2022: at least 24 events, at least eight per year, at least five
  long and five short;
- selection 2023: at least 10 events, at least four per half, at least two long
  and two short;
- active decision share in `[3%, 40%]`;
- maximum month share no more than 20% in train and 30% in selection.

The family advances only if at least eight variants pass and every score
archetype and window pair remains represented. The executable family predicate
requires all 24 frozen IDs as input and fails closed on a missing ID or
non-boolean gate. Failure retires the family without threshold, sign, hold, or
window repair.

## Later economic selection

Only source-supported variants may open BTC outcomes. Train is 2021–2022 and
selection is 2023, reset independently with strict split containment. Starting
equity is 1.0 and units are `side*equity_before_entry_cost/entry_open_price`. Base costs
deduct 2 bps of executed notional at entry and exit; stress costs deduct 4 bps
at each side. At every Binance funding timestamp after entry and at or before
exit, cash changes by `-units*funding_mark*funding_rate`.

Equity is marked on every five-minute close, including flat intervals. Within
each fully held bar after entry and strictly before exit, the strict path visits
the favorable extreme before the adverse extreme (long: high then low; short:
low then high), maximizing possible peak-to-trough drawdown. Exit occurs at the
scheduled bar open before that bar's extremes, followed by exit cost.
Absolute return is `E1/E0-1`. CAGR is
`(E1/E0)^(365.2425/full_calendar_days)-1`. Strict MDD is the maximum decline
from running peak over that full marked/intrabar path. Absolute return, CAGR,
strict MDD, CAGR/strict-MDD, and trade count are always reported together.
The evaluator must receive exactly every UTC five-minute bar-open in the full
split. It must also receive exactly one BTCUSDT funding row at each UTC 00:00,
08:00, and 16:00 timestamp in the split; omitted bars or funding rows fail the
run before metrics are computed.

The single champion is selected from train-and-selection qualifiers by highest
2023 CAGR/strict-MDD, then lower 2023 strict MDD, then lexical variant ID. The
24-variant family requires a one-sided Romano-Wolf max-t test on 2023 UTC daily
net log-equity returns. The statistic is `sqrt(n)*mean(r)/std(r,ddof=1)` under
`mean(r)<=0`; each series is mean-centered and all variants use the same
circular seven-day block indices for 100,000 draws with seed `20260720`.
The JSON preregistration embeds all 24 exact variant IDs as the test family.
Source-unsupported, train-ineligible, or zero-variance variants receive
adjusted `p=1`; controls are diagnostic-only.
The champion must have adjusted `p <= 0.10`.

OOS uses two executable seals. Before any 2024+ news request,
`results/gnrc_oos_source_access_seal_2026-07-20.json` must bind the champion,
selection report, source downloader, evaluator, output path, fixed
`[2024-01-01, 2026-07-01)` interval, and no-interim declaration. After the
outcome-blind source download and before any OOS BTC market/funding read,
`results/gnrc_oos_market_access_seal_2026-07-20.json` must additionally bind the
source-access seal hash, exact source-output hash, source manifest hash, exactly
912 daily rows, and assert that neither source feature values nor any outcome
was inspected. Seal paths are canonical repository-relative paths; their real
file hashes are checked. The hashed gzip CSV is structurally parsed: its schema,
canonical dates, `+48h15m` availability timestamps, and exact sorted unique
daily grid over `[2024-01-01, 2026-07-01)` must match. The market seal must
preserve every shared source-seal field and have a non-earlier timestamp. Both
use the code-enforced `gnrc_oos_access_seal_v1` schema. Final
OOS goals are positive absolute return, `CAGR/strict MDD >= 3`, strict MDD
`<= 15%`, and at least 50 trades.

## RLLM boundary

RLLM is **not** part of the GNRC primary claim: abstention changes included
trades and sizing changes exposure, so it is another strategy family. Any RLLM
overlay requires a separate preregistration with model, prompt, parser,
checkpoint, training-data hashes, multiplicity correction, and a holdout not
opened by deterministic GNRC. GNRC's primary OOS cannot be reused for an RLLM
claim. Until that separate freeze, all RLLM output is exploratory only; it may
never retime, reverse, or create an event.
