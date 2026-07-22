# BIRB-120 mechanism decision — 2026-07-23

## Decision

Preregister one new source-family-seen, outcome-blind singleton:
**BIRB-120 — SEC Bitcoin Issuer Reactivation Breadth**.

This decision was made without deriving the exact BIRB incidence and without
reading BTC bars, funding, future returns, PnL, CAGR, or MDD. The next step may
open only the frozen 2018–2023 SEC metadata rows to build source-support and
novelty clocks. Calendar 2024 and later remains sealed.

## Why this mechanism, not another semantic-prompt repair

The SEC EDGAR source has a causal UTC acceptance clock, broad issuer coverage,
and no BTC outcomes have been opened for either prior SEC candidate. However,
the two previous candidates failed their frozen Gemma synthetic interfaces:

- `EBCT-72` attempted balance-sheet constraint-state extraction;
- `BPAX-120` attempted customer Bitcoin product-access extraction.

Neither failure authorizes prompt, casing, model, or threshold repair. BIRB is
therefore a new mechanism and identity. It uses only immutable filing metadata
that was already source-audited: accession, CIK set, amendment flag, and
official `acceptanceDateTime`. It does not fetch filing bodies and does not use
an LLM to manufacture source labels.

## Economic hypothesis

A new Bitcoin filing from a habitual filer is often repeated publicity or
routine reporting. A filing after the issuer has been absent from the frozen
Bitcoin-hit stream for at least 365 elapsed days is different: it marks renewed
public-company engagement after a full annual reporting cycle. Three distinct
issuer reactivations inside seven elapsed days form a cross-issuer diffusion
event rather than one company's repetition.

The preregistered hypothesis is long-only delayed assimilation. On the third
distinct reactivation, enter long BTC for 120 elapsed hours. Filing prose is
not used, so BIRB makes no claim that each filing is individually positive.
The breadth requirement is the directional object. First-ever issuer births
are excluded from the primary and retained as a mechanism control because they
do not demonstrate re-engagement.

## Frozen source clock

1. Use only non-amendment `6-K` and `8-K` accessions from the frozen SEC source.
2. Deduplicate by accession and assign the smallest numeric CIK as issuer key.
3. Historical ready time is official UTC `acceptanceDateTime + 60 minutes`.
4. Process accessions in `(ready_time, accession)` order. Equal ready times are
   simultaneous and may not use intra-batch ordering.
5. An issuer reactivation occurs only when its immediately previous eligible
   Bitcoin-hit accession is at least 365 elapsed days earlier. First-ever hits
   and gaps below 365 days are not primary events.
6. Maintain the distinct reactivated issuers in `(current_ready - 7 days,
   current_ready]`. Signal at the first atomic batch that moves the count from
   below three to at least three. Expiry alone never emits a signal.
7. A CIK may contribute at most once to a seven-day breadth episode. A new
   signal requires the state to return below three and later cross again.
8. Entry is `ceil_5m(signal_ready) + 5 minutes`, including when the signal is
   exactly on the grid. Exit is 120 elapsed hours later. Exposure is 0.5x long.
9. Reserve accepted events globally on `[entry, exit)`. Suppressed candidates
   are not queued. Split-crossing trades are skipped. No stop, take-profit,
   trailing exit, or side override is allowed.

## Frozen splits and gates

- source warmup: 2018–2019;
- train: 2020–2022;
- selection: 2023;
- sealed: 2024 onward.

Source support must pass before a BTC row is opened: at least 24 train events,
8 selection events, 6 in each train year, 3 in each 2023 half, at least 18
distinct train issuers and 6 selection issuers, no month above 20%, no quarter
above 40%, at least 8 active train quarters, and no gap above 150 days. Exact
entry Jaccard and ±12-hour containment are checked against the frozen prior
microstructure bundle, the Trollbox semantic clock, and current live-portfolio
pure clocks.

Source controls are fixed before incidence:

- `first_ever_birth_breadth`: same 3-in-7-day rule using first-ever CIK hits;
- `any_mention_breadth`: any eligible accession, one issuer contribution per
  episode;
- `repeat_filer_breadth`: gaps below 90 days;
- `single_reactivation`: each reactivation directly emits;
- `stale_30d`: shift each reactivation ready time by 30 elapsed days before
  breadth construction;
- `year_cik_permutation`: deterministic within-year permutation of issuer keys;
- `threshold_two` and `threshold_four`: report-only breadth specificity checks.

The primary must not collapse to birth, repeat-filer, any-mention, single-event,
or stale clocks under the preregistered overlap/retention caps. Permutation may
not preserve the primary event identities beyond its cap.

If source support passes, a separately committed strict evaluator must use
realized funding, 6 bp per side base cost, 10 bp stress cost, full-calendar
CAGR, and high-water/intratrade strict MDD. Train and selection each require
positive absolute return, `CAGR / strict MDD >= 3`, strict MDD at most 15%,
positive stress return, minimum trade support, positive 2023 H1/H2 return, and
failure of direction-flip, deterministic-random-side, first-ever-birth,
any-mention, repeat-filer, single-reactivation, stale, and time-shift controls.

## RLLM boundary

RLLM is not allowed to create the clock, infer or reverse a side, or alter the
hold. Only after the deterministic train and selection policy passes may a
frozen small model receive causal source summaries, completed market context,
and current position state to choose between `TRADE_FIXED_LONG` and `ABSTAIN`.
Its reward must penalize strict drawdown and turnover. This preserves the
deductive/abstention strength of an LLM without delegating arithmetic,
timestamping, or execution semantics to it.

## Prior-research disclosure and stopping rule

This is the third hypothesis using the audited SEC Bitcoin 8-K/6-K source
family. Source metadata and aggregate source counts have been opened. The exact
BIRB reactivation/breadth incidence and all BIRB market outcomes remain
unopened. The two earlier semantic candidates stopped before SEC body
classification or BTC outcomes, so their failure is interface evidence, not
economic evidence.

Any identity, source, causality, support, novelty, train, or selection failure
retires `BIRB-120`. Gap, breadth window, threshold, side, or hold repair requires
a new candidate identity preregistered before access.

