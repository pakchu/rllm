# High-volatility alpha search resumed-exhaustion audit — 2026-08-18

## Decision

The resumed search is blocked before another honest preregistration. This is
not a claim that no BTC alpha exists. It records that the current repository
sources and frozen 2023–2026 protocol no longer contain an untested independent
mechanism that can be selected without reusing known failed outcomes to alter a
threshold, side, clock, hold, subset, weight, or control.

Deribit and all option-surface inputs remained excluded after the user's
instruction. No option data was used by the candidates in this audit.

## Repeated blocking condition

The same condition persisted for more than three consecutive resumed goal
turns:

1. independently defined source objects generally passed source support and
   Gross9, but fixed-direction price-path signals reversed in 2024 test;
2. source-observable regime routers and causal mature-label adapters destroyed
   train edge or failed calendar-half/significance gates; and
3. independent external relative-value transitions either lacked source
   incidence or reproduced the same train-pass/test-fail pattern.

Further local work now requires choosing a rule because its already opened
train/test aggregate repairs a failure. That is p-hacking under the frozen
terminal-first-failure contracts, not continued clean alpha discovery.

## Resumed candidate evidence

| Family | Candidate | Terminal gate | Evidence |
|---|---|---:|---|
| directional veto | `HVCASPCWMV-8` | test | `a9e40477` |
| directional veto | `HVDEMWMV-24` | test | `021fa005` |
| variance/OI veto | `HVCASPCVAIV-8` | train stress | `d3e6bd82` |
| variance/OI router | `HVCVAROIR-8` | train | `6b887194` |
| BTC factor residual recentering | `HVBFRRCR-8` | source | `b4432679` |
| BTC factor residual flip | `HVBFRRFC-8` | train | `fbf117ff` |
| cross-venue median consensus | `HVCVMRS-8` | train calendar half | `510c3180` |
| cash-led median dominance | `HVCVMSD-8` | source | `98debcec` |
| cross-venue median spread | `HVCVMSSD-8` | Gross9 | `779e3c68` |
| spot median shift | `HVSMRSR-8` | train | `f3255871` |
| spot underwater duration | `HVSAUD-8` | test | `f3ef7e7b` |
| mature-label spot adapter | `HVSAUDCA-8` | train | `d786d45f` |
| long-memory autocorrelation router | `HVSAUDAR-8` | train | `ec83485b` |
| equity-duration rotation | `HVERDR-24` | train | `df9fdbee` |
| credit-quality transition | `HVCQTR-24` | test | `8ae8b8ca` |
| energy-commodity transition | `HVECTR-24` | train | `666d6d64` |
| risk-parity transition | `HVRPCTR-24` | source | `d604d65a` |
| credit × variation router | `HVCQVAR-24` | train | `68c38e0f` |
| prehistory causal credit adapter | `HVCQCA-24` | train | `1ef03dc4` |

Every listed economic candidate used 0.5 gross, exact held funding, 6 bp per
notional side base cost, 10 bp stress cost, full-calendar CAGR, and global-HWM
held-five-minute favorable-then-adverse strict MDD. Each stopped at its first
failed frozen gate, and every artifact was committed and pushed.

## Strongest near-misses remain terminal

- `HVCVMRS-8` passed return, ratio, stress, gross-move, MDD, and sign-flip
  gates, but its first train calendar half was `-0.03996%`. Its source, side,
  or subset cannot be repaired.
- `HVCQTR-24` had train CAGR/MDD `8.1637`, stress ratio `7.0342`, mean gross
  move `101.72 bp`, and sign-flip `p=0.0151`, but test returned `-0.8828%`,
  with its second test half at `-4.7454%`.
- `HVCELV-8` was positive in test before this resumed audit, but test stress
  return, gross move, ratio, significance, and second-half gates failed. It is
  not an admissible pool sleeve or repair base.

These results cannot authorize a post-hoc threshold, phase rule, portfolio
weight, side inversion, or calendar filter.

## Exact unblock conditions

The goal can resume without weakening the protocol when one of these external
conditions becomes true:

1. a genuinely independent, legally usable point-in-time source with complete
   causal 2023–2026 coverage is added (excluding option data unless the user
   explicitly reauthorizes it);
2. a new forward source accumulates enough history for unchanged
   train/test/eval/final support gates;
3. a newly published mechanism uses an observable not represented by the
   exhausted price-path, spot/perpetual, cross-alt, funding, OI, flow,
   cross-asset, calendar, and causal-adapter families; or
4. the user explicitly changes the objective by relaxing the no-repair,
   full-window, source-completeness, Gross9, or economic gates.

Until then, creating another candidate from known failures would reduce rather
than improve evidentiary quality.

## Verification state

- Branch: `codex/gross9-structural-clock-bundle-20260731`
- All candidate-specific targeted tests passed before each terminal commit.
- Worktree and upstream were clean at the start of this audit.
- Latest terminal candidate commit before this audit: `1ef03dc4`.

## Post-resumption new-mechanism audit

After the user explicitly requested a completely new mechanism, the search was
resumed as a fresh blocked audit. Three additional mechanism classes were
preregistered and evaluated without option data:

| New mechanism | Candidate | Terminal gate | Evidence |
|---|---|---:|---|
| SPY/TLT consensus polarity transition | `HVRPCTR-24` | source incidence | `d604d65a` |
| credit transition × completed BTC variation phase | `HVCQVAR-24` | train | `68c38e0f` |
| prehistory-seeded mature-label credit response adapter | `HVCQCA-24` | train | `1ef03dc4` |

The latter two passed source support and every Gross9 sleeve. `HVCQCA-24`
also passed train return, base CAGR/MDD, stress return, stress CAGR/MDD, mean
gross move, and strict MDD, but failed the frozen first-calendar-half and
weekly sign-flip gates. Altering memory, maturity, side, early-history
treatment, or a calendar subset after this result would be an explicit repair.

This post-resumption audit therefore satisfies the repeated-blocker threshold:
the same absence of a remaining independent, outcome-uncontaminated local
mechanism persisted for at least three fresh continuation turns after the user
resumed the blocked goal.
