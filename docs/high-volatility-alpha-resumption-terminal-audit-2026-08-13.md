# High-volatility alpha resumption terminal audit — 2026-08-13

## Decision

Do not resume either apparent unfinished candidate discovered by filename-level
inventory. Both were already terminal under their frozen no-repair contracts.

This audit opened no new source values, candidate incidence, Gross9 rows,
execution prices, funding rows, post-entry returns, PnL, CAGR, or MDD.

## HVBLSFX-12

`HVBLSFX-12` appeared to have a preregistration without a source-support
artifact. The committed predecessor chain instead contains a stricter
outcome-blind model-integrity stop:

- preregistration commit: `d89db044`;
- terminal seal: `50004472`;
- artifact:
  `results/high_volatility_bls_fx_reaction_transmission_relay_model_integrity_failure_2026-08-13.json`;
- frozen minimum strict-prior history: 60 selected releases;
- selected releases before train: 30;
- maximum possible ranked train events: 0; and
- decision: `terminal_preregistered_history_floor_failure`.

The release history, universe, rank minimum, train boundary, source, side,
clock, and hold may not be repaired.

## HVEBPR-24

`HVEBPR-24` appeared to have passed support without a Gross9 result. The same
terminal source commit records that support statistics passed but exact source
reproduction failed:

- preregistration commit: `ae5eb615`;
- evaluator freeze: `ce09ba0c`;
- terminal seal: `5f5f5d22`;
- support artifact:
  `results/high_volatility_equity_breadth_participation_relay_support_2026-08-12.json`;
- terminal artifact:
  `results/high_volatility_equity_breadth_participation_relay_source_reproduction_failure_2026-08-12.json`; and
- failure class: mutable Yahoo adjusted-history responses did not reproduce the
  hash-pinned raw and derived source objects.

Passing source-support statistics do not override failed source reproduction.
Changing provider, adjustment history, symbols, or source binding would repair
the frozen candidate and is forbidden.

## DeFi lending-rate axis

The next source-blind screen also rejected Maker/Sky DSR or SSR as the next
candidate. The official DSS contract surface establishes that the savings rate
is a governance-set `Pot` parameter and that savings accumulation is triggered
through `Pot.drip`. It does not provide a free, provider-independent historical
archive with causal publication timestamps for the complete 2023–2026 window.
Reconstructing the state requires an archive Ethereum transport or a separately
versioned governance-spell interpretation. The repository's Aave V3 rate-ledger
and governance-log families are already terminal transport boundaries, so
switching to a different lending protocol or transport after those failures
would be a source-family substitution rather than an independent alpha
mechanism.

Primary documentation inspected before any historical values or event
incidence:

- `https://github.com/sky-ecosystem/dss`;
- `https://github.com/sky-ecosystem/dss/wiki/Actions`; and
- `https://github.com/sky-ecosystem/governance-portal-v2`.

No DSR/SSR candidate is preregistered from this screen.

## Research boundary

The next admissible candidate must have all of the following before source
incidence is opened:

1. a genuinely independent observable rather than a provider/protocol repair;
2. a stable causal archive over the complete frozen window;
3. a direct high-volatility BTC directional-return rationale;
4. plausible stage support under the unchanged `8/12/12/8`, side-balance, and
   month-concentration gates; and
5. an exact formula, side, decision clock, hold, controls, and stop rule frozen
   in a committed preregistration.

Until such an object is identified, rejecting weak or unreproducible axes is
preferred to manufacturing another candidate from an already terminal family.
