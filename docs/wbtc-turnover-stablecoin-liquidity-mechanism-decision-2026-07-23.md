# WTSL-168-SOURCE-SEEN mechanism decision — 2026-07-23

## Decision

Freeze one new candidate, **WTSL-168-SOURCE-SEEN**: WBTC bridge turnover is a
materiality clock, while combined Ethereum USDC/USDT issuance-redemption flow
supplies the trade direction.

This is not a repair of rejected WCDR-2016. WCDR used the *sign* of WBTC
mint/burn flow and required an opposite USDC sign. WTSL deliberately discards
WBTC direction. It asks whether unusually active movement through the wrapped
collateral boundary makes an independently measured dollar-liquidity impulse
more relevant for BTC over the next day.

## Research-boundary disclosure

This freeze is **market-outcome blind, but not source-incidence blind**.
Before this document was written, a source-only probe of this exact mechanism
reported 236 non-overlapping pre-2024 candidates: 189 LONG / 47 SHORT, with
168 in 2021, 43 in 2022, and 25 in 2023; the 2023 side split was
12 LONG / 13 SHORT. No BTC OHLC, funding,
future return, PnL, CAGR, strict MDD, or post-2023 contract-event value was
opened for WTSL.

Those counts are disclosed rather than presented as confirmatory evidence.
The source-support build must reproduce them exactly. Market performance has
not been used to select any WTSL threshold, direction, delay, hold, cost, or
control. WTSL is the second candidate in the current WBTC/stablecoin source
family after WCDR-2016.

## Economic mechanism

The two inputs have separate roles:

1. **WBTC turnover gate.** Large mint/burn gross flow indicates that BTC-linked
   collateral is crossing a custody/settlement boundary. Gross flow is used
   because mint and burn can coexist during migration and netting; its sign is
   forbidden from deciding the WTSL side.
2. **Stablecoin impulse.** Net USDC mint plus USDT issue is a positive dollar
   liquidity impulse; net USDC burn plus USDT redeem is negative. A positive
   impulse maps to LONG and a negative impulse maps to SHORT.
3. **Interaction claim.** Stablecoin flow alone is broad. The WBTC turnover
   gate should select periods in which that flow can transmit into BTC rather
   than merely describe routine token administration.

`destroyed_black_funds` is not an economic redemption. Any such USDT event in
the current stablecoin window vetoes the anchor instead of entering net or
gross flow.

## Frozen causal state

At UTC anchors `00:00`, `06:00`, `12:00`, and `18:00`:

- decision time: `D`;
- source cutoff: `C = D - 6 elapsed hours`;
- all source ordering and windows use only `available_at`;
- current WBTC and stablecoin windows are `(C - 168h, C]`;
- block timestamps are identity/audit fields only and are forbidden as clocks.

### WBTC activity state

The current WBTC window must have:

- positive gross mint+burn amount;
- at least 2 semantic events;
- at least 2 distinct nonzero actor addresses;
- largest actor gross share at most 0.85.

Define 1,460 strictly prior six-hour endpoints `C - 6h` through `C - 365d`.
At each endpoint compute WBTC gross amount over its own prior 168 elapsed
hours, including exact zeros. The current WBTC gross must be greater than or
equal to the median of those 1,460 values. For the even sample, compare
`2 * current_gross` with the sum of the two middle order statistics; no float
rounding is allowed.

A candidate is eligible only when the complete 372-day activity-history
horizon is inside the frozen source coverage. This prevents pre-source time
from being silently treated as zero.

### Stablecoin direction state

Within the current 168-hour window:

- USDC `mint` and USDT `issue` have sign `+1`;
- USDC `burn` and USDT `redeem` have sign `-1`;
- both contracts use 6 decimals, so raw amounts may be summed exactly;
- USDT `destroyed_black_funds` contributes neither net nor gross and vetoes
  the anchor when present;
- gross eligible amount must be positive;
- `abs(net_raw) / gross_raw >= 0.05`, evaluated as integer arithmetic.

Side is `LONG` when stablecoin net is positive and `SHORT` when it is negative.
Zero net or any failed condition means no candidate. WBTC net sign cannot
change or veto the side.

## Frozen execution

- entry: first BTCUSDT 5-minute bar open at `D + 10 minutes`;
- exit: bar open exactly 288 five-minute bars later (24 elapsed hours);
- fixed notional exposure: 0.5x;
- one global position; no pyramiding;
- anchors are processed chronologically and accepted only when entry is at or
  after the prior accepted exit;
- trades crossing a research-window boundary are skipped, never truncated;
- no stop, take-profit, trailing exit, threshold search, or side reversal.

## Frozen controls

The same source artifacts and causal clocks must produce:

1. `direction_flip`: exact primary entries with both sides reversed;
2. `stablecoin_only_direct`: stablecoin rule without the WBTC activity gate;
3. `wbtc_signed_placebo`: valid WBTC activity state with side from WBTC net
   sign, never used by primary;
4. `stale_24h` and `stale_48h`: both source cutoffs delayed by exactly one
   or two holding periods;
5. `actor_cap_60`: primary with WBTC top-actor share capped at 0.60;
6. `no_black_funds_veto`: primary while ignoring only the black-funds veto;
7. `usdc_only_direct`: WBTC gate retained, direction and imbalance computed
   from USDC mint/burn only;
8. `usdt_only_direct`: WBTC gate retained, direction and imbalance computed
   from USDT issue/redeem only, with the same black-funds veto;
9. `year_amount_permutation`: deterministic within source/event/year amount
   permutation, preserving identities, timestamps, signs, and amount multisets;
10. `deterministic_random_side`: exact primary entries with a SHA-256 fixed,
    side-count-matched permutation.

Primary must beat `stablecoin_only_direct` on train and selection risk-adjusted
performance. Full qualification by direction-flip, either stale clock, or random-side
control rejects the mechanism. USDC-only and USDT-only controls identify which
stablecoin actually carries any result; they cannot replace primary after
outcomes are opened.

## Windows and staged access

- source warm-up/median history: 2020 only;
- train outcomes: `[2021-01-01, 2023-01-01)`;
- selection outcomes: `[2023-01-01, 2024-01-01)`;
- every source value and outcome at or after `2024-01-01` remains sealed until
  the complete pre-2024 sequence passes.

Sequence:

1. hash-bind this decision and both source artifacts;
2. reproduce the disclosed source-only clock and all controls;
3. freeze the strict evaluator before opening BTC outcomes;
4. evaluate train only;
5. open selection only if every train gate passes;
6. extend the same contract/topic/feature policy after 2023 only if train and
   selection pass;
7. then evaluate immutable test 2024, eval 2025, and recent 2026.

## Source-support integrity gates

The disclosed counts must reproduce exactly before outcomes are opened. In
addition, both train and selection clocks must satisfy all prespecified
structural gates:

- train total at least 120 and selection total at least 20;
- each of 2021 and 2022 at least 20 candidates;
- each train half-year at least 8 and each selection half-year at least 6;
- train at least 24 candidates per side and selection at least 8 per side;
- no UTC month above 20% and no UTC quarter above 40% of its split;
- no more than 20 consecutive accepted candidates on one side;
- at least 10 distinct WBTC actors in train and 5 in selection.

Because source incidence was seen, passing these gates is a reproducibility and
minimum-identifiability check, not confirmatory evidence. Failure still retires
the candidate without threshold repair or market-outcome access.

## Strict economic gates

Each opened primary split must satisfy all of:

- positive absolute return over the full calendar split;
- full-calendar CAGR / strict intratrade MDD at least 3.0;
- strict MDD at most 15%;
- positive return under 10 bp notional cost per side;
- realized funding included;
- at least 20 executed trades and at least 6 per side;
- calendar-month clustered sign-flip `p <= 0.10`;
- primary CAGR/MDD strictly above `stablecoin_only_direct`.

Base cost is 6 bp of notional per side. CAGR includes every idle day in the
calendar split. Strict MDD uses the highest marked-to-market equity reached
before or during a position, so entry gaps and intratrade drawdowns count.

## RLLM boundary

No LLM/RL component may create entries, reverse stablecoin direction, alter the
24-hour hold, or optimize these rules before deterministic train and selection
pass. A later compact RLLM may only choose `TRADE_FIXED_SIDE` or `ABSTAIN` from
causal, bucketed source-state tokens plus current-position/time-to-exit state.
Its reward must penalize strict drawdown and turnover, and it must beat the
frozen deterministic WTSL policy on untouched test/eval windows.

## Stopping rule

Any source identity, disclosed-count reproduction, causality, integrity,
train, or staged-selection failure retires WTSL-168-SOURCE-SEEN without threshold repair.
A different direction, window, hold, asset set, or gate requires a new identity
and a new market-outcome-blind freeze.
