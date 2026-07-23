# DCLB-864 mechanism decision — dollar-collateral liquidity/bank relay

## Decision

Freeze **DCLB-864 — Dollar-Collateral Liquidity/Bank Relay** before decoding a
DCLB joint source state or any new BTC outcome.

DCLB observes one state at each eligible archived H.8 release. A fresh H.4.1
stock impulse and an H.8-anchored ON RRP interval-flow impulse form one
continuous macro-liquidity side. The H.8 bank state must be valid and labels
whether commercial-bank transmission supports or opposes that side. The
deterministic composer fixes the side; a later RLLM may only trade it or
abstain.

This is a source-informed, outcome-blind candidate. H.4.1, ON RRP, H.8, and
related predecessor outcomes are already known. DCLB may establish only a
candidate-level sequential test, not a pristine global-market discovery.

## Immutable documents and sources

Boundary:

```text
docs/dollar-collateral-liquidity-bank-relay-boundary-2026-07-24.md
SHA256 fed61a096acf0186f153f0cc4e939cec39652fa68eba1fe5afd480121572bf24
```

### H.4.1 allowlist

Source:

```text
data/federal_reserve_h41_net_liquidity_2018_2023/
federal_reserve_h41_net_liquidity_2018-01-04_2023-12-28.csv.gz
SHA256 224883dad01b9d7f17d52eb87f3d7ef9890c8dd055a6c36577a534d2afe69621
header SHA256 4bd522eddda52fefa94c9722f6015596fcde80769c59441046bc0438e1d314d9
```

Build manifest:

```text
data/federal_reserve_h41_net_liquidity_2018_2023/build_manifest.json
SHA256 1ec212a85de0e49c5a0c2d35b8b22be86eb7d62989f7a0098be1bb1274b2a99b
```

The evaluator may decode only:

```text
release_date
observation_date
available_at_utc
net_liquidity_usd_millions
```

### ON RRP allowlist

Source:

```text
data/new_york_fed_overnight_rrp_2018_2023/
new_york_fed_overnight_rrp_2018-01-01_2023-12-31.csv.gz
SHA256 49f67ed44b7eb81fd35c17a8209cf14d6a8019d7e9f77fce8c343d1a7fb66b27
header SHA256 81a388d6e36c5e84c166b5fe111d3766ea5c6b56ac83895ed3541a6c05a01e9c
```

Build manifest:

```text
data/new_york_fed_overnight_rrp_2018_2023/build_manifest.json
SHA256 4f87e2219da71c94832c8708086ba01387efc145e3488b62cd3b3d07c62d8fee
```

The evaluator may decode only:

```text
operation_date
result_available_at_utc
total_amount_accepted_usd
source_complete
quarantine_reason
```

`quarantine_reason` is used only to prove that an incomplete row is explicitly
quarantined. Its text may not become a model token.

### H.8 allowlist

Source:

```text
data/fed_h8_deposit_migration_2017_2023/
fed_h8_deposit_migration_2017_2023.csv.gz
SHA256 c8d1bfb0bbd13ef6d35f09ad7367ef8d2d5bb28981376223b735746ade68a572
header SHA256 b9c20c15035b90266cb47b8465922fde5f1062c3634050b0f006ee6263b978e8
```

Build manifest:

```text
data/fed_h8_deposit_migration_2017_2023/build_manifest.json
SHA256 1f0a194e628ab9c44c23fc4a923145dcf89a62bface745cc36872eeee919eda9
```

The evaluator may decode only:

```text
release_date
release_time_utc
release_weekday
sa_large_other_deposits_prior
sa_large_other_deposits_latest
sa_small_other_deposits_prior
sa_small_other_deposits_latest
sa_small_borrowings_prior
sa_small_borrowings_latest
sa_small_cash_assets_prior
sa_small_cash_assets_latest
nsa_large_other_deposits_prior
nsa_large_other_deposits_latest
nsa_small_other_deposits_prior
nsa_small_other_deposits_latest
nsa_small_borrowings_prior
nsa_small_borrowings_latest
nsa_small_cash_assets_prior
nsa_small_cash_assets_latest
```

Every loader must call `pandas.read_csv(usecols=exact_allowlist)`. Loading a
wider frame and dropping columns is forbidden. Retained H.4.1 and H.8 numeric
levels must be finite and strictly positive. For ON RRP,
`total_amount_accepted_usd` must be blank if and only if
`source_complete == false` and `quarantine_reason` is nonempty; otherwise it
must be finite and nonnegative, including a valid zero. Dates and availability
timestamps must be unique where appropriate, strictly increasing,
timezone-aware, and earlier than 2024.

## H.8 anchor and execution clock

Every archived H.8 row defines a potential anchor, including actual Thursday,
Friday, and Monday irregular releases. No weekday is synthesized or deleted.

For release date `D`:

```text
decision_time = D 17:00:00 America/New_York
entry_time    = D 17:05:00 America/New_York
exit_time_utc = entry_time_utc + 4,320 elapsed minutes
exposure      = [entry_time_utc, exit_time_utc)
```

The elapsed-time UTC definition is authoritative across daylight-saving
transitions. No wall-clock normalization or DST-shortened/lengthened hold is
allowed.

The row is invalid unless:

- `decision_time > release_time_utc`;
- all required H.8 source primitives are valid;
- the exact release date is not one of the four source-methodology exclusions;
  and
- H.4.1 and ON RRP states described below are causally available.

Frozen H.8 exclusions:

```text
2020-10-02
2023-03-31
2023-06-30
2023-12-15
```

The exclusions come from the committed H.8 source audit, not candidate
incidence or BTC outcomes. Excluded rows still delimit ON RRP intervals and
remain in H.8 robust history; they cannot emit any DCLB control or primary
event from their own source state. An excluded release may host execution of a
valid state originated by the immediately preceding release for the frozen
one-release-delay control only.

Raw candidates are constructed over the complete pre-2024 source, sorted by
entry and immutable identity, globally non-overlap-reserved, and only then
split-contained. Entry equal to the previous accepted exit is allowed.

## H.4.1 stock-relief state

On the independent H.4.1 source calendar:

```text
h41_delta[t] = log(net_liquidity[t] / net_liquidity[t-1])
```

For the current delta, use exactly the previous 104 finite H.4.1 deltas,
excluding current:

```text
h41_num[t] =
    2*count(prior_delta < h41_delta[t])
    + count(prior_delta == h41_delta[t])

h41_center_num[t] = h41_num[t] - 104
h41_relief[t]     = h41_center_num[t] / 104
```

`h41_num` is an integer in `[0,208]`. Positive is dollar-liquidity relief;
negative is stress; zero is neutral. The current delta enters history only
after its numerator is fixed.

Every finite H.4.1 delta is appended after processing even when fewer than 104
prior deltas make the current row unrankable. Therefore delta 105 is the first
rankable delta: deltas 1 through 104 form its strictly-prior history.

For H.8 decision `j`, select the latest H.4.1 row whose `available_at_utc` is
at or before `decision_time[j]`. It is fresh only if:

```text
decision_time[j-1] < h41.available_at_utc <= decision_time[j]
```

The first H.8 anchor has no previous decision and is ineligible. Reusing the
same H.4.1 release at two anchors, carrying a missing release, or selecting a
later archive row is forbidden.

## H.8-anchored ON RRP interval-relief state

Let `T[j]` be the frozen 17:00 ET H.8 decision timestamp, including excluded
H.8 rows. Define half-open availability intervals over every archived ON RRP
row, including quarantined rows:

```text
I[j] = {archived ON RRP rows r:
        T[j-1] < r.result_available_at_utc <= T[j]}
```

An interval is complete only when:

- it contains between 3 and 7 normal-operation rows inclusive;
- every row has `source_complete == true`;
- every accepted amount is finite and nonnegative; and
- no row has a nonempty quarantine reason.

For a complete interval:

```text
rrp_level[j] =
    log1p(mean(total_amount_accepted_usd over I[j]) / 1_000_000_000)
```

The mean, rather than sum, prevents a holiday-shortened operation count from
becoming a mechanical liquidity signal.

The interval change is valid only when both adjacent intervals are complete:

```text
rrp_delta[j] = rrp_level[j] - rrp_level[j-1]
```

Any incomplete/quarantined interval:

1. emits no level or delta;
2. clears the ON RRP rank history after the row's containing interval; and
3. requires 13 new consecutive valid `rrp_delta` observations before a new
   rank can be emitted.

This is the source audit's no-bridge rule. The 13-observation window is one
H.8 quarter and is fixed before incidence.

For a current valid delta, use exactly the previous 13 consecutive valid
deltas since the last reset:

```text
rrp_num[j] =
    2*count(prior_delta < rrp_delta[j])
    + count(prior_delta == rrp_delta[j])

rrp_center_num[j] = rrp_num[j] - 13
rrp_relief[j]     = -rrp_center_num[j] / 13
```

`rrp_num` is an integer in `[0,26]`. The sign is negated because increasing ON
RRP uptake removes reserves, while declining uptake is provisional relief.
The current delta enters history only after its rank is fixed.

Every finite post-reset ON RRP interval delta is appended after processing
even when fewer than 13 prior deltas make it unrankable. The 14th valid delta
after a reset is therefore the first rankable delta. No delta whose construction
touches an incomplete interval is appended.

## H.8 bank-transmission state

Use the seasonally adjusted levels printed in each dated H.8 release:

```text
migration_bp =
    10000*log(large_other_deposits_latest / large_other_deposits_prior)
  - 10000*log(small_other_deposits_latest / small_other_deposits_prior)

borrowings_bp =
    10000*log(small_borrowings_latest / small_borrowings_prior)

cash_stress_bp =
   -10000*log(small_cash_assets_latest / small_cash_assets_prior)
```

Each component receives a strictly-prior robust z-score against exactly the
previous 104 H.8 component observations:

```text
z = (current - median(prior104)) /
    (1.4826 * median(abs(prior104 - median(prior104))))
```

Zero MAD or a non-finite value makes the H.8 state invalid. Then:

```text
h8_stress = mean(z_migration, z_borrowings, z_cash_stress)
h8_stress_sign = sign(h8_stress)
h8_relief_sign = -h8_stress_sign
```

The state is valid only when `h8_stress != 0` and at least two component
z-score signs equal `h8_stress_sign`. No absolute tail, quantile, magnitude
threshold, current-vintage history, or not-seasonally-adjusted substitution is
allowed.

Every finite H.8 component observation is appended to its own history after
processing, even when the current row is not yet rankable or another component
makes the composite state invalid. Observation 105 is the first with a
strictly-prior 104-observation robust baseline. Excluded release rows remain
in these histories exactly as stated above.

The `nsa_h8` control recomputes the same three components, prior-104 robust
z-scores, consensus rule, and relief sign using the corresponding frozen NSA
fields. Its NSA history is strictly prior and separate from SA history. NSA
values never enter the primary state.

## Primary macro side and bank relation

The H.4.1 and ON RRP ranks are combined with exact integer arithmetic:

```text
macro_integer =
    13*h41_center_num - 104*rrp_center_num
```

This is exactly proportional to `h41_relief + rrp_relief`; no float comparison
chooses the sign.

An anchor is primary-eligible only when:

- H.4.1 is fresh and rank-complete;
- the ON RRP interval state is complete and rank-complete;
- the H.8 bank state is valid;
- the H.8 release is not excluded; and
- `macro_integer != 0`.

Direction:

```text
side_sign = sign(macro_integer)
side      = LONG  if side_sign == +1
side      = SHORT if side_sign == -1
```

Bank-transmission relation:

```text
h8_relief_sign == side_sign -> BANK_SUPPORTS
h8_relief_sign != side_sign -> BANK_OPPOSES
```

Both relations belong to the primary clock. Neither subset may replace the
full primary after source incidence or outcomes are known. H.8 is
mechanistically binding because an invalid bank state makes the opportunity
ineligible and its relation is a mandatory RLLM input.

Additional frozen relation tokens:

- `MACRO_CONCORDANT`: H.4.1 and ON RRP relief signs agree non-neutrally;
- `MACRO_DISCORDANT_H41_DOMINANT`: signs oppose and the macro side equals H.4.1;
- `MACRO_DISCORDANT_RRP_DOMINANT`: signs oppose and the macro side equals RRP;
- `HAS_NEUTRAL_COMPONENT`: either component is neutral;
- macro strength: `WEAK` when
  `abs(macro_integer) <= 13*52`, otherwise `STRONG`;
- H.8 agreement: `TWO_OF_THREE` or `THREE_OF_THREE`;
- H.4.1 age at decision: `SAME_DAY`, `ONE_DAY`, `TWO_TO_THREE_DAYS`, or
  `FOUR_PLUS_DAYS`;
- ON RRP interval counts: `THREE_TO_FOUR`, `FIVE`, or `SIX_TO_SEVEN`; and
- prior primary-side transition: `NO_PRIOR`, `PERSIST`, or `FLIP`.

The strength split is the exact midpoint of the H.4.1 centered-rank numerator
after common-denominator conversion. It is a prompt token only and never an
eligibility threshold.

## Frozen source-only controls

Each independently scheduled control uses the same H.8 decisions, source
validity, 864-bar hold, global reservation, and split containment. Controls
diagnose and may not replace primary:

1. `primary`: exact composite and bank-validity rule above.
2. `h41_only`: nonzero H.4.1 centered rank fixes side; ON RRP and H.8 must
   still be valid.
3. `rrp_interval_only`: nonzero ON RRP relief rank fixes side; H.4.1 and H.8
   must still be valid.
4. `h8_only`: valid H.8 relief sign fixes side while H.4.1 and ON RRP remain
   causally valid.
5. `macro_concordant_only`: primary subset with `MACRO_CONCORDANT`.
6. `macro_discordant_only`: primary subset with either discordant relation.
7. `bank_supports_only`: primary subset with `BANK_SUPPORTS`.
8. `bank_opposes_only`: primary subset with `BANK_OPPOSES`.
9. `stale_h41_one_release`: replace current H.4.1 rank numerator with the
   immediately preceding emitted H.4.1 rank numerator on the independent
   H.4.1 source calendar that is available by the current decision.
10. `stale_rrp_one_interval`: replace current ON RRP rank numerator with the
    immediately preceding emitted ON RRP interval rank numerator in the same
    uninterrupted post-quarantine segment.
11. `exact_direction_flip`: opposite primary side on accepted primary
    timestamps.
12. `deterministic_random_side`: primary timestamps; SHA-256 of
    `DCLB-864|YYYY-MM-DDTHH:MM:SSZ`, encoded as UTF-8 with the timestamp
    serialized by UTC `strftime("%Y-%m-%dT%H:%M:%SZ")`; LONG iff the first
    digest byte is below 128.
13. `one_h8_release_execution_delay`: raw primary state and side enter at the
    next archived H.8 decision's 17:05 ET entry, exit 864 bars later, then
    repeat global reservation. An unavailable terminal next release is omitted.
14. `nsa_h8`: keep the primary macro side, H.4.1 state, ON RRP state, clock,
    and reservation, but replace SA H.8 validity and relation with the exact
    NSA replay defined above. An invalid NSA replay emits no control event.

Stale controls are independently reserved. Same-clock flip and random controls
reuse the accepted primary timestamps. Excluded H.8 releases may be the
execution date of the delayed control because the delayed control carries an
already valid prior state; its entry must still be strictly later than the
current archived H.8 release timestamp.

## Live fail-flat source parity

Any later live implementation must predeclare the expected H.4.1, ON RRP, and
H.8 publication calendars and apply the archived parser, exact allowlists,
availability rules, quarantine semantics, rank warm-ups, and interval resets
without substitution. For each expected publication:

- a missing or late release, schema drift, response-integrity failure,
  availability-time ambiguity, or quarantine mismatch emits no source update
  and no candidate event;
- no stale carry, alternate endpoint, interpolation, forward fill, imputation,
  or calendar synthesis is permitted;
- the live parser records the retrieval time and response SHA-256 in an
  append-only provenance ledger before accepting the update; and
- a source or schema migration is allowed only after a separately committed,
  prospective compatibility decision and shadow replay. It cannot alter an
  already scheduled event.

The live route must fail flat rather than approximate the historical feature.

## Source-support gates

The source-only evaluator must be committed and hash-bound before decoding any
DCLB joint state. It evaluates globally reserved, split-contained clocks.
All fourteen controls enumerated above are required controls.

Train:

```text
[2020-01-01T00:00:00Z, 2023-01-01T00:00:00Z)
```

All are required:

- at least 75 primary events;
- at least 12 events in each of 2020, 2021, and 2022;
- at least 24 active calendar months;
- LONG and SHORT each at least 20%;
- no month above 12%;
- no quarter above 24%;
- maximum entry gap at most 60 calendar days;
- maximum same-side run at most 12; and
- every required control has at least one contained event.

Selection:

```text
[2023-01-01T00:00:00Z, 2024-01-01T00:00:00Z)
```

All are required:

- at least 20 primary events;
- at least 7 events in each half;
- at least 2 events in each quarter;
- at least 8 active calendar months;
- LONG and SHORT each at least 20%;
- no month above 25%;
- maximum entry gap at most 75 calendar days;
- maximum same-side run at most 10; and
- every required control has at least one contained event.

Relational composition must pass separately in train and selection:

- `BANK_SUPPORTS` and `BANK_OPPOSES` each at least 20%;
- `MACRO_CONCORDANT` and all discordant relations combined each at least 20%;
- `WEAK` and `STRONG` macro strength each at least 15%;
- `TWO_OF_THREE` and `THREE_OF_THREE` H.8 agreement each at least 10%;
- H.4.1-only and RRP-only same-entry same-side reproduction each at most 85%;
- each stale control reproduction at most 85%; and
- deterministic-random reproduction at most 60%.

Same-side reproduction is same entry and side matches divided by primary
count. An empty denominator, missing control, undefined statistic, source/hash
drift, timing violation, overlap, or failed gate retires DCLB before comparator
rows or outcomes.

## Frozen comparator cohort

Comparator hashes and exact headers are validated before source decoding.
Comparator row values may be decoded only after complete source-support and
relational-composition passes. Each selected group is evaluated separately.

The following prospective common-window policy is binding:

```text
docs/novelty-comparator-common-window-policy-2026-07-23.md
SHA256 928bce6e04fb34001478b4b4ea84156580b661c88a0f0338065a891c009bd580
```

The exact novelty window for candidate and every comparator group is:

```text
[2020-01-01T00:00:00Z, 2024-01-01T00:00:00Z)
```

Only intervals fully contained in that window are eligible. Raw artifacts are
validated in full before containment filtering; boundary-crossing intervals
are reported and excluded whole. Every metric below uses this same complete
window rather than an artifact-specific observed prefix.

### FLCC H.4.1

```text
results/federal_liquidity_component_concordance_preregistered_clock_2026-07-17.csv.gz
SHA256 7ebb0450422d9265e46c596e0b6415b6a8816c66f5e0cbb9ccda14ca6cb4c67c
header SHA256 38d354aa0f63efa58bf5181cdc8cdecc4d9fc2f1a6eda8df8933c95f79cffdb7
```

Compare `clock_name == primary` separately for:

```text
FLCC-H4-Q60
FLCC-H4-Q65
FLCC-H8-Q60
FLCC-H8-Q65
```

Decoder `usecols` is exactly:

```text
candidate_id,clock_name,entry_time,exit_time,side
```

Each selected FLCC group must have at least 90 fully contained rows.

### ORFR ON RRP

```text
results/overnight_rrp_flow_release_clocks_2026-07-17.csv.gz
SHA256 7242d9870627dfc0cf067ff87d9664a1576dd374cb8985e927b40f15d1e3d480
header SHA256 3a45759e0b14eeef01ddfb5146ca03515a562846412c98fa1ca1aca7e285528e
```

Compare separately:

```text
primary
one_day_delta_tail
one_release_delay
```

Decoder `usecols` is exactly:

```text
control,entry_time,exit_time,side
```

Each selected ORFR group must have at least 150 fully contained rows.

### ORPB participant breadth

```text
results/overnight_rrp_participant_breadth_support_clocks_2026-07-21.csv.gz
SHA256 ef21323229801f11557e0c2d9d4465f7d58b13569552d656d64fdb7d440622ed
header SHA256 257ee9b477b9c62e9c287d03269d813c3a8b4b6286d836ab61ed5a925a2fd3f4
```

Compare `candidate_id == ORPB-21` and `control == primary`.

Decoder `usecols` is exactly:

```text
candidate_id,control,entry_time,exit_time,side
```

The selected ORPB group must have at least 180 fully contained rows.

### H8DM bank stress

```text
results/fed_h8_deposit_migration_preregistered_clock_2026-07-18.csv.gz
SHA256 20405f79b86861adcc784c81223baae1c40fdf3c73edda339578471a6a6d1b40
header SHA256 58dd787ede642429260f05ca2bc0918a22f2a83eb778686de49b279d8a1cf8b3
```

Compare `clock_mode == primary`.

Decoder `usecols` is exactly:

```text
clock_mode,entry_time,exit_time,side
```

The selected H8DM group must have at least 90 fully contained rows.

### BDRC bank/repo concordance

```text
results/bank_deposit_secured_repo_concordance_clocks_2026-07-20.csv.gz
SHA256 1ff3a6075e3ceff928e1dd19d05880dbe9dbab0e07d79b853146d7b4c8f6cabc
header SHA256 ef3cd7e042ff592bd4747ecd9bbf47b66cc7ab587db60a834efb502a11c7a605
```

Compare `clock_name == primary` and `clock_name == h8_only` separately.

Decoder `usecols` is exactly:

```text
clock_name,entry_time,exit_time,side
```

The BDRC `primary` group must have at least 50 fully contained rows and the
BDRC `h8_only` group at least 120.

Every comparator decoder may read only the exact `usecols` declared above.
Empty or below-floor required extraction, invalid side, duplicate entry,
invalid interval, self-overlap, hash/header drift, or undefined correlation
fails.

On the exact common window declared above:

- same-H.8 clocks (H8DM and BDRC): exact-entry Jaccard at most 0.60,
  same-entry same-side reproduction at most 0.75, and absolute signed
  occupied-exposure Pearson at most 0.65;
- all other clocks: exact-entry Jaccard at most 0.20, maximum-cardinality
  one-to-one ±6-hour tolerant Jaccard at most 0.35, and absolute signed
  occupied-exposure Pearson at most 0.45; and
- ±7-calendar-day tolerant Jaccard is report-only for every comparator.

The complete common five-minute grid is used for occupied exposure. Metrics
are never computed on only a comparator's observed prefix.

## Outcome and RLLM sequence

Only source-support and novelty passes authorize a separately committed
economic/RLLM evaluator.

Frozen execution economics:

- instrument: Binance USD-M BTCUSDT perpetual;
- fixed gross exposure: 0.5x account equity;
- base fee plus slippage: 6 bp per notional side;
- stress fee plus slippage: 10 bp per notional side;
- exact realized funding on `[entry, exit)`;
- full-calendar CAGR including idle cash;
- strict MDD from the global pre-entry high-water mark, entry cost, funding,
  every held five-minute high/low with favorable-before-adverse ordering,
  virtual adverse exit cost, and realized exit; and
- no stop, take profit, early exit, leverage selection, or overlapping trade.

Sequential roles:

```text
source warm-up: 2017-2019
RLLM fit:       2020-2021
inner test:     2022
sealed eval:    2023
post-2023:      requires separately audited source extension before outcomes
```

The cheap deterministic primary is reported before LLM compute. Model,
tokenizer, quantization, LoRA/RL algorithm, reward, checkpoint rule, and seed
must be frozen in the separate evaluator before any fit/eval reward is opened.
No 2022 checkpoint choice may inspect 2023.

Every final report must show absolute return beside CAGR, strict MDD, ratio,
trade count, side count, gross edge, funding, base/stress costs, and clustered
significance.

Minimum sealed-eval qualification:

- positive absolute return at base and stress cost;
- positive absolute return in both halves;
- CAGR / strict MDD at least 3.0;
- strict MDD at most 15%;
- at least 12 executed trades, at least 4 per side, and at least 4 per half;
- mean gross underlying movement above 20 bp; and
- weekly-cluster sign-flip `p <= 0.10`.

The fit/test battery must use at least equally strict full-period return,
ratio, MDD, side-support, stress-cost, and annual-stability gates. Exact model
selection and checkpoint gates are frozen before any economic row is opened.

## RLLM token and action boundary

Allowed compact tokens:

- fixed side;
- H.4.1 direction and transition;
- ON RRP interval direction and transition;
- macro concordance/dominance and strength;
- H.8 relief/stress, component agreement, and bank-support relation;
- source age/count/validity buckets; and
- current position state.

Forbidden:

- raw levels, deltas, z-scores, ranks, or rank numerators;
- source dates, timestamps, weekdays, release IDs, row IDs, URLs, or hashes;
- BTC price, return, funding, future path, label, reward, PnL, CAGR, MDD, or
  split identity;
- candidate creation, side reversal, hold/leverage/time selection; and
- any prompt text revealing aggregate outcome summaries.

Action space:

```text
TRADE_FIXED_SIDE
ABSTAIN
```

The RLLM may reason deductively about whether central-bank stock relief,
facility flow, and bank transmission form a coherent relay. It cannot invent
direction or recover calendar memorization.

## Failure and stop rule

The mandatory order is:

1. commit this mechanism;
2. commit a write-once preregistration without source-row decoding;
3. commit and test an outcome-blind support/novelty evaluator;
4. retire DCLB unchanged on the first source, support, composition, or novelty
   failure;
5. only a full pass may freeze the economic/RLLM evaluator;
6. stop at the first failed fit, test, eval, or post-extension gate.

No failed threshold, rank window, quarantine reset, source relation, side,
clock, hold, token, comparator, model, reward, or checkpoint may be repaired
inside DCLB-864.
