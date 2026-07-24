# TPCT-120 mechanism decision — tri-party composition topology

## Decision

Freeze one source-support-seen, market-outcome-unseen candidate:

**TPCT-120 — Tri-Party Composition Topology Policy**.

TPCT is one text-only policy over a delayed private tri-party repo composition
state. It chooses exactly one action:

```text
ABSTAIN
LONG
SHORT
```

No source primitive, rank, token, threshold, or prior OFR candidate owns a
direction. The policy receives only a fixed twelve-token relational grammar.
Raw values, numeric ranks, dates, prices, returns, labels, prior actions,
position state, executability, and portfolio PnL are forbidden model inputs.

This document is binding to:

```text
docs/tri-party-composition-topology-boundary-2026-07-24.md
SHA256 a3e8d5c0dd2da652b4be93627a935f064c1d144560b124613f34f95287d28159
commit 51cbcd0cff26ce7c25f9cf94ea78932d616b4af9
```

This mechanism decodes no TPCT source value, rank, token, incidence, market,
funding, comparator, return, PnL, label, 2023 TPCT covariate, or post-2023 row.

## Research-history boundary

The OFR preliminary source and prior source-support results are seen:

- `RVFC-72` opened cross-venue dispersion/concentration incidence and failed
  source support;
- `RMSR-72` opened a cross-venue collateral first-passage race and failed
  source support;
- `RCRE-72` opened a GCF-minus-TRIV1 signed interaction and failed quadrant
  support;
- `DMSH-168` opened a DVP maturity handoff and failed source support; and
- the source audit disclosed hashes, schema, row/date counts, missingness, and
  release-clock facts.

No OFR candidate opened BTC prices, funding, future returns, PnL, CAGR, or MDD.
TPCT is therefore market-outcome-unseen but explicitly
**source-support-seen**. The prior failures informed the decision to test a
dense within-segment grammar. This disclosure is permanent.

TPCT is not a threshold repair:

- only `TRIV1` is used; DVP, GCF, and `TRI` are forbidden;
- there is no sparse source threshold, vote, precursor, terminal, quadrant,
  source-owned side, or confirmation state;
- every valid rank-ready delayed state is eligible before action-independent
  reservation;
- the grammar describes simultaneous private tri-party composition rather than
  a prior cross-venue or DVP object; and
- the 120-hour hold and split-gap accounting were fixed before TPCT values.

No prior OFR threshold, failed state, event, side, or control may be introduced
later as a gate.

## Frozen source and metadata

Primary source:

```text
data/ofr_repo_preliminary_2019_2023/
  ofr_repo_preliminary_observations_2019_2023.csv.gz
SHA256 6bb24b9a3d6ed6d266b0c885a9d9888bcccb3045df66d37d83031af66c6dc02a
header SHA256 743cb319f6fe1d2722cf8f249ff981c87b24def885a8776b0017f03cb060959c
```

The header hash is over the decompressed header without CR/LF:

```text
mnemonic,observation_date,available_at_utc,value,disclosure_edit,segment,measure,subset,series_name
```

Metadata:

```text
data/ofr_repo_preliminary_2019_2023/
  ofr_repo_preliminary_metadata_2019_2023.json.gz
SHA256 19a04e82eb5d8ddc6c3cb8dc64694438abd6b1987951470bb317659d9c53ef4f
```

Canonical selected-metadata hash:

```text
e75d656e6ae322eeb0a44ef9e52450af21c54c3a17936c0791ec9fa4421c8edc
```

It is SHA-256 of UTF-8 JSON over the sixteen metadata objects in the exact
allowlist order below, with sorted keys, separators `(",",":")`, and
`ensure_ascii=False`.

Source manifest:

```text
data/ofr_repo_preliminary_2019_2023/build_manifest.json
file SHA256 f937f567e1789ecb39a2b84d6288b2cbab931da4e9f1f4e51addea4b3423b705
canonical manifest 802b83a9478711cd29d5b606d9e12eb1e90890e37f5908d4de64d7dd71f6d449
```

Governing source audit:

```text
docs/ofr-repo-preliminary-source-audit-2026-07-23.md
```

Exact mnemonic order:

```text
REPO-TRIV1_AR_OO-P
REPO-TRIV1_TV_OO-P
REPO-TRIV1_AR_B27-P
REPO-TRIV1_TV_B27-P
REPO-TRIV1_AR_B830-P
REPO-TRIV1_TV_B830-P
REPO-TRIV1_AR_G30-P
REPO-TRIV1_TV_G30-P
REPO-TRIV1_AR_T-P
REPO-TRIV1_TV_T-P
REPO-TRIV1_AR_AG-P
REPO-TRIV1_TV_AG-P
REPO-TRIV1_AR_CORD-P
REPO-TRIV1_TV_CORD-P
REPO-TRIV1_AR_O-P
REPO-TRIV1_TV_O-P
```

The metadata validator must prove for every selected object:

- `segment == "TRIV1"`;
- measure is the mnemonic's exact `AR` or `TV`;
- subset is the mnemonic's exact `OO`, `B27`, `B830`, `G30`, `T`, `AG`,
  `CORD`, or `O`;
- vintage and vintage approach are both `Preliminary`;
- release is the OFR U.S. Repo Markets Data Release with daily observation and
  release frequency;
- rate units are `Percent`/`Rate`;
- volume units are `USD`/`Volume`; and
- the exact series name and description identify tri-party repo excluding
  Federal Reserve transactions.

The frozen payload metadata defines tenor names as `Overnight/Open`,
`2 - 7 Days`, `8 - 30 Days`, and `>30 Days`. It defines collateral names as
`U.S. Treasury Securities`, `Federal Agency and GSE Securities`,
`Corporate Debt`, and `Other Collateral`. Those payload-specific definitions
govern historical reconstruction. The current general mnemonic glossary's
`30+ Days` wording is a disclosed version mismatch. It cannot rewrite the
frozen payload. Any live or extension metadata drift fails closed.

The exact source loader may parse only the header allowlist above. Load-and-drop
is forbidden. It must reject path, compressed hash, decompressed header,
metadata subset, source-manifest, schema, type, order, duplicate, or semantic
drift.

Forbidden source fields and families include:

```text
TOT
LE30
DVP
GCF
TRI
outstanding volume
final or as-of vintage
TRI-minus-TRIV1 residual
market price, funding, premium, OI, liquidation, return, label, action,
reward, PnL, portfolio, comparator, or 2024+ data
```

## Pre-2023 parser seal

The source-support builder must hash the complete frozen source artifact. For
each physical row it may parse mnemonic, observation date, and availability
text before value conversion. From that non-value clock metadata, derive:

```text
unreserved_entry = ceil_5m(available_at_utc) + 5 minutes
unreserved_exit  = unreserved_entry + 120 hours
```

It may convert the `value` text only when:

```text
unreserved_entry >= 2020-09-10T00:00:00Z
unreserved_exit  <  2023-01-01T00:00:00Z
```

Every vector whose deterministic unreserved entry or exit falls outside that
strict pre-2023 window remains sealed even when its observation date is in
2022.

For sealed rows it may parse only enough non-value text to:

1. verify that the physical file remains within its already published
   2019–2023 manifest boundary; and
2. skip the row without converting, comparing, counting by TPCT mnemonic, or
   emitting any candidate-specific statistic.

It may not decode the `value` field, build a required-series vector, measure
TPCT completeness, compute a primitive, rank, token, reservation, calendar
distribution, signature, or unseen vocabulary for any sealed row.

No sealed-boundary TPCT source statistic may appear in the preregistration or
source-support report. The first stage allowed to decode those late-2022/2023
values is the single immutable final-policy evaluation stage after every
pre-2023 policy artifact is hash-frozen.

## Exact source vector

For source observation date `d`, require exactly one row for every selected
mnemonic. Every row must:

- have the exact expected `segment`, `measure`, `subset`, and `series_name`;
- have `disclosure_edit` exactly text `0`;
- have nonempty exact-decimal `value`;
- have the exact frozen availability:

  ```text
  max(d 00:00:00 UTC + 8 elapsed calendar days,
      2020-09-10 00:00:00 UTC)
  ```

- contain no NaN, infinity, exponent, leading plus, thousands separator, or
  negative zero representation; and
- parse exactly as `fractions.Fraction(value_text)`.

All eight `TV` values must be strictly positive. Every `AR` value must be
finite; negative rates are permitted because sign is economically meaningful
and the source schema does not prohibit them.

A date with at least one selected row but not a complete valid vector is an
invalid source date. It breaks token-transition continuity at the maximum
availability among selected rows present for that date. A date with no
selected row is not a TPCT source date and does not manufacture a break.

There is no fill, interpolation, zero substitution, epsilon, clipping,
winsorization, final-vintage repair, neighboring-bucket substitution, or total
series reconciliation.

## Equal-availability batches

Rows sharing one `available_at_utc` form one causal batch. For each batch:

1. construct complete/invalid vectors without using a row outside the batch;
2. compute every complete vector's primitive values;
3. compute ranks only against complete vectors whose availability is strictly
   earlier than the batch;
4. never let one current-batch row enter another current-batch reference;
5. allow only the greatest complete observation date to become a decision row,
   and only if the batch contains no invalid selected-source date;
6. compute that row's transition tokens only against the immediately previous
   rank-complete decision from a strictly earlier availability batch;
7. finalize that row's token state and reservation; and
8. only then append complete current-batch vectors to future histories in
   ascending observation-date order.

An invalid selected-source date in the batch breaks continuity after the
batch and prevents a decision opportunity from that batch. This rule is
source-value- and token-blind.

The large 2020-09-10 publication-floor batch may seed later strict-prior
history but cannot manufacture same-timestamp sequential decisions.

## Exact ten primitives

Use these mnemonic abbreviations:

```text
v_X = exact transaction volume for subset X
r_X = exact average rate for subset X
```

Tenor totals:

```text
V_TENOR = v_OO + v_B27 + v_B830 + v_G30
V_TERM  = v_B27 + v_B830 + v_G30
```

Collateral totals:

```text
V_COLLATERAL = v_T + v_AG + v_CORD + v_O
V_GOV        = v_T + v_AG
V_PRIVATE    = v_CORD + v_O
```

All denominators are strictly positive because every component volume is
strictly positive.

Primitive order and formulas:

```text
1. OVERNIGHT_SHARE =
       v_OO / V_TENOR

2. NEAR_TERM_SHARE =
       v_B27 / V_TENOR

3. MEDIUM_TERM_SHARE =
       v_B830 / V_TENOR

4. LONG_TERM_SHARE =
       v_G30 / V_TENOR

5. TERM_PREMIUM =
       (v_B27*r_B27 + v_B830*r_B830 + v_G30*r_G30) / V_TERM
       - r_OO

6. GOVERNMENT_SHARE =
       V_GOV / V_COLLATERAL

7. TREASURY_WITHIN_GOV =
       v_T / V_GOV

8. CORPORATE_WITHIN_PRIVATE =
       v_CORD / V_PRIVATE

9. PRIVATE_COLLATERAL_PREMIUM =
       (v_CORD*r_CORD + v_O*r_O) / V_PRIVATE
       - (v_T*r_T + v_AG*r_AG) / V_GOV

10. CONCENTRATION_GAP =
       [(v_OO/V_TENOR)^2
        +(v_B27/V_TENOR)^2
        +(v_B830/V_TENOR)^2
        +(v_G30/V_TENOR)^2]
       -
       [(v_T/V_COLLATERAL)^2
        +(v_AG/V_COLLATERAL)^2
        +(v_CORD/V_COLLATERAL)^2
        +(v_O/V_COLLATERAL)^2]
```

Every addition, multiplication, division, subtraction, square, equality, and
ordering operation is exact rational arithmetic. Binary floating point is
forbidden in source support.

The four tenor shares must sum exactly to one. The four collateral shares used
inside `CONCENTRATION_GAP` must sum exactly to one. These are arithmetic
identities, not source-total reconciliation.

## Strictly prior ranks

Primitive key order:

```text
OVERNIGHT
NEAR_TERM
MEDIUM_TERM
LONG_TERM
TERM_PREMIUM
GOVERNMENT_SHARE
TREASURY_WITHIN_GOV
CORPORATE_WITHIN_PRIVATE
PRIVATE_PREMIUM
CONCENTRATION_GAP
```

Every primitive uses exactly the previous 252 complete source vectors by
observation-date order, restricted to vectors whose availability is strictly
earlier than the current batch. There is no shorter-history fallback,
expanding alternative, weekday conditioning, year conditioning, calendar
fill, or fitted transform.

For exact rational current value `x` and exact 252-value prior `R`:

```text
rank(x;R) =
    (count(R < x) + 0.5*count(R == x)) / 252
```

Ranks are exact fractions in `[0,1]`. The current value enters history only
after every current-batch rank is fixed. Histories are independent by
primitive.

A **rank-complete decision state** is a complete batch decision row with all
ten ranks. The first state after startup or a continuity break is
predecessor-only and emits no policy token row. Reservation/split suppression
does not break continuity; invalid selected-source vectors do.

## Exact twelve-token grammar

The grammar avoids raw LOW/MID/HIGH values. Six tokens compare rolling ranks;
six describe the joint topology and its transition.

### Pair relation

For named pair `(left,right)`:

```text
d = rank(left) - rank(right)

d >  1/6 -> LEFT
d < -1/6 -> RIGHT
otherwise -> BALANCED
```

Exact `+1/6` and `-1/6` are `BALANCED`.

The six semantic pair tokens are:

```text
maturity_wings:
  OVERNIGHT_LEADS | BALANCED | LONG_TERM_LEADS

term_belly:
  NEAR_TERM_LEADS | BALANCED | MEDIUM_TERM_LEADS

term_volume_rate:
  LONG_TERM_VOLUME_LEADS | BALANCED | TERM_RATE_LEADS

collateral_volume_rate:
  GOVERNMENT_VOLUME_LEADS | BALANCED | PRIVATE_RATE_LEADS

safe_risky_composition:
  TREASURY_LEADS | BALANCED | CORPORATE_LEADS

rate_surface:
  TERM_RATE_LEADS | BALANCED | PRIVATE_RATE_LEADS
```

Pair inputs:

```text
maturity_wings =
  (OVERNIGHT, LONG_TERM)

term_belly =
  (NEAR_TERM, MEDIUM_TERM)

term_volume_rate =
  (LONG_TERM, TERM_PREMIUM)

collateral_volume_rate =
  (GOVERNMENT_SHARE, PRIVATE_PREMIUM)

safe_risky_composition =
  (TREASURY_WITHIN_GOV, CORPORATE_WITHIN_PRIVATE)

rate_surface =
  (TERM_PREMIUM, PRIVATE_PREMIUM)
```

### High and low leaders

`high_leader` is the identity of the unique maximum rank. `low_leader` is the
identity of the unique minimum rank. An exact tie for the relevant extreme is
`TIE`. There is no epsilon.

Vocabulary:

```text
OVERNIGHT
NEAR_TERM
MEDIUM_TERM
LONG_TERM
TERM_PREMIUM
GOVERNMENT_SHARE
TREASURY_WITHIN_GOV
CORPORATE_WITHIN_PRIVATE
PRIVATE_PREMIUM
CONCENTRATION_GAP
TIE
```

### Rank breadth

```text
high = count(rank > 0.5)
low  = count(rank < 0.5)
breadth = high-low

breadth >=  3 -> HIGH_BROAD
breadth <= -3 -> LOW_BROAD
otherwise     -> MIXED
```

Exact rank `0.5` contributes to neither side.

### Extreme occupancy

Count ranks strictly below `1/6` or strictly above `5/6`:

```text
0..3  -> COMPACT
4..6  -> FOCUSED
7..10 -> FRACTURED
```

Exact `1/6` and `5/6` are not extreme.

### Order transition

For all 45 unordered primitive pairs, encode current and previous strict rank
order as `-1`, `0`, or `+1`. Count pair states that changed, including entry
into or out of an exact tie:

```text
0..5  -> STABLE
6..14 -> ROTATING
15..45 -> RESET
```

### Leader transition

Compare current and previous high/low leaders:

```text
any current/previous leader is TIE -> TIE_INVOLVED
neither changed                    -> BOTH_STABLE
only high changed                  -> HIGH_ROTATED
only low changed                   -> LOW_ROTATED
both changed                       -> BOTH_ROTATED
```

### Canonical token order

```text
1. maturity_wings
2. term_belly
3. term_volume_rate
4. collateral_volume_rate
5. safe_risky_composition
6. rate_surface
7. high_leader
8. low_leader
9. rank_breadth
10. extreme_occupancy
11. order_transition
12. leader_transition
```

The prompt may contain only these twelve `KEY=VALUE` tokens, one task
identifier, and one neutral-code action-option order.

Forbidden model inputs:

```text
raw source values or numeric ranks
observation date, availability, year, month, weekday, mnemonic, row identity
BTC price, return, funding, premium, OI, Kimchi premium, DXY, or future path
prior action, action history, current position, executability, portfolio
conflict, reward, PnL, CAGR, MDD, or split identity
RVFC/RMSR/RCRE/DMSH states, sides, events, controls, or outcomes
source path, hash, transport, revision, or metadata identity
free-form rationale, chain of thought, generated feature, or hidden state
```

Current position is deterministically flat at every globally reserved TPCT
opportunity because reservation precedes inference and TPCT positions never
overlap. External portfolio conflicts are post-policy deterministic execution
guards. Unknown tokens, stale state, model errors, nonfinite scores, or
unexecutable orders force `ABSTAIN`.

## Exact causal execution clock

For the complete decision vector:

```text
raw_signal_available =
    max(available_at_utc across the sixteen rows)

ceil_5m(t) =
    smallest UTC timestamp >= t divisible by 300 elapsed seconds

signal_available = ceil_5m(raw_signal_available)
entry            = signal_available + 5 elapsed minutes
exit             = entry + 120 elapsed hours
```

An exact five-minute `raw_signal_available` is not advanced by the ceiling.
The separate five-minute inference/order buffer always remains.

Execution:

- Binance USD-M BTCUSDT perpetual;
- exact five-minute open at entry and exit;
- fixed `0.5x` account gross;
- exactly 1,440 held five-minute bars / 120 elapsed hours;
- scheduled exit only;
- no stop, take-profit, trailing, pyramiding, dynamic size, model exit,
  source-side override, or external regime gate;
- reserve `[entry,exit)` globally before policy inference;
- accept a reservation only when `entry >= prior_reserved_exit`;
- abstention, scoring failure, market defect, or later portfolio conflict does
  not release the reservation;
- suppress rather than queue every overlapping later state; and
- reserve before split containment.

Split intervals are half-open `[start,end)`. A reservation is emitted only
when:

```text
entry >= split_start
exit  < split_end
```

The strict exit inequality is frozen. A reservation rejected by containment
still consumes its globally reserved interval.

Suppressed or split-rejected states remain primitive/rank/token predecessors.

## Frozen temporal roles

```text
source history                  [2019-01-01, first eligible decision)
policy train                    [2020-09-10, 2022-01-01)
development selection           [2022-01-01, 2023-01-01)
untouched candidate evaluation  [2023-01-01, 2024-01-01)
sealed                          [2024-01-01, ...)
```

Split assignment is by reserved entry. The full strict-contained hold must
remain in one split.

Every cheap policy and Gemma adapter fits only policy-train rows. The final
algorithm/checkpoint is selected only on 2022. No monthly, rolling, continual,
or eval-label adaptation is allowed.

2023 source covariates, tokens, actions, market, and funding remain sealed
until the final pre-2023 policy hash freeze.

## Source-only support gate

The source-support builder may decode only the source/metadata/manifest
allowlists and values whose deterministic unreserved hold is strictly
contained before 2023. It may not load market, funding, label, action, reward,
return, PnL, or any sealed-boundary TPCT value.

Counts use token-ready, globally reserved, split-contained opportunities.

### Integrity and incidence

All must pass:

- every frozen source, metadata, manifest, canonical-hash, header, schema,
  semantic, exact-decimal, availability, and allowlist check;
- exact ten-primitive rational identities;
- exact current-batch exclusion and 252-row strict-prior histories;
- deterministic greatest-observation-date batch tie-break;
- continuity break and predecessor rules;
- exact five-minute ceiling, latency, 120-hour hold, strict split containment,
  and action-independent global reservation;
- at least 75 policy-train opportunities;
- at least 15 train opportunities whose entries are in 2020;
- at least 50 train opportunities whose entries are in 2021;
- at least 55 development-selection opportunities in 2022;
- at least three active train execution months in 2020;
- at least eleven active execution months in each of 2021 and 2022;
- at least 23 opportunities per half-year in each of 2021 and 2022;
- at least ten opportunities per quarter in each of 2021 and 2022;
- maximum single-month share at most 20% separately in policy train and 2022;
- maximum consecutive emitted-entry gap within the same split at most ten
  elapsed days;
- first emitted train entry at most 21 days after train start;
- first emitted 2022 entry at most 15 days after 2022 start;
- last emitted train exit at most 15 days before train end;
- last emitted 2022 exit at most 15 days before 2022 end;
- cross-boundary blackout from last emitted train entry to first emitted 2022
  entry at most 20 elapsed days;
- no sealed-boundary TPCT value or candidate statistic decoded; and
- no raw value, rank, action, side, market, return, or outcome column in the
  emitted clock.

The interior-gap gate includes calendar holidays and invalid source vectors.
Only the cross-split pair is excluded from it and remains subject to the
separate 20-day cap.

### Token support

Policy train and 2022 must each satisfy:

- every value of every pair token occurs at least three times and has at least
  3% share;
- no pair-token value exceeds 85%;
- at least five non-tie high-leader and five non-tie low-leader identities in
  policy train;
- at least four non-tie identities for each leader token in 2022;
- no non-tie leader identity exceeds 50%;
- `TIE` is at most 20% for each leader token;
- every rank-breadth value occurs at least three times and none exceeds 90%;
- every extreme-occupancy value occurs at least twice and none exceeds 92%;
- every order-transition value occurs at least twice and none exceeds 92%;
- at least four leader-transition values occur in train and at least three in
  2022;
- no leader-transition value exceeds 85%;
- largest exact twelve-token signature share is at most 15%;
- no token is missing or invalid; and
- every token value appearing in 2022 already appears in policy train.

The builder must not inspect whether any 2023 token value would be unseen.

Required synthetic tests:

- path/hash/header/metadata/manifest/allowlist enforcement;
- exact decimal and forbidden representation rejection;
- 16-row completeness, duplicate, disclosure-edit, and positive-volume rules;
- metadata version drift and non-overlap proof;
- conservative availability and publication-floor batch behavior;
- equal-batch current exclusion and greatest-date tie-break;
- invalid-date continuity breaks;
- exact rational primitives and simplex identities;
- 252-row midrank ties and no fallback;
- pair threshold equality;
- exact leader ties, breadth, occupancy, order transition, leader transition;
- predecessor inclusion despite reservation or split suppression;
- five-minute ceiling and latency;
- strict half-open reservation and split containment;
- cross-boundary versus interior-gap accounting; and
- hard proof that 2023 values are not converted.

Any failure retires TPCT-120 unchanged before comparator or market outcomes.

## Frozen source-only controls

These controls are generated only after primary source support passes. They
diagnose grammar mechanics and can never replace TPCT:

1. `one_decision_stale` — apply the immediately previous token state at each
   current reservation;
2. `five_decision_stale` — apply the fifth previous token state;
3. `year_primitive_permutation` — independently permute each complete
   primitive inside observation year with destination order fixed by
   `SHA256("TPCT-120|year_primitive_permutation|<year>|<primitive>|<date>")`,
   then recompute ranks/tokens;
4. `joint_year_state_permutation` — permute complete ten-primitive vectors
   within year with
   `SHA256("TPCT-120|joint_year_state_permutation|<year>|<date>")`;
5. `pair_orientation_flip` — exchange every non-`BALANCED` left/right pair
   value without refitting; and
6. `leader_role_flip` — exchange `high_leader` and `low_leader` values while
   retaining their key positions, without refitting.

Permutation controls use only pre-2023 source values and current causal
availability. They may not qualify as TPCT.

## Pre-outcome clock novelty

Comparator rows remain closed until source and token support pass. The
source-only novelty stage then reads only comparator policy/group ID, entry,
exit, decision/admission if present, and side/action only when required to
validate schema. It may not read comparator source features or outcomes.

Frozen comparator cohort:

| Comparator | Artifact | SHA-256 |
|---|---|---|
| RVFC primary and source controls | `results/ofr_repo_venue_fragmentation_consensus_clocks_2026-07-23.csv.gz` | `b2d251b147feecc98def94610834c7b11f078f35e580f6e7a950fc55bfe0724e` |
| RMSR primary and source controls | `results/ofr_repo_mix_shock_resolution_race_clocks_2026-07-23.csv.gz` | `bed2f522bd68fd389cb0af3d655980999324c0e6adb4f1dc027c6b6ea2cc5ae6` |
| RCRE primary and source controls | `results/ofr_repo_collateral_routing_efficiency_clocks_2026-07-23.csv.gz` | `cbe4e5f6fc52b66062abbf931e46ea4aa0d1f3c0157ffd365d0638aa573c2826` |
| DMSH primary and source controls | `results/ofr_dvp_maturity_stock_flow_handoff_clocks_2026-07-23.csv.gz` | `0cfb881b4e3a0123111eeab904eba7bee074767b9c1315f74e7bddf54e3371c3` |
| Federal-liquidity components | `results/federal_liquidity_component_concordance_preregistered_clock_2026-07-17.csv.gz` | `7ebb0450422d9265e46c596e0b6415b6a8816c66f5e0cbb9ccda14ca6cb4c67c` |
| Frozen live sleeves | `results/cchr_live_portfolio_pure_clocks_2026-07-21.csv.gz` | `73d6efbd35b3be64b0fa04fa9c8cb2db25866ef884f19b1ae673949e22a42b08` |

Use the common intersection of declared pre-2023 coverage. For every required
nonempty comparator group:

- exact-entry Jaccard at most 0.20;
- maximum-cardinality one-to-one 24-hour tolerant entry Jaccard at most 0.50;
- absolute Pearson correlation of unsigned occupied time on the common
  five-minute grid at most 0.75; and
- zero variance, undefined correlation, missing required common coverage,
  duplicate entry, non-contained interval, schema drift, or hash drift fails.

Against each frozen live sleeve, stricter limits apply:

- exact-entry Jaccard at most 0.10;
- 24-hour tolerant Jaccard at most 0.35; and
- absolute unsigned occupied-time correlation at most 0.60.

Tolerant matching is chronological maximum-cardinality one-to-one matching.
Failure retires TPCT before any BTC or funding row is opened.

## Frozen accounting

Market:

```text
data/binance_um_kline_reference_btc_2020_2023/
  BTCUSDT_5m_2020-01-01_2023-12-31.csv.gz
SHA256 e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d

data/binance_um_kline_reference_btc_2020_2023/build_manifest.json
SHA256 c04fbbd299cc748a6745c0ef030787da4d560833c744c81c98dd8840efc7913e
```

Funding:

```text
data/binance_um_btcusdt_funding_marks_2020_2023.csv.gz
SHA256 3284bbb6bb67946acb673c6b67459543e217f752589e1d47b6c7c3b659f733e6

results/binance_um_btcusdt_funding_marks_2020_2023_manifest_2026-07-17.json
SHA256 a0b2d27e1aa8cf2d9ab8cb659b598ee0a6d7bd25401c9e10ae92d1a74415845b
```

Costs and quantity:

```text
leverage                         0.5
base cost/notional/side          0.0006
stress cost/notional/side        0.0010
quantity                         entry_equity*0.5/entry_open
```

Stress replaces base cost. Quantity remains fixed until scheduled exit.

For settlement `entry <= time <= exit`:

```text
funding_cash =
    -side * fixed_quantity * settlement_mark_price * funding_rate
```

Positive funding exactly at entry or exit is dropped; negative boundary
funding is retained. Interior funding is retained.

Strict MDD uses one global/pre-entry high-water mark and marks:

1. post-entry cost;
2. favorable held OHLC extreme;
3. adverse held OHLC extreme with retained funding and virtual adverse exit
   cost; and
4. scheduled-exit equity.

Favorable before adverse is deliberately conservative. Full-calendar CAGR
uses the complete half-open declared split, including warmup, idle cash, and
abstention:

```text
years = (split_end-split_start)/(365.25 days)
CAGR  = final_equity**(1/years)-1
```

Reports always include absolute return, full-calendar CAGR, strict MDD,
CAGR/strict-MDD, trades, LONG/SHORT counts, action shares, side contributions,
mean signed gross move, active months, halves, quarters, funding, costs,
stress, delay controls, and clustered uncertainty.

Weekly-cluster sign flip:

- group net compounded account returns by UTC ISO entry week;
- use independent `numpy.random.default_rng(20260724)` per split/policy;
- use 100,000 Rademacher sign draws; and
- `p=(1+count(null>=observed))/100001`.

No-trade/no-cluster policies return `1.0`.

### Familywise selection correction

Every development-stage policy comparison uses one shared max-stat null. Place
each policy's weekly net returns on the union of nonempty UTC weeks, using zero
for flat weeks, and compute:

```text
t_policy =
    mean(weekly_return)
    / (std(weekly_return,ddof=1)/sqrt(number_of_union_weeks))
```

Zero variance returns negative infinity. Each of 100,000 draws applies the
same Rademacher sign to the same week for every frozen policy and retains the
maximum null `t`. The adjusted one-sided p-value is:

```text
p_max =
    (1 + count(max_null_t >= observed_selected_t)) / 100001
```

The family includes every cheap primary, exact-memory, prior-only,
quarter-prior, single-token, group-only, ablation, shuffled-label,
shuffled-utility, circular-shift, masked-token adapter, SFT, DPO checkpoint,
and orientation-flipped inference policy emitted at that stage. A seen policy
cannot be removed.

The immutable 2023 policy reports an ordinary one-policy p-value because no
2023 policy is selected. The overall discovery remains explicitly exploratory
until unchanged 2024 confirmation.

## Train-only utility and labels

For each policy-train opportunity and action:

```text
U(ABSTAIN) = 0

U(trade) =
    log(max(account_multiplier,1e-12))
  - (1/3)*local_held_path_strict_drawdown
  - 0.0005
```

The final term is an account-level hurdle, not an execution cost. Utility uses
the same base costs, funding, and 120-hour path accounting as evaluation.

Oracle tie priority:

```text
ABSTAIN, LONG, SHORT
```

SFT target is the unique oracle action after tie priority.

DPO creates every unordered action pair whose absolute utility difference is
at least `0.0003`; the higher-utility action is chosen and the lower rejected.
No outcome-dependent sampling, class balancing, direction balancing,
hard-negative mining, source mirroring, or synthetic label is allowed.

Before GPU work:

- no SFT target exceeds 90%;
- LONG and SHORT each form at least 15% of SFT targets;
- every action is preferred in at least 10% of retained DPO pairs; and
- the DPO set is nonempty.

## Cheap causal baselines

Tokens are nominal and never ordinal integers.

Representation:

- one-hot every policy-train-observed token value;
- one-hot all 66 unordered token-pair conjunctions;
- retain features occurring at least five times in fit data; and
- one unpenalized intercept where supported.

Unknown downstream token values force `ABSTAIN`.

Frozen policies:

1. always abstain;
2. always long;
3. always short;
4. exact-signature majority-oracle memory, unseen signatures abstain;
5. categorical Naive Bayes oracle-action classifier, Laplace alpha `1.0`;
6. separate LONG/SHORT ridge utility regressions, alpha `100.0`;
7. separate LONG/SHORT Extra Trees utility regressions:
   - 512 estimators,
   - squared-error criterion,
   - max depth 4,
   - min split 16,
   - min leaf 8,
   - sqrt max features,
   - no bootstrap,
   - seed 20260724;
8. fit-majority three-action prior;
9. fit admission-prior plus direction-prior constant policy;
10. UTC calendar-quarter majority prior `Q1..Q4`, diagnostic only;
11. 32 shuffled-label Naive Bayes controls, seeds `20260724..20260755`;
12. 32 independently shuffled-utility ridge controls, same seeds;
13. 16 circular chronological label/utility shifts:

    ```text
    [7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67]
    ```

14. twelve single-token ridge policies;
15. twelve leave-one-token-out ridge policies;
16. five group-only ridge policies:
    - pair relations;
    - leaders;
    - breadth/occupancy;
    - transitions;
    - current topology without transitions; and
17. five leave-one-group-out ridge policies over the same groups.

Every learned primary and later Gemma checkpoint emits without refitting:

18. pair-orientation-flipped inference; and
19. leader-role-flipped inference.

Prior, seasonal, shuffled, circular, single-token, group-only, ablation, and
orientation controls cannot qualify as TPCT.

### 2022 cheap learnability gate

At least one of Naive Bayes, ridge contextual utility, or Extra Trees
contextual utility must satisfy unchanged on 2022:

- positive absolute return;
- `CAGR/strict-MDD >= 1.0`;
- strict MDD at most 15%;
- positive H1 and H2 return;
- at least 30 trades;
- at least ten trades in each half;
- at least eight LONG and eight SHORT trades;
- positive LONG and SHORT contribution separately;
- no action above 90%;
- positive stress-cost return;
- positive one-hour-delay return;
- familywise weekly-cluster `p_max < 0.20`;
- higher return and ratio than always abstain, always long, always short, and
  exact-signature memory;
- higher return and ratio than every prior-only, quarter, shuffled, and
  circular control;
- higher return and ratio than the strongest single-token or group-only
  policy;
- higher return and ratio than both orientation-flipped inference controls;
  and
- no single-token majority-action policy reproduces more than 75% of selected
  non-abstain actions on matching opportunities.

Select by higher ratio, higher return, lower MDD, then lexicographically
smaller policy ID. Failure retires TPCT before GPU.

## Frozen single-Gemma RLLM

### Model and artifact

```text
model      google/gemma-4-E2B-it
revision   3e22461f65e89153144f8adb70e3b8c2cc9845a7
loader     transformers.AutoModelForMultimodalLM
processor  transformers.AutoProcessor
trust_remote_code=False
text only
thinking disabled
```

Runtime-used snapshot files:

| File | SHA-256 |
|---|---|
| `chat_template.jinja` | `0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5` |
| `config.json` | `1b28f3d2c3100f6c594754b81107428bd7b822a7f48272ca681dae9d2ec38330` |
| `generation_config.json` | `d4226bbe3117d2d253ba4609720ba82c6c4ce4627a9a6ae05387c78983ac03de` |
| `model.safetensors` | `2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550` |
| `processor_config.json` | `32bdf45d2ad4cc29a0822ddd157a182de76644f0419a6228d151495256e9813c` |
| `tokenizer.json` | `cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f` |
| `tokenizer_config.json` | `9f4fec4b1dc6ecddf8f4a92e9caea5971c0e67d81309f3f9066a2bee8c362633` |

Official model card:

<https://huggingface.co/google/gemma-4-E2B-it>

Gemma 4 is multimodal, but TPCT must instantiate no image/audio/video tensor,
encoder input, image token, multimodal example, analyzer model, second trader
model, free-form rationale, or hidden-reasoning target.

Runtime:

```text
torch             2.9.0
transformers git  5d7ff4393ab99aa7cadf4cccd1f814dbb799f2bb
trl               0.29.0
peft              0.18.1
bitsandbytes      0.49.2
numpy             2.2.6
pandas            2.3.3
scikit-learn      1.7.2
```

Quantization:

```text
load_in_4bit=True
bnb_4bit_quant_type="nf4"
bnb_4bit_use_double_quant=True
bnb_4bit_compute_dtype=torch.bfloat16
```

LoRA:

```text
r=8
alpha=16
dropout=0.05
bias="none"
task_type="CAUSAL_LM"
target_regex=.*language_model.*\.(q_proj|k_proj|v_proj|o_proj)$
expected trainable parameters=2,678,784
```

Memory/artifact gates:

- exactly one visible CUDA device;
- BF16 support required;
- inference peak allocated at most 7.0 GiB;
- inference peak reserved at most 7.25 GiB;
- training peak reserved at most 24 GiB;
- training peak allocated at most 20 GiB;
- each adapter/checkpoint at most 256 MiB; and
- retained final SFT plus selected DPO at most 1 GiB.

### Neutral action scoring

Serialize the twelve tokens in canonical order. End the text-only user prompt:

```text
TASK=TPCT_ACTION
OPTIONS:
<one permutation of the three fixed mapping lines below>
Return exactly CHOICE=<one option>.
```

The fixed neutral mapping lines are:

```text
Q1=ABSTAIN
Q2=LONG
Q3=SHORT
```

Every state is scored under all six permutations of the displayed option
lines. The only valid completions are `CHOICE=Q1`, `CHOICE=Q2`, and
`CHOICE=Q3`. Generation is forbidden.

For each permutation and action completion:

1. compute conditional completion-token log probability;
2. exclude prompt and special tokens;
3. divide by completion-token count;
4. average each action over all six option orders;
5. compute the same mean with the adapter disabled;
6. use `adapter_delta = adapted - base`; and
7. subtract the policy-train mean adapter delta separately for each action.

Offsets are frozen from policy train and never recomputed downstream. Choose
the maximum calibrated action score. Ties within absolute `1e-12` use priority
`ABSTAIN`, `LONG`, `SHORT`. Any malformed/nonfinite/missing score abstains.

Maximum prompt plus completion length is 384 tokens. Truncation is forbidden.

One masked-token prior adapter uses the identical labels and recipe but
replaces every value with literal `MASKED` while retaining keys and option
orders. It can never qualify. The selected TPCT checkpoint must beat it.

### SFT

Use all policy-train oracle actions in chronological order and all six option
orders. Cycle that deterministic ordered sequence only as needed to fill the
exact optimizer-step budget; do not sample, balance, or drop based on outcome.

```text
optimizer             AdamW
learning_rate         1e-4
betas                 (0.9,0.999)
epsilon               1e-8
weight_decay          0.01
scheduler             cosine
warmup_steps          8
max_grad_norm         1.0
optimizer_steps       64
per_device_batch      1
gradient_accumulation 8
packing               false
completion_only_loss  true
bf16                  true
seed                  20260724
```

Final SFT initializes DPO. No SFT checkpoint is selected.

### DPO

Use every qualifying train-only pair and every option order in deterministic
chronological/action-pair order. Cycle only to fill the exact step budget. The
reference is final SFT with DPO updates disabled.

```text
loss                  sigmoid
beta                  0.1
label_smoothing       0.0
optimizer             AdamW
learning_rate         5e-6
betas                 (0.9,0.999)
epsilon               1e-8
weight_decay          0.01
scheduler             cosine
warmup_steps          8
max_grad_norm         1.0
optimizer_steps       96
per_device_batch      1
gradient_accumulation 8
bf16                  true
seed                  20260724
checkpoints           [24,48,72,96]
```

Each checkpoint is evaluated once on 2022. It qualifies only with:

- positive absolute return;
- `CAGR/strict-MDD >= 2.0`;
- strict MDD at most 15%;
- positive H1 and H2 return;
- at least 30 trades;
- at least ten trades in each half;
- at least eight LONG and eight SHORT trades;
- positive LONG and SHORT contribution separately;
- no action above 90%;
- positive stress-cost return;
- positive one-hour-delay return;
- familywise weekly-cluster `p_max < 0.10`;
- return and ratio above every cheap primary, prior, quarter, shuffled,
  circular, single-token, group-only, and ablation policy;
- return and ratio above both orientation-flipped inference controls;
- return and ratio above the masked-token prior adapter;
- ratio at least 0.25 above the strongest frozen non-RLLM policy;
- no one token value contains more than 65% of non-abstain actions; and
- no single-token majority policy reproduces more than 75% of non-abstain
  actions.

Select by higher ratio, higher return, lower MDD, then earlier optimizer step.
Failure retires TPCT before 2023. Retain final SFT and selected DPO only.

## Final pre-2023 policy freeze

Before opening one 2023 TPCT value, bind:

- source, metadata, manifest, mechanism, preregistration, and code hashes;
- exact token vocabulary learned from policy train;
- every cheap-policy artifact and 2022 result;
- base model snapshot and runtime hashes;
- final SFT and selected DPO adapter hashes;
- train-only calibration offsets;
- prompt/token/action serialization;
- selected 2022 policy and all controls;
- complete pre-2023 opportunity/action clocks; and
- unknown-token, runtime-failure, and live-staleness abstention behavior.

No 2022 failure may be deferred to 2023.

## Untouched 2023 inference, novelty, and outcome gate

One committed evaluator opens 2023 in this order without returning control to
the researcher between substeps:

1. decode the sealed late-2022/2023 TPCT source vectors needed to construct
   2023 reservations;
2. reproduce all pre-2023 values, tokens, reservations, and actions
   byte-for-byte;
3. infer the unchanged 2023 action clock, with unseen tokens abstaining;
4. run signed/unsigned clock novelty using only action-clock comparator fields;
5. if novelty passes, load 2023 market and funding rows; and
6. emit one immutable source/economic/control report.

The 2023 source must independently satisfy, without repair:

- at least 55 split-contained reserved opportunities;
- at least eleven active reservation months;
- at least 23 opportunities in each half;
- at least ten opportunities per quarter;
- maximum month share at most 20%;
- maximum within-split entry gap at most ten elapsed days;
- first entry and last exit boundary distances at most 15 elapsed days;
- every token is valid;
- unseen pre-2023 token values abstain; and
- no source/schema/version/replay drift.

Policy novelty over common pre-2024 coverage:

- exact-entry Jaccard at most 0.10 against each frozen live sleeve;
- 24-hour tolerant entry Jaccard at most 0.30;
- absolute signed occupied-exposure correlation at most 0.35;
- absolute unsigned occupied-time correlation at most 0.60; and
- every required comparator has valid nonempty common coverage.

The unchanged TPCT policy must then satisfy:

- positive absolute return;
- `CAGR/strict-MDD >= 3.0`;
- strict MDD at most 15%;
- positive H1 and H2 return;
- at least 30 trades;
- at least ten trades in each half;
- at least eight LONG and eight SHORT trades;
- positive LONG and SHORT contribution separately;
- at least eight active execution months;
- maximum single execution-month share at most 20%;
- no action above 90%;
- at least 20 nonempty UTC entry-week clusters;
- one-policy weekly-cluster one-sided `p < 0.05`;
- mean signed gross underlying move at least 40 bp per trade;
- positive stress-cost return;
- positive one-hour-delay return;
- positive return under every neutral-option-order audit;
- return and ratio above the frozen strongest cheap policy;
- ratio at least 0.50 above that cheap policy;
- return and ratio above both orientation controls and the masked-token prior;
- no one token value contains more than 65% of non-abstain actions; and
- no single-token majority policy reproduces more than 75% of non-abstain
  actions.

One-day delayed entry is mandatory reporting, not a gate.

Any failure retires TPCT-120 unchanged.

## Required 2024 confirmation

A 2023 pass is not production authorization because TPCT was selected after
several source-support attempts. It authorizes one official
preliminary-vintage 2024 extension.

The extension must be separately fetched, raw-response/hash frozen, and
source-audited after the 2023 report commit. Before opening 2024 outcomes it
must reproduce all pre-2024 source rows, primitives, ranks, tokens,
reservations, and actions byte-for-byte.

The unchanged policy must pass every 2023 source, economic, risk, direction,
cost, delay, and significance gate on full-calendar 2024. Combined 2023–2024
weekly-cluster p must be below 0.01. Failure retires TPCT; no retraining,
continual update, prompt repair, checkpoint change, threshold adjustment, or
leverage increase is allowed.

Only an unchanged 2024 pass may authorize later 2025 and 2026-YTD reports.

## Live boundary

Live source use must:

- query only official preliminary OFR endpoints;
- persist raw responses, request URL, retrieval UTC, response hash, metadata
  version, and publication clock;
- use the later of actual receipt/validation time and the frozen conservative
  eight-day clock;
- reject revisions to a state already acted upon;
- fail flat on a missing, partial, duplicated, disclosure-edited, stale,
  semantically drifted, or non-replayable vector;
- use the frozen tokenizer/model/adapter/scoring path;
- abstain on any unknown token or runtime fault; and
- place orders only through deterministic risk/execution guards.

The model never sets size, hold, stop, leverage, timestamp, or exit.

## Mandatory sequence

1. commit this mechanism;
2. commit canonical preregistration and synthetic tests;
3. commit source-only builder and tests;
4. execute pre-2023 source support exactly once;
5. retire unchanged on any source/token failure;
6. run and freeze pre-outcome clock novelty;
7. if it passes, commit and hash-freeze the economic evaluator and cheap
   baselines;
8. open only policy-train and 2022 outcomes;
9. retire before GPU on cheap learnability failure;
10. train one frozen Gemma 4 E2B SFT and four DPO checkpoints;
11. select one checkpoint on 2022 and freeze every pre-2023 artifact;
12. run the atomic 2023 source/novelty/outcome evaluation exactly once;
13. fetch and evaluate 2024 only after an unchanged 2023 pass; and
14. commit every completed unit with hashes and fresh tests.

## Outcome boundary

At this mechanism commit:

```text
source artifact bytes hashed                 yes
source/manifest aggregate metadata read      yes
selected metadata objects read               yes
TPCT source values decoded                   0
TPCT primitives/ranks derived                0
TPCT token rows derived                      0
TPCT opportunity rows derived                0
TPCT comparator rows decoded                 0
market rows loaded                           0
funding rows loaded                          0
future-return rows loaded                    0
return or PnL fields                         0
2023 TPCT values decoded                     0
post-2023 source rows loaded                 0
model labels created                         0
model training runs                          0
```

Status:

```text
mechanism_frozen_before_TPCT_values
```
