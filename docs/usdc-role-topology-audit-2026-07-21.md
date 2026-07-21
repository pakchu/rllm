# USDC role-topology audit — 2026-07-21

## Verdict

The finalized 2020–2023 USDC source contains a real directed
`mint recipient -> burn caller` closure, so the relation is semantically
testable. **Retire it before temporal pairing**, however: promoting that
relation after AMTR failed would be a post-source carveout, while ordinary
non-tailored actor-breadth and concentration standards already fail. No
temporal pair, comparator, BTC price, funding, future return, PnL, absolute
return, CAGR, or strict MDD was opened.

The topology is extremely centralized. Only two addresses ever appear in
both roles, and one of those roles accounts for more than 99% of the eligible
mint and burn legs. The minimum five operational roles and maximum 40% primary
role concentration already frozen for AMTR's source screen are not met. A
later candidate therefore cannot be described as broad participant
coordination or cross-entity flow. At most, it would retest the state transition
of the same dominant operational hub through a newly selected field relation.

## Reproduced source-only facts

| Item | Result |
|---|---:|
| Eligible USDC mint/burn rows | 265,585 |
| Distinct mint callers | 4 |
| Distinct burn callers | 8 |
| Distinct mint recipients | 6,311 |
| Mint caller ∩ burn caller | 4 |
| Mint recipient ∩ burn caller | **2** |
| Mint recipient ∩ mint caller | 0 |
| All three roles | 0 |

For the two full-period recipient/burner roles:

| Descriptive topology measure | Result |
|---|---:|
| Distinct mint callers into those roles | 2 |
| Distinct directed caller→recipient edges | 2 |
| Mint legs | 84,406 (85.23% of mint events) |
| Mint amount share | 99.43% |
| Burn legs | 143,076 (85.90% of burn events) |
| Burn amount share | 96.75% |
| Largest role share of eligible mint legs | 99.69% |
| Largest role share of eligible burn legs | 99.75% |

Eligible mint legs occur in every year from 2020 through 2023. Eligible burn
legs occur from 2021 through 2023. These are full-panel descriptive counts,
not causal pair incidence.

## Causal and semantic boundary

Circle's contract event meanings are asymmetric:

- `Mint(minter, to, amount)` identifies an authorized caller and a possibly
  different recipient;
- `Burn(burner, amount)` identifies an authorized caller burning its own
  balance.

Therefore matching `mint.to == burn.burner` is a directed operational-balance
closure. It is materially different from AMTR's unrestricted `cross_minter`
control, which only required different callers and ignored where the mint was
delivered. It still does **not** identify a customer, exchange, beneficial
owner, fiat deposit, or BTC buyer.

The set intersection above is computed over the complete source interval only
to establish structural feasibility. Using that full-period membership as a
historical feature would leak future role discovery and is forbidden. A causal
candidate may recognize the relationship only when its two event legs have
actually become available at block `N+64`; it may not preload the two addresses
from this audit.

## Rejection boundary

This audit authorizes neither a mechanism freeze nor a clock. Specifically:

1. the two-role overlap fails the pre-existing five-role breadth standard;
2. greater than 99% eligible-leg concentration fails the pre-existing 40%
   primary-role concentration standard;
3. AMTR controls were explicitly prohibited from authorizing a repaired
   mechanism after incidence, and selecting its mint-recipient relation now
   would violate that boundary; and
4. full-period role membership remains forbidden as a historical feature.

No temporal matching is needed to establish those failures. The next action is
a new independently motivated source or mechanism axis, not a looser graph
gate or an outcome test of this one.

## Artifacts

- evaluator: `training/audit_usdc_role_topology.py`
- result: `results/usdc_role_topology_audit_2026-07-21.json`
- tests: `tests/test_audit_usdc_role_topology.py` and
  `tests/test_usdc_role_topology_audit_artifact.py`
- source: `data/ethereum_stablecoin_issuance_redemption_2020_2023/ethereum_usdt_usdc_issuance_redemption_2020_2023.csv.gz`

Contract references:

- <https://github.com/circlefin/stablecoin-evm/blob/master/contracts/v1/FiatTokenV1.sol>
- <https://developers.circle.com/stablecoins/usdc-contract-addresses>
