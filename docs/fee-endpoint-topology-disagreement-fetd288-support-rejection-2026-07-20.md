# FETD-288 support rejection (2026-07-20)

## Decision

Reject `FETD-288` permanently at the outcome-blind support gate.  Do not build
or run a market/funding evaluator for this policy, do not open train or
selection returns, and do not repair its threshold, side mapping, packet size,
rank history, hold, latency, support floors, or calendar rules.

This decision follows the frozen stopping rule in
[`fee-endpoint-topology-disagreement-fetd288-preregistration-2026-07-20.md`](fee-endpoint-topology-disagreement-fetd288-preregistration-2026-07-20.md).
The canonical aggregate-only evidence is
[`fee_endpoint_topology_disagreement_support_2026-07-20.json`](../results/fee_endpoint_topology_disagreement_support_2026-07-20.json).

## Evidence boundary

- Frozen confirmed-ledger source values through 2023 were read once by the
  canonical builder.
- No post-2023 source row, BTC market row, funding row, premium/OI row,
  liquidation/order-book row, return, PnL, or performance value was loaded.
- No event row or feature value was published.  The artifact contains only
  aggregate source/support evidence and cryptographic clock commitments.
- Support artifact file SHA-256:
  `03ba910a314ba6efb647f6588dff603261d414e5114680ca33bdc27d59aed035`.
- Support manifest hash:
  `24902cbe9869d2c5dc3443047d31c7ad0a1650d23822da6158d7e4b5ee758c27`.

## Frozen incidence result

| Window / period | Accepted entries |
|---|---:|
| Train 2021-2022 | 82 |
| 2021 | 21 |
| 2021 H1 | 0 |
| 2021 H2 | 21 |
| 2022 | 61 |
| 2022 H1 | 35 |
| 2022 H2 | 26 |
| Selection 2023 | 37 |
| 2023 H1 | 16 |
| 2023 H2 | 21 |
| 2023 Q1 | 13 |
| 2023 Q2 | 3 |
| 2023 Q3 | 9 |
| 2023 Q4 | 12 |

Side balance itself passed: train had 41 long and 41 short entries; selection
had 16 long and 21 short entries.  The one-bar delayed control dropped zero
entries at split boundaries.

## Failed preregistered gates

Five conjunctive support checks failed:

1. `train_each_year_minimum`: 2021 had 21 entries, below 32.
2. `train_each_half_year_minimum`: 2021 H1 had zero entries, below 14.
3. `train_maximum_month_share`: `17.0732%`, above `15%`.
4. `selection_each_quarter_minimum`: 2023 Q2 had 3 entries, below 6.
5. `selection_maximum_month_share`: `21.6216%`, above `20%`.

The overall counts of 82 train and 37 selection entries are not sufficient to
override these failures.  The signal is temporally clustered and absent from a
complete train half-year, so it cannot provide the preregistered dispersion
needed for a statistically defensible outcome test.

## Research conclusion

The confirmed-fee/endpoint-disagreement mechanism remains economically
plausible, but this exact singleton is too concentrated in time.  Treating the
failed distribution as a prompt to loosen the rank threshold or alter packet
timing would be post-observation repair and is forbidden.  Any future research
must start as a materially different mechanism with a new causal hypothesis
and clean preregistration; it must not be presented as a repaired FETD-288.
