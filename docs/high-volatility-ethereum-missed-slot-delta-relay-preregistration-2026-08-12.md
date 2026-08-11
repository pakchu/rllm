# HVEMSD-24 preregistration

`HVEMSD-24` tests an immutable cross-chain liveness shock in July-like BTC
volatility.  For each post-Merge Ethereum UTC day it subtracts the number of
canonical execution blocks from the fixed 7,200-slot schedule.  A tail increase
in missed slots maps short BTC and a tail decrease maps long BTC, conditional on
causally elevated prior-24-hour BTC variation.

The source is reconstructed only after this preregistration from two fixed
Ethereum JSON-RPC hosts.  Both hosts must agree on every retained day-boundary
block and confirmation anchor.  No mutable explorer chart, fee, gas, transfer,
validator identity, ETH return, BTC outcome, funding value, Gross9 row, prior
event, or promoted control selects the rule.

The decision waits until `00:20 UTC`, entry is the exact `00:25 UTC` BTCUSDT
five-minute open, and hold is 24 elapsed hours at 0.5 gross exposure.  The
absolute missed-slot-count change requires strict-prior rank `>=0.70`; completed
BTC variation requires strict-prior rank `>=0.65`.  The first genuine source,
support, novelty, or economic failure is terminal without repair.
