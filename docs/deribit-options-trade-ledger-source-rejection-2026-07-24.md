# DOTL-v1 source rejection — 2026-07-24

## Decision

**Retire DOTL-v1 — Deribit BTC Options Trade Ledger unchanged.**

The one-shot verifier opened the prospectively fixed historical source request
and stopped on the first canonical-schema violation:

```text
mark_price must be positive
```

No alternate interval, field relaxation, row deletion, null substitution,
zero-to-epsilon mapping, historical endpoint change, or second run is
authorized.

## Frozen execution identity

Source boundary:

```text
docs/deribit-options-trade-ledger-source-boundary-2026-07-24.md
SHA256 82532408ae44dac0cdc907e181fcd130879f6ec67479e443f471a2d984ca72ee
```

Verifier commit:

```text
d7599b681ead2de89089d9acc3e0656e5ed0d072
```

The verifier had passed:

- independent boundary review with zero remaining P0/P1 findings;
- independent implementation review with zero remaining P0/P1 findings;
- Python compilation; and
- 37 targeted tests after the disk-usage regression fix.

The real command was:

```text
.venv/bin/python -m training.verify_deribit_options_trade_ledger
```

It ran from a HEAD-clean worktree and reserved its immutable one-shot sentinel
before the first source response.

## What failed

The boundary classified present `mark_price` as an auxiliary execution-time
decimal that nevertheless had to be finite and strictly positive. The frozen
2021-01-04 historical-day request returned a row for which that condition was
false. The verifier terminated while parsing the historical page and did not
complete a page, compute source incidence, or proceed to the live surfaces.

Although the first authorized successor was forbidden from using auxiliary
fields, changing `mark_price` from positive to nonnegative after observing this
failure would still be a post-result schema repair. DOTL-v1 therefore remains
rejected rather than being redefined as a hard-fields-only pass.

## Unopened evidence

The rejection occurred before:

- completion or replay of the fixed historical day;
- the fixed 2026-07-24 13:00–13:20 UTC WebSocket capture;
- recent REST parity;
- any cross-horizon option-flow state or candidate incidence;
- DEWH/DEHR/OPDR/DVOL comparator clocks;
- BTC execution prices or returns;
- funding, PnL, reward, model, CAGR, or strict MDD; and
- any 2024-or-later source.

Every outcome flag in the terminal report is false. No historical raw response
was persisted.

## Immutable artifacts

One-shot sentinel:

```text
results/.deribit_options_trade_ledger_source_parity_2026-07-24.started
SHA256 6f2b354de703cc73c281e42531c2efd1dbc4081154c60d26ae0e319ff7e57dd4
```

Terminal report:

```text
results/deribit_options_trade_ledger_source_parity_2026-07-24.json
SHA256 cda7563480d421f9056c257884c5a493ee4993b357df179143f5350d0dfd7c10
manifest_hash_without_self
4966f8fe1b16f27777900e4947d959fa6811b7a15d05be5780176dee2a3b8e6b
```

## No-repair boundary

The following are explicitly forbidden:

- allowing zero only because this run exposed it;
- dropping or ignoring the offending field under the DOTL-v1 identity;
- selecting a different old day or shorter interval;
- querying another Deribit endpoint for the same intended ledger;
- replaying after code, dependency, clock, or transport changes;
- renaming a hard-fields-only extraction as a new DOTL candidate; or
- using the failure row, page position, or source error as an economic feature.

The next alpha search must leave the exact Deribit option-trade-ledger source
identity and return to a prospectively frozen, outcome-unseen source axis.
