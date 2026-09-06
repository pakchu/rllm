# Approved G9 + macro1 + dollar short0.5: execution preparation

## Scope

Portfolio choice is approved in `configs/approved/g9_macro1_dollar_short05_2026-09-07.json`. The approved original G9 ratios remain, macro coefficient1.0 and dollar short0.5 are added; failed-rebound short is excluded.

**Offline/paper preparation is available. Actual broker execution is not enabled or connected.** This is not a claim that the existing live service has switched strategies.

## Implemented

- Pure macro adapter: completed-hour features, 75/25 frozen components, daily UTC refresh and hourly target maintenance at HH:05; explicit closed-bar cutoff and DXY availability. Genuine source inputs only, no fabricated availability flags.
- Pure dollar-short adapter: original144-window feature contract and hourly signal at HH:55, next-bar entry HH:00,12h hold with no separate TP/SL.
- Existing G9 pure scorers are reused; no existing order runner is invoked.
- A separate net planner maintains signed virtual sleeve quantities and produces aggregate one-way deltas. Opposing sleeve intents offset before fee estimates; post-fee4.5x net cap and lot-step rounding are applied. Carried sleeve quantities are not routinely resized by other sleeves (risk override remains explicit).
- One-way mode, fresh account snapshot, account/virtual-state equality and approved weights/evidence are mandatory. Hedge-mode or ambiguous attribution fails closed.
- Plan creation never mutates state. Only ideal complete **paper** fills advance paper state. Plan hashes, state revisions and complete per-sleeve signal-ID histories prevent duplicate/replayed commits.
- Time/observed-price barrier exit planning is supported. It does not assume a late order fills at an earlier historical stop/TP price.
- `--live` is unconditionally rejected. There are no credentials, exchange clients or order-submission methods in the new runner/planner.

## Executable offline command

Run from repository root with a genuine enriched5m CSV and an explicit **paper** snapshot:

```bash
python -m execution.approved_portfolio_dry_run \
  --market /path/to/enriched_5m.csv.gz \
  --snapshot /path/to/paper_snapshot.json \
  --output /path/to/new_plan.json \
  --paper-fill-state-output /path/to/new_paper_state.json
```

For the next paper step, pass the previous paper state with `--state`. Every output path must be new; existing/live state files are never overwritten. The snapshot supplies `symbol`, `position_mode`, `asof`, `equity`, `mark_price`, and `net_units`. Equity includes the paper/account valuation at that time; this planner is not a performance backtester. The example lot step is not an exchange-certified filter; real futures filters and minimum notionals require broker integration.

Recorded network-disabled snapshot smokes are under `research/approved_runtime_preparation/verified`. Snapshot equities are fictional10000 units, not actual account balances. Market CSV fixtures are locally generated from existing research snapshots, hash-recorded and intentionally ignored by git; staging copies are test data only.

## Remaining production requirements

1. Authoritative broker fill/cancel/partial-fill attribution and restart recovery for the net ledger.
2. An explicit migration/reconciliation plan from the current hedge-mode/per-sleeve executor; do not infer ownership or flatten positions automatically.
3. Fresh market/account feeds, validated futures lot/minimum-notional filters and end-to-end latency/clock parity.
4. Production barrier monitoring even when signal scoring fails, plus operational fault/soak validation.
5. Deployment/account/operator controls. No service switch or actual order submission is performed by this preparation.

## Server changes

`ubuntu-server` resolves to the observed host `pakchu-server`; active checkout is `/home/master/rllm`. Six original modified files were preserved in commit `4219204e` with identical working-file bytes and hooks disabled. A backup branch records that snapshot.

The active mainnet process and active files were not restarted/replaced. The isolated `/home/master/rllm-prepared-20260907` branch was rebased onto origin/main. Separate fixes (`619677ff`) require real fill-report evidence before complete strategy attribution and fail closed on failed live-style lot-rule lookups. These preparation-branch fixes do not alter the running checkout.

Only final schema-v3 paper states are accepted; older incomplete replay-history schemas are rejected. Selection approval remains distinct from execution readiness.
