# ubuntu-server: preserved changes and isolated preparation

## Completed

- Inspected actual host alias `ubuntu-server` (`pakchu-server`), checkout `/home/master/rllm`.
- Preserved six modified files in `4219204e`; active file bytes were unchanged and git hooks were disabled. Backup branch: `backup/server-uncommitted-20260907`.
- Reviewed existing changes: strategy/fill fee attribution, partial-fill and restart recovery, cancellation uncertainty, lot sizing, HTF resample origin, and audit/test updates.
- Fixed two additional hazards only in isolated preparation: missing fill evidence cannot produce complete strategy PnL, and failed live-style lot-rule lookup cannot silently fall back. Commit `619677ff`.
- Created `/home/master/rllm-prepared-20260907`, branch `codex/server-prepared-20260907`. Rebase-pulled origin/main there, then merged the approved strategy preparation branch. Validated merge: `9c76b408`.
- Merged server source:114 tests plus7 subtests passed on both server and local review, with network disabled. Three server paper smokes passed and all expected signal adapters were ready.
- Sparse checkout excludes only prepared `data/` and `checkpoints/`; prepared size is425MiB. Active data was not removed. Server still has about5.2GiB free /95% usage.

## Not performed

**No live strategy switch, service restart, exchange order, account-mode change, position migration, or production database migration.** The old mainnet process was still present and the active checkout still matched `4219204e` byte-for-byte.

The new executable is an offline/paper planning surface. `--live` is rejected. Approval of the portfolio composition does not solve actual fill attribution, hedge-mode migration, live source/lot-filter validation or operational barrier monitoring; production readiness remains false.

Only the isolated worktree was updated/rebased. Rewriting the running checkout would not be equivalent to harmless preparation because the running strategy may import or reread files.

Evidence: `research/server_preparation/report_2026-09-07.json`. Paper runner and usage: `docs/approved-portfolio-execution-preparation-2026-09-07.md`.
