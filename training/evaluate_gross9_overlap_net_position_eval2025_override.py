"""Open eval2025 under the explicit post-terminal diagnostic override."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from training import build_gross9_overlap_net_position_config as net_config
from training import evaluate_gross9_async_active_veto_train_economics as train_sources
from training import evaluate_gross9_overlap_net_position_portfolio as original_evaluator
from training import evaluate_gross9_qtr_distill_economics as fixed_ledger
from training import optimize_gross9_overlap_portfolio as optimizer
from training import preregister_gross9_overlap_net_position_eval2025_override as freeze_builder

POLICY_ID = freeze_builder.POLICY_ID
PROTOCOL_VERSION = "gross9_overlap_net_position_eval2025_stop_override_economics_v2"
FREEZE = freeze_builder.DEFAULT_OUTPUT
DEFAULT_OUTPUT = Path(
    "results/gross9_overlap_net_position_eval2025_diagnostic_v2_2026-09-04.json"
)


def canonical_hash(value: Any) -> str:
    return freeze_builder.canonical_hash(value)


def sha256_file(path: str | Path) -> str:
    return freeze_builder.sha256_file(path)


def load_freeze(path: Path = FREEZE) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{POLICY_ID} freeze must be a JSON object")
    freeze_builder.validate(value)
    if value != freeze_builder.build():
        raise RuntimeError(f"{POLICY_ID} freeze no longer matches bound inputs/code")
    return value


def public_source_receipt(source: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(source)
    identity = value.get("database_identity")
    if isinstance(identity, Mapping):
        value["database_identity"] = {
            "identity_sha256": canonical_hash(identity),
            "server_version_num": str(identity.get("server_version_num")),
            "transaction_snapshot": str(identity.get("transaction_snapshot")),
            "network_and_database_names_redacted": True,
        }
    if value.get("database_environment_source") is not None:
        value["database_environment_source"] = "redacted_local_env_file"
    return value


def normalize_funding_clock(raw_funding: pd.DataFrame) -> pd.DataFrame:
    funding = raw_funding.copy()
    funding["date"] = pd.to_datetime(funding["date"], utc=True, errors="raise").dt.floor(
        "5min"
    )
    if funding["date"].duplicated().any():
        raise RuntimeError(f"{POLICY_ID} duplicate funding buckets after normalization")
    return funding.sort_values("date").reset_index(drop=True)


def load_normalized_eval2025_sources(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    env_source = original_evaluator._ensure_db_environment()
    market = train_sources.load_market_hash_bound(start, end)
    _, raw_funding, database_receipt = original_evaluator.load_postgres_extracts(
        start,
        end,
        include_market=False,
    )
    raw_receipt = original_evaluator._frame_receipt(raw_funding)
    funding = normalize_funding_clock(raw_funding)
    fixed_ledger.validate_market(market, start, end)
    fixed_ledger.validate_funding(funding, start, end)
    source = {
        **database_receipt,
        "database_environment_source": env_source,
        "market_source": {
            "mode": "hash_bound_gzip_physical_prefix",
            "path": str(train_sources.econ.v1.MARKET),
            "sha256": train_sources.econ.v1.MARKET_SHA,
        },
        "market_extract": original_evaluator._frame_receipt(market),
        "funding_raw_extract": raw_receipt,
        "funding_time_normalization": {
            "operation": "floor_to_5min",
            "rows": int(len(funding)),
            "rate_and_mark_price_changed": False,
            "duplicate_normalized_buckets": 0,
        },
        "funding_extract": original_evaluator._frame_receipt(funding),
    }
    return market, funding, source


def run(
    output: str | Path = DEFAULT_OUTPUT,
    *,
    freeze_path: Path = FREEZE,
) -> dict[str, Any]:
    frozen = load_freeze(freeze_path)
    original_freeze = original_evaluator.load_validation_freeze(
        freeze_builder.ORIGINAL_FREEZE
    )
    start = original_evaluator._utc("2025-01-01T00:00:00Z")
    end = original_evaluator._utc("2026-01-01T00:00:00Z")
    clock, clock_receipts = original_evaluator.load_portfolio_clock(
        original_freeze,
        "eval2025",
        start,
        end,
    )
    market, funding, source = load_normalized_eval2025_sources(
        start,
        end,
    )
    primary = fixed_ledger.evaluate_primary(clock, market, funding, start, end)
    base_raw = fixed_ledger.simulate_portfolio(
        clock,
        market,
        funding,
        start,
        end,
        fixed_ledger.BASE_COST,
    )
    legacy_risk = optimizer.exposure_and_turnover(clock, start, end)
    net_risk = net_config.net_exposure_metrics(clock, start, end)
    episodes = original_evaluator.aggregate_net_signed_episode_count(clock)
    checks = original_evaluator.oos_checks(
        "eval2025",
        primary,
        net_risk,
        episodes,
        original_freeze["gates"]["oos"],
    )
    diagnostic_passed = all(checks.values())
    zero_cost = fixed_ledger.public_metric(
        fixed_ledger.simulate_portfolio(
            clock,
            market,
            funding,
            start,
            end,
            0.0,
        )
    )
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "stage": "eval2025_diagnostic",
        "window": [original_evaluator._iso_z(start), original_evaluator._iso_z(end)],
        "override_freeze": {
            "path": str(freeze_path),
            "sha256": sha256_file(freeze_path),
            "manifest_hash": frozen["manifest_hash"],
        },
        "original_chain": frozen["original_chain"],
        "fixed_portfolio": {
            "sleeve_weights": frozen["fixed_portfolio"]["sleeve_weights"],
            "clock_receipts": clock_receipts,
            "weights_changed": False,
            "reranked_repaired_or_substituted": False,
        },
        "source": public_source_receipt(source),
        "physical_rows_opened": {
            "market": int(len(market)),
            "funding": int(len(funding)),
            "portfolio_clock": int(len(clock)),
        },
        "primary": primary,
        "zero_cost_diagnostic": zero_cost,
        "source_shape": {
            "intervals": int(primary["base"]["intervals"]),
            "long_intervals": int(primary["base"]["long_intervals"]),
            "short_intervals": int(primary["base"]["short_intervals"]),
            "aggregate_net_signed_episodes": int(episodes),
            "active_iso_weeks": original_evaluator.active_iso_weeks(clock),
        },
        "net_position_risk": net_risk,
        "legacy_nonnet_risk_disclosure": legacy_risk,
        "operating_cost_disclosure": original_evaluator.operating_cost_disclosure(
            clock,
            base_raw,
            primary,
            legacy_risk,
            start,
            end,
        ),
        "monthly_stability": optimizer.evaluate_monthly_stability(
            clock,
            market,
            funding,
            start,
            end,
        ),
        "reported_performance_checks": checks,
        "diagnostic_performance_checks_passed": diagnostic_passed,
        "turnover_frequency_or_cost_gate_applied": False,
        "original_test2024_relabelled_as_pass": False,
        "original_protocol_advance": False,
        "advance_beyond_eval2025": False,
        "final2026_outcomes_opened": False,
        "live_capital_authorized": False,
        "order_submission_enabled": False,
        "status": (
            "post_terminal_eval2025_diagnostic_pass"
            if diagnostic_passed
            else "post_terminal_eval2025_diagnostic_fail"
        ),
        "decision": "diagnostic_only_no_further_advance",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=FREEZE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    if args.verify_only:
        frozen = load_freeze(args.freeze)
        print(
            json.dumps(
                {
                    "policy_id": POLICY_ID,
                    "freeze_manifest_hash": frozen["manifest_hash"],
                    "verified": True,
                    "eval2025_outcomes_opened": False,
                }
            )
        )
        return 0
    result = run(args.output, freeze_path=args.freeze)
    print(
        json.dumps(
            {
                "stage": result["stage"],
                "diagnostic_performance_checks_passed": result[
                    "diagnostic_performance_checks_passed"
                ],
                "output": str(args.output),
                "failed_checks": [
                    name
                    for name, passed in result["reported_performance_checks"].items()
                    if not passed
                ],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
