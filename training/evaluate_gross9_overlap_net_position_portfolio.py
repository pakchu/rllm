"""Sequential holdout/OOS evaluator for G9-OVERLAP-NET-PORT-1.

The fixed train rank-one portfolio is evaluated without reranking, repair, or
substitution.  A stage may open data only after the frozen validation artifact
and every predecessor stage authorize it.  Turnover and costs are disclosed
but are not rejection gates.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from preprocessing.live_db_features import load_env_file, postgres_url_from_env
from training import build_gross9_overlap_net_position_config as net_config
from training import evaluate_gross9_async_active_veto_train_economics as train_sources
from training import evaluate_gross9_qtr_distill_economics as fixed_ledger
from training import optimize_gross9_overlap_portfolio as optimizer
from training import preregister_gross9_overlap_net_position_validation as validation

POLICY_ID = validation.POLICY_ID
PROTOCOL_VERSION = "gross9_overlap_net_position_sequential_economics_v1"
FREEZE = validation.DEFAULT_OUTPUT
STAGES = validation.STAGES
PREDECESSOR = {
    "test2024": "holdout_dec2023",
    "eval2025": "test2024",
    "final2026": "eval2025",
}
OUTPUTS = {
    stage: Path(f"results/gross9_overlap_net_position_{stage}_2026-09-03.json")
    for stage in STAGES
}
DB_ENV_KEYS = ("PG_USER", "PG_PASSWORD", "PG_HOST", "PG_PORT", "PG_DB_NAME")
MARKET_SQL = """
SELECT
    date_bin('5 minutes', ts, TIMESTAMPTZ '1970-01-01 00:00:00+00') AS date,
    (array_agg(open ORDER BY ts))[1] AS open,
    max(high) AS high,
    min(low) AS low,
    (array_agg(close ORDER BY ts DESC))[1] AS close,
    count(*) AS source_rows
FROM bars_binance
WHERE interval = '1m'
  AND symbol = :symbol
  AND ts >= :start
  AND ts < :query_end
GROUP BY 1
ORDER BY 1
""".strip()
FUNDING_SQL = """
SELECT funding_time AS date, funding_rate, mark_price
FROM funding_rates_binance
WHERE symbol = :symbol
  AND funding_time >= :start
  AND funding_time < :end
ORDER BY funding_time
""".strip()
DATABASE_IDENTITY_SQL = """
SELECT
    current_database() AS database_name,
    inet_server_addr()::text AS server_address,
    inet_server_port() AS server_port,
    current_setting('server_version_num') AS server_version_num,
    txid_current_snapshot()::text AS transaction_snapshot
""".strip()


def canonical_hash(value: Any) -> str:
    return validation.canonical_hash(value)


def sha256_file(path: str | Path) -> str:
    return validation.sha256_file(path)


def _utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _iso_z(value: Any) -> str:
    return _utc(value).isoformat().replace("+00:00", "Z")


def load_validation_freeze(path: Path = FREEZE) -> dict[str, Any]:
    value = validation.load_hashed_json(path)
    validation.validate(value)
    if value != validation.build():
        raise RuntimeError(f"{POLICY_ID} validation freeze no longer matches frozen inputs/code")
    return value


def validation_freeze_receipt(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "manifest_hash": value["manifest_hash"],
    }


def verify_predecessor(
    stage: str,
    expected_freeze_receipt: Mapping[str, Any],
    expected_implementation: Mapping[str, Any],
    outputs: Mapping[str, Path] = OUTPUTS,
) -> dict[str, Any] | None:
    predecessor_stage = PREDECESSOR.get(stage)
    if predecessor_stage is None:
        return None

    def verify_stage_chain(expected_stage: str) -> tuple[Path, dict[str, Any]]:
        path = Path(outputs[expected_stage])
        if not path.is_file():
            raise RuntimeError(f"{POLICY_ID} missing predecessor {expected_stage}: {path}")
        value = validation.load_hashed_json(path)
        if (
            value.get("policy_id") != POLICY_ID
            or value.get("stage") != expected_stage
            or value.get("passed") is not True
            or value.get("advance_to_next_stage") is not True
            or value.get("validation_freeze") != expected_freeze_receipt
            or value.get("implementation") != expected_implementation
        ):
            raise RuntimeError(f"{POLICY_ID} predecessor did not authorize {stage}")
        parent_stage = PREDECESSOR.get(expected_stage)
        if parent_stage is None:
            if value.get("predecessor") is not None:
                raise RuntimeError(f"{POLICY_ID} predecessor chain drift at {expected_stage}")
        else:
            parent_path, parent = verify_stage_chain(parent_stage)
            expected_parent_receipt = {
                "stage": parent_stage,
                "path": str(parent_path),
                "sha256": sha256_file(parent_path),
                "manifest_hash": parent["manifest_hash"],
            }
            if value.get("predecessor") != expected_parent_receipt:
                raise RuntimeError(f"{POLICY_ID} predecessor chain drift at {expected_stage}")
        return path, value

    path, value = verify_stage_chain(predecessor_stage)
    return {
        "stage": predecessor_stage,
        "path": str(path),
        "sha256": sha256_file(path),
        "manifest_hash": value["manifest_hash"],
    }


def _ensure_db_environment() -> str:
    if all(os.environ.get(key) for key in DB_ENV_KEYS):
        return "process_environment"
    candidates = []
    configured = os.environ.get("RLLM_DB_ENV")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(
        [
            Path(".env"),
            Path("/home/pakchu/rllm/.env"),
            Path("/home/pakchu/upbit-usdt/.env"),
        ]
    )
    for path in candidates:
        if not path.is_file():
            continue
        load_env_file(path)
        if all(os.environ.get(key) for key in DB_ENV_KEYS):
            return str(path)
    missing = [key for key in DB_ENV_KEYS if not os.environ.get(key)]
    raise RuntimeError(f"{POLICY_ID} missing PostgreSQL environment keys: {missing}")


def _database_identity() -> dict[str, str]:
    return {
        "configured_host": os.environ["PG_HOST"],
        "configured_port": os.environ["PG_PORT"],
        "configured_database": os.environ["PG_DB_NAME"],
    }


def load_postgres_extracts(
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    include_market: bool,
) -> tuple[pd.DataFrame | None, pd.DataFrame, dict[str, Any]]:
    from sqlalchemy import create_engine, text

    engine = create_engine(
        postgres_url_from_env(),
        connect_args={"connect_timeout": 10},
        isolation_level="REPEATABLE READ",
    )
    extracted_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    with engine.connect() as connection, connection.begin():
        identity_row = connection.execute(text(DATABASE_IDENTITY_SQL)).mappings().one()
        market: pd.DataFrame | None = None
        if include_market:
            market = pd.read_sql_query(
                text(MARKET_SQL),
                connection,
                params={
                    "symbol": "BTCUSDT",
                    "start": start.to_pydatetime(),
                    "query_end": (end + pd.Timedelta(minutes=5)).to_pydatetime(),
                },
            )
        funding = pd.read_sql_query(
            text(FUNDING_SQL),
            connection,
            params={
                "symbol": "BTCUSDT",
                "start": start.to_pydatetime(),
                "end": end.to_pydatetime(),
            },
        )
    engine.dispose()
    if market is not None:
        market["date"] = pd.to_datetime(market["date"], utc=True, errors="raise")
        if not market["source_rows"].eq(5).all():
            raise RuntimeError(f"{POLICY_ID} PostgreSQL 5m market aggregation is incomplete")
        market = market[["date", "open", "high", "low", "close"]]
    funding["date"] = pd.to_datetime(funding["date"], utc=True, errors="raise")
    funding = funding[["date", "funding_rate", "mark_price"]]
    identity = {key: str(value) for key, value in identity_row.items()}
    receipt = {
        "mode": "postgres_repeatable_read_exact_extract",
        "symbol": "BTCUSDT",
        "tables": ["bars_binance", "funding_rates_binance"]
        if include_market
        else ["funding_rates_binance"],
        "database_identity": {**_database_identity(), **identity},
        "isolation_level": "REPEATABLE READ",
        "extracted_at_utc": extracted_at,
        "query_contract": {
            "market_sql_sha256": hashlib.sha256(MARKET_SQL.encode()).hexdigest()
            if include_market
            else None,
            "funding_sql_sha256": hashlib.sha256(FUNDING_SQL.encode()).hexdigest(),
            "database_identity_sql_sha256": hashlib.sha256(
                DATABASE_IDENTITY_SQL.encode()
            ).hexdigest(),
            "parameters": {
                "symbol": "BTCUSDT",
                "start": _iso_z(start),
                "end": _iso_z(end),
                "market_query_end": _iso_z(end + pd.Timedelta(minutes=5))
                if include_market
                else None,
            },
        },
    }
    return market, funding, receipt


def dataframe_hash(frame: pd.DataFrame) -> str:
    normalized = frame.copy()
    for column in normalized.columns:
        if pd.api.types.is_datetime64_any_dtype(normalized[column]):
            normalized[column] = pd.to_datetime(normalized[column], utc=True).astype("int64")
    row_hashes = pd.util.hash_pandas_object(normalized, index=False).to_numpy(dtype="uint64")
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()


def _frame_receipt(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(frame)),
        "first_time": _iso_z(frame["date"].iloc[0]),
        "last_time": _iso_z(frame["date"].iloc[-1]),
        "content_sha256": dataframe_hash(frame),
        "hash_method": "pandas_hash_pandas_object_uint64_sha256",
    }


def load_stage_sources(
    stage: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if stage == "holdout_dec2023":
        market = train_sources.load_market_hash_bound(start, end)
        funding = train_sources.load_train_funding_hash_bound(start, end)
        source: dict[str, Any] = {
            "mode": "hash_bound_gzip_physical_prefix",
            "market": {
                "path": str(train_sources.econ.v1.MARKET),
                "sha256": train_sources.econ.v1.MARKET_SHA,
            },
            "funding": {
                "path": str(train_sources.econ.TRAIN_FUNDING),
                "sha256": train_sources.econ.TRAIN_FUNDING_SHA,
            },
        }
    else:
        env_source = _ensure_db_environment()
        postgres_market, funding, database_receipt = load_postgres_extracts(
            start,
            end,
            include_market=stage == "final2026",
        )
        if postgres_market is None:
            market = train_sources.load_market_hash_bound(start, end)
            market_receipt = {
                "mode": "hash_bound_gzip_physical_prefix",
                "path": str(train_sources.econ.v1.MARKET),
                "sha256": train_sources.econ.v1.MARKET_SHA,
            }
        else:
            market = postgres_market
            market_receipt = {"mode": "postgres_exact_1m_to_5m"}
        source = {
            **database_receipt,
            "database_environment_source": env_source,
            "market_source": market_receipt,
        }
    fixed_ledger.validate_market(market, start, end)
    fixed_ledger.validate_funding(funding, start, end)
    source.update(
        {
            "market_extract": _frame_receipt(market),
            "funding_extract": _frame_receipt(funding),
        }
    )
    return market, funding, source


def load_portfolio_clock(
    freeze: Mapping[str, Any],
    stage: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    split = STAGES[stage][0]
    rows: list[pd.DataFrame] = []
    receipts: list[dict[str, Any]] = []
    for record in freeze["frozen_inputs"]["selected_clocks"]:
        path = Path(record["path"])
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"{POLICY_ID} selected clock hash drift: {path}")
        required = {"split", "entry_time", "exit_time", "side"}
        stage_rows = []
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not required.issubset(reader.fieldnames or []):
                raise RuntimeError(f"{POLICY_ID} selected clock schema drift: {path}")
            for row in reader:
                if row["split"] != split:
                    continue
                entry_time = _utc(row["entry_time"])
                exit_time = _utc(row["exit_time"])
                if entry_time >= start and exit_time <= end:
                    stage_rows.append(
                        {
                            "entry_time": entry_time,
                            "exit_time": exit_time,
                            "side": row["side"],
                        }
                    )
        sleeve_clock = pd.DataFrame(
            stage_rows,
            columns=["entry_time", "exit_time", "side"],
        )
        sleeve_clock["entry_time"] = pd.to_datetime(
            sleeve_clock["entry_time"], utc=True, errors="raise"
        )
        sleeve_clock["exit_time"] = pd.to_datetime(
            sleeve_clock["exit_time"], utc=True, errors="raise"
        )
        sleeve_clock["side"] = pd.to_numeric(
            sleeve_clock["side"], errors="raise"
        ).astype(int)
        sleeve_clock = sleeve_clock.sort_values(["entry_time", "exit_time"])
        if not sleeve_clock["side"].isin([-1, 1]).all():
            raise RuntimeError(f"{POLICY_ID} selected clock side drift: {path}")
        if not (sleeve_clock["entry_time"] < sleeve_clock["exit_time"]).all():
            raise RuntimeError(f"{POLICY_ID} non-positive selected clock interval: {path}")
        if len(sleeve_clock) > 1 and (
            sleeve_clock["entry_time"].iloc[1:].to_numpy()
            < sleeve_clock["exit_time"].iloc[:-1].to_numpy()
        ).any():
            raise RuntimeError(f"{POLICY_ID} intra-sleeve overlap: {path}")
        sleeve_clock = sleeve_clock.reset_index(drop=True)
        if not sleeve_clock.empty:
            sleeve_clock = sleeve_clock.copy()
            sleeve_clock["sleeve"] = record["sleeve_id"]
            sleeve_clock["weight"] = float(record["weight"])
            rows.append(sleeve_clock[["sleeve", "weight", "entry_time", "exit_time", "side"]])
        receipts.append(
            {
                "sleeve_id": record["sleeve_id"],
                "weight": float(record["weight"]),
                "path": str(path),
                "sha256": record["sha256"],
                "full_clock_rows": int(record["rows"]),
                "stage_rows": int(len(sleeve_clock)),
            }
        )
    if not rows:
        raise RuntimeError(f"{POLICY_ID} empty portfolio clock for {stage}")
    clock = fixed_ledger.normalize_portfolio_clock(
        pd.concat(rows, ignore_index=True),
        require_four_sleeves=False,
    )
    return clock, receipts


def aggregate_net_signed_episode_count(clock: pd.DataFrame) -> int:
    events: dict[pd.Timestamp, list[float]] = {}
    for row in clock.itertuples(index=False):
        signed = float(row.weight) * int(row.side)
        events.setdefault(_utc(row.entry_time), []).append(signed)
        events.setdefault(_utc(row.exit_time), []).append(-signed)
    net = 0.0
    previous_sign = 0
    episodes = 0
    for timestamp in sorted(events):
        net += sum(events[timestamp])
        if abs(net) < 1e-12:
            net = 0.0
        sign = 1 if net > 0 else -1 if net < 0 else 0
        if sign != 0 and sign != previous_sign:
            episodes += 1
        previous_sign = sign
    return episodes


def active_iso_weeks(clock: pd.DataFrame) -> int:
    iso = clock["entry_time"].dt.isocalendar()
    return len(set(zip(iso["year"].astype(int), iso["week"].astype(int), strict=True)))


def operating_cost_disclosure(
    clock: pd.DataFrame,
    base_raw: Mapping[str, Any],
    primary: Mapping[str, Any],
    risk: Mapping[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    actual_net_turnover = sum(
        abs(float(row["delta_q"]))
        * float(row["open"])
        / max(float(row["equity_pre"]), 1e-12)
        for row in base_raw["transition_rows"]
        if abs(float(row["delta_q"])) > 1e-12
    )
    nonzero_events = sum(
        abs(float(row["delta_q"])) > 1e-12 for row in base_raw["transition_rows"]
    )
    days = max((end - start).total_seconds() / 86_400.0, 1e-12)
    pre_net_turnover = float(risk["turnover_weight"])
    per_sleeve = []
    for sleeve, group in clock.groupby("sleeve", sort=True):
        turnover = float(2.0 * group["weight"].abs().sum())
        per_sleeve.append(
            {
                "sleeve_id": str(sleeve),
                "intervals": int(len(group)),
                "pre_net_turnover_weight": turnover,
                "pre_net_turnover_share": turnover / pre_net_turnover if pre_net_turnover else 0.0,
            }
        )
    base = primary["base"]
    stress = primary["stress"]
    return {
        "classification": "high_cost_disclosure_not_rejection",
        "current_rejection_gates": [],
        "measurement_window_days": days,
        "pre_net_turnover_weight": pre_net_turnover,
        "actual_aggregate_net_turnover_weight": float(actual_net_turnover),
        "actual_aggregate_net_turnover_weight_per_day": float(actual_net_turnover / days),
        "annualized_aggregate_net_turnover_x": float(actual_net_turnover / days * 365.25),
        "netting_savings_share": (
            float(1.0 - actual_net_turnover / pre_net_turnover)
            if pre_net_turnover
            else 0.0
        ),
        "nonzero_net_execution_events": int(nonzero_events),
        "nonzero_net_execution_events_per_day": float(nonzero_events / days),
        "base_fee_cash": float(base["total_fees"]),
        "base_funding_cash": float(base["total_funding"]),
        "base_fee_less_funding_cash": float(base["total_fees"] - base["total_funding"]),
        "stress_fee_cash": float(stress["total_fees"]),
        "stress_funding_cash": float(stress["total_funding"]),
        "stress_fee_less_funding_cash": float(
            stress["total_fees"] - stress["total_funding"]
        ),
        "per_sleeve": per_sleeve,
    }


def holdout_checks(
    primary: Mapping[str, Any],
    risk: Mapping[str, float],
    intervals: int,
    weeks: int,
    gates: Mapping[str, Any],
) -> dict[str, bool]:
    base = primary["base"]
    stress = primary["stress"]
    return {
        "minimum_intervals": intervals >= int(gates["minimum_intervals"]),
        "minimum_active_iso_weeks": weeks >= int(gates["minimum_active_iso_weeks"]),
        "absolute_return_positive": float(base["absolute_return_pct"]) > 0.0,
        "stress_absolute_return_positive": float(stress["absolute_return_pct"]) > 0.0,
        "strict_mdd_max": float(base["strict_mdd_pct"]) <= float(gates["strict_mdd_max_pct"]),
        "mean_abs_net_position_cap": float(risk["mean_abs_net_position"])
        <= float(gates["mean_abs_net_position_max"]),
        "max_abs_net_position_cap": float(risk["max_abs_net_position"])
        <= float(gates["max_abs_net_position_max"]),
    }


def oos_checks(
    stage: str,
    primary: Mapping[str, Any],
    risk: Mapping[str, float],
    signed_episodes: int,
    gates: Mapping[str, Any],
) -> dict[str, bool]:
    base = primary["base"]
    stress = primary["stress"]
    minimum_episodes = int(gates["aggregate_net_signed_episode_min"][stage])
    return {
        "absolute_return_positive": float(base["absolute_return_pct"]) > 0.0,
        "cagr_to_strict_mdd_min": float(base["cagr_to_strict_mdd"])
        >= float(gates["cagr_to_strict_mdd_min"]),
        "strict_mdd_max": float(base["strict_mdd_pct"]) <= float(gates["strict_mdd_max_pct"]),
        "mean_exposure_weighted_gross_edge_min": float(
            base["mean_exposure_weighted_gross_edge_bp"]
        )
        >= float(gates["mean_exposure_weighted_gross_edge_min_bp"]),
        "stress_absolute_return_positive": float(stress["absolute_return_pct"]) > 0.0,
        "stress_cagr_to_strict_mdd_min": float(stress["cagr_to_strict_mdd"])
        >= float(gates["stress_cagr_to_strict_mdd_min"]),
        "each_calendar_half_positive": all(
            float(row["absolute_return_pct"]) > 0.0
            for row in primary["calendar_halves"].values()
        ),
        "weekly_cluster_signflip_one_sided_p_max": float(
            primary["cluster_signflip"]["pvalue"]
        )
        <= float(gates["weekly_cluster_signflip_one_sided_p_max"]),
        "aggregate_net_signed_episode_min": signed_episodes >= minimum_episodes,
        "mean_abs_net_position_cap": float(risk["mean_abs_net_position"])
        <= float(gates["mean_abs_net_position_max"]),
        "max_abs_net_position_cap": float(risk["max_abs_net_position"])
        <= float(gates["max_abs_net_position_max"]),
    }


def run(
    stage: str,
    output: str | Path | None = None,
    *,
    freeze_path: Path = FREEZE,
    outputs: Mapping[str, Path] = OUTPUTS,
) -> dict[str, Any]:
    if stage not in STAGES:
        raise RuntimeError(f"{POLICY_ID} unknown stage: {stage}")
    freeze = load_validation_freeze(freeze_path)
    freeze_receipt = validation_freeze_receipt(freeze_path, freeze)
    predecessor = verify_predecessor(
        stage,
        freeze_receipt,
        freeze["implementation"],
        outputs,
    )
    split, start_value, end_value = STAGES[stage]
    start = _utc(start_value)
    end = _utc(end_value)
    clock, clock_receipts = load_portfolio_clock(freeze, stage, start, end)
    market, funding, source = load_stage_sources(stage, start, end)

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
    episodes = aggregate_net_signed_episode_count(clock)
    weeks = active_iso_weeks(clock)
    if stage == "holdout_dec2023":
        checks = holdout_checks(
            primary,
            net_risk,
            int(primary["base"]["intervals"]),
            weeks,
            freeze["gates"]["holdout_dec2023"],
        )
    else:
        checks = oos_checks(
            stage,
            primary,
            net_risk,
            episodes,
            freeze["gates"]["oos"],
        )
    passed = all(checks.values())
    final_stage = stage == list(STAGES)[-1]
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "stage": stage,
        "split": split,
        "window": [_iso_z(start), _iso_z(end)],
        "validation_freeze": freeze_receipt,
        "implementation": freeze["implementation"],
        "predecessor": predecessor,
        "fixed_portfolio": {
            "sleeve_weights": {
                row["sleeve_id"]: row["weight"]
                for row in freeze["frozen_inputs"]["selected_clocks"]
            },
            "clock_receipts": clock_receipts,
            "reranked_repaired_or_substituted": False,
        },
        "source": source,
        "physical_rows_opened": {
            "market": int(len(market)),
            "funding": int(len(funding)),
            "portfolio_clock": int(len(clock)),
        },
        "primary": primary,
        "source_shape": {
            "intervals": int(primary["base"]["intervals"]),
            "long_intervals": int(primary["base"]["long_intervals"]),
            "short_intervals": int(primary["base"]["short_intervals"]),
            "aggregate_net_signed_episodes": int(episodes),
            "aggregate_net_signed_episode_definition": (
                "nonzero sign episodes of sum(active side*weight) after atomic same-timestamp netting"
            ),
            "active_iso_weeks": int(weeks),
        },
        "net_position_risk": net_risk,
        "legacy_nonnet_risk_disclosure": legacy_risk,
        "operating_cost_disclosure": operating_cost_disclosure(
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
        "checks": checks,
        "waived_cost_gates_applied_as_rejections": False,
        "passed": passed,
        "advance_to_next_stage": passed and not final_stage,
        "later_stage_outcomes_opened": False,
        "live_capital_authorized": False,
        "order_submission_enabled": False,
        "status": (
            "all_oos_pass_shadow_not_live"
            if passed and final_stage
            else "stage_pass_advance_authorized"
            if passed
            else "terminal_reject_no_repair"
        ),
        "decision": "pass" if passed else "terminal_reject_no_repair",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    destination = Path(output) if output is not None else Path(outputs[stage])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--freeze", type=Path, default=FREEZE)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    if args.verify_only:
        freeze = load_validation_freeze(args.freeze)
        freeze_receipt = validation_freeze_receipt(args.freeze, freeze)
        predecessor = verify_predecessor(
            args.stage,
            freeze_receipt,
            freeze["implementation"],
        )
        print(
            json.dumps(
                {
                    "stage": args.stage,
                    "verified": True,
                    "freeze_manifest_hash": freeze["manifest_hash"],
                    "predecessor": predecessor,
                    "outcomes_opened": False,
                }
            )
        )
        return 0
    result = run(args.stage, args.output, freeze_path=args.freeze)
    print(
        json.dumps(
            {
                "stage": args.stage,
                "passed": result["passed"],
                "output": str(args.output or OUTPUTS[args.stage]),
                "failed_checks": [
                    name for name, passed in result["checks"].items() if not passed
                ],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
