"""Write-once staged evaluator for the frozen CLBR-24 liquidation alpha."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from training.preregister_coinm_liquidation_burst_release import SPLITS, canonical_hash


SUPPORT_COMMIT = "33b57afd4c7656d274debada375772a83bc9f8bd"
EXECUTION_SOURCE_COMMIT = "1af8b80207fe8d3290c0890837daf9201576ccc3"
SUPPORT_MANIFEST_HASH = (
    "114071968cfba0bb40cf7fa44b283a3b0312d5de6551dec549999d80b8cbb27b"
)
EVALUATOR_SOURCE = Path("training/evaluate_coinm_liquidation_burst_release.py")
EVALUATOR_FREEZE = Path(
    "results/coinm_liquidation_burst_release_evaluator_freeze_2026-07-19.json"
)
SUPPORT_RESULT = Path(
    "results/coinm_liquidation_burst_release_support_2026-07-19.json"
)
EXECUTION_MANIFEST = Path("results/clbr_execution_sources_2023_2024_manifest.json")
COMBINED_CLOCKS = Path(
    "data/coinm_liquidation_burst_release_clocks_2023_2024.csv.gz"
)
SPLIT_CLOCK_DIR = Path("data/coinm_liquidation_burst_release_clocks_split_2023_2024")
RESULT_PATHS = {
    stage: Path(
        f"results/coinm_liquidation_burst_release_{stage}_2026-07-19.json"
    )
    for stage in SPLITS
}
STATIC_INPUT_SHA256 = {
    "training/preregister_coinm_liquidation_burst_release.py": (
        "7e59e8582717d9f2e44737dd94f6fda2beabf398bdee4af1e28e147c7875b6b6"
    ),
    "docs/coinm-liquidation-burst-release-preregistration-2026-07-19.md": (
        "d1d7a358bcdf14c319e57b2663ae688992fadb8f8e07f7205e2bf95e999d2ebc"
    ),
    "tests/test_preregister_coinm_liquidation_burst_release.py": (
        "d52ed9a475841c72f830e3d2a682018955bfa6734ce36192a628c274fc61c7de"
    ),
    "tests/test_coinm_liquidation_burst_release_support_artifact.py": (
        "d347501fd6ecae20402fbdf399c900bb761bfaa14040349a1e3f48f066b0290b"
    ),
    str(COMBINED_CLOCKS): (
        "df619a5ffc3b849d3c35fc7112641c33105ba76c81cbb7b8c7f3c975fd80bee0"
    ),
    str(SUPPORT_RESULT): (
        "362c1b45fd52b278e2c7f3f06214812fd02b5a1a311aae716ad3c8621852ead3"
    ),
    "training/build_clbr_execution_sources.py": (
        "08cab3889241fa935097e8847a7335d2382100a4de9cda26bfdbef44e43905f5"
    ),
    "docs/clbr-execution-source-contract-2026-07-19.md": (
        "d1e5874277b4e0bd54dde923a0913e6bb152b0128b038fd1dc880174b2aa8e65"
    ),
    "tests/test_build_clbr_execution_sources.py": (
        "a8005ef36056ea73c1c1a2f2cf789eec5d3a28b314be628abf919f2d44f26c9d"
    ),
    "tests/test_clbr_execution_sources_artifact.py": (
        "4c30b207a76502463bee6450614e0b4be778c893b79304e9b0e941e81cc8f257"
    ),
    str(EXECUTION_MANIFEST): (
        "50b86d6ab896a1c913ee83311f416f67392f29f6fb5a143f59f3abc08448d0c6"
    ),
    "docs/clbr-strict-evaluator-contract-2026-07-19.md": (
        "f24e839de2a2e56eb0fb117b0cb315af19aad9728be86586ccd2fe323ce991ee"
    ),
}
EXECUTION_FILE_SHA256 = {
    "train": {
        "market": "fa78e344e576ed3d1e911325613bce1465bfc76c259c0a3733cb350e1cdac2e4",
        "funding": "b94daae411b41d447e52dd0490a269ffd28eaf9316ddbea0da8a6a293d7d44ce",
    },
    "test": {
        "market": "3cbc1198ee32b5d77cdfa468bdaf9ed34af346a962b7026c70c70f3ff0ba7af7",
        "funding": "4b16e60417d30592679d41eeac2d08231c0bd37d337a73dbe9b8c0e43d285414",
    },
    "eval": {
        "market": "212a441e2e8213eda528e2cd586853515785f51ee4291ef8bf8f05ae0d6e52f4",
        "funding": "07dc50bbdff43f6704d819bea0ef0e32c5ff93d7072cdb8252753c122bec8fbd",
    },
}
EXPECTED_MARKET_ROWS = {"train": 32_256, "test": 52_704, "eval": 52_704}
EXPECTED_FUNDING_ROWS = {"train": 336, "test": 549, "eval": 549}
EXPECTED_CLOCK_ROWS = {"train": 40, "test": 128, "eval": 109}
STAGES = tuple(SPLITS)
BAR = pd.Timedelta(minutes=5)
YEAR_SECONDS = 365.2425 * 24.0 * 60.0 * 60.0
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EvaluationConfig:
    combined_clocks: str = str(COMBINED_CLOCKS)
    split_clock_dir: str = str(SPLIT_CLOCK_DIR)
    support_result: str = str(SUPPORT_RESULT)
    execution_manifest: str = str(EXECUTION_MANIFEST)
    freeze_output: str = str(EVALUATOR_FREEZE)
    train_result: str = str(RESULT_PATHS["train"])
    test_result: str = str(RESULT_PATHS["test"])
    eval_result: str = str(RESULT_PATHS["eval"])
    leverage: float = 1.0
    base_cost_rate_per_side: float = 0.0006
    stress_cost_rate_per_side: float = 0.0012
    bootstrap_mean_block_trades: int = 8
    bootstrap_resamples: int = 10_000
    bootstrap_seed: int = 20_260_719


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _timestamp(value: Any) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if bool(pd.isna(parsed)):
        raise ValueError("timestamp cannot be NaT")
    return cast(pd.Timestamp, parsed)


def _stable_hash(payload: dict[str, Any], hash_field: str) -> str:
    stable = {
        key: value
        for key, value in payload.items()
        if key != hash_field
    }
    return canonical_hash(stable)


def _write_json_exclusive(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _gzip_csv_bytes(frame: pd.DataFrame) -> bytes:
    raw = io.BytesIO()
    with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as zipped:
        zipped.write(frame.to_csv(index=False).encode())
    return raw.getvalue()


def _write_gzip_csv_exclusive(path: str | Path, frame: pd.DataFrame) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(_gzip_csv_bytes(frame))


def _require_canonical_config(cfg: EvaluationConfig) -> None:
    if asdict(cfg) != asdict(EvaluationConfig()):
        raise ValueError("all evaluator paths and parameters are frozen")


def _result_path(cfg: EvaluationConfig, stage: str) -> Path:
    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage}")
    return Path(getattr(cfg, f"{stage}_result"))


def _split_clock_path(cfg: EvaluationConfig, stage: str) -> Path:
    return Path(cfg.split_clock_dir) / f"{stage}_clocks.csv.gz"


def _verify_static_dependencies() -> tuple[dict[str, Any], dict[str, Any]]:
    for path, expected in STATIC_INPUT_SHA256.items():
        if _sha256(path) != expected:
            raise ValueError(f"frozen dependency changed: {path}")

    support = _read_json(SUPPORT_RESULT)
    if support.get("manifest_hash") != SUPPORT_MANIFEST_HASH:
        raise ValueError("support manifest hash changed")
    protocol = support.get("protocol", {})
    if protocol.get("outcomes_opened") is not False:
        raise ValueError("support stage opened outcomes")
    if protocol.get("market_prices_opened") is not False:
        raise ValueError("support stage opened executable prices")
    if support.get("clocks", {}).get("sha256") != STATIC_INPUT_SHA256[
        str(COMBINED_CLOCKS)
    ]:
        raise ValueError("support points to different clocks")

    execution = _read_json(EXECUTION_MANIFEST)
    execution_protocol = execution.get("protocol", {})
    if execution_protocol.get("outcomes_opened") is not False:
        raise ValueError("execution source stage opened outcomes")
    if execution_protocol.get("strategy_returns_computed") is not False:
        raise ValueError("execution source stage computed returns")
    if execution_protocol.get("clbr_clocks_loaded") is not False:
        raise ValueError("execution source stage loaded CLBR clocks")
    for stage in STAGES:
        for source_kind in ("market", "funding"):
            meta = execution["files"][stage][source_kind]
            if meta.get("sha256") != EXECUTION_FILE_SHA256[stage][source_kind]:
                raise ValueError(f"execution manifest changed {stage} {source_kind}")
    return support, execution


def _load_combined_clocks(cfg: EvaluationConfig, support: dict[str, Any]) -> pd.DataFrame:
    if _sha256(cfg.combined_clocks) != STATIC_INPUT_SHA256[str(COMBINED_CLOCKS)]:
        raise ValueError("combined clock bytes changed")
    time_columns = [
        "burst_time",
        "release_time",
        "feature_available_time",
        "entry_time",
        "planned_exit_time",
    ]
    clocks = pd.read_csv(
        cfg.combined_clocks, compression="gzip", parse_dates=time_columns
    )
    if len(clocks) != int(support["clocks"]["rows"]):
        raise ValueError("combined clock count changed")
    if not cast(pd.Series, clocks["candidate"]).eq("CLBR-24").all():
        raise ValueError("combined clocks contain another candidate")
    if not cast(pd.Series, clocks["direction"]).isin((-1, 1)).all():
        raise ValueError("combined clocks contain an invalid direction")
    if not cast(pd.Series, clocks["entry_time"]).is_monotonic_increasing:
        raise ValueError("combined clocks are not chronological")
    if cast(pd.Series, clocks["entry_time"]).duplicated().any():
        raise ValueError("combined clocks contain duplicate entries")
    if not cast(pd.Series, clocks["entry_time"]).ge(
        clocks["feature_available_time"]
    ).all():
        raise ValueError("clock entry precedes feature availability")
    return clocks


def _expected_split_clock_artifacts(
    cfg: EvaluationConfig, support: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    clocks = _load_combined_clocks(cfg, support)
    metadata: dict[str, Any] = {}
    frames: dict[str, pd.DataFrame] = {}
    for stage, (start_text, end_text) in SPLITS.items():
        start = _timestamp(start_text)
        end = _timestamp(end_text)
        subset = cast(pd.DataFrame, clocks.loc[clocks["split"].eq(stage)].copy())
        if len(subset) != EXPECTED_CLOCK_ROWS[stage]:
            raise ValueError(f"{stage} clock support changed")
        if not cast(pd.Series, subset["entry_time"]).ge(start).all() or not cast(
            pd.Series, subset["planned_exit_time"]
        ).lt(end).all():
            raise ValueError(f"{stage} clock crosses its physical boundary")
        entries = cast(pd.Series, subset["entry_time"]).reset_index(drop=True)
        exits = cast(pd.Series, subset["planned_exit_time"]).reset_index(drop=True)
        if len(subset) > 1 and not entries.iloc[1:].reset_index(drop=True).ge(
            exits.iloc[:-1].reset_index(drop=True)
        ).all():
            raise ValueError(f"{stage} clocks overlap")
        path = _split_clock_path(cfg, stage)
        metadata[stage] = {
            "path": str(path),
            "sha256": hashlib.sha256(_gzip_csv_bytes(subset)).hexdigest(),
            "rows": int(len(subset)),
            "start_inclusive": str(start),
            "end_exclusive": str(end),
        }
        frames[stage] = subset
    return metadata, frames


def _freeze_split_clocks(
    cfg: EvaluationConfig, support: dict[str, Any]
) -> dict[str, Any]:
    metadata, frames = _expected_split_clock_artifacts(cfg, support)
    for stage in STAGES:
        path = _split_clock_path(cfg, stage)
        _write_gzip_csv_exclusive(path, frames[stage])
        if _sha256(path) != metadata[stage]["sha256"]:
            raise ValueError(f"{stage} split clock write is not deterministic")
    return metadata


def _build_freeze_report(
    cfg: EvaluationConfig,
    split_clocks: dict[str, Any],
    *,
    created_at: str,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": created_at,
        "support_commit": SUPPORT_COMMIT,
        "execution_source_commit": EXECUTION_SOURCE_COMMIT,
        "support_manifest_hash": SUPPORT_MANIFEST_HASH,
        "evaluation_source": str(EVALUATOR_SOURCE),
        "evaluation_source_sha256": _sha256(EVALUATOR_SOURCE),
        "static_input_sha256": STATIC_INPUT_SHA256,
        "execution_file_sha256": EXECUTION_FILE_SHA256,
        "split_clocks": split_clocks,
        "config": asdict(cfg),
        "execution_contract": {
            "position_sizing": "fixed quantity = pre-entry equity * leverage / entry open",
            "held_bars": "[entry_position, planned_exit_position)",
            "time_exit": "planned-exit bar open",
            "gap_stop": "funding at exact bar-open boundary, then open-price stop; no post-exit high/low",
            "intrabar_stop": "bar funding first, favorable extreme, then stop fill; timestamp next-open minus 1ns",
            "funding_inclusion": "entry_time < funding_time <= exit_time",
            "funding_cash": "-direction * fixed_quantity * settlement_mark * funding_rate",
            "strict_mdd": "global HWM including entry/exit fees, funding marks, and favorable-before-adverse held-bar extremes",
            "calendar_year_seconds": YEAR_SECONDS,
        },
        "bootstrap_contract": {
            "method": "circular stationary block bootstrap of net trade returns under centered null",
            "mean_block_trades": cfg.bootstrap_mean_block_trades,
            "resamples": cfg.bootstrap_resamples,
            "seed": cfg.bootstrap_seed,
            "p_value": "(1 + count(centered_bootstrap_mean >= observed_mean)) / (B + 1)",
        },
        "promotion_gates": promotion_gate_contract(),
        "opened_windows": [],
        "sealed_windows": list(STAGES),
        "candidate_returns_computed_before_freeze": False,
        "simulation_run": False,
        "mutable_parameters": [],
    }
    report["freeze_hash"] = _stable_hash(report, "freeze_hash")
    return report


def freeze_evaluator(cfg: EvaluationConfig = EvaluationConfig()) -> dict[str, Any]:
    """Freeze evaluator bytes and physically split source-only clocks."""

    _require_canonical_config(cfg)
    if Path(cfg.freeze_output).exists():
        raise ValueError("evaluator freeze already exists and cannot be replaced")
    existing_results = [str(_result_path(cfg, stage)) for stage in STAGES if _result_path(cfg, stage).exists()]
    if existing_results:
        raise ValueError(f"stage results already exist before freeze: {existing_results}")
    existing_clocks = [str(_split_clock_path(cfg, stage)) for stage in STAGES if _split_clock_path(cfg, stage).exists()]
    if existing_clocks:
        raise ValueError(f"split clocks already exist before freeze: {existing_clocks}")

    support, execution = _verify_static_dependencies()
    for stage in STAGES:
        for source_kind in ("market", "funding"):
            path = execution["files"][stage][source_kind]["path"]
            expected = EXECUTION_FILE_SHA256[stage][source_kind]
            if _sha256(path) != expected:
                raise ValueError(f"frozen {stage} {source_kind} bytes changed")
    split_clocks = _freeze_split_clocks(cfg, support)
    report = _build_freeze_report(
        cfg,
        split_clocks,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    _write_json_exclusive(cfg.freeze_output, report)
    return report


def verify_evaluator_freeze(
    cfg: EvaluationConfig = EvaluationConfig(),
) -> dict[str, Any]:
    """Verify sealed metadata without reading any split market/funding frame."""

    _require_canonical_config(cfg)
    support, _execution = _verify_static_dependencies()
    freeze = _read_json(cfg.freeze_output)
    expected_split_clocks, _frames = _expected_split_clock_artifacts(cfg, support)
    created_at = freeze.get("created_at")
    if not isinstance(created_at, str) or not created_at:
        raise ValueError("evaluator freeze lacks its creation timestamp")
    expected_freeze = _build_freeze_report(
        cfg,
        expected_split_clocks,
        created_at=created_at,
    )
    if freeze != expected_freeze:
        raise ValueError("evaluator freeze does not reproduce from frozen inputs")
    for stage, expected in expected_split_clocks.items():
        path = _split_clock_path(cfg, stage)
        if str(path) != expected["path"] or _sha256(path) != expected["sha256"]:
            raise ValueError(f"{stage} split clock bytes changed after freeze")
    return freeze


def _validate_market(
    market: pd.DataFrame, stage: str, start: pd.Timestamp, end: pd.Timestamp
) -> None:
    expected = pd.Series(
        pd.date_range(start, end, freq="5min", inclusive="left"), name="date"
    )
    if len(market) != EXPECTED_MARKET_ROWS[stage] or not market["date"].equals(expected):
        raise ValueError(f"{stage} market is not the exact five-minute grid")
    prices = market[["open", "high", "low", "close"]].to_numpy(float)
    if not np.isfinite(prices).all() or (prices <= 0.0).any():
        raise ValueError(f"{stage} market contains invalid prices")
    if not cast(pd.Series, market["high"]).ge(
        market[["open", "close"]].max(axis=1)
    ).all() or not cast(pd.Series, market["low"]).le(
        market[["open", "close"]].min(axis=1)
    ).all():
        raise ValueError(f"{stage} market violates OHLC envelope")


def _validate_funding(
    funding: pd.DataFrame, stage: str, start: pd.Timestamp, end: pd.Timestamp
) -> None:
    times = cast(pd.Series, funding["funding_time"])
    if len(funding) != EXPECTED_FUNDING_ROWS[stage]:
        raise ValueError(f"{stage} funding count changed")
    if times.duplicated().any() or not times.is_monotonic_increasing:
        raise ValueError(f"{stage} funding times are invalid")
    if not times.ge(start).all() or not times.lt(end).all():
        raise ValueError(f"{stage} funding crosses its boundary")
    values = funding[["funding_rate", "settlement_mark_price"]].to_numpy(float)
    if not np.isfinite(values).all() or (funding["settlement_mark_price"] <= 0.0).any():
        raise ValueError(f"{stage} funding contains invalid values")


def _validate_clocks(
    clocks: pd.DataFrame, stage: str, start: pd.Timestamp, end: pd.Timestamp
) -> None:
    if len(clocks) != EXPECTED_CLOCK_ROWS[stage]:
        raise ValueError(f"{stage} clock count changed")
    if not cast(pd.Series, clocks["split"]).eq(stage).all():
        raise ValueError(f"{stage} split clock contains another stage")
    entries = cast(pd.Series, clocks["entry_time"])
    exits = cast(pd.Series, clocks["planned_exit_time"])
    if entries.duplicated().any() or not entries.is_monotonic_increasing:
        raise ValueError(f"{stage} entries are invalid")
    if not entries.ge(start).all() or not exits.lt(end).all():
        raise ValueError(f"{stage} clocks cross the stage boundary")
    if not exits.sub(entries).eq(pd.Timedelta(hours=2)).all():
        raise ValueError(f"{stage} hold differs from frozen 24 bars")
    if not cast(pd.Series, clocks["direction"]).isin((-1, 1)).all():
        raise ValueError(f"{stage} direction is invalid")
    stop = cast(pd.Series, clocks["stop_price"])
    if not np.isfinite(stop.to_numpy(float)).all() or not stop.gt(0.0).all():
        raise ValueError(f"{stage} stop is invalid")


def _load_stage_inputs(
    stage: str, cfg: EvaluationConfig, freeze: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """Load only the requested stage's market, funding, and split clock files."""

    execution = _read_json(cfg.execution_manifest)
    start = _timestamp(SPLITS[stage][0])
    end = _timestamp(SPLITS[stage][1])
    market_meta = execution["files"][stage]["market"]
    funding_meta = execution["files"][stage]["funding"]
    clock_meta = freeze["split_clocks"][stage]
    for kind, meta, expected in (
        ("market", market_meta, EXECUTION_FILE_SHA256[stage]["market"]),
        ("funding", funding_meta, EXECUTION_FILE_SHA256[stage]["funding"]),
        ("clock", clock_meta, clock_meta["sha256"]),
    ):
        if _sha256(meta["path"]) != expected:
            raise ValueError(f"{stage} {kind} bytes changed")
    market = pd.read_csv(
        market_meta["path"], compression="gzip", parse_dates=["date"]
    )
    funding = pd.read_csv(
        funding_meta["path"], compression="gzip", parse_dates=["funding_time"]
    )
    clock_times = [
        "burst_time",
        "release_time",
        "feature_available_time",
        "entry_time",
        "planned_exit_time",
    ]
    clocks = pd.read_csv(
        clock_meta["path"], compression="gzip", parse_dates=clock_times
    )
    _validate_market(market, stage, start, end)
    _validate_funding(funding, stage, start, end)
    _validate_clocks(clocks, stage, start, end)
    return market, funding, clocks, start, end


def _find_exit(
    market: pd.DataFrame,
    entry_position: int,
    planned_exit_position: int,
    direction: int,
    stop_price: float,
) -> tuple[int, pd.Timestamp, float, str]:
    for position in range(entry_position, planned_exit_position):
        bar = market.iloc[position]
        bar_time = cast(pd.Timestamp, bar["date"])
        bar_open = float(bar["open"])
        if (direction > 0 and bar_open <= stop_price) or (
            direction < 0 and bar_open >= stop_price
        ):
            return position, bar_time, bar_open, "gap_stop"
        if (direction > 0 and float(bar["low"]) <= stop_price) or (
            direction < 0 and float(bar["high"]) >= stop_price
        ):
            return (
                position,
                _timestamp(bar_time + BAR - pd.Timedelta(1, unit="ns")),
                stop_price,
                "intrabar_stop",
            )
    exit_bar = market.iloc[planned_exit_position]
    return (
        planned_exit_position,
        cast(pd.Timestamp, exit_bar["date"]),
        float(exit_bar["open"]),
        "time_exit",
    )


def simulate_strict(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    clocks: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    leverage: float,
    cost_rate_per_side: float,
) -> dict[str, Any]:
    if leverage <= 0.0 or cost_rate_per_side < 0.0:
        raise ValueError("leverage and cost must be non-negative")
    positions = {
        cast(pd.Timestamp, value): index
        for index, value in enumerate(cast(pd.Series, market["date"]))
    }
    funding_times = cast(pd.Series, funding["funding_time"])
    realized_equity = 1.0
    high_water_mark = 1.0
    maximum_drawdown = 0.0
    records: list[dict[str, Any]] = []
    invalid_stops = 0

    def update_path(value: float) -> None:
        nonlocal high_water_mark, maximum_drawdown
        if not np.isfinite(value):
            raise ValueError("non-finite strict equity path")
        high_water_mark = max(high_water_mark, value)
        if high_water_mark > 0.0:
            maximum_drawdown = max(
                maximum_drawdown, (high_water_mark - value) / high_water_mark
            )

    for clock in clocks.to_dict(orient="records"):
        entry_time = _timestamp(clock["entry_time"])
        planned_exit_time = _timestamp(clock["planned_exit_time"])
        entry_position = positions.get(entry_time)
        planned_exit_position = positions.get(planned_exit_time)
        if entry_position is None or planned_exit_position is None:
            raise ValueError("frozen clock is absent from the market grid")
        if planned_exit_position - entry_position != 24:
            raise ValueError("frozen hold is not exactly 24 bars")
        direction = int(clock["direction"])
        stop_price = float(clock["stop_price"])
        entry_price = float(market.iloc[entry_position]["open"])
        if (direction > 0 and entry_price <= stop_price) or (
            direction < 0 and entry_price >= stop_price
        ):
            invalid_stops += 1
            continue

        exit_position, exit_time, exit_price, exit_reason = _find_exit(
            market,
            entry_position,
            planned_exit_position,
            direction,
            stop_price,
        )
        pre_entry_equity = realized_equity
        quantity = pre_entry_equity * leverage / entry_price
        entry_fee = quantity * entry_price * cost_rate_per_side
        cash = pre_entry_equity - entry_fee
        funding_cash = 0.0
        included_funding = funding.loc[
            funding_times.gt(entry_time) & funding_times.le(exit_time)
        ].copy()
        next_funding = 0
        update_path(cash)

        def apply_funding_through(upper: pd.Timestamp) -> None:
            nonlocal cash, funding_cash, next_funding
            while next_funding < len(included_funding):
                event = included_funding.iloc[next_funding]
                event_time = cast(pd.Timestamp, event["funding_time"])
                if event_time > upper:
                    break
                settlement_mark = float(event["settlement_mark_price"])
                cash_flow = (
                    -direction
                    * quantity
                    * settlement_mark
                    * float(event["funding_rate"])
                )
                cash += cash_flow
                funding_cash += cash_flow
                update_path(cash + direction * quantity * (settlement_mark - entry_price))
                next_funding += 1

        last_held_position = (
            exit_position if exit_reason != "time_exit" else planned_exit_position - 1
        )
        for position in range(entry_position, last_held_position + 1):
            bar = market.iloc[position]
            bar_time = cast(pd.Timestamp, bar["date"])
            apply_funding_through(bar_time)
            if exit_reason == "gap_stop" and position == exit_position:
                update_path(cash + direction * quantity * (exit_price - entry_price))
                break

            accounting_bar_end = _timestamp(
                bar_time + BAR - pd.Timedelta(1, unit="ns")
            )
            apply_funding_through(min(accounting_bar_end, exit_time))
            favorable_price = float(bar["high"] if direction > 0 else bar["low"])
            update_path(cash + direction * quantity * (favorable_price - entry_price))
            if exit_reason == "intrabar_stop" and position == exit_position:
                update_path(cash + direction * quantity * (stop_price - entry_price))
                break
            adverse_price = float(bar["low"] if direction > 0 else bar["high"])
            update_path(cash + direction * quantity * (adverse_price - entry_price))

        if exit_reason == "time_exit":
            apply_funding_through(exit_time)
            update_path(cash + direction * quantity * (exit_price - entry_price))
        if next_funding != len(included_funding):
            raise ValueError("funding event was not applied before exit")

        exit_fee = quantity * exit_price * cost_rate_per_side
        gross_pnl = direction * quantity * (exit_price - entry_price)
        realized_equity = cash + gross_pnl - exit_fee
        update_path(realized_equity)
        net_return = realized_equity / pre_entry_equity - 1.0
        records.append(
            {
                "entry_time": str(entry_time),
                "exit_time": str(exit_time),
                "planned_exit_time": str(planned_exit_time),
                "direction": direction,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "bars_held": (
                    exit_position - entry_position
                    if exit_reason == "gap_stop"
                    else exit_position - entry_position + (exit_reason != "time_exit")
                ),
                "pre_entry_equity": pre_entry_equity,
                "fixed_quantity": quantity,
                "entry_fee": entry_fee,
                "exit_fee": exit_fee,
                "funding_cash": funding_cash,
                "funding_events": int(len(included_funding)),
                "gross_pnl": gross_pnl,
                "net_return": net_return,
                "post_exit_equity": realized_equity,
            }
        )

    years = (end - start).total_seconds() / YEAR_SECONDS
    if years <= 0.0:
        raise ValueError("evaluation window must have positive duration")
    absolute_return = realized_equity - 1.0
    cagr = realized_equity ** (1.0 / years) - 1.0 if realized_equity > 0.0 else -1.0
    ratio = cagr / max(maximum_drawdown, 1e-12)
    net_returns = np.asarray([float(row["net_return"]) for row in records], dtype=float)
    directions = np.asarray([int(row["direction"]) for row in records], dtype=int)
    exposure_seconds = sum(
        (pd.Timestamp(row["exit_time"]) - pd.Timestamp(row["entry_time"])).total_seconds()
        for row in records
    )
    return {
        "metrics": {
            "absolute_return_pct": absolute_return * 100.0,
            "cagr_pct": cagr * 100.0,
            "strict_mdd_pct": maximum_drawdown * 100.0,
            "cagr_to_strict_mdd": ratio,
            "candidate_clocks": int(len(clocks)),
            "executable_trades": int(len(records)),
            "invalid_crossed_stops": int(invalid_stops),
            "long_trades": int((directions > 0).sum()),
            "short_trades": int((directions < 0).sum()),
            "win_rate_pct": (
                float((net_returns > 0.0).mean() * 100.0) if len(net_returns) else 0.0
            ),
            "mean_net_trade_bps": (
                float(net_returns.mean() * 10_000.0) if len(net_returns) else 0.0
            ),
            "exposure_pct": exposure_seconds / (end - start).total_seconds() * 100.0,
            "calendar_start": str(start),
            "calendar_end_exclusive": str(end),
            "calendar_years": years,
            "cost_rate_per_side": cost_rate_per_side,
            "leverage": leverage,
        },
        "trades": records,
        "net_trade_returns": net_returns,
    }


def stationary_bootstrap_p_value(
    returns: np.ndarray,
    *,
    mean_block_trades: int,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("bootstrap requires at least two finite trade returns")
    if mean_block_trades < 1 or resamples < 1:
        raise ValueError("bootstrap parameters must be positive")
    observed = float(values.mean())
    centered = values - observed
    restart_probability = 1.0 / mean_block_trades
    rng = np.random.default_rng(seed)
    exceedances = 0
    n = len(values)
    for _ in range(resamples):
        index = int(rng.integers(n))
        total = float(centered[index])
        for _position in range(1, n):
            if float(rng.random()) < restart_probability:
                index = int(rng.integers(n))
            else:
                index = (index + 1) % n
            total += float(centered[index])
        if total / n >= observed:
            exceedances += 1
    return {
        "method": "circular_stationary_block_bootstrap_centered_null",
        "observed_mean_net_return": observed,
        "mean_block_trades": mean_block_trades,
        "resamples": resamples,
        "seed": seed,
        "exceedances": exceedances,
        "one_sided_p_value": (1 + exceedances) / (resamples + 1),
    }


def promotion_gate_contract() -> dict[str, Any]:
    return {
        "train": {
            "absolute_return_positive": True,
            "minimum_cagr_to_strict_mdd": 2.0,
            "maximum_strict_mdd_pct": 15.0,
            "minimum_executable_trades": 30,
            "both_directions_required": True,
        },
        "test": {
            "absolute_return_positive": True,
            "minimum_cagr_to_strict_mdd": 2.0,
            "maximum_strict_mdd_pct": 15.0,
            "minimum_executable_trades": 60,
            "both_directions_required": True,
            "stress_absolute_return_positive": True,
            "maximum_bootstrap_p_value": 0.10,
        },
        "eval": {
            "absolute_return_positive": True,
            "minimum_cagr_to_strict_mdd": 3.0,
            "maximum_strict_mdd_pct": 15.0,
            "minimum_executable_trades": 60,
            "both_directions_required": True,
            "stress_absolute_return_positive": True,
        },
    }


def _evaluate_gates(
    stage: str,
    base: dict[str, Any],
    stress: dict[str, Any],
    bootstrap: dict[str, Any],
) -> dict[str, Any]:
    metrics = base["metrics"]
    stress_metrics = stress["metrics"]
    ratio_floor = 3.0 if stage == "eval" else 2.0
    trade_floor = 30 if stage == "train" else 60
    checks = {
        "absolute_return_positive": metrics["absolute_return_pct"] > 0.0,
        "cagr_to_strict_mdd": metrics["cagr_to_strict_mdd"] >= ratio_floor,
        "strict_mdd": metrics["strict_mdd_pct"] <= 15.0,
        "executable_trades": metrics["executable_trades"] >= trade_floor,
        "both_directions": metrics["long_trades"] > 0 and metrics["short_trades"] > 0,
    }
    if stage in ("test", "eval"):
        checks["stress_absolute_return_positive"] = (
            stress_metrics["absolute_return_pct"] > 0.0
        )
    if stage == "test":
        checks["bootstrap_p_value"] = bootstrap["one_sided_p_value"] <= 0.10
    return {"checks": checks, "passes": bool(all(checks.values()))}


def _compute_stage_report(
    stage: str,
    cfg: EvaluationConfig,
    freeze: dict[str, Any],
    *,
    created_at: str,
) -> dict[str, Any]:
    market, funding, clocks, start, end = _load_stage_inputs(stage, cfg, freeze)
    base = simulate_strict(
        market,
        funding,
        clocks,
        start=start,
        end=end,
        leverage=cfg.leverage,
        cost_rate_per_side=cfg.base_cost_rate_per_side,
    )
    stress = simulate_strict(
        market,
        funding,
        clocks,
        start=start,
        end=end,
        leverage=cfg.leverage,
        cost_rate_per_side=cfg.stress_cost_rate_per_side,
    )
    bootstrap = stationary_bootstrap_p_value(
        base["net_trade_returns"],
        mean_block_trades=cfg.bootstrap_mean_block_trades,
        resamples=cfg.bootstrap_resamples,
        seed=cfg.bootstrap_seed,
    )
    promotion = _evaluate_gates(stage, base, stress, bootstrap)
    stage_index = STAGES.index(stage)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": created_at,
        "stage": stage,
        "evaluator_freeze_hash": freeze["freeze_hash"],
        "evaluation_source_sha256": freeze["evaluation_source_sha256"],
        "protocol": {
            "opened_windows": list(STAGES[: stage_index + 1]),
            "sealed_windows": list(STAGES[stage_index + 1 :]),
            "loaded_market_windows": [stage],
            "loaded_funding_windows": [stage],
            "loaded_clock_windows": [stage],
            "parameters_mutated_after_freeze": False,
        },
        "window": {"start_inclusive": str(start), "end_exclusive": str(end)},
        "base": {"metrics": base["metrics"], "trades": base["trades"]},
        "stress": {"metrics": stress["metrics"]},
        "bootstrap": bootstrap,
        "promotion": promotion,
    }
    report["result_hash"] = _stable_hash(report, "result_hash")
    return report


def _verify_prior_result(
    stage: str, cfg: EvaluationConfig, freeze: dict[str, Any]
) -> dict[str, Any]:
    result = _read_json(_result_path(cfg, stage))
    if result.get("result_hash") != _stable_hash(result, "result_hash"):
        raise ValueError(f"{stage} result hash changed")
    created_at = result.get("created_at")
    if not isinstance(created_at, str) or not created_at:
        raise ValueError(f"{stage} result lacks its creation timestamp")
    expected = _compute_stage_report(
        stage,
        cfg,
        freeze,
        created_at=created_at,
    )
    if result != expected:
        raise ValueError(f"{stage} result does not reproduce from frozen inputs")
    if result["promotion"]["passes"] is not True:
        raise ValueError(f"{stage} failed; later windows remain sealed")
    return result


def evaluate_stage(
    stage: str, cfg: EvaluationConfig = EvaluationConfig()
) -> dict[str, Any]:
    _require_canonical_config(cfg)
    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage}")
    freeze = verify_evaluator_freeze(cfg)
    output = _result_path(cfg, stage)
    if output.exists():
        raise ValueError(f"{stage} result already exists and cannot be replaced")
    stage_index = STAGES.index(stage)
    for later in STAGES[stage_index + 1 :]:
        if _result_path(cfg, later).exists():
            raise ValueError(f"later stage exists before {stage}: {later}")
    for prior in STAGES[:stage_index]:
        if not _result_path(cfg, prior).exists():
            raise ValueError(f"{prior} must pass before opening {stage}")
        _verify_prior_result(prior, cfg, freeze)

    report = _compute_stage_report(
        stage,
        cfg,
        freeze,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    _write_json_exclusive(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", *STAGES))
    args = parser.parse_args()
    if args.action == "freeze":
        report = freeze_evaluator()
        summary = {
            "freeze_hash": report["freeze_hash"],
            "opened_windows": report["opened_windows"],
            "sealed_windows": report["sealed_windows"],
        }
    else:
        report = evaluate_stage(args.action)
        summary = {
            "stage": args.action,
            "base": report["base"]["metrics"],
            "stress_absolute_return_pct": report["stress"]["metrics"][
                "absolute_return_pct"
            ],
            "bootstrap_p_value": report["bootstrap"]["one_sided_p_value"],
            "promotion": report["promotion"],
        }
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
