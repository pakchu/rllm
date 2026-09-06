"""Export the frozen pre-2025 Gross9 sleeve clocks without co-resident replays.

Each sleeve reconstruction runs in its own interpreter.  The parent process
only validates the small clock CSV, writes byte-deterministic gzip output, and
records the frozen source and authority bindings.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Callable

import pandas as pd


PROTOCOL_VERSION = "gross9_pre2025_structural_clocks_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path("data/gross9_pre2025_structural_clocks_2026-08-08")
DEFAULT_MANIFEST = DEFAULT_OUTPUT_DIR / "manifest.json"
ANCHOR_PATH = Path("results/gross9_pre2025_authoritative_anchor_2026-07-28.json")
ANCHOR_SHA256 = "329878d90b6cd9c731eb4871ac041256f95f03c14dd261ada681d3a370709875"
EXPECTED_WEIGHTS = {
    "cand_rex_veto_7": 1.6,
    "fresh_kimchi_fx": 2.0,
    "frozen_annual_rank7": 3.0,
    "markov_transition_long": 2.0,
    "rex_taker_low_range_position": 0.4,
}
EXPECTED_COUNTS = {
    "cand_rex_veto_7": {"train": 308, "test2024": 64},
    "fresh_kimchi_fx": {"train": 117, "test2024": 30},
    "frozen_annual_rank7": {"train": 19, "test2024": 22},
    "markov_transition_long": {"train": 143, "test2024": 22},
    "rex_taker_low_range_position": {"train": 274, "test2024": 65},
}
CLOCK_COLUMNS = ["split", "entry_time", "exit_time", "side"]

# Paths are repository-relative identities.  Large ignored inputs may resolve
# from the authoritative /home/pakchu/rllm mirror, but the bytes must match.
INPUT_BINDINGS = {
    "market": (
        "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz",
        "a77cd0ae5b88b3c95e509d8d2610773d34af3afdc9170c63d88564bc3d0b990c",
    ),
    "market_with_oi": (
        "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz",
        "dbc9e53b09551b469168fe19cc750c5c3ea86278db3055d079103f7654050192",
    ),
    "funding": (
        "data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz",
        "4d381be086e275bacaf31df431dc31307a71a26b3947b7082efffc10bb129dd7",
    ),
    "premium": (
        "data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz",
        "b45fcc5a3cf75c8e594effe61a698c4652f841b1d304107e9669524e0fc9d0d7",
    ),
    "rank7_spot_premium": (
        "data/cache_spot_premium_5m_2020-01-01_2026-06-01.csv.gz",
        "c21ed3f52804a3f879ef5167b6b81ee0fd7dd262f2e9d87ee5c1c25dbad73617",
    ),
    "rex_veto_source": (
        "data/rex_event_reasoning_policy_sft_20260712.jsonl",
        "2f5f477ed7ffd6063bd25b1fdbcb6cbaa804685be43b4522b7105dfba1b75d48",
    ),
    "rex_veto_scan": (
        "results/rex_failure_veto_alpha_scan_2026-07-12.json",
        "e84f580c8a2dff0c35b11d2a3ff2c1db916f1c6a31225e471d4e3698f11f71cc",
    ),
    "rex_taker_train": (
        "data/rex_pullback_reclaim_q075_h144_ranker_train_2021_2023.jsonl",
        "07f6c4bb43ac92b341ce1a1b54ea6a429983611000148ad6966b81ea4a086df0",
    ),
    "rex_taker_test": (
        "data/rex_pullback_reclaim_q075_h144_ranker_test_2024.jsonl",
        "b1f5abf59c901ac109823a50063665ef455e75e70e90135acda77755ab8e5371",
    ),
    "rex_taker_eval": (
        "data/rex_pullback_reclaim_q075_h144_ranker_eval_2025_2026h1.jsonl",
        "bbe13d845d8dffcbb3e6c9b0f348390bd9d089c2d7b7bd6bccbafb91e75d9ce7",
    ),
    "rank7_capacity_evidence": (
        "results/expanding_extratrees_rank7_leverage_battery_2026-07-27.json",
        "e079f7a70d4e5eea7de962cf5daad93fd634fdf5779854d1783f83a837dc41ab",
    ),
    "rank7_capacity_preregistration": (
        "results/expanding_extratrees_rank7_leverage_battery_preregistration_2026-07-27.json",
        "3fb86f5fc3b22d33451a66a797dec50ca2e0208997a8ee224410df2781fcdc29",
    ),
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_frozen_input(relative_path: str | Path) -> Path:
    relative = Path(relative_path)
    local = REPOSITORY_ROOT / relative
    if local.is_file():
        return local.resolve()
    mirror = Path("/home/pakchu/rllm") / relative
    if mirror.is_file():
        return mirror.resolve()
    raise FileNotFoundError(relative)


def validate_frozen_inputs() -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for name, (identity, expected_hash) in INPUT_BINDINGS.items():
        resolved = resolve_frozen_input(identity)
        observed_hash = sha256_file(resolved)
        if observed_hash != expected_hash:
            raise RuntimeError(f"Gross9 frozen input hash drift: {name}")
        records[name] = {
            "path": identity,
            "resolved_path": str(resolved),
            "sha256": observed_hash,
            "size_bytes": resolved.stat().st_size,
        }
    return records


def validate_anchor() -> dict[str, Any]:
    path = REPOSITORY_ROOT / ANCHOR_PATH
    if sha256_file(path) != ANCHOR_SHA256:
        raise RuntimeError("Gross9 authoritative anchor hash drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("weights") != EXPECTED_WEIGHTS or payload.get("gross") != 9.0:
        raise RuntimeError("Gross9 authoritative anchor weights drift")
    source_counts = payload.get("selection_stats", {})
    observed = {
        split: source_counts.get(split, {}).get("trades_by_sleeve")
        for split in ("train", "test2024")
    }
    transposed = {
        sleeve: {split: observed[split].get(sleeve) for split in observed}
        for sleeve in EXPECTED_WEIGHTS
    }
    if transposed != EXPECTED_COUNTS:
        raise RuntimeError("Gross9 authoritative anchor counts drift")
    return payload


def _split_bounds() -> tuple[tuple[str, pd.Timestamp, pd.Timestamp], ...]:
    return (
        ("train", pd.Timestamp("2020-09-01", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC")),
        ("test2024", pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC")),
    )


def _fresh_kimchi_fx() -> pd.DataFrame:
    from training.audit_fresh_kimchi_orthogonal_alpha import (
        Config,
        build_candidate_context,
        candidate_schedule,
    )

    context = build_candidate_context(Config(exclude_from="2025-01-01"))
    dates = pd.DatetimeIndex(pd.to_datetime(context["market"]["date"], utc=True))
    rows = []
    for split, start, end in _split_bounds():
        for trade in candidate_schedule(context, start=str(start.date()), end=str(end.date())):
            side = 1 if str(trade.side).lower() == "long" else -1
            rows.append((split, dates[trade.entry_position], dates[trade.exit_position], side))
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def _frozen_annual_rank7() -> pd.DataFrame:
    from training.audit_fresh_kimchi_orthogonal_alpha import (
        Config,
        build_rank7_context,
        rank7_schedule,
    )

    # The frozen annual model authenticates itself against the already-published
    # complete schedule hashes.  Replaying only a truncated prefix changes the
    # cadence verifier, so reconstruct the frozen context unchanged and export
    # only the two pre-2025 authority windows below.
    context = build_rank7_context(Config())
    market = context["base"]["context"]["market"]
    dates = pd.DatetimeIndex(pd.to_datetime(market["date"], utc=True))
    rows = []
    for split, start, end in _split_bounds():
        for trade in rank7_schedule(context, start=str(start.date()), end=str(end.date())):
            side = 1 if str(trade.side).lower() == "long" else -1
            rows.append((split, dates[trade.entry_position], dates[trade.exit_position], side))
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def _cand_rex_veto_7() -> pd.DataFrame:
    from training.build_rex_event_reasoning_policy_data import _build_light_rex_features
    from training.portfolio_opt_all_discovered_alpha_gross10 import (
        SCAN_FILES,
        _rex_row_matches,
        load_json,
    )

    market = pd.read_csv(resolve_frozen_input(INPUT_BINDINGS["market_with_oi"][0]), compression="gzip")
    market["date"] = pd.to_datetime(market["date"], utc=True)
    features = _build_light_rex_features(market)
    report = load_json(SCAN_FILES["rex_veto"])
    cells: list[dict[str, Any]] = []
    seen: set[str] = set()
    for bucket in ("top", "tte_top"):
        for row in report.get(bucket, [])[:50]:
            key = json.dumps(row.get("gates", []), sort_keys=True)
            if key not in seen:
                seen.add(key)
                cells.append(row)
    cell = cells[7]
    source = [
        json.loads(line)
        for line in resolve_frozen_input(INPUT_BINDINGS["rex_veto_source"][0]).read_text().splitlines()
        if line.strip()
    ]
    rows = []
    for split, start, end in _split_bounds():
        next_allowed = 0
        for row in source:
            entry_position = int(row.get("signal_pos", -1))
            exit_position = entry_position + 145
            if entry_position < next_allowed or entry_position < 0 or exit_position >= len(market):
                continue
            if not (start <= market.date.iloc[entry_position] < end and start <= market.date.iloc[exit_position] < end):
                continue
            side = str((row.get("base_event") or {}).get("base_side", "")).lower()
            if side not in {"long", "short"} or not _rex_row_matches(cell.get("gates", []), features, row):
                continue
            rows.append((split, market.date.iloc[entry_position + 1], market.date.iloc[exit_position], 1 if side == "long" else -1))
            next_allowed = exit_position + 1
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def _markov_transition_long() -> pd.DataFrame:
    import numpy as np
    from training import portfolio_opt_added_alpha_update as portfolio
    from training.gross9_structural_clock_primitives import attach_open_interest, load_market

    market = load_market(
        portfolio.Config.input_csv,
        funding_path=portfolio.Config.funding_csv,
        premium_path=portfolio.Config.premium_csv,
        exclude_from="2025-01-01",
    )
    oi = pd.read_csv(
        resolve_frozen_input(INPUT_BINDINGS["market_with_oi"][0]),
        usecols=["date", "open_interest"],
    )
    market = attach_open_interest(market, oi)
    features = portfolio.feature_frame(market)
    active = portfolio.markov_active(market, features)
    dates = pd.DatetimeIndex(pd.to_datetime(market.date, utc=True))
    positions = np.arange(143, max(0, len(market) - 576 - 2), 12, dtype=np.int64)
    rows = []
    for split, start, end in _split_bounds():
        next_allowed = 0
        for position in positions:
            position = int(position)
            exit_position = position + 577
            if position < next_allowed or not active[position] or exit_position >= len(market):
                continue
            if not (start <= dates[position] < end and start <= dates[exit_position] < end):
                continue
            rows.append((split, dates[position + 1], dates[exit_position], 1))
            next_allowed = exit_position + 1
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def _rex_taker_low_range_position() -> pd.DataFrame:
    from training.audit_rex8640_usdkrw_gate import gate_match
    from training.portfolio_opt_added_alpha_update import REX_GATES

    market = pd.read_csv(
        resolve_frozen_input(INPUT_BINDINGS["market"][0]),
        usecols=["date"],
        compression="gzip",
    )
    market["date"] = pd.to_datetime(market.date, utc=True)
    rows_by: dict[tuple[int, str], dict[str, Any]] = {}
    for key in ("rex_taker_train", "rex_taker_test", "rex_taker_eval"):
        for line in resolve_frozen_input(INPUT_BINDINGS[key][0]).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows_by[(int(row["signal_pos"]), str(row["date"]))] = row
    source = sorted(rows_by.values(), key=lambda row: int(row["signal_pos"]))
    rows = []
    for split, start, end in _split_bounds():
        next_allowed = 0
        for row in source:
            entry_position = int(row["signal_pos"])
            exit_position = entry_position + 145
            if entry_position < next_allowed or exit_position >= len(market):
                continue
            if not (start <= market.date.iloc[entry_position] < end and start <= market.date.iloc[exit_position] < end):
                continue
            stamp = pd.Timestamp(row["date"])
            stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
            if stamp != market.date.iloc[entry_position] or not gate_match(row, list(REX_GATES)):
                continue
            side = str((row.get("action") or {}).get("side", "")).lower()
            if side not in {"long", "short"}:
                continue
            rows.append((split, market.date.iloc[entry_position + 1], market.date.iloc[exit_position], 1 if side == "long" else -1))
            next_allowed = exit_position + 1
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


SLEEVE_BUILDERS: dict[str, Callable[[], pd.DataFrame]] = {
    "cand_rex_veto_7": _cand_rex_veto_7,
    "fresh_kimchi_fx": _fresh_kimchi_fx,
    "frozen_annual_rank7": _frozen_annual_rank7,
    "markov_transition_long": _markov_transition_long,
    "rex_taker_low_range_position": _rex_taker_low_range_position,
}


def validate_clock(frame: pd.DataFrame, sleeve: str) -> pd.DataFrame:
    if list(frame.columns) != CLOCK_COLUMNS:
        raise RuntimeError(f"Gross9 clock schema drift: {sleeve}")
    result = frame.copy()
    result["entry_time"] = pd.to_datetime(result["entry_time"], utc=True, errors="raise")
    result["exit_time"] = pd.to_datetime(result["exit_time"], utc=True, errors="raise")
    result["side"] = pd.to_numeric(result["side"], errors="raise").astype(int)
    if set(result["split"]) - {"train", "test2024"}:
        raise RuntimeError(f"Gross9 split drift: {sleeve}")
    # Two authoritative Fresh-Kimchi rows are zero-duration barrier exits.
    if set(result["side"]) - {-1, 1} or not (result.exit_time >= result.entry_time).all():
        raise RuntimeError(f"Gross9 interval drift: {sleeve}")
    if result.duplicated(CLOCK_COLUMNS).any():
        raise RuntimeError(f"Gross9 duplicate clock rows: {sleeve}")
    result = result.sort_values(["entry_time", "exit_time", "side"], kind="mergesort").reset_index(drop=True)
    counts = {key: int(value) for key, value in result["split"].value_counts().items()}
    if counts != EXPECTED_COUNTS[sleeve]:
        raise RuntimeError(f"Gross9 authoritative count mismatch: {sleeve}: {counts}")
    return result


def deterministic_csv_bytes(frame: pd.DataFrame) -> bytes:
    output = frame.copy()
    for column in ("entry_time", "exit_time"):
        output[column] = output[column].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return output.to_csv(index=False, lineterminator="\n").encode("utf-8")


def write_deterministic_gzip(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(deterministic_csv_bytes(frame), mtime=0))


def _worker(sleeve: str, output: Path) -> None:
    if sleeve not in SLEEVE_BUILDERS:
        raise ValueError(f"unknown Gross9 sleeve: {sleeve}")
    validate_anchor()
    validate_frozen_inputs()
    frame = validate_clock(SLEEVE_BUILDERS[sleeve](), sleeve)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(deterministic_csv_bytes(frame))


def run(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output_dir = Path(output_dir)
    anchor = validate_anchor()
    input_records = validate_frozen_inputs()
    clock_records: dict[str, dict[str, Any]] = {}
    env = dict(os.environ)
    env.update({"PYTHONHASHSEED": "0"})
    with tempfile.TemporaryDirectory(prefix="gross9-clock-export-") as temporary:
        temporary_root = Path(temporary)
        for sleeve in EXPECTED_WEIGHTS:
            raw_path = temporary_root / f"{sleeve}.csv"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "training.export_gross9_structural_clocks",
                    "--worker-sleeve",
                    sleeve,
                    "--worker-output",
                    str(raw_path),
                ],
                cwd=REPOSITORY_ROOT,
                env=env,
                check=True,
            )
            frame = validate_clock(pd.read_csv(raw_path), sleeve)
            destination = output_dir / f"{sleeve}.csv.gz"
            write_deterministic_gzip(frame, destination)
            clock_records[sleeve] = {
                "path": destination.as_posix(),
                "sha256": sha256_file(destination),
                "rows": len(frame),
                "counts": EXPECTED_COUNTS[sleeve],
                "weight": EXPECTED_WEIGHTS[sleeve],
                "first_entry": frame.entry_time.min().isoformat(),
                "last_exit": frame.exit_time.max().isoformat(),
            }
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "authority": {
            "path": ANCHOR_PATH.as_posix(),
            "sha256": ANCHOR_SHA256,
            "protocol_hash": anchor["protocol_hash"],
            "gross": anchor["gross"],
            "weights": EXPECTED_WEIGHTS,
        },
        "reconstruction_boundary": {
            "end_exclusive": "2025-01-01T00:00:00Z",
            "sleeves_run_in_isolated_subprocesses": True,
            "maximum_concurrent_sleeves": 1,
            "ovepr_economic_metrics_computed": False,
            "gross9_rank7_existing_schedule_stats_replayed_for_hash_authentication": True,
        },
        "inputs": input_records,
        "producer": {
            "path": Path(__file__).relative_to(REPOSITORY_ROOT).as_posix(),
            "sha256": sha256_file(__file__),
        },
        "clocks": clock_records,
        "all_authoritative_counts_verified": True,
    }
    manifest = {**core, "manifest_hash": canonical_hash(core)}
    manifest_path = output_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--worker-sleeve", choices=tuple(SLEEVE_BUILDERS))
    parser.add_argument("--worker-output", type=Path)
    arguments = parser.parse_args()
    if arguments.worker_sleeve:
        if arguments.worker_output is None:
            parser.error("--worker-output is required with --worker-sleeve")
        _worker(arguments.worker_sleeve, arguments.worker_output)
        return
    manifest = run(arguments.output_dir)
    print(json.dumps({"manifest": str(arguments.output_dir / "manifest.json"), "manifest_hash": manifest["manifest_hash"]}))


if __name__ == "__main__":
    main()
