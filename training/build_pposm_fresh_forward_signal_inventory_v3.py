"""Run the interval-corrected, outcome-blind fresh-forward PPOSM inventory."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from training import build_pposm_fresh_forward_signal_inventory_v2 as v2
from training import preregister_pposm_fresh_forward_signal_inventory_v3 as prereg

POLICY_ID = prereg.POLICY_ID
DEFAULT_PREREG = prereg.DEFAULT_OUTPUT
DEFAULT_MANIFEST = prereg.DEFAULT_MANIFEST
DEFAULT_CACHE = prereg.DEFAULT_CACHE
DEFAULT_OUTPUT = Path("results/pposm_fresh_forward_signal_inventory_v3_2026-09-05.json")
FORWARD_START = prereg.FORWARD_START
FORWARD_END_EXCLUSIVE = prereg.FORWARD_END_EXCLUSIVE
PARITY_ATOL = prereg.PARITY_ATOL
PARITY_RTOL = prereg.PARITY_RTOL
ROUTES = v2.ROUTES


@dataclass(frozen=True)
class Config:
    preregistration: Path = DEFAULT_PREREG
    manifest: Path = DEFAULT_MANIFEST
    cache: Path = DEFAULT_CACHE
    output: Path = DEFAULT_OUTPUT
    env_file: Path = prereg.SOURCE_ROOT / ".env"


def runtime_code_hashes() -> dict[str, str]:
    hashes = prereg.code_hashes()
    hashes["builder_module"] = prereg.sha256_file(__file__)
    return hashes


def expected_preregistration_hash(payload: dict[str, Any]) -> str:
    core = {
        key: value
        for key, value in payload.items()
        if key not in {"created_at", "preregistration_hash"}
    }
    return prereg.sha256_bytes(prereg.canonical_json(core).encode())


def validate_preregistration(
    path: str | Path, *, manifest: str | Path, cache: str | Path
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("policy_id") != POLICY_ID:
        raise RuntimeError("wrong v3 preregistration policy_id")
    if payload.get("preregistration_hash") != expected_preregistration_hash(payload):
        raise RuntimeError("v3 preregistration hash mismatch")
    expected = prereg.build_preregistration(
        prereg.Config(manifest=Path(manifest), cache=Path(cache))
    )
    payload_core = {
        key: value
        for key, value in payload.items()
        if key not in {"created_at", "preregistration_hash"}
    }
    expected_core = {
        key: value
        for key, value in expected.items()
        if key not in {"created_at", "preregistration_hash"}
    }
    if payload_core != expected_core:
        raise RuntimeError("v3 preregistration contract differs from runtime")
    if payload.get("code_hashes") != runtime_code_hashes():
        raise RuntimeError("v3 runtime code hashes differ from preregistration")
    return payload


def forward_grid_frame(db_market_5m: pd.DataFrame) -> pd.DataFrame:
    if "date" not in db_market_5m.columns:
        return db_market_5m.iloc[0:0].copy()
    dates = v2._coerce_utc_naive(db_market_5m["date"])
    start = pd.Timestamp(FORWARD_START).tz_localize(None)
    end = pd.Timestamp(FORWARD_END_EXCLUSIVE).tz_localize(None)
    return db_market_5m.loc[(dates >= start) & (dates < end)].copy().reset_index(drop=True)


def check_source_completeness(
    *,
    btcusdt_1m: pd.DataFrame,
    premium_1m: pd.DataFrame,
    funding: pd.DataFrame,
    db_market_5m: pd.DataFrame,
    canonical_funding: pd.DataFrame,
    alias_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    raw = v2.check_raw_source_completeness(
        btcusdt_1m=btcusdt_1m,
        premium_1m=premium_1m,
        funding=funding,
        canonical_funding=canonical_funding,
        alias_diagnostics=alias_diagnostics,
    )
    checked = forward_grid_frame(db_market_5m)
    forward = v2._grid_check(
        checked,
        column="date",
        start=FORWARD_START,
        end=FORWARD_END_EXCLUSIVE,
        freq="5min",
    )
    forward.update(
        {
            "full_frame_rows": int(len(db_market_5m)),
            "checked_frame_rows": int(len(checked)),
            "warmup_rows_excluded": int(len(db_market_5m) - len(checked)),
            "full_frame_hash": v2.frame_hash(db_market_5m),
            "checked_frame_hash": v2.frame_hash(checked),
            "slice_policy": prereg.FORWARD_GRID_CHECK_POLICY,
        }
    )
    checks = {**raw["checks"], "forward_5m": forward}
    failed = [name for name, value in checks.items() if not bool(value.get("passed"))]
    return {
        "passed": not failed,
        "failed": failed,
        "policy": v2.COMPLETENESS_POLICY,
        "forward_grid_check_policy": prereg.FORWARD_GRID_CHECK_POLICY,
        "checks": checks,
    }


def build_from_frames(
    *,
    preregistration: str | Path,
    manifest: str | Path,
    cache: str | Path,
    btcusdt_1m: pd.DataFrame,
    premium_1m: pd.DataFrame,
    funding: pd.DataFrame,
    output: str | Path | None = None,
) -> dict[str, Any]:
    frozen = validate_preregistration(preregistration, manifest=manifest, cache=cache)
    canonical_funding, alias_diagnostics = v2.canonicalize_funding_aliases(funding)
    raw = v2.check_raw_source_completeness(
        btcusdt_1m=btcusdt_1m,
        premium_1m=premium_1m,
        funding=funding,
        canonical_funding=canonical_funding,
        alias_diagnostics=alias_diagnostics,
    )
    db_market: pd.DataFrame | None = None
    checked_market: pd.DataFrame | None = None
    if raw["passed"]:
        db_market = v2.resample_market_bars(btcusdt_1m, "5min")
        checked_market = forward_grid_frame(db_market)
        completeness = check_source_completeness(
            btcusdt_1m=btcusdt_1m,
            premium_1m=premium_1m,
            funding=funding,
            db_market_5m=db_market,
            canonical_funding=canonical_funding,
            alias_diagnostics=alias_diagnostics,
        )
    else:
        completeness = raw
    source_row_counts = {
        "btcusdt_1m": len(btcusdt_1m),
        "premium_1m": len(premium_1m),
        "funding_raw": len(funding),
        "funding_canonical": len(canonical_funding) if alias_diagnostics.get("passed") else None,
        "db_5m_market_full": len(db_market) if db_market is not None else None,
        "db_5m_market_forward_checked": len(checked_market) if checked_market is not None else None,
        "db_5m_market_enriched": None,
    }
    source_hashes = {
        "btcusdt_1m": v2.frame_hash(btcusdt_1m),
        "premium_1m": v2.frame_hash(premium_1m),
        "funding_raw": v2.frame_hash(funding),
        "funding_canonical": v2.frame_hash(canonical_funding) if alias_diagnostics.get("passed") else None,
        "db_5m_market_full": v2.frame_hash(db_market) if db_market is not None else None,
        "db_5m_market_forward_checked": v2.frame_hash(checked_market) if checked_market is not None else None,
    }
    result_base = {
        "policy_id": POLICY_ID,
        "preregistration_hash": frozen["preregistration_hash"],
        "frozen_manifest_sha256": prereg.sha256_file(manifest),
        "cache_sha256": prereg.sha256_file(cache),
        "query_hashes": v2.query_hashes(),
        "code_hashes": runtime_code_hashes(),
        "source_row_counts": source_row_counts,
        "source_hashes": source_hashes,
        "funding_alias_diagnostics": alias_diagnostics,
        "source_completeness": completeness,
        "opened_outcomes": False,
        "trained": False,
    }
    if not completeness["passed"]:
        result = v2.terminal_result(
            result_base, terminal="source_incomplete", reason=completeness
        )
    else:
        cache_market, cache_features, frozen_manifest = v2.load_frozen_cache_bundle(
            manifest, cache
        )
        db_enriched, db_features = v2._features_from_5m_market(
            db_market, canonical_funding, premium_1m
        )
        result_base["source_row_counts"]["db_5m_market_enriched"] = len(db_enriched)
        result_base["source_hashes"]["db_5m_market_enriched"] = v2.frame_hash(db_enriched)
        parity = v2.compare_parity(
            v2.parity_decision_frame(cache_features, cache_market["date"]),
            v2.parity_decision_frame(db_features, db_market["date"]),
            atol=float(frozen["terminal_gate"]["parity_atol"]),
            rtol=float(frozen["terminal_gate"]["parity_rtol"]),
        )
        result_base["parity"] = parity
        if not parity["passed"]:
            result = v2.terminal_result(
                result_base, terminal="source_mismatch", reason=parity
            )
        else:
            merged_market = v2.merge_cache_db_markets(cache_market, db_enriched)
            merged_features = v2.features_from_enriched_market(merged_market)
            dates = pd.to_datetime(merged_market["date"])
            live_prefix = v2.live_decision_features(merged_features)
            decisions_prefix = v2.decision_mask(
                dates, "live_hour_signal_bar", window_size=144
            )
            _, base_thresholds = v2._fit_active(live_prefix, dates, decisions_prefix)
            if base_thresholds != frozen_manifest.get("base_thresholds"):
                raise RuntimeError("combined historical prefix changed frozen base thresholds")
            prefix = (dates < pd.Timestamp("2024-01-01")).to_numpy(bool)
            feature_hash = v2.pposm.feature_hash(
                v2.pposm.state_feature_frame(live_prefix), prefix
            )
            if feature_hash != frozen_manifest.get("feature_prefix_hash"):
                raise RuntimeError("combined historical prefix changed frozen state feature hash")
            signals, summary = v2.build_signal_inventory(
                merged_market, merged_features, frozen_manifest
            )
            result = {
                **result_base,
                "terminal": "pass",
                "forward_counted": True,
                "signals": signals,
                "summary": summary,
            }
    result["result_hash"] = v2.row_hash(
        {key: value for key, value in result.items() if key != "result_hash"}
    )
    if output is not None:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n",
            encoding="utf-8",
        )
    return result


def run(cfg: Config = Config()) -> dict[str, Any]:
    engine = v2.sql_engine_from_env_file(cfg.env_file)
    try:
        frames = v2.query_db_frames(engine)
    finally:
        if hasattr(engine, "dispose"):
            engine.dispose()
    return build_from_frames(
        preregistration=cfg.preregistration,
        manifest=cfg.manifest,
        cache=cfg.cache,
        btcusdt_1m=frames["btcusdt_1m"],
        premium_1m=frames["premium_1m"],
        funding=frames["funding"],
        output=cfg.output,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--env-file", type=Path, default=Config.env_file)
    return parser.parse_args()


def main() -> None:
    payload = run(Config(**vars(parse_args())))
    print(
        json.dumps(
            {
                "terminal": payload["terminal"],
                "forward_counted": payload["forward_counted"],
                "summary": payload["summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
