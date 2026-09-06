"""Run the cache-tail-corrected, outcome-blind PPOSM signal inventory."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from training import build_pposm_fresh_forward_signal_inventory_v2 as v2
from training import build_pposm_fresh_forward_signal_inventory_v3 as v3
from training import preregister_pposm_fresh_forward_signal_inventory_v4 as prereg

POLICY_ID = prereg.POLICY_ID
DEFAULT_PREREG = prereg.DEFAULT_OUTPUT
DEFAULT_MANIFEST = prereg.DEFAULT_MANIFEST
DEFAULT_CACHE = prereg.DEFAULT_CACHE
DEFAULT_OUTPUT = Path("results/pposm_fresh_forward_signal_inventory_v4_2026-09-05.json")
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
    core = {key: value for key, value in payload.items() if key not in {"created_at", "preregistration_hash"}}
    return prereg.sha256_bytes(prereg.canonical_json(core).encode())


def validate_preregistration(path: str | Path, *, manifest: str | Path, cache: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("policy_id") != POLICY_ID:
        raise RuntimeError("wrong v4 preregistration policy_id")
    if payload.get("preregistration_hash") != expected_preregistration_hash(payload):
        raise RuntimeError("v4 preregistration hash mismatch")
    expected = prereg.build_preregistration(
        prereg.Config(manifest=Path(manifest), cache=Path(cache))
    )
    strip = lambda item: {key: value for key, value in item.items() if key not in {"created_at", "preregistration_hash"}}
    if strip(payload) != strip(expected):
        raise RuntimeError("v4 preregistration contract differs from runtime")
    if payload.get("code_hashes") != runtime_code_hashes():
        raise RuntimeError("v4 runtime code hashes differ from preregistration")
    return payload


def load_frozen_cache_bundle(manifest: str | Path, cache: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    frozen_manifest = json.loads(Path(manifest).read_text(encoding="utf-8"))
    execution = dict(frozen_manifest.get("frozen_execution_config", {}))
    execution["input_csv"] = str(cache)
    strategy_cfg = v2.pposm.Config(**execution, manifest_output=str(manifest))
    market, features, _, _ = v2._load_bundle(
        strategy_cfg,
        cutoff=prereg.CACHE_LOADER_CUTOFF_ARGUMENT,
        premium_tolerance=strategy_cfg.live_premium_tolerance,
    )
    observed_last = pd.Timestamp(pd.to_datetime(market["date"]).max())
    expected_last = pd.Timestamp(prereg.CACHE_EXPECTED_LAST).tz_localize(None)
    if observed_last != expected_last:
        raise RuntimeError(
            f"immutable cache tail changed: observed={observed_last}, expected={expected_last}"
        )
    return market, features, frozen_manifest


def build_full_context_parity_candidate(
    cache_market: pd.DataFrame, db_enriched: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    hybrid_market = v2.merge_cache_db_markets(
        cache_market, db_enriched, cutoff=prereg.QUERY_START
    )
    return hybrid_market, v2.features_from_enriched_market(hybrid_market)


def check_final_merged_grid(merged_market: pd.DataFrame) -> dict[str, Any]:
    if merged_market.empty or "date" not in merged_market:
        return {"passed": False, "reason": "empty_or_missing_date"}
    start = pd.Timestamp(pd.to_datetime(merged_market["date"]).min()).isoformat()
    check = v2._grid_check(
        merged_market,
        column="date",
        start=start,
        end=prereg.FORWARD_END_EXCLUSIVE,
        freq="5min",
    )
    return {
        **check,
        "cache_precedence_before": prereg.DB_PRECEDENCE_FROM,
        "db_precedence_from": prereg.DB_PRECEDENCE_FROM,
        "frame_hash": v2.frame_hash(merged_market),
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
        checked_market = v3.forward_grid_frame(db_market)
        completeness = v3.check_source_completeness(
            btcusdt_1m=btcusdt_1m,
            premium_1m=premium_1m,
            funding=funding,
            db_market_5m=db_market,
            canonical_funding=canonical_funding,
            alias_diagnostics=alias_diagnostics,
        )
    else:
        completeness = raw
    result_base = {
        "policy_id": POLICY_ID,
        "preregistration_hash": frozen["preregistration_hash"],
        "frozen_manifest_sha256": prereg.sha256_file(manifest),
        "cache_sha256": prereg.sha256_file(cache),
        "query_hashes": v2.query_hashes(),
        "code_hashes": runtime_code_hashes(),
        "source_row_counts": {
            "btcusdt_1m": len(btcusdt_1m),
            "premium_1m": len(premium_1m),
            "funding_raw": len(funding),
            "funding_canonical": len(canonical_funding) if alias_diagnostics.get("passed") else None,
            "db_5m_market_full": len(db_market) if db_market is not None else None,
            "db_5m_market_forward_checked": len(checked_market) if checked_market is not None else None,
            "db_5m_market_enriched": None,
        },
        "source_hashes": {
            "btcusdt_1m": v2.frame_hash(btcusdt_1m),
            "premium_1m": v2.frame_hash(premium_1m),
            "funding_raw": v2.frame_hash(funding),
            "funding_canonical": v2.frame_hash(canonical_funding) if alias_diagnostics.get("passed") else None,
            "db_5m_market_full": v2.frame_hash(db_market) if db_market is not None else None,
            "db_5m_market_forward_checked": v2.frame_hash(checked_market) if checked_market is not None else None,
        },
        "funding_alias_diagnostics": alias_diagnostics,
        "source_completeness": completeness,
        "opened_outcomes": False,
        "trained": False,
    }
    if not completeness["passed"]:
        result = v2.terminal_result(result_base, terminal="source_incomplete", reason=completeness)
    else:
        cache_market, cache_features, frozen_manifest = load_frozen_cache_bundle(manifest, cache)
        db_enriched, _ = v2._features_from_5m_market(db_market, canonical_funding, premium_1m)
        result_base["source_row_counts"]["db_5m_market_enriched"] = len(db_enriched)
        result_base["source_hashes"]["db_5m_market_enriched"] = v2.frame_hash(db_enriched)
        hybrid_market, hybrid_features = build_full_context_parity_candidate(cache_market, db_enriched)
        cache_parity = v2.parity_decision_frame(
            cache_features,
            cache_market["date"],
            start=prereg.PARITY_START,
            end=prereg.PARITY_END_EXCLUSIVE,
        )
        hybrid_parity = v2.parity_decision_frame(
            hybrid_features,
            hybrid_market["date"],
            start=prereg.PARITY_START,
            end=prereg.PARITY_END_EXCLUSIVE,
        )
        parity = v2.compare_parity(
            cache_parity,
            hybrid_parity,
            atol=float(frozen["terminal_gate"]["parity_atol"]),
            rtol=float(frozen["terminal_gate"]["parity_rtol"]),
        )
        parity["expected_rows"] = prereg.CONTEXT_AND_SEAM_POLICY["parity_expected_hourly_rows"]
        parity["full_context_candidate"] = True
        result_base["parity"] = parity
        if not parity["passed"] or parity.get("checked_rows") != parity["expected_rows"]:
            result = v2.terminal_result(result_base, terminal="source_mismatch", reason=parity)
        else:
            merged_market = v2.merge_cache_db_markets(
                cache_market, db_enriched, cutoff=prereg.DB_PRECEDENCE_FROM
            )
            final_grid = check_final_merged_grid(merged_market)
            result_base["final_merged_grid"] = final_grid
            if not final_grid["passed"]:
                result = v2.terminal_result(result_base, terminal="source_incomplete", reason=final_grid)
            else:
                merged_features = v2.features_from_enriched_market(merged_market)
                dates = pd.to_datetime(merged_market["date"])
                live_prefix = v2.live_decision_features(merged_features)
                decisions_prefix = v2.decision_mask(dates, "live_hour_signal_bar", window_size=144)
                _, base_thresholds = v2._fit_active(live_prefix, dates, decisions_prefix)
                if base_thresholds != frozen_manifest.get("base_thresholds"):
                    raise RuntimeError("combined historical prefix changed frozen base thresholds")
                prefix = (dates < pd.Timestamp("2024-01-01")).to_numpy(bool)
                feature_hash = v2.pposm.feature_hash(v2.pposm.state_feature_frame(live_prefix), prefix)
                if feature_hash != frozen_manifest.get("feature_prefix_hash"):
                    raise RuntimeError("combined historical prefix changed frozen state feature hash")
                signals, summary = v2.build_signal_inventory(merged_market, merged_features, frozen_manifest)
                result = {**result_base, "terminal": "pass", "forward_counted": True, "signals": signals, "summary": summary}
    result["result_hash"] = v2.row_hash({key: value for key, value in result.items() if key != "result_hash"})
    if output is not None:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n", encoding="utf-8")
    return result


def run(cfg: Config = Config()) -> dict[str, Any]:
    engine = v2.sql_engine_from_env_file(cfg.env_file)
    try:
        frames = v2.query_db_frames(engine)
    finally:
        if hasattr(engine, "dispose"):
            engine.dispose()
    return build_from_frames(preregistration=cfg.preregistration, manifest=cfg.manifest, cache=cfg.cache, btcusdt_1m=frames["btcusdt_1m"], premium_1m=frames["premium_1m"], funding=frames["funding"], output=cfg.output)


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
    print(json.dumps({"terminal": payload["terminal"], "forward_counted": payload["forward_counted"], "summary": payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
