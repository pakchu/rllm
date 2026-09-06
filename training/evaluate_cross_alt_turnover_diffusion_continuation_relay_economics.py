"""Sequential strict economics for frozen CATDCR-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from training import evaluate_options_led_volatility_expansion_premium_relay_economics as legacy
from training import evaluate_options_led_volatility_expansion_premium_relay_economics_v5 as engine


POLICY_ID = "CATDCR-8"
BAR = pd.Timedelta(minutes=5)
LEVERAGE = 0.5
BASE_COST = 0.0006
STRESS_COST = 0.0010
ENV_FILE = "/home/pakchu/rllm/.env"
PREREG = Path("results/cross_alt_turnover_diffusion_continuation_relay_preregistration_2026-08-10.json")
PREREG_SHA = "190840cd984208a873a727cff2796d4b8a6c671f5415ef8e229826e034429f27"
SUPPORT = Path("results/cross_alt_turnover_diffusion_continuation_relay_support_2026-08-10.json")
SUPPORT_SHA = "72bd5de1dd14d66c5b96fe4e94ff442a2cb65ce94c2bf5d16ab6ff955cfb62eb"
NOVELTY = Path("results/cross_alt_turnover_diffusion_continuation_relay_gross9_novelty_2026-08-10.json")
NOVELTY_SHA = "a0c40a56c574fdd6c3d8a3a50f1da25198e071ef9f7d7fc23c8141483869be08"
CLOCK = Path("data/cross_alt_turnover_diffusion_continuation_relay_clocks_2023_2026.csv.gz")
CLOCK_SHA = "87cb5d729e637301b7265dec4071770e5fde428858f88d0c0d909b2fcb387624"
CONTROL_DIR = Path("data/cross_alt_turnover_diffusion_continuation_relay_controls_2023_2026")
FREEZE = Path("results/cross_alt_turnover_diffusion_continuation_relay_economic_evaluator_freeze_2026-08-10.json")
STAGES = {
    "train": ("train", "2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"),
    "test": ("test", "2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"),
    "eval": ("eval", "2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
    "final": ("final", "2026-01-01T00:00:00Z", "2026-08-01T00:00:00Z"),
}
PREDECESSOR = {"test": "train", "eval": "test", "final": "eval"}
OUTPUTS = {
    stage: Path(f"results/cross_alt_turnover_diffusion_continuation_relay_{stage}_economics_2026-08-10.json")
    for stage in STAGES
}
CONTROLS = ("no_diffusion_gate", "raw_turnover_entropy", "one_block_stale_diffusion", "direction_flip")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def verify(stage: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if stage not in STAGES:
        raise ValueError(stage)
    expected = ((PREREG, PREREG_SHA), (SUPPORT, SUPPORT_SHA), (NOVELTY, NOVELTY_SHA), (CLOCK, CLOCK_SHA))
    if any(sha256(path) != digest for path, digest in expected):
        raise RuntimeError("CATDCR frozen predecessor hash drift")
    novelty = load_json(NOVELTY)
    if novelty.get("advance_to_economic_outcomes") is not True or novelty.get("evidence_boundary", {}).get("outcomes_opened") is not False:
        raise RuntimeError("CATDCR novelty did not authorize economics")
    freeze = load_json(FREEZE)
    freeze_core = {key: value for key, value in freeze.items() if key != "manifest_hash"}
    if freeze.get("manifest_hash") != canonical_hash(freeze_core):
        raise RuntimeError("economic evaluator freeze manifest drift")
    if freeze.get("evaluator", {}).get("sha256") != sha256(Path(__file__)):
        raise RuntimeError("economic evaluator code drift")
    if freeze.get("outcomes_opened") is not False:
        raise RuntimeError("economic evaluator was not frozen outcome-blind")
    if stage in PREDECESSOR:
        predecessor = OUTPUTS[PREDECESSOR[stage]]
        if not predecessor.is_file():
            raise RuntimeError(f"missing predecessor: {predecessor}")
        report = load_json(predecessor)
        core = {key: value for key, value in report.items() if key != "manifest_hash"}
        if report.get("manifest_hash") != canonical_hash(core) or report.get("passed") is not True:
            raise RuntimeError("economic predecessor did not pass")
    return novelty, freeze


def postgres_engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env
    load_env_file(ENV_FILE)
    return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def load_postgres_funding(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    from sqlalchemy import text
    db = postgres_engine()
    query = text("SELECT funding_time AS date,funding_rate,mark_price FROM funding_rates_binance WHERE symbol=:symbol AND funding_time>=:start AND funding_time<:end ORDER BY funding_time")
    with db.connect() as connection:
        frame = pd.read_sql_query(query, connection, params={"symbol": "BTCUSDT", "start": start.to_pydatetime(), "end": end.to_pydatetime()})
    db.dispose(); frame["date"] = pd.to_datetime(frame["date"], utc=True)
    return frame[["date", "funding_rate", "mark_price"]]


def load_postgres_sources(start: pd.Timestamp, end: pd.Timestamp):
    from sqlalchemy import text
    db = postgres_engine()
    bars = text("SELECT date_bin('5 minutes',ts,TIMESTAMPTZ '1970-01-01 00:00:00+00') AS date,(array_agg(open ORDER BY ts))[1] AS open,max(high) AS high,min(low) AS low,(array_agg(close ORDER BY ts DESC))[1] AS close,count(*) AS source_rows FROM bars_binance WHERE interval='1m' AND symbol=:symbol AND ts>=:start AND ts<:query_end GROUP BY 1 ORDER BY 1")
    funds = text("SELECT funding_time AS date,funding_rate,mark_price FROM funding_rates_binance WHERE symbol=:symbol AND funding_time>=:start AND funding_time<:end ORDER BY funding_time")
    with db.connect() as connection:
        market = pd.read_sql_query(bars, connection, params={"symbol": "BTCUSDT", "start": start.to_pydatetime(), "query_end": (end + BAR).to_pydatetime()})
        funding = pd.read_sql_query(funds, connection, params={"symbol": "BTCUSDT", "start": start.to_pydatetime(), "end": end.to_pydatetime()})
    db.dispose(); market["date"] = pd.to_datetime(market["date"], utc=True); funding["date"] = pd.to_datetime(funding["date"], utc=True)
    if not market["source_rows"].eq(5).all():
        raise RuntimeError("Postgres 5m source incomplete")
    return market[["date", "open", "high", "low", "close"]], funding[["date", "funding_rate", "mark_price"]], {"mode": "postgres_exact_1m_to_5m", "tables": ["bars_binance", "funding_rates_binance"], "symbol": "BTCUSDT"}


def load_sources(stage: str, start: pd.Timestamp, end: pd.Timestamp):
    if stage == "final":
        return load_postgres_sources(start, end)
    market = engine.load_csv_market(start, end)
    if stage == "train":
        funding = engine.load_train_funding(start, end)
        source = {"mode": "hash_bound_gzip_physical_prefix", "market_sha256": legacy.MARKET_SHA, "funding_marks_sha256": engine.TRAIN_FUNDING_SHA}
    else:
        funding = load_postgres_funding(start, end)
        source = {"mode": "hash_bound_gzip_market_plus_postgres_exact_funding", "market_sha256": legacy.MARKET_SHA, "funding_table": "funding_rates_binance", "symbol": "BTCUSDT"}
    return market, funding, source


def public_metrics(report: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if key != "trade_rows"}


def evaluate_primary(clock: pd.DataFrame, market: pd.DataFrame, funding: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    base = engine.simulate(clock, market, funding, start, end, BASE_COST)
    stress = engine.simulate(clock, market, funding, start, end, STRESS_COST)
    midpoint = start + (end - start) / 2
    halves = {}
    for name, left, right in (("first", start, midpoint), ("second", midpoint, end)):
        subset = clock[(clock.entry_time >= left) & (clock.exit_time <= right)]
        halves[name] = public_metrics(engine.simulate(subset, market, funding, left, right, BASE_COST))
    return {"base": public_metrics(base), "stress": public_metrics(stress), "cluster_signflip": engine.cluster_p(base["trade_rows"]), "calendar_halves": halves}


def evaluate_control(clock: pd.DataFrame, market: pd.DataFrame, funding: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    return {"base": public_metrics(engine.simulate(clock, market, funding, start, end, BASE_COST)), "stress": public_metrics(engine.simulate(clock, market, funding, start, end, STRESS_COST))}


def run(stage: str, output: Path | None = None) -> dict[str, Any]:
    novelty, freeze = verify(stage)
    split, start_text, end_text = STAGES[stage]
    start, end = legacy._utc(start_text), legacy._utc(end_text)
    market, funding, source = load_sources(stage, start, end)
    engine.validate_market(market, start, end); engine.validate_funding(funding, start, end)
    primary_clock = legacy.load_clock(CLOCK, split, start, end)
    primary = evaluate_primary(primary_clock, market, funding, start, end)
    support = load_json(SUPPORT); controls = {}
    for name in CONTROLS:
        path = CONTROL_DIR / f"{name}.csv.gz"
        if sha256(path) != support["controls"][name]["sha256"]:
            raise RuntimeError(f"control hash drift: {name}")
        controls[name] = evaluate_control(legacy.load_clock(path, split, start, end), market, funding, start, end)
    base_report, stress_report = primary["base"], primary["stress"]
    checks = {
        "absolute_return_positive": base_report["absolute_return_pct"] > 0,
        "cagr_to_strict_mdd_min_3": base_report["cagr_to_strict_mdd"] >= 3,
        "strict_mdd_max_15": base_report["strict_mdd_pct"] <= 15,
        "mean_gross_move_min_20bp": base_report["mean_gross_underlying_bp"] >= 20,
        "cluster_signflip_p_max_0_1": primary["cluster_signflip"]["pvalue"] <= 0.1,
        "stress_absolute_return_positive": stress_report["absolute_return_pct"] > 0,
        "stress_cagr_to_strict_mdd_min_2_5": stress_report["cagr_to_strict_mdd"] >= 2.5,
        "each_calendar_half_positive": all(item["absolute_return_pct"] > 0 for item in primary["calendar_halves"].values()),
    }
    passed = all(checks.values())
    predecessor = None if stage == "train" else {"stage": PREDECESSOR[stage], "path": str(OUTPUTS[PREDECESSOR[stage]]), "sha256": sha256(OUTPUTS[PREDECESSOR[stage]])}
    core = {
        "protocol_version": "catdcr_8_sequential_economics_v1", "policy_id": POLICY_ID,
        "stage": stage, "window": [start_text, end_text], "predecessor": predecessor,
        "evaluator_freeze": {"path": str(FREEZE), "sha256": sha256(FREEZE), "manifest_hash": freeze["manifest_hash"]},
        "novelty_authorization": {"path": str(NOVELTY), "sha256": NOVELTY_SHA, "manifest_hash": novelty["manifest_hash"]},
        "accounting": {"quantity": "side*0.5*pre_entry_equity/entry_open, fixed through exit", "same_open_transition": "exit and exit cost first, then next entry and entry cost", "funding": "cash=-fixed_quantity*settlement_mark*rate for entry<=time<exit", "strict_mdd": "global peak, every held favorable then adverse OHLC, funding cash, virtual adverse exit cost, actual exit cost", "cagr": "full calendar including idle time"},
        "source": source, "physical_rows_opened": {"market": len(market), "funding": len(funding), "primary_clock": len(primary_clock)},
        "later_stage_outcomes_opened": False, "primary": primary, "controls_diagnostic_only": controls,
        "checks": checks, "passed": passed, "advance_to_next_stage": passed and stage != "final",
        "decision": "pass" if passed else "terminal_reject_no_repair",
    }
    report = {**core, "manifest_hash": canonical_hash(core)}
    destination = output or OUTPUTS[stage]
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--stage", choices=tuple(STAGES), required=True); parser.add_argument("--output", type=Path); parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify(args.stage); print(json.dumps({"stage": args.stage, "verified": True, "outcomes_opened": False}))
    else:
        result = run(args.stage, args.output); print(json.dumps({"stage": args.stage, "passed": result["passed"], "output": str(args.output or OUTPUTS[args.stage])}))
