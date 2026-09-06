"""Strict staged economics and train-only selection for HVFPR-6."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from training import (
    evaluate_options_led_volatility_expansion_premium_relay_economics as legacy,
)
from training import (
    evaluate_options_led_volatility_expansion_premium_relay_economics_v5 as engine,
)

POLICY_ID = "HVFPR-6"
BAR = pd.Timedelta(minutes=5)
LEVERAGE = 0.5
BASE_COST = 0.0006
STRESS_COST = 0.0010
FAMILY_SIZE = 2
NORMAL_WEEKLY_P_MAX = 0.10
TRAIN_BONFERRONI_RAW_P_MAX = 0.05
ENV_FILE = "/home/pakchu/rllm/.env"

PREREG = Path("results/high_volatility_flow_priority_router_preregistration_2026-08-16.json")
PREREG_SHA = "c9a6c2799155fa89bf6fdecdfd66a97e5777a468efe5ad290e643eb8704a21c8"
SUPPORT = Path("results/high_volatility_flow_priority_router_support_2026-08-16.json")
SUPPORT_SHA = "5f0a95c151f12ac19ecc4b20fb0e3bc49ea49d47ff9e8b160379bf7ca42d93e6"
GROSS9 = Path("results/high_volatility_flow_priority_router_gross9_novelty_2026-08-16.json")
GROSS9_SHA = "50d8a109f56d7539de9abfa34af1ba2b80057734c2c4eba1b5731d8a2a68473d"

FAMILY_ORDER = (
    "HVAFC-6__THEN__HVELR-6__THEN__RIVSCR-6__ELIGIBLE_BY__HVTCCR-8",
    "RIVSCR-6__THEN__HVELR-6__THEN__HVAFC-6__ELIGIBLE_BY__HVTCCR-8",
)
ELIGIBLE_CANDIDATES = FAMILY_ORDER
CLOCKS = {
    FAMILY_ORDER[0]: {
        "path": Path("data/high_volatility_flow_priority_router_clocks_2023_2026/HVAFC-6__THEN__HVELR-6__THEN__RIVSCR-6__ELIGIBLE_BY__HVTCCR-8.csv.gz"),
        "sha256": "dcd2aaaf153d6e299cbcf3abf9236995d526cd4b33c44cb95178446ec0034a6a",
        "rows": 160,
    },
    FAMILY_ORDER[1]: {
        "path": Path("data/high_volatility_flow_priority_router_clocks_2023_2026/RIVSCR-6__THEN__HVELR-6__THEN__HVAFC-6__ELIGIBLE_BY__HVTCCR-8.csv.gz"),
        "sha256": "56c79c267dff57c38aeb972fdfeb07ce7c269d06e6dea36e24c40fe8a8a90b94",
        "rows": 160,
    },
}

STAGES = {
    "train": ("train", "2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"),
    "test": ("test", "2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"),
    "eval": ("eval", "2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
    "final": ("final", "2026-01-01T00:00:00Z", "2026-08-01T00:00:00Z"),
}
PREDECESSOR = {"test": "train", "eval": "test", "final": "eval"}
OUTPUTS = {
    stage: Path(f"results/high_volatility_flow_priority_router_{stage}_economics_2026-08-16.json")
    for stage in STAGES
}


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"HVFPR-6 artifact is not a JSON object: {path}")
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError(f"HVFPR-6 manifest drift: {path}")
    return value


def _verify_frozen_authorization() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    for path, expected, label in (
        (PREREG, PREREG_SHA, "preregistration"),
        (SUPPORT, SUPPORT_SHA, "source support"),
        (GROSS9, GROSS9_SHA, "Gross9 novelty"),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"HVFPR-6 {label} hash drift")
    for candidate, binding in CLOCKS.items():
        if sha256(binding["path"]) != binding["sha256"]:
            raise RuntimeError(f"HVFPR-6 family clock hash drift: {candidate}")

    preregistration = load_manifest(PREREG)
    support = load_manifest(SUPPORT)
    gross9 = load_manifest(GROSS9)
    if (
        preregistration.get("policy_id") != POLICY_ID
        or tuple(preregistration.get("candidate_family", ())) != FAMILY_ORDER
        or preregistration.get("candidate_family_size") != FAMILY_SIZE
        or preregistration.get("familywise_multiplicity", {}).get("number_of_hypotheses") != FAMILY_SIZE
        or preregistration.get("familywise_multiplicity", {}).get("winner_raw_weekly_signflip_p_max")
        != TRAIN_BONFERRONI_RAW_P_MAX
    ):
        raise RuntimeError("HVFPR-6 preregistered economics drift")

    candidates = support.get("candidates", {})
    if (
        support.get("policy_id") != POLICY_ID
        or support.get("eligible_routers_for_combination_gross9") != list(ELIGIBLE_CANDIDATES)
        or support.get("eligible_router_count") != len(ELIGIBLE_CANDIDATES)
        or support.get("advance_to_combination_gross9") is not True
        or support.get("advance_to_economic_outcomes") is not False
    ):
        raise RuntimeError("HVFPR-6 source-supported candidate drift")
    for candidate, binding in CLOCKS.items():
        expected_clock = {
            "path": binding["path"].as_posix(),
            "sha256": binding["sha256"],
            "rows": binding["rows"],
        }
        if (
            candidates.get(candidate, {}).get("support_passed") is not True
            or candidates.get(candidate, {}).get("advance_to_combination_gross9") is not True
            or candidates.get(candidate, {}).get("advance_to_economic_outcomes") is not False
            or candidates.get(candidate, {}).get("clock") != expected_clock
        ):
            raise RuntimeError(f"HVFPR-6 source candidate drift: {candidate}")

    gross9_results = gross9.get("router_results", {})
    if (
        gross9.get("policy_id") != POLICY_ID
        or gross9.get("eligible_routers_for_economics") != list(ELIGIBLE_CANDIDATES)
        or gross9.get("all_source_supported_routers_passed") is not True
        or gross9.get("advance_to_economic_outcomes") is not True
        or gross9.get("evidence_boundary", {}).get("outcomes_opened") is not False
        or set(gross9_results) != set(ELIGIBLE_CANDIDATES)
    ):
        raise RuntimeError("HVFPR-6 Gross9 did not authorize economics")
    for candidate in ELIGIBLE_CANDIDATES:
        record = gross9_results.get(candidate, {})
        if (
            record.get("source_support_passed") is not True
            or record.get("every_gross9_sleeve_passed") is not True
            or record.get("gross9_novelty_status") != "passed"
            or record.get("advance_to_economic_outcomes") is not True
        ):
            raise RuntimeError(f"HVFPR-6 Gross9 candidate drift: {candidate}")
    return preregistration, support, gross9


def _load_passing_stage_report(stage: str) -> dict[str, Any]:
    path = OUTPUTS[stage]
    if not path.is_file():
        raise RuntimeError(f"missing HVFPR-6 predecessor: {path}")
    report = load_manifest(path)
    winner = report.get("frozen_train_winner")
    if (
        report.get("policy_id") != POLICY_ID
        or report.get("stage") != stage
        or report.get("candidate") != winner
        or winner not in ELIGIBLE_CANDIDATES
        or report.get("passed") is not True
        or report.get("substitution_authorized") is not False
    ):
        raise RuntimeError("HVFPR-6 predecessor did not pass with the frozen winner")
    earlier_stage = PREDECESSOR.get(stage)
    if earlier_stage is None:
        if report.get("predecessor") is not None or report.get("selection", {}).get("raw_train_rank_one") != winner:
            raise RuntimeError("HVFPR-6 train predecessor chain drift")
    else:
        earlier = _load_passing_stage_report(earlier_stage)
        earlier_path = OUTPUTS[earlier_stage]
        expected_link = {
            "stage": earlier_stage,
            "path": earlier_path.as_posix(),
            "sha256": sha256(earlier_path),
            "manifest_hash": earlier["manifest_hash"],
        }
        if report.get("predecessor") != expected_link or earlier.get("frozen_train_winner") != winner:
            raise RuntimeError("HVFPR-6 predecessor chain drift")
    return report


def verify(stage: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Verify all frozen authority and predecessor passes before opening prices."""
    if stage not in STAGES:
        raise ValueError(stage)
    _, _, gross9 = _verify_frozen_authorization()
    predecessor = None if stage == "train" else _load_passing_stage_report(PREDECESSOR[stage])
    return gross9, predecessor


def postgres_engine():
    from sqlalchemy import create_engine

    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(ENV_FILE)
    return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def load_postgres_funding(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    from sqlalchemy import text

    database = postgres_engine()
    query = text(
        "SELECT funding_time AS date,funding_rate,mark_price "
        "FROM funding_rates_binance WHERE symbol=:symbol "
        "AND funding_time>=:start AND funding_time<:end ORDER BY funding_time"
    )
    try:
        with database.connect() as connection:
            frame = pd.read_sql_query(query, connection, params={"symbol": "BTCUSDT", "start": start.to_pydatetime(), "end": end.to_pydatetime()})
    finally:
        database.dispose()
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    return frame[["date", "funding_rate", "mark_price"]]


def load_postgres_sources(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    from sqlalchemy import text

    database = postgres_engine()
    bars = text(
        "SELECT date_bin('5 minutes',ts,TIMESTAMPTZ '1970-01-01 00:00:00+00') "
        "AS date,(array_agg(open ORDER BY ts))[1] AS open,max(high) AS high,"
        "min(low) AS low,(array_agg(close ORDER BY ts DESC))[1] AS close,"
        "count(*) AS source_rows FROM bars_binance WHERE interval='1m' "
        "AND symbol=:symbol AND ts>=:start AND ts<:query_end GROUP BY 1 ORDER BY 1"
    )
    funds = text(
        "SELECT funding_time AS date,funding_rate,mark_price FROM funding_rates_binance "
        "WHERE symbol=:symbol AND funding_time>=:start AND funding_time<:end ORDER BY funding_time"
    )
    try:
        with database.connect() as connection:
            market = pd.read_sql_query(bars, connection, params={"symbol": "BTCUSDT", "start": start.to_pydatetime(), "query_end": (end + BAR).to_pydatetime()})
            funding = pd.read_sql_query(funds, connection, params={"symbol": "BTCUSDT", "start": start.to_pydatetime(), "end": end.to_pydatetime()})
    finally:
        database.dispose()
    market["date"] = pd.to_datetime(market["date"], utc=True)
    funding["date"] = pd.to_datetime(funding["date"], utc=True)
    if not market["source_rows"].eq(5).all():
        raise RuntimeError("Postgres 5m source incomplete")
    source = {"mode": "postgres_exact_1m_to_5m", "tables": ["bars_binance", "funding_rates_binance"], "symbol": "BTCUSDT"}
    return market[["date", "open", "high", "low", "close"]], funding[["date", "funding_rate", "mark_price"]], source


def load_sources(stage: str, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Open only the physical market/funding prefix for the requested stage."""
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


def public_metrics(report: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if key != "trade_rows"}


def evaluate_primary(clock: pd.DataFrame, market: pd.DataFrame, funding: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    base = engine.simulate(clock, market, funding, start, end, BASE_COST)
    stress = engine.simulate(clock, market, funding, start, end, STRESS_COST)
    midpoint = start + (end - start) / 2
    halves = {}
    for name, left, right in (("first", start, midpoint), ("second", midpoint, end)):
        subset = clock[(clock["entry_time"] >= left) & (clock["exit_time"] <= right)]
        halves[name] = public_metrics(engine.simulate(subset, market, funding, left, right, BASE_COST))
    return {
        "base": public_metrics(base),
        "stress": public_metrics(stress),
        "cluster_signflip": engine.cluster_p(base["trade_rows"]),
        "calendar_halves": halves,
    }


def economic_checks(stage: str, primary: Mapping[str, Any]) -> dict[str, bool]:
    base, stress = primary["base"], primary["stress"]
    weekly_p = primary["cluster_signflip"]["pvalue"]
    checks = {
        "absolute_return_positive": base["absolute_return_pct"] > 0,
        "cagr_to_strict_mdd_min_3": base["cagr_to_strict_mdd"] >= 3,
        "strict_mdd_max_15": base["strict_mdd_pct"] <= 15,
        "mean_gross_move_min_20bp": base["mean_gross_underlying_bp"] >= 20,
        "cluster_signflip_p_max_0_1": weekly_p <= NORMAL_WEEKLY_P_MAX,
        "stress_absolute_return_positive": stress["absolute_return_pct"] > 0,
        "stress_cagr_to_strict_mdd_min_2_5": stress["cagr_to_strict_mdd"] >= 2.5,
        "each_calendar_half_positive": all(half["absolute_return_pct"] > 0 for half in primary["calendar_halves"].values()),
    }
    if stage == "train":
        checks["train_bonferroni_raw_weekly_p_max_0_05"] = weekly_p <= TRAIN_BONFERRONI_RAW_P_MAX
    return checks


def rank_train(candidate_results: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Rank raw train outcomes, with immutable full-family order as final tie-break."""
    family_index = {candidate: index for index, candidate in enumerate(FAMILY_ORDER)}
    return sorted(
        candidate_results,
        key=lambda candidate: (
            -candidate_results[candidate]["primary"]["base"]["cagr_to_strict_mdd"],
            -candidate_results[candidate]["primary"]["base"]["absolute_return_pct"],
            family_index[candidate],
        ),
    )


def _load_clock(candidate: str, split: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return legacy.load_clock(CLOCKS[candidate]["path"], split, start, end)


def run(stage: str, output: Path | None = None) -> dict[str, Any]:
    gross9, predecessor_report = verify(stage)
    split, start_text, end_text = STAGES[stage]
    start, end = legacy._utc(start_text), legacy._utc(end_text)
    market, funding, source = load_sources(stage, start, end)
    engine.validate_market(market, start, end)
    engine.validate_funding(funding, start, end)

    candidates_opened = ELIGIBLE_CANDIDATES if stage == "train" else (predecessor_report["frozen_train_winner"],)
    candidate_results: dict[str, Any] = {}
    clock_rows: dict[str, int] = {}
    for candidate in candidates_opened:
        clock = _load_clock(candidate, split, start, end)
        primary = evaluate_primary(clock, market, funding, start, end)
        candidate_results[candidate] = {"primary": primary, "checks": economic_checks(stage, primary)}
        clock_rows[candidate] = len(clock)

    if stage == "train":
        ranking = rank_train(candidate_results)
        winner = ranking[0]
        winner_checks = candidate_results[winner]["checks"]
        passed = all(winner_checks.values())
        frozen_train_winner = winner if passed else None
    else:
        ranking = None
        winner = candidates_opened[0]
        winner_checks = candidate_results[winner]["checks"]
        passed = all(winner_checks.values())
        frozen_train_winner = winner

    predecessor = None
    if stage != "train":
        predecessor_path = OUTPUTS[PREDECESSOR[stage]]
        predecessor = {
            "stage": PREDECESSOR[stage],
            "path": predecessor_path.as_posix(),
            "sha256": sha256(predecessor_path),
            "manifest_hash": predecessor_report["manifest_hash"],
        }
    decision = "terminal_reject_no_substitution" if not passed else ("pass_final" if stage == "final" else "pass")
    winner_primary = candidate_results[winner]["primary"]
    core = {
        "protocol_version": "hvfpr_6_strict_staged_economics_v1",
        "policy_id": POLICY_ID,
        "candidate": winner,
        "stage": stage,
        "window": [start_text, end_text],
        "predecessor": predecessor,
        "frozen_bindings": {
            "preregistration": {"path": PREREG.as_posix(), "sha256": PREREG_SHA},
            "source_support": {"path": SUPPORT.as_posix(), "sha256": SUPPORT_SHA},
            "gross9_novelty": {"path": GROSS9.as_posix(), "sha256": GROSS9_SHA, "manifest_hash": gross9["manifest_hash"]},
            "family_clocks": {
                candidate: {"path": binding["path"].as_posix(), "sha256": binding["sha256"], "rows": binding["rows"]}
                for candidate, binding in CLOCKS.items()
            },
        },
        "selection": {
            "full_bonferroni_family": list(FAMILY_ORDER),
            "source_and_gross9_eligible_candidates": list(ELIGIBLE_CANDIDATES),
            "train_candidates_evaluated": list(ELIGIBLE_CANDIDATES) if stage == "train" else [],
            "raw_train_ranking": ranking,
            "raw_train_rank_one": winner if stage == "train" else frozen_train_winner,
            "ranking_order": ["descending base cagr_to_strict_mdd", "descending base absolute_return_pct", "ascending frozen candidate_family order"],
            "train_gate_passed": passed if stage == "train" else True,
            "failure_is_terminal": True,
            "substitution_authorized": False,
        },
        "frozen_train_winner": frozen_train_winner,
        "substitution_authorized": False,
        "accounting": {
            "quantity": "side*0.5*pre_entry_equity/entry_open, fixed through exit",
            "gross_exposure": LEVERAGE,
            "same_open_transition": "exit and exit cost first, then next entry and entry cost",
            "funding": "cash=-fixed_quantity*settlement_mark*rate for entry<=time<exit",
            "base_cost_per_notional_side": BASE_COST,
            "stress_cost_per_notional_side": STRESS_COST,
            "strict_mdd": "global peak, every held 5m favorable then adverse OHLC, funding cash, virtual adverse exit cost, actual exit cost",
            "cagr": "full calendar including idle time",
            "mean_gross_move": "mean signed underlying entry-open to exit-open move",
            "weekly_signflip": "one-sided UTC-week cluster sign-flip Monte Carlo",
            "calendar_halves": "each exact calendar half evaluated independently at base cost",
        },
        "source": source,
        "physical_rows_opened": {"market": len(market), "funding": len(funding), "candidate_clocks": clock_rows},
        "requested_stage_outcomes_opened": stage,
        "later_stage_outcomes_opened": False,
        "candidate_results": candidate_results,
        "primary": winner_primary,
        "checks": winner_checks,
        "passed": passed,
        "advance_to_next_stage": passed and stage != "final",
        "decision": decision,
    }
    report = {**core, "manifest_hash": canonical_hash(core)}
    destination = output or OUTPUTS[stage]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify(args.stage)
        print(json.dumps({"stage": args.stage, "verified": True, "outcomes_opened": False}))
    else:
        result = run(args.stage, args.output)
        print(json.dumps({"stage": args.stage, "candidate": result["candidate"], "passed": result["passed"], "frozen_train_winner": result["frozen_train_winner"], "decision": result["decision"], "output": str(args.output or OUTPUTS[args.stage])}))
