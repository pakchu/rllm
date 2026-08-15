"""Deterministic outcome-blind source support for HVLTFIR-8."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import build_high_volatility_large_ticket_temporal_clustering_relay_support as ticket
from training import build_high_volatility_quarter_hour_order_flow_relay_support as common
from training import preregister_high_volatility_large_ticket_funding_innovation_router as prereg


START = pd.Timestamp("2023-04-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
PREREG_SHA = "d0cca3a65193fc1e60e34ddf2458a01b4165ba2a32b708acb158b9b507852fa7"
REGISTRATION = prereg.build()
POLICY = REGISTRATION["policy"]
STAGES = {key: tuple(map(pd.Timestamp, value)) for key, value in REGISTRATION["stages"].items()}
GATES = REGISTRATION["source_support_gates"]
FUNDING_QUERY = """
SELECT funding_time, min(funding_rate) AS funding_rate, count(*) AS source_rows
FROM funding_rates_binance
WHERE symbol='BTCUSDT' AND funding_time>=:start AND funding_time<:end
GROUP BY funding_time ORDER BY funding_time
""".strip()

ROOT = Path("data/high_volatility_large_ticket_funding_innovation_router_sources_2023_2026")
PANEL = ROOT / "hourly_states.csv.gz"
MANIFEST = ROOT / "manifest.json"
CLOCK = Path("data/high_volatility_large_ticket_funding_innovation_router_clocks_2023_2026.csv.gz")
SPLIT_DIR = Path("data/high_volatility_large_ticket_funding_innovation_router_split_clocks_2023_2026")
RESULT = Path("results/high_volatility_large_ticket_funding_innovation_router_support_2026-08-16.json")
BUILDER = Path(__file__).relative_to(Path.cwd())
DEPENDENCY = Path("training/build_high_volatility_large_ticket_temporal_clustering_relay_support.py")
PANEL_COLUMNS = (
    "decision_time", "feature_available_time", "source_valid", "ticket_onset", "eligible",
    "funding_time", "previous_funding_time", "funding_rate", "previous_funding_rate",
    "funding_innovation", "funding_age_minutes", "funding_gap_minutes", "price_direction",
    "routed_side", "large_ticket_clustering", "clustering_rank", "realized_variation",
    "variation_rank", "block_return", "final_hour_return",
)
CLOCK_COLUMNS = (
    "candidate", "control", "split", "decision_time", "feature_available_time",
    "entry_time", "exit_time", "side", *PANEL_COLUMNS[5:],
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source() -> tuple[pd.DataFrame, pd.DataFrame]:
    from sqlalchemy import text

    database = ticket.postgres_engine()
    try:
        with database.connect() as connection:
            hourly = pd.read_sql_query(
                text(ticket.QUERY), connection,
                params={"start": START.to_pydatetime(), "end": END.to_pydatetime()},
            )
            funding = pd.read_sql_query(
                text(FUNDING_QUERY), connection,
                params={"start": (START - pd.Timedelta(hours=9)).to_pydatetime(), "end": END.to_pydatetime()},
            )
    finally:
        database.dispose()
    return hourly, funding


def prepare_funding(raw: pd.DataFrame) -> pd.DataFrame:
    expected = ["funding_time", "funding_rate", "source_rows"]
    if raw.columns.tolist() != expected:
        raise RuntimeError("HVLTFIR-8 funding schema drift")
    frame = raw.copy()
    frame["funding_time"] = pd.to_datetime(frame["funding_time"], utc=True, errors="raise")
    frame["funding_rate"] = pd.to_numeric(frame["funding_rate"], errors="raise")
    frame["source_rows"] = pd.to_numeric(frame["source_rows"], errors="raise")
    if frame["funding_time"].duplicated().any() or not frame["funding_time"].is_monotonic_increasing:
        raise RuntimeError("HVLTFIR-8 funding timestamp drift")
    frame["previous_funding_time"] = frame["funding_time"].shift(1)
    frame["previous_funding_rate"] = frame["funding_rate"].shift(1)
    frame["previous_source_rows"] = frame["source_rows"].shift(1)
    frame["funding_innovation"] = frame["funding_rate"] - frame["previous_funding_rate"]
    frame["funding_gap_minutes"] = (
        frame["funding_time"] - frame["previous_funding_time"]
    ).dt.total_seconds() / 60.0
    frame["funding_pair_valid"] = (
        frame["source_rows"].eq(1)
        & frame["previous_source_rows"].eq(1)
        & np.isfinite(frame[["funding_rate", "previous_funding_rate", "funding_innovation", "funding_gap_minutes"]]).all(axis=1)
        & frame["funding_gap_minutes"].between(POLICY["funding_gap_min_minutes"], POLICY["funding_gap_max_minutes"], inclusive="both")
        & frame["funding_innovation"].ne(0)
    )
    return frame


def build_panel(raw: tuple[pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    hourly, funding_raw = raw
    states = ticket.build_states(hourly).copy()
    ticket_onset, price_side = ticket.conditions(states, "primary")
    states["ticket_onset"] = ticket_onset
    states["price_direction"] = price_side.fillna(0).astype(int)
    funding = prepare_funding(funding_raw)
    panel = pd.merge_asof(
        states.sort_values("decision_time"), funding.sort_values("funding_time"),
        left_on="decision_time", right_on="funding_time", direction="backward", allow_exact_matches=True,
    )
    panel["funding_age_minutes"] = (
        panel["decision_time"] - panel["funding_time"]
    ).dt.total_seconds() / 60.0
    funding_fresh = panel["funding_age_minutes"].ge(0) & panel["funding_age_minutes"].lt(POLICY["funding_max_age_hours"] * 60)
    panel["source_valid"] = states["source_valid"].to_numpy(bool) & panel["funding_pair_valid"].eq(True) & funding_fresh
    panel["eligible"] = panel["source_valid"] & panel["ticket_onset"]
    panel["routed_side"] = -np.sign(panel["funding_innovation"]).fillna(0).astype(int)
    panel["feature_available_time"] = panel[["decision_time", "funding_time"]].max(axis=1)
    if not panel.loc[panel["eligible"], "routed_side"].isin([-1, 1]).all():
        raise RuntimeError("HVLTFIR-8 routed side drift")
    return panel.loc[:, PANEL_COLUMNS]


def stage_for(entry: pd.Timestamp, exit_: pd.Timestamp) -> str | None:
    return next((name for name, (start, end) in STAGES.items() if start <= entry and exit_ <= end), None)


def build_clock(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    reserved_until: pd.Timestamp | None = None
    for row in panel.loc[panel["eligible"]].itertuples(index=False):
        decision = pd.Timestamp(row.decision_time)
        entry = decision + pd.Timedelta(minutes=POLICY["entry_delay_minutes"])
        exit_ = entry + pd.Timedelta(hours=POLICY["hold_hours"])
        if reserved_until is not None and entry < reserved_until:
            continue
        split = stage_for(entry, exit_)
        if split is None:
            continue
        if row.feature_available_time > entry or row.routed_side not in (-1, 1):
            raise RuntimeError("HVLTFIR-8 availability drift")
        reserved_until = exit_
        rows.append({
            "candidate": prereg.POLICY_ID, "control": "primary", "split": split,
            "decision_time": decision, "feature_available_time": row.feature_available_time,
            "entry_time": entry, "exit_time": exit_, "side": int(row.routed_side),
            **{column: getattr(row, column) for column in PANEL_COLUMNS[5:]},
        })
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def support_stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    subset = clock.loc[clock["split"].eq(split)]
    if subset.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs = int(subset["side"].eq(1).sum()); shorts = int(subset["side"].eq(-1).sum())
    months = pd.to_datetime(subset["entry_time"], utc=True).dt.strftime("%Y-%m").value_counts()
    return {"events": len(subset), "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / len(subset), "max_month_share": int(months.max()) / len(subset)}


def run() -> dict[str, Any]:
    if sha256_file(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVLTFIR-8 preregistration hash drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text()); prereg.validate(registration)
    raw = load_source(); panel = build_panel(raw); clock = build_clock(panel)
    split_clocks = {name: clock.loc[clock["split"].eq(name)].copy() for name in STAGES}
    common.immutable(PANEL, common.csv_gz(panel)); common.immutable(CLOCK, common.csv_gz(clock))
    for name, frame in split_clocks.items(): common.immutable(SPLIT_DIR / f"{name}.csv.gz", common.csv_gz(frame))
    queries = {"hourly_ticket_inputs": ticket.QUERY, "funding": FUNDING_QUERY}
    source_core = {
        "protocol_version": "hvltfir_8_sources_v1", "queries": queries,
        "query_sha256": {key: hashlib.sha256(value.encode()).hexdigest() for key, value in queries.items()},
        "tables": ["bars_binance", "funding_rates_binance"], "symbol": "BTCUSDT",
        "window": [START.isoformat(), END.isoformat()], "physical_rows": {"hourly": len(raw[0]), "funding": len(raw[1])},
        "builder": {"path": str(BUILDER), "sha256": sha256_file(BUILDER)},
        "dependency": {"path": str(DEPENDENCY), "sha256": sha256_file(DEPENDENCY)},
        "panel": {"path": str(PANEL), "sha256": sha256_file(PANEL), "rows": len(panel), "valid_rows": int(panel["source_valid"].sum())},
        "outcomes_opened": False, "execution_prices_opened": False, "held_interval_funding_opened": False,
        "gross9_rows_opened": False, "no_imputation": True,
    }
    manifest = {**source_core, "manifest_hash": prereg.canonical_hash(source_core)}
    common.immutable(MANIFEST, common.json_bytes(manifest))
    support = {name: support_stats(clock, name) for name in STAGES}
    checks = {key: passed for name, values in support.items() for key, passed in (
        (f"{name}_minimum_events", values["events"] >= GATES["minimum_events"][name]),
        (f"{name}_side_balance", values["minority_side_share"] >= GATES["minority_side_share_min"]),
        (f"{name}_month_concentration", values["max_month_share"] <= GATES["max_month_share"]),
    )}
    passed = all(checks.values())
    core = {
        "protocol_version": "hvltfir_8_source_support_v1", "policy_id": prereg.POLICY_ID,
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]},
        "source_manifest": {"path": str(MANIFEST), "sha256": sha256_file(MANIFEST), "manifest_hash": manifest["manifest_hash"]},
        "completed_preentry_sources_opened": True, "candidate_incidence_opened": True,
        "postentry_return_pnl_execution_price_opened": False, "held_interval_funding_values_opened": False, "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha256_file(CLOCK), "rows": len(clock)},
        "split_artifacts": {name: {"path": str(SPLIT_DIR / f"{name}.csv.gz"), "sha256": sha256_file(SPLIT_DIR / f"{name}.csv.gz"), "rows": len(frame)} for name, frame in split_clocks.items()},
        "support": support, "support_checks": checks, "support_passed": passed,
        "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False,
        "decision": "pass_to_gross9_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": prereg.canonical_hash(core)}
    common.immutable(RESULT, common.json_bytes(result)); return result


if __name__ == "__main__":
    report = run(); print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2))
