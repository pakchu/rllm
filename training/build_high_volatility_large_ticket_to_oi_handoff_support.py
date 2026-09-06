"""Deterministic outcome-blind source support for HVLTOIH-8."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import build_high_volatility_large_ticket_temporal_clustering_relay_support as ticket
from training import build_high_volatility_oi_price_coactivity_sponsorship_relay_support as oi_price
from training import build_high_volatility_quarter_hour_order_flow_relay_support as common
from training import preregister_high_volatility_large_ticket_to_oi_handoff as prereg


START = pd.Timestamp("2023-04-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
PREREG_SHA = "bf73cd49a6045de13e5b24d131b3f6d87b7c9a1bd3b259ee22dbcb30c8368da3"
REGISTRATION = prereg.build()
POLICY = REGISTRATION["policy"]
STAGES = {key: tuple(map(pd.Timestamp, value)) for key, value in REGISTRATION["stages"].items()}
GATES = REGISTRATION["source_support_gates"]

ROOT = Path("data/high_volatility_large_ticket_to_oi_handoff_sources_2023_2026")
PANEL = ROOT / "handoff_states.csv.gz"
MANIFEST = ROOT / "manifest.json"
CLOCK = Path("data/high_volatility_large_ticket_to_oi_handoff_clocks_2023_2026.csv.gz")
SPLIT_DIR = Path("data/high_volatility_large_ticket_to_oi_handoff_split_clocks_2023_2026")
RESULT = Path("results/high_volatility_large_ticket_to_oi_handoff_support_2026-08-16.json")
BUILDER = Path(__file__).relative_to(Path.cwd())
DEPENDENCIES = (
    Path("training/build_high_volatility_oi_price_coactivity_sponsorship_relay_support.py"),
    Path("training/build_high_volatility_large_ticket_temporal_clustering_relay_support.py"),
)
PANEL_COLUMNS = (
    "decision_time", "feature_available_time", "source_valid", "handoff_eligible", "handoff_onset",
    "ticket_state_time",
    "completed_return_3h", "oi_variation", "gross_oi_activity", "coactivity",
    "gross_oi_activity_rank", "coactivity_rank", "oi_variation_rank",
    "lagged_large_ticket_clustering", "lagged_clustering_rank", "lagged_ticket_variation",
    "lagged_ticket_variation_rank", "lagged_block_return_6h", "lagged_final_hour_return",
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


def load_source() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from sqlalchemy import text

    bars_5m, open_interest = oi_price.load_sources()
    database = ticket.postgres_engine()
    try:
        with database.connect() as connection:
            hourly = pd.read_sql_query(
                text(ticket.QUERY), connection,
                params={"start": START.to_pydatetime(), "end": END.to_pydatetime()},
            )
    finally:
        database.dispose()
    return bars_5m, open_interest, hourly


def build_panel(raw: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    bars_5m, open_interest, hourly = raw
    oi_features = oi_price.build_features(bars_5m, open_interest).rename(columns={
        "completed_return": "completed_return_3h",
        "realized_variation": "oi_variation",
        "variation_rank": "oi_variation_rank",
        "source_valid": "oi_source_valid",
    })
    ticket_states = ticket.build_states(hourly).rename(columns={
        "source_valid": "ticket_source_valid",
        "realized_variation": "ticket_variation",
        "variation_rank": "ticket_variation_rank",
        "block_return": "block_return_6h",
    })
    keep_oi = [
        "decision_time", "oi_source_valid", "completed_return_3h", "oi_variation",
        "gross_oi_activity", "coactivity", "gross_oi_activity_rank", "coactivity_rank",
        "oi_variation_rank",
    ]
    keep_ticket = [
        "decision_time", "ticket_source_valid", "large_ticket_clustering", "clustering_rank",
        "ticket_variation", "ticket_variation_rank", "block_return_6h", "final_hour_return",
    ]
    panel = oi_features[keep_oi].merge(ticket_states[keep_ticket], on="decision_time", how="left", validate="one_to_one")
    ticket_columns = keep_ticket[1:]
    panel["ticket_state_time"] = panel["decision_time"].shift(1)
    for column in ticket_columns:
        panel[f"lagged_{column}"] = panel[column].shift(1)
    numeric = keep_oi[2:] + [f"lagged_{column}" for column in keep_ticket[2:]]
    panel["source_valid"] = (
        panel["oi_source_valid"].eq(True)
        & panel["lagged_ticket_source_valid"].eq(True)
        & panel["decision_time"].sub(panel["ticket_state_time"]).eq(pd.Timedelta(hours=POLICY["handoff_lag_hours"]))
        & np.isfinite(panel[numeric]).all(axis=1)
        & panel[["completed_return_3h", "lagged_block_return_6h", "lagged_final_hour_return"]].ne(0).all(axis=1)
    )
    side = np.sign(panel["completed_return_3h"])
    direction = (
        side.ne(0)
        & np.sign(panel["lagged_block_return_6h"]).eq(side)
        & np.sign(panel["lagged_final_hour_return"]).eq(side)
    )
    panel["handoff_eligible"] = (
        panel["source_valid"]
        & panel["coactivity_rank"].ge(POLICY["coactivity_rank_min"])
        & panel["gross_oi_activity_rank"].ge(POLICY["gross_oi_activity_rank_min"])
        & panel["oi_variation_rank"].ge(POLICY["oi_variation_rank_min"])
        & panel["lagged_clustering_rank"].ge(POLICY["clustering_rank_min"])
        & panel["lagged_ticket_variation_rank"].ge(POLICY["ticket_variation_rank_min"])
        & direction
    )
    prior_valid = panel["source_valid"].shift(1, fill_value=False)
    prior_exact = panel["decision_time"].diff().eq(pd.Timedelta(hours=3))
    panel["handoff_onset"] = (
        panel["handoff_eligible"]
        & ~panel["handoff_eligible"].shift(1, fill_value=False)
        & prior_valid
        & prior_exact
    )
    panel["feature_available_time"] = panel["decision_time"]
    return panel.loc[:, PANEL_COLUMNS]


def stage_for(entry: pd.Timestamp, exit_: pd.Timestamp) -> str | None:
    return next((name for name, (start, end) in STAGES.items() if start <= entry and exit_ <= end), None)


def build_clock(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    reserved_until: pd.Timestamp | None = None
    for row in panel.loc[panel["handoff_onset"]].itertuples(index=False):
        decision = pd.Timestamp(row.decision_time)
        entry = decision + pd.Timedelta(minutes=POLICY["entry_delay_minutes"])
        exit_ = entry + pd.Timedelta(hours=POLICY["hold_hours"])
        if reserved_until is not None and entry < reserved_until:
            continue
        split = stage_for(entry, exit_)
        if split is None:
            continue
        side = int(np.sign(row.completed_return_3h))
        if side not in (-1, 1) or row.feature_available_time > entry:
            raise RuntimeError("HVLTOIH-8 side or availability drift")
        reserved_until = exit_
        values = {column: getattr(row, column) for column in PANEL_COLUMNS[5:]}
        rows.append({
            "candidate": prereg.POLICY_ID, "control": "primary", "split": split,
            "decision_time": decision, "feature_available_time": row.feature_available_time,
            "entry_time": entry, "exit_time": exit_, "side": side, **values,
        })
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def support_stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    subset = clock.loc[clock["split"].eq(split)]
    if subset.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs = int(subset["side"].eq(1).sum())
    shorts = int(subset["side"].eq(-1).sum())
    months = pd.to_datetime(subset["entry_time"], utc=True).dt.strftime("%Y-%m").value_counts()
    return {"events": len(subset), "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / len(subset), "max_month_share": int(months.max()) / len(subset)}


def run() -> dict[str, Any]:
    if sha256_file(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVLTOIH-8 preregistration hash drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    prereg.validate(registration)
    raw = load_source()
    panel = build_panel(raw)
    clock = build_clock(panel)
    split_clocks = {name: clock.loc[clock["split"].eq(name)].copy() for name in STAGES}
    common.immutable(PANEL, common.csv_gz(panel))
    common.immutable(CLOCK, common.csv_gz(clock))
    for name, frame in split_clocks.items():
        common.immutable(SPLIT_DIR / f"{name}.csv.gz", common.csv_gz(frame))
    queries = {"five_minute_bars": oi_price.BAR_QUERY, "open_interest": oi_price.OI_QUERY, "hourly_ticket_inputs": ticket.QUERY}
    source_core = {
        "protocol_version": "hvltoih_8_sources_v1", "queries": queries,
        "query_sha256": {key: hashlib.sha256(value.encode()).hexdigest() for key, value in queries.items()},
        "tables": ["bars_binance", "open_interest_binance"], "symbol": "BTCUSDT",
        "window": [START.isoformat(), END.isoformat()],
        "physical_rows": {"five_minute_bars": len(raw[0]), "open_interest": len(raw[1]), "hourly_ticket_inputs": len(raw[2])},
        "builder": {"path": str(BUILDER), "sha256": sha256_file(BUILDER)},
        "dependencies": {str(path): sha256_file(path) for path in DEPENDENCIES},
        "panel": {"path": str(PANEL), "sha256": sha256_file(PANEL), "rows": len(panel), "valid_rows": int(panel["source_valid"].sum())},
        "prior_component_event_rows_reused": False,
        "outcomes_opened": False, "execution_prices_opened": False,
        "held_interval_funding_opened": False, "gross9_rows_opened": False, "no_imputation": True,
    }
    manifest = {**source_core, "manifest_hash": prereg.canonical_hash(source_core)}
    common.immutable(MANIFEST, common.json_bytes(manifest))
    support = {name: support_stats(clock, name) for name in STAGES}
    checks = {
        key: passed
        for name, values in support.items()
        for key, passed in (
            (f"{name}_minimum_events", values["events"] >= GATES["minimum_events"][name]),
            (f"{name}_side_balance", values["minority_side_share"] >= GATES["minority_side_share_min"]),
            (f"{name}_month_concentration", values["max_month_share"] <= GATES["max_month_share"]),
        )
    }
    passed = all(checks.values())
    core = {
        "protocol_version": "hvltoih_8_source_support_v1", "policy_id": prereg.POLICY_ID,
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]},
        "source_manifest": {"path": str(MANIFEST), "sha256": sha256_file(MANIFEST), "manifest_hash": manifest["manifest_hash"]},
        "completed_preentry_sources_opened": True, "candidate_incidence_opened": True,
        "postentry_return_pnl_execution_price_opened": False,
        "held_interval_funding_values_opened": False, "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha256_file(CLOCK), "rows": len(clock)},
        "split_artifacts": {name: {"path": str(SPLIT_DIR / f"{name}.csv.gz"), "sha256": sha256_file(SPLIT_DIR / f"{name}.csv.gz"), "rows": len(frame)} for name, frame in split_clocks.items()},
        "support": support, "support_checks": checks, "support_passed": passed,
        "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False,
        "decision": "pass_to_gross9_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": prereg.canonical_hash(core)}
    common.immutable(RESULT, common.json_bytes(result))
    return result


if __name__ == "__main__":
    report = run()
    print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2))
