"""Deterministic outcome-blind source support for HVCEDOIR-24."""
from __future__ import annotations

# The displacement geometry and clock are intentionally inherited from the
# preregistered HVCEDR implementation; this module adds only the exact current
# auction open-interest expansion condition.
from training.build_high_volatility_causal_extreme_displacement_reversal_support import *  # noqa: F401,F403
from training import build_high_volatility_causal_extreme_displacement_reversal_support as base
from training import preregister_high_volatility_causal_extreme_displacement_open_interest_reversal as prereg

PREREG_SHA = "ddf463605f12e6188e849d9094e4282449784f799a5c4e981b8fce295ae72c0d"
REGISTRATION = prereg.build()
POLICY = REGISTRATION["policy"]
STAGES = {key: tuple(map(pd.Timestamp, value)) for key, value in REGISTRATION["stages"].items()}
GATES = REGISTRATION["source_support_gates"]
OI_QUERY = """SELECT ts,sum_open_interest FROM open_interest_binance WHERE symbol='BTCUSDT' AND period='5m' AND ts>=:start AND ts<:end ORDER BY ts"""

ROOT = Path("data/high_volatility_causal_extreme_displacement_open_interest_reversal_sources_2020_2026")
PANEL = ROOT / "settlement_states.csv.gz"
MANIFEST = ROOT / "manifest.json"
CLOCK = Path("data/high_volatility_causal_extreme_displacement_open_interest_reversal_clocks_2020_2026.csv.gz")
SPLIT_DIR = Path("data/high_volatility_causal_extreme_displacement_open_interest_reversal_split_clocks_2020_2026")
RESULT = Path("results/high_volatility_causal_extreme_displacement_open_interest_reversal_support_2026-08-16.json")
BUILDER = Path(__file__).relative_to(Path.cwd())
PANEL_COLUMNS = base.PANEL_COLUMNS[:-1] + ("oi_start", "oi_end", "oi_change", "eligible")
CLOCK_COLUMNS = base.CLOCK_COLUMNS + ("oi_change",)


def load_source() -> tuple[pd.DataFrame, pd.DataFrame]:
    from sqlalchemy import text
    database = postgres_engine()
    try:
        with database.connect() as connection:
            bars = pd.read_sql_query(text(BAR_QUERY), connection, params={"start": START, "end": END})
            oi = pd.read_sql_query(
                text(OI_QUERY), connection,
                params={"start": START - pd.Timedelta("4h"), "end": END + pd.Timedelta("5m")},
            )
    finally:
        database.dispose()
    return bars, oi


def prepare_open_interest(raw: pd.DataFrame) -> pd.Series:
    if raw.columns.tolist() != ["ts", "sum_open_interest"]:
        raise RuntimeError("HVCEDOIR-24 OI schema drift")
    frame = raw.copy()
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="raise")
    frame["sum_open_interest"] = pd.to_numeric(frame["sum_open_interest"], errors="raise")
    if frame["ts"].duplicated().any() or not np.isfinite(frame["sum_open_interest"]).all() or not frame["sum_open_interest"].ge(0).all():
        raise RuntimeError("HVCEDOIR-24 invalid OI source")
    return frame.sort_values("ts").set_index("ts")["sum_open_interest"]


def build_panel(raw: tuple[pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    bars, oi_raw = raw
    panel = base.build_panel(bars).copy()
    oi = prepare_open_interest(oi_raw)
    decisions = pd.to_datetime(panel["decision_time"], utc=True)
    panel["oi_start"] = (decisions - pd.Timedelta("4h")).map(oi)
    panel["oi_end"] = decisions.map(oi)
    positive = panel["oi_start"].gt(0) & panel["oi_end"].gt(0)
    panel["oi_change"] = np.nan
    panel.loc[positive, "oi_change"] = np.log(
        panel.loc[positive, "oi_end"] / panel.loc[positive, "oi_start"]
    )
    panel["eligible"] = panel["eligible"] & panel["oi_change"].gt(0)
    return panel.loc[:, PANEL_COLUMNS]


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
        side = int(row.reversal_side)
        if side not in (-1, 1) or row.feature_available_time > entry or not row.oi_change > 0:
            raise RuntimeError("HVCEDOIR-24 side, OI, or availability drift")
        reserved_until = exit_
        rows.append({
            "candidate": prereg.POLICY_ID, "control": "primary", "split": split,
            "decision_time": decision, "feature_available_time": row.feature_available_time,
            "entry_time": entry, "exit_time": exit_, "side": side,
            "dominant_displacement": float(row.dominant_displacement),
            "displacement_rank": float(row.displacement_rank),
            "realized_variation": float(row.realized_variation),
            "variation_rank": float(row.variation_rank), "oi_change": float(row.oi_change),
        })
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def run() -> dict[str, Any]:
    if sha256_file(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVCEDOIR-24 preregistration hash drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    prereg.validate(registration)
    raw = load_source()
    panel = build_panel(raw)
    clock = build_clock(panel)
    split_clocks = {name: clock.loc[clock["split"].eq(name)].copy() for name in STAGES}
    common.immutable(PANEL, common.csv_gz(panel)); common.immutable(CLOCK, common.csv_gz(clock))
    for name, frame in split_clocks.items():
        common.immutable(SPLIT_DIR / f"{name}.csv.gz", common.csv_gz(frame))
    source_core = {
        "protocol_version": "hvcedoir_24_sources_v1",
        "queries": {"bars": BAR_QUERY, "open_interest": OI_QUERY},
        "query_sha256": {"bars": hashlib.sha256(BAR_QUERY.encode()).hexdigest(), "open_interest": hashlib.sha256(OI_QUERY.encode()).hexdigest()},
        "tables": ["bars_binance", "open_interest_binance"], "symbol": "BTCUSDT",
        "window": [START.isoformat(), END.isoformat()],
        "physical_rows": {"bars": len(raw[0]), "open_interest": len(raw[1])},
        "builder": {"path": str(BUILDER), "sha256": sha256_file(BUILDER)},
        "panel": {"path": str(PANEL), "sha256": sha256_file(PANEL), "rows": len(panel), "valid_rows": int(panel["source_valid"].sum()), "oi_expansion_rows": int(panel["oi_change"].gt(0).sum())},
        "outcomes_opened": False, "execution_prices_opened": False, "held_interval_funding_opened": False, "gross9_rows_opened": False, "no_imputation": True,
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
        "protocol_version": "hvcedoir_24_source_support_v1", "policy_id": prereg.POLICY_ID,
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
