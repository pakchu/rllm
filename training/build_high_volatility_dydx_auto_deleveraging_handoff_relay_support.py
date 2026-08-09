"""Open source-only OOS incidence for preregistered HVDADH-8."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_dydx_auto_deleveraging_handoff_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


ENV_FILE = "/home/pakchu/rllm/.env"
START = pd.Timestamp("2023-11-14T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
BAR_START = START - pd.Timedelta(hours=6)
PREREG_SHA256 = "6eb22a261c56c75fbfec0b05af64159bf0522349ea8ff2c864c7548bb31fdf95"
API = "https://indexer.dydx.trade/v4/trades/perpetualMarket/BTC-USD"
OPENAPI = "https://indexer.dydx.trade/docs/swagger-ui-init.js"
SOURCE_DIR = Path("data/high_volatility_dydx_auto_deleveraging_handoff_relay_sources_2023_2026")
FORCED_TRADES = SOURCE_DIR / "forced_trades.csv.gz"
PAGE_MANIFEST = SOURCE_DIR / "page_manifest.json"
PANEL = SOURCE_DIR / "hourly_source_panel.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/high_volatility_dydx_auto_deleveraging_handoff_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_dydx_auto_deleveraging_handoff_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_dydx_auto_deleveraging_handoff_relay_support_2026-08-10.json")
QUERY = """
SELECT ts,open,high,low,close
FROM bars_binance
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
ORDER BY ts
"""
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
ACCEPTED_TYPES = {"LIMIT", "LIQUIDATED", "DELEVERAGED", "TWAP_SUBORDER"}
FORCED_TYPES = {"LIQUIDATED", "DELEVERAGED"}
CONTROLS = (
    "no_volatility_gate",
    "no_forced_notional_gate",
    "always_fade_forced_flow",
    "one_hour_stale_features",
    "direction_flip",
    "forced_long",
)
CLOCK_COLUMNS = (
    "candidate", "control", "split", "decision_time", "feature_available_time",
    "entry_time", "exit_time", "side", "forced_buy_notional", "forced_sell_notional",
    "forced_notional", "forced_flow", "deleveraged_notional", "deleveraged_state",
    "forced_notional_rank", "realized_variation", "realized_variation_rank",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def strict_prior_midrank(
    values: pd.Series, lookback: int = 720, minimum: int = 480
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    output = pd.Series(np.nan, index=numeric.index, dtype=float)
    history: list[float] = []
    for index, current in numeric.items():
        prior = history[-lookback:]
        if math.isfinite(current) and len(prior) >= minimum:
            array = np.asarray(prior)
            output.at[index] = (
                np.sum(array < current) + 0.5 * np.sum(array == current)
            ) / len(array)
        if math.isfinite(current):
            history.append(current)
    return output


def postgres_engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(ENV_FILE)
    return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def http_get(url: str, timeout: int = 60) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "rllm-source-audit/1"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"dYdX HTTP status {response.status}")
        return response.read()


def parse_trade(row: Any) -> dict[str, Any]:
    if not isinstance(row, dict) or set(row) != {
        "id", "side", "size", "price", "type", "createdAt", "createdAtHeight"
    }:
        raise ValueError("dYdX trade schema invalid")
    trade_id = row["id"]
    side = row["side"]
    trade_type = row["type"]
    if not isinstance(trade_id, str) or not trade_id or side not in {"BUY", "SELL"}:
        raise ValueError("dYdX trade identity invalid")
    if trade_type not in ACCEPTED_TYPES:
        raise ValueError("dYdX trade type invalid")
    size = float(row["size"])
    price = float(row["price"])
    height = int(row["createdAtHeight"])
    created_at = pd.Timestamp(row["createdAt"])
    if (
        not math.isfinite(size) or size <= 0 or not math.isfinite(price) or price <= 0
        or height <= 0 or created_at.tzinfo is None
    ):
        raise ValueError("dYdX trade value invalid")
    return {
        "id": trade_id,
        "side": side,
        "size": size,
        "price": price,
        "notional": size * price,
        "type": trade_type,
        "created_at": created_at.tz_convert("UTC"),
        "created_at_height": height,
    }


def download_forced_trades() -> tuple[pd.DataFrame, dict[str, Any], bytes]:
    openapi = http_get(OPENAPI)
    text = openapi.decode("utf-8")
    for token in ('"LIQUIDATED"', '"DELEVERAGED"', '"createdBeforeOrAtHeight"'):
        if token not in text:
            raise RuntimeError(f"dYdX OpenAPI token missing: {token}")
    boundary_url = API + "?" + urllib.parse.urlencode({
        "limit": 1, "createdBeforeOrAt": END.isoformat().replace("+00:00", "Z")
    })
    boundary_raw = http_get(boundary_url)
    boundary = json.loads(boundary_raw)
    if not isinstance(boundary, dict) or len(boundary.get("trades", [])) != 1:
        raise RuntimeError("dYdX upper boundary lookup invalid")
    cursor = parse_trade(boundary["trades"][0])["created_at_height"]
    pages: list[dict[str, Any]] = []
    forced: list[dict[str, Any]] = []
    seen: set[str] = set()
    previous_cursor = cursor + 1
    reached_lower_boundary = False
    while cursor > 0:
        if cursor >= previous_cursor:
            raise RuntimeError("dYdX cursor did not decrease")
        previous_cursor = cursor
        url = API + "?" + urllib.parse.urlencode({
            "limit": 1000, "createdBeforeOrAtHeight": cursor
        })
        raw = http_get(url)
        payload = json.loads(raw)
        rows = payload.get("trades") if isinstance(payload, dict) else None
        if not isinstance(rows, list) or not rows:
            raise RuntimeError("dYdX empty page before lower boundary")
        parsed = [parse_trade(row) for row in rows]
        ids = [row["id"] for row in parsed]
        if len(ids) != len(set(ids)) or seen.intersection(ids):
            raise RuntimeError("dYdX repeated trade id")
        seen.update(ids)
        heights = [row["created_at_height"] for row in parsed]
        if heights != sorted(heights, reverse=True) or max(heights) > cursor:
            raise RuntimeError("dYdX page ordering invalid")
        minimum_height = min(heights)
        if len(parsed) == 1000 and all(height == minimum_height for height in heights):
            raise RuntimeError("dYdX single-height page overflow")
        pages.append({
            "cursor": cursor,
            "rows": len(parsed),
            "maximum_height": max(heights),
            "minimum_height": minimum_height,
            "newest_created_at": max(row["created_at"] for row in parsed).isoformat(),
            "oldest_created_at": min(row["created_at"] for row in parsed).isoformat(),
            "response_sha256": hashlib.sha256(raw).hexdigest(),
        })
        for row in parsed:
            if START <= row["created_at"] < END and row["type"] in FORCED_TYPES:
                forced.append(row)
        if min(row["created_at"] for row in parsed) < START:
            reached_lower_boundary = True
            break
        cursor = minimum_height - 1
        if len(pages) % 100 == 0:
            print(f"dYdX pages={len(pages)} cursor={cursor}", flush=True)
        time.sleep(0.02)
    if not reached_lower_boundary:
        raise RuntimeError("dYdX lower boundary not reached")
    frame = pd.DataFrame(forced, columns=(
        "id", "side", "size", "price", "notional", "type", "created_at", "created_at_height"
    )).sort_values(["created_at", "id"]).reset_index(drop=True)
    manifest = {
        "boundary_url": boundary_url,
        "boundary_response_sha256": hashlib.sha256(boundary_raw).hexdigest(),
        "pages": pages,
        "page_count": len(pages),
        "all_trade_ids_seen": len(seen),
        "forced_trade_rows": len(frame),
        "reached_lower_boundary": reached_lower_boundary,
    }
    return frame, manifest, openapi


def feature_panel(forced: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    market = bars.copy()
    market["ts"] = pd.to_datetime(market.ts, utc=True)
    for column in ("open", "high", "low", "close"):
        market[column] = pd.to_numeric(market[column], errors="coerce")
    market = market.drop_duplicates("ts", keep=False).set_index("ts").sort_index()
    trades = forced.copy()
    if not trades.empty:
        trades["created_at"] = pd.to_datetime(trades.created_at, utc=True)
    rows: list[dict[str, Any]] = []
    for decision in pd.date_range(START, END, freq="1h", inclusive="left"):
        expected = pd.date_range(decision - pd.Timedelta(hours=6), decision, freq="1min", inclusive="left")
        window = market.reindex(expected)
        ohlc = window[["open", "high", "low", "close"]]
        valid = bool(
            decision >= START + pd.Timedelta(hours=6)
            and len(window) == 360
            and np.isfinite(ohlc).all(axis=1).all()
            and ohlc.gt(0).all(axis=1).all()
            and window.high.ge(window[["open", "close"]].max(axis=1)).all()
            and window.low.le(window[["open", "close"]].min(axis=1)).all()
            and window.high.ge(window.low).all()
        )
        variation = float("nan")
        if valid:
            close = window.close.to_numpy(float)
            variation = float(np.square(np.diff(np.log(close))).sum())
            valid = math.isfinite(variation) and variation > 0
        selected = trades[
            trades.created_at.ge(decision - pd.Timedelta(hours=6))
            & trades.created_at.lt(decision)
        ] if not trades.empty else trades
        buy = float(selected.loc[selected.side.eq("BUY"), "notional"].sum()) if len(selected) else 0.0
        sell = float(selected.loc[selected.side.eq("SELL"), "notional"].sum()) if len(selected) else 0.0
        deleveraged = float(selected.loc[selected.type.eq("DELEVERAGED"), "notional"].sum()) if len(selected) else 0.0
        rows.append({
            "decision_time": decision,
            "source_valid": valid,
            "forced_buy_notional": buy,
            "forced_sell_notional": sell,
            "forced_notional": buy + sell,
            "forced_flow": buy - sell,
            "deleveraged_notional": deleveraged,
            "deleveraged_state": deleveraged > 0,
            "realized_variation": variation,
        })
    panel = pd.DataFrame(rows)
    panel["forced_notional_rank"] = strict_prior_midrank(
        panel.forced_notional.where(panel.source_valid)
    )
    panel["realized_variation_rank"] = strict_prior_midrank(
        panel.realized_variation.where(panel.source_valid)
    )
    return panel


def candidate_clock(panel: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    frame = panel.copy()
    forced_rank = frame.forced_notional_rank
    variation_rank = frame.realized_variation_rank
    forced_flow = frame.forced_flow
    deleveraged = frame.deleveraged_state
    valid = frame.source_valid
    feature_available_time = frame.decision_time
    if control == "one_hour_stale_features":
        forced_rank = forced_rank.shift(1)
        variation_rank = variation_rank.shift(1)
        forced_flow = forced_flow.shift(1)
        deleveraged = deleveraged.shift(1, fill_value=False)
        valid = valid.shift(1, fill_value=False)
        feature_available_time = frame.decision_time - pd.Timedelta(hours=1)
    volatility_gate = pd.Series(True, index=frame.index) if control == "no_volatility_gate" else variation_rank.ge(0.65)
    forced_gate = pd.Series(True, index=frame.index) if control == "no_forced_notional_gate" else forced_rank.ge(0.80)
    eligible_state = valid & forced_flow.ne(0) & forced_gate & volatility_gate
    onset = eligible_state & ~eligible_state.shift(1, fill_value=False)
    flow_sign = pd.Series(np.where(forced_flow.gt(0), 1, -1), index=frame.index)
    side = pd.Series(np.where(deleveraged, flow_sign, -flow_sign), index=frame.index)
    if control == "always_fade_forced_flow":
        side = -flow_sign
    elif control == "direction_flip":
        side = -side
    elif control == "forced_long":
        side = pd.Series(1, index=frame.index)
    rows: list[dict[str, Any]] = []
    reserved_until: pd.Timestamp | None = None
    for index in frame.index[onset]:
        decision = pd.Timestamp(frame.at[index, "decision_time"])
        entry = decision + pd.Timedelta(minutes=5)
        exit_ = entry + pd.Timedelta(hours=8)
        if reserved_until is not None and entry < reserved_until:
            continue
        split = next(
            (name for name, (start, end) in SPLITS.items() if entry >= start and exit_ <= end),
            None,
        )
        if split is None:
            continue
        reserved_until = exit_
        rows.append({
            "candidate": "HVDADH-8",
            "control": control,
            "split": split,
            "decision_time": decision,
            "feature_available_time": pd.Timestamp(feature_available_time.at[index]),
            "entry_time": entry,
            "exit_time": exit_,
            "side": int(side.at[index]),
            "forced_buy_notional": float(frame.at[index, "forced_buy_notional"]),
            "forced_sell_notional": float(frame.at[index, "forced_sell_notional"]),
            "forced_notional": float(frame.at[index, "forced_notional"]),
            "forced_flow": float(forced_flow.at[index]),
            "deleveraged_notional": float(frame.at[index, "deleveraged_notional"]),
            "deleveraged_state": bool(deleveraged.at[index]),
            "forced_notional_rank": float(forced_rank.at[index]),
            "realized_variation": float(frame.at[index, "realized_variation"]),
            "realized_variation_rank": float(variation_rank.at[index]),
        })
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def stats(candidate: pd.DataFrame, split: str) -> dict[str, float | int]:
    subset = candidate[candidate.split.eq(split)]
    if subset.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "deleveraged_events": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs = int(subset.side.eq(1).sum())
    shorts = int(subset.side.eq(-1).sum())
    months = subset.entry_time.dt.strftime("%Y-%m").value_counts()
    return {
        "events": len(subset),
        "longs": longs,
        "shorts": shorts,
        "deleveraged_events": int(subset.deleveraged_state.sum()),
        "minority_side_share": min(longs, shorts) / len(subset),
        "max_month_share": int(months.max()) / len(subset),
    }


def run() -> dict[str, Any]:
    from sqlalchemy import text

    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA256:
        raise RuntimeError("HVDADH preregistration artifact drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    if registration != prereg.build():
        raise RuntimeError("HVDADH preregistration payload drift")
    forced, page_manifest, openapi = download_forced_trades()
    database = postgres_engine()
    with database.connect() as connection:
        bars = pd.read_sql_query(
            text(QUERY), connection,
            params={"start": BAR_START.to_pydatetime(), "end": END.to_pydatetime()},
        )
    database.dispose()
    panel = feature_panel(forced, bars)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(forced, FORCED_TRADES)
    PAGE_MANIFEST.write_text(json.dumps(page_manifest, indent=2, allow_nan=False) + "\n")
    _write_gzip_csv(panel, PANEL)
    source_core = {
        "protocol_version": "hvdadh_8_source_materialization_v1",
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA256, "manifest_hash": registration["manifest_hash"]},
        "official_indexer": API,
        "official_openapi": OPENAPI,
        "openapi_sha256": hashlib.sha256(openapi).hexdigest(),
        "window": [START.isoformat(), END.isoformat()],
        "page_manifest": {"path": str(PAGE_MANIFEST), "sha256": sha(PAGE_MANIFEST), "pages": page_manifest["page_count"], "all_trade_ids_seen": page_manifest["all_trade_ids_seen"]},
        "forced_trades": {"path": str(FORCED_TRADES), "sha256": sha(FORCED_TRADES), "rows": len(forced)},
        "bars_query": QUERY,
        "bars_rows": len(bars),
        "panel": {"path": str(PANEL), "sha256": sha(PANEL), "rows": len(panel), "valid_rows": int(panel.source_valid.sum())},
        "no_imputation": True,
        "oos_postentry_return_pnl_execution_price_opened": False,
        "gross9_rows_opened": False,
    }
    source = {**source_core, "manifest_hash": canonical_hash(source_core)}
    SOURCE_MANIFEST.write_text(json.dumps(source, indent=2, allow_nan=False) + "\n")
    primary = candidate_clock(panel)
    controls = {name: candidate_clock(panel, name) for name in CONTROLS}
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(primary, CLOCK)
    for name, candidate in controls.items():
        _write_gzip_csv(candidate, CONTROL_DIR / f"{name}.csv.gz")
    support = {name: stats(primary, name) for name in SPLITS}
    checks: dict[str, bool] = {}
    for name, values in support.items():
        checks[f"{name}_minimum_events"] = values["events"] >= MINIMUM[name]
        checks[f"{name}_side_balance"] = values["minority_side_share"] >= 0.20
        checks[f"{name}_month_concentration"] = values["max_month_share"] <= 0.45
    passed = all(checks.values())
    core = {
        "protocol_version": "hvdadh_8_oos_source_support_v1",
        "policy_id": "HVDADH-8",
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA256, "manifest_hash": registration["manifest_hash"]},
        "source_manifest": {"path": str(SOURCE_MANIFEST), "sha256": sha(SOURCE_MANIFEST), "manifest_hash": source["manifest_hash"]},
        "completed_preentry_sources_opened": True,
        "oos_postentry_return_pnl_execution_price_opened": False,
        "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(candidate), "promotion_authorized": False} for name, candidate in controls.items()},
        "support": support,
        "support_checks": checks,
        "support_passed": passed,
        "advance_to_gross9_novelty": passed,
        "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    result = run()
    print(json.dumps({"passed": result["support_passed"], "support": result["support"]}, indent=2))
