"""Open source-only OOS incidence for preregistered HVSSIM-24."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_sha256_settlement_intensity_migration_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


ENV_FILE = "/home/pakchu/rllm/.env"
START = pd.Timestamp("2023-01-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
PREREG_SHA256 = "c5d0c02e1748cf7f9e5ff3cc824bf0165ef73f474c1fe59b5c5c13b193a6cdd0"
API = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
SOURCE_DIR = Path("data/high_volatility_sha256_settlement_intensity_migration_relay_sources_2023_2026")
PAIR_PANEL = SOURCE_DIR / "btc_bch_settlement_intensity_daily.csv.gz"
FEATURE_PANEL = SOURCE_DIR / "daily_feature_panel.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/high_volatility_sha256_settlement_intensity_migration_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_sha256_settlement_intensity_migration_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_sha256_settlement_intensity_migration_relay_support_2026-08-10.json")
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
CONTROLS = (
    "no_volatility_gate",
    "no_migration_gate",
    "btc_settlement_intensity_change_only",
    "one_observation_stale_features",
    "direction_flip",
    "forced_long",
)
CLOCK_COLUMNS = (
    "candidate", "control", "split", "observation_time", "feature_available_time",
    "decision_time", "entry_time", "exit_time", "side", "btc_tx_count", "btc_active_addresses",
    "bch_tx_count", "bch_active_addresses", "btc_settlement_intensity", "bch_settlement_intensity",
    "relative_settlement_intensity", "intensity_migration", "absolute_migration_rank",
    "btc_settlement_intensity_change", "realized_variation", "realized_variation_rank",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def strict_prior_midrank(
    values: pd.Series, lookback: int = 365, minimum: int = 180
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


def http_json(url: str) -> tuple[dict[str, Any], str]:
    request = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": "rllm-source-audit/1"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 200:
            raise RuntimeError(f"Coin Metrics HTTP status {response.status}")
        raw = response.read()
    payload = json.loads(raw)
    if not isinstance(payload, dict) or "error" in payload:
        raise RuntimeError(f"Coin Metrics payload error: {payload.get('error')}")
    return payload, hashlib.sha256(raw).hexdigest()


def source_url() -> str:
    return API + "?" + urllib.parse.urlencode({
        "assets": "btc,bch",
        "metrics": "TxCnt,AdrActCnt,AssetEODCompletionTime",
        "frequency": "1d",
        "start_time": START.strftime("%Y-%m-%d"),
        "end_time": (END - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        "page_size": 10000,
    })


def parse_source_row(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != {
        "asset", "time", "TxCnt", "AdrActCnt", "AssetEODCompletionTime"
    }:
        raise ValueError("Coin Metrics row schema invalid")
    asset = raw["asset"]
    if asset not in {"btc", "bch"}:
        raise ValueError("Coin Metrics asset invalid")
    observation = pd.Timestamp(raw["time"])
    available = pd.Timestamp(int(raw["AssetEODCompletionTime"]), unit="s", tz="UTC")
    tx_count = float(raw["TxCnt"])
    active_addresses = float(raw["AdrActCnt"])
    if (
        observation.tzinfo is None or observation != observation.floor("1d")
        or not math.isfinite(tx_count) or tx_count <= 0
        or not math.isfinite(active_addresses) or active_addresses <= 0
        or available < observation + pd.Timedelta(days=1)
    ):
        raise ValueError("Coin Metrics row value invalid")
    return {
        "asset": asset,
        "observation_time": observation.tz_convert("UTC"),
        "available_at": available,
        "tx_count": tx_count,
        "active_addresses": active_addresses,
    }


def download_pair_panel() -> tuple[pd.DataFrame, dict[str, Any]]:
    first = source_url()
    url: str | None = first
    seen_urls: set[str] = set()
    rows: list[dict[str, Any]] = []
    pages: list[dict[str, Any]] = []
    while url:
        if url in seen_urls:
            raise RuntimeError("Coin Metrics pagination loop")
        seen_urls.add(url)
        payload, digest = http_json(url)
        raw_rows = payload.get("data")
        if not isinstance(raw_rows, list):
            raise ValueError("Coin Metrics data field invalid")
        rows.extend(parse_source_row(row) for row in raw_rows)
        pages.append({"url": url, "rows": len(raw_rows), "response_sha256": digest})
        next_url = payload.get("next_page_url")
        if next_url is None:
            url = None
        elif isinstance(next_url, str):
            url = urllib.parse.urljoin(first, next_url)
        else:
            raise ValueError("Coin Metrics next_page_url invalid")
    frame = pd.DataFrame(rows)
    if frame.duplicated(["asset", "observation_time"], keep=False).any():
        raise RuntimeError("Coin Metrics duplicate asset/day")
    expected = pd.date_range(START, END, freq="1d", inclusive="left")
    by_asset: dict[str, pd.DataFrame] = {}
    for asset in ("btc", "bch"):
        subset = frame[frame.asset.eq(asset)].set_index("observation_time").sort_index()
        if not subset.index.equals(expected):
            raise RuntimeError(f"Coin Metrics incomplete {asset} daily grid")
        by_asset[asset] = subset
    pair = pd.DataFrame({
        "observation_time": expected,
        "btc_available_at": by_asset["btc"].available_at.to_numpy(),
        "bch_available_at": by_asset["bch"].available_at.to_numpy(),
        "btc_tx_count": by_asset["btc"].tx_count.to_numpy(float),
        "btc_active_addresses": by_asset["btc"].active_addresses.to_numpy(float),
        "bch_tx_count": by_asset["bch"].tx_count.to_numpy(float),
        "bch_active_addresses": by_asset["bch"].active_addresses.to_numpy(float),
    })
    pair["feature_available_time"] = pair[["btc_available_at", "bch_available_at"]].max(axis=1)
    pair["decision_time"] = pair.feature_available_time.dt.ceil("5min")
    return pair, {"source_url": first, "pages": pages, "page_count": len(pages), "raw_rows": len(frame)}


def feature_panel(pair: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    market = bars.copy()
    market["ts"] = pd.to_datetime(market.ts, utc=True)
    for column in ("open", "high", "low", "close"):
        market[column] = pd.to_numeric(market[column], errors="coerce")
    market = market.drop_duplicates("ts", keep=False).set_index("ts").sort_index()
    frame = pair.copy()
    frame["btc_settlement_intensity"] = frame.btc_tx_count / frame.btc_active_addresses
    frame["bch_settlement_intensity"] = frame.bch_tx_count / frame.bch_active_addresses
    positive_intensity = frame.btc_settlement_intensity.gt(0) & frame.bch_settlement_intensity.gt(0)
    frame["relative_settlement_intensity"] = np.log(
        frame.btc_settlement_intensity.where(positive_intensity)
        / frame.bch_settlement_intensity.where(positive_intensity)
    )
    frame["intensity_migration"] = frame.relative_settlement_intensity.diff(3)
    frame["btc_settlement_intensity_change"] = np.log(
        frame.btc_settlement_intensity.where(frame.btc_settlement_intensity.gt(0))
    ).diff(3)
    variations: list[float] = []
    valid: list[bool] = []
    for observation in frame.observation_time:
        expected = pd.date_range(observation, observation + pd.Timedelta(days=1), freq="1min", inclusive="left")
        window = market.reindex(expected)
        ohlc = window[["open", "high", "low", "close"]]
        good = bool(
            len(window) == 1440
            and np.isfinite(ohlc).all(axis=1).all()
            and ohlc.gt(0).all(axis=1).all()
            and window.high.ge(window[["open", "close"]].max(axis=1)).all()
            and window.low.le(window[["open", "close"]].min(axis=1)).all()
            and window.high.ge(window.low).all()
        )
        variation = float("nan")
        if good:
            variation = float(np.square(np.diff(np.log(window.close.to_numpy(float)))).sum())
            good = math.isfinite(variation) and variation > 0
        variations.append(variation)
        valid.append(good and bool(positive_intensity.iloc[len(valid)]))
    frame["source_valid"] = valid
    frame["realized_variation"] = variations
    frame["absolute_migration_rank"] = strict_prior_midrank(
        frame.intensity_migration.abs().where(frame.source_valid)
    )
    frame["realized_variation_rank"] = strict_prior_midrank(
        frame.realized_variation.where(frame.source_valid)
    )
    return frame


def candidate_clock(panel: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    frame = panel.copy()
    migration = frame.intensity_migration
    migration_rank = frame.absolute_migration_rank
    variation_rank = frame.realized_variation_rank
    available = frame.feature_available_time
    decision = frame.decision_time
    valid = frame.source_valid
    btc_change = frame.btc_settlement_intensity_change
    if control == "one_observation_stale_features":
        migration = migration.shift(1)
        migration_rank = migration_rank.shift(1)
        variation_rank = variation_rank.shift(1)
        valid = valid.shift(1, fill_value=False)
        btc_change = btc_change.shift(1)
        available = frame.feature_available_time.shift(1)
    volatility_gate = pd.Series(True, index=frame.index) if control == "no_volatility_gate" else variation_rank.ge(0.65)
    migration_gate = pd.Series(True, index=frame.index) if control == "no_migration_gate" else migration_rank.ge(0.80)
    side_source = btc_change if control == "btc_settlement_intensity_change_only" else migration
    eligible_state = valid & side_source.ne(0) & side_source.notna() & migration_gate & volatility_gate
    onset = eligible_state & ~eligible_state.shift(1, fill_value=False)
    side = pd.Series(np.where(side_source.gt(0), 1, -1), index=frame.index)
    if control == "direction_flip":
        side = -side
    elif control == "forced_long":
        side = pd.Series(1, index=frame.index)
    rows: list[dict[str, Any]] = []
    reserved_until: pd.Timestamp | None = None
    for index in frame.index[onset]:
        decision_time = pd.Timestamp(decision.at[index])
        entry = decision_time + pd.Timedelta(minutes=5)
        exit_ = entry + pd.Timedelta(hours=24)
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
            "candidate": "HVSSIM-24",
            "control": control,
            "split": split,
            "observation_time": frame.at[index, "observation_time"],
            "feature_available_time": available.at[index],
            "decision_time": decision_time,
            "entry_time": entry,
            "exit_time": exit_,
            "side": int(side.at[index]),
            "btc_tx_count": float(frame.at[index, "btc_tx_count"]),
            "btc_active_addresses": float(frame.at[index, "btc_active_addresses"]),
            "bch_tx_count": float(frame.at[index, "bch_tx_count"]),
            "bch_active_addresses": float(frame.at[index, "bch_active_addresses"]),
            "btc_settlement_intensity": float(frame.at[index, "btc_settlement_intensity"]),
            "bch_settlement_intensity": float(frame.at[index, "bch_settlement_intensity"]),
            "relative_settlement_intensity": float(frame.at[index, "relative_settlement_intensity"]),
            "intensity_migration": float(migration.at[index]),
            "absolute_migration_rank": float(migration_rank.at[index]),
            "btc_settlement_intensity_change": float(btc_change.at[index]),
            "realized_variation": float(frame.at[index, "realized_variation"]),
            "realized_variation_rank": float(variation_rank.at[index]),
        })
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def stats(candidate: pd.DataFrame, split: str) -> dict[str, float | int]:
    subset = candidate[candidate.split.eq(split)]
    if subset.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs = int(subset.side.eq(1).sum())
    shorts = int(subset.side.eq(-1).sum())
    months = subset.entry_time.dt.strftime("%Y-%m").value_counts()
    return {
        "events": len(subset),
        "longs": longs,
        "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(subset),
        "max_month_share": int(months.max()) / len(subset),
    }


def run() -> dict[str, Any]:
    from sqlalchemy import text

    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA256:
        raise RuntimeError("HVSSIM preregistration artifact drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    if registration != prereg.build():
        raise RuntimeError("HVSSIM preregistration payload drift")
    pair, transport = download_pair_panel()
    database = postgres_engine()
    with database.connect() as connection:
        bars = pd.read_sql_query(
            text(QUERY), connection,
            params={"start": START.to_pydatetime(), "end": END.to_pydatetime()},
        )
    database.dispose()
    panel = feature_panel(pair, bars)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(pair, PAIR_PANEL)
    _write_gzip_csv(panel, FEATURE_PANEL)
    source_core = {
        "protocol_version": "hvssim_24_source_materialization_v1",
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA256, "manifest_hash": registration["manifest_hash"]},
        "transport": transport,
        "pair_panel": {"path": str(PAIR_PANEL), "sha256": sha(PAIR_PANEL), "rows": len(pair)},
        "bars_query": QUERY,
        "bars_rows": len(bars),
        "feature_panel": {"path": str(FEATURE_PANEL), "sha256": sha(FEATURE_PANEL), "rows": len(panel), "valid_rows": int(panel.source_valid.sum())},
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
        "protocol_version": "hvssim_24_oos_source_support_v1",
        "policy_id": "HVSSIM-24",
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
