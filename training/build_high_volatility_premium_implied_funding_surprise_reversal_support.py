"""Materialize source-only HVPIFSR-8 clocks from completed settlement inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import preregister_high_volatility_premium_implied_funding_surprise_reversal as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


ENV = Path("/home/pakchu/rllm/.env")
PREREG_SHA = "1a030336273190e8d0d0460763e7566c0ccdc990265b257a3107ea36f12609ac"
START = pd.Timestamp("2023-01-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
SOURCE_DIR = Path("data/high_volatility_premium_implied_funding_surprise_reversal_sources_2023_2026")
PANEL = SOURCE_DIR / "settlement_states.csv.gz"
MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/high_volatility_premium_implied_funding_surprise_reversal_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_premium_implied_funding_surprise_reversal_controls_2023_2026")
RESULT = Path("results/high_volatility_premium_implied_funding_surprise_reversal_support_2026-08-13.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), END),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_surprise_tail",
    "no_variation_gate",
    "unweighted_premium_mean",
    "one_settlement_stale_surprise",
    "direction_flip",
    "forced_long",
)
COLUMNS = (
    "candidate", "control", "split", "decision_time", "feature_available_time",
    "entry_time", "exit_time", "side", "funding_rate", "premium_average",
    "implied_funding_proxy", "funding_surprise", "surprise_rank", "btc_variation",
    "variation_rank",
)
FUNDING_QUERY = """
SELECT funding_time AS decision_time,funding_rate
FROM funding_rates_binance
WHERE symbol='BTCUSDT' AND funding_time>=:start AND funding_time<:end
ORDER BY funding_time
"""
PREMIUM_QUERY = """
SELECT ts,open,high,low,close
FROM bars_binance_premium
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
ORDER BY ts
"""
PRICE_QUERY = """
SELECT ts,open,high,low,close
FROM bars_binance
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
ORDER BY ts
"""


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_prior_midrank(values: pd.Series, maximum: int = 270, minimum: int = 180) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    output = pd.Series(np.nan, index=values.index, dtype=float)
    history: list[float] = []
    for index, value in values.items():
        prior = np.asarray(history[-maximum:], dtype=float)
        if np.isfinite(value) and len(prior) >= minimum:
            output.at[index] = ((prior < value).sum() + 0.5 * (prior == value).sum()) / len(prior)
        if np.isfinite(value):
            history.append(float(value))
    return output


def implied_funding(premium_average: float) -> float:
    return float(premium_average + np.clip(0.0001 - premium_average, -0.0005, 0.0005))


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from sqlalchemy import text
    from preprocessing.live_db_features import sqlalchemy_engine_from_env

    engine = sqlalchemy_engine_from_env(ENV)
    params = {"start": (START - pd.Timedelta(days=1)).to_pydatetime(), "end": END.to_pydatetime()}
    try:
        with engine.connect() as connection:
            funding = pd.read_sql_query(
                text(FUNDING_QUERY), connection,
                params={"start": START.to_pydatetime(), "end": END.to_pydatetime()},
            )
            premium = pd.read_sql_query(text(PREMIUM_QUERY), connection, params=params)
            price = pd.read_sql_query(text(PRICE_QUERY), connection, params=params)
    finally:
        engine.dispose()
    funding["decision_time"] = pd.to_datetime(funding.decision_time, utc=True)
    funding["funding_rate"] = pd.to_numeric(funding.funding_rate, errors="coerce")
    if (
        funding.decision_time.duplicated().any()
        or not funding.decision_time.is_monotonic_increasing
        or not np.isfinite(funding.funding_rate).all()
    ):
        raise RuntimeError("HVPIFSR funding source drift")
    return funding, normalize_bars(premium, signed=True), normalize_bars(price, signed=False)


def normalize_bars(frame: pd.DataFrame, *, signed: bool) -> pd.DataFrame:
    result = frame.copy()
    result["ts"] = pd.to_datetime(result.ts, utc=True)
    for column in ("open", "high", "low", "close"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result.ts.duplicated().any() or not result.ts.is_monotonic_increasing:
        raise RuntimeError("HVPIFSR bar timestamp drift")
    finite = np.isfinite(result[["open", "high", "low", "close"]]).all(axis=1)
    coherent = result.high.ge(result[["open", "close"]].max(axis=1)) & result.low.le(
        result[["open", "close"]].min(axis=1)
    ) & result.high.ge(result.low)
    positive = pd.Series(True, index=result.index) if signed else result[["open", "high", "low", "close"]].gt(0).all(axis=1)
    result["valid"] = finite & coherent & positive
    return result.set_index("ts")


def build_panel(funding: pd.DataFrame, premium: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    weights = np.arange(1.0, 481.0)
    for settlement in funding.itertuples(index=False):
        decision = pd.Timestamp(settlement.decision_time)
        premium_grid = pd.date_range(decision - pd.Timedelta(hours=8), decision, freq="1min", inclusive="left")
        price_grid = pd.date_range(decision - pd.Timedelta(days=1), decision, freq="1min", inclusive="left")
        premium_window = premium.reindex(premium_grid)
        price_window = price.reindex(price_grid)
        valid = bool(
            len(premium_window) == 480
            and premium_window.valid.fillna(False).all()
            and len(price_window) == 1440
            and price_window.valid.fillna(False).all()
        )
        weighted_average = unweighted_average = variation = float("nan")
        if valid:
            premium_close = premium_window.close.to_numpy(float)
            price_close = price_window.close.to_numpy(float)
            weighted_average = float(np.average(premium_close, weights=weights))
            unweighted_average = float(np.mean(premium_close))
            variation = float(np.sqrt(np.square(np.diff(np.log(price_close))).sum()))
            valid = bool(np.isfinite([weighted_average, unweighted_average, variation]).all() and variation > 0)
        weighted_implied = implied_funding(weighted_average) if valid else float("nan")
        unweighted_implied = implied_funding(unweighted_average) if valid else float("nan")
        rows.append({
            "decision_time": decision,
            "funding_rate": float(settlement.funding_rate),
            "premium_average": weighted_average,
            "unweighted_premium_average": unweighted_average,
            "implied_funding_proxy": weighted_implied,
            "unweighted_implied_funding_proxy": unweighted_implied,
            "funding_surprise": float(settlement.funding_rate) - weighted_implied,
            "unweighted_funding_surprise": float(settlement.funding_rate) - unweighted_implied,
            "btc_variation": variation,
            "base_source_valid": valid,
        })
    panel = pd.DataFrame(rows)
    panel["surprise_rank"] = strict_prior_midrank(panel.funding_surprise.abs().where(panel.base_source_valid))
    panel["unweighted_surprise_rank"] = strict_prior_midrank(
        panel.unweighted_funding_surprise.abs().where(panel.base_source_valid)
    )
    panel["variation_rank"] = strict_prior_midrank(panel.btc_variation.where(panel.base_source_valid))
    panel["source_valid"] = (
        panel.base_source_valid
        & np.isfinite(panel[["funding_surprise", "surprise_rank", "btc_variation", "variation_rank"]]).all(axis=1)
        & panel.funding_surprise.ne(0)
    )
    return panel


def candidate_clock(panel: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    used = panel.shift(1) if control == "one_settlement_stale_surprise" else panel
    surprise_column = "unweighted_funding_surprise" if control == "unweighted_premium_mean" else "funding_surprise"
    rank_column = "unweighted_surprise_rank" if control == "unweighted_premium_mean" else "surprise_rank"
    surprise = used[surprise_column]
    surprise_rank = used[rank_column]
    valid = used.source_valid.eq(True) & surprise.notna() & surprise.ne(0) & surprise_rank.notna()
    eligible = valid.copy()
    if control != "no_surprise_tail":
        eligible &= surprise_rank.ge(0.75)
    if control != "no_variation_gate":
        eligible &= panel.variation_rank.ge(0.65)
    onset = eligible & ~eligible.shift(1, fill_value=False) & valid.shift(1, fill_value=False)
    side = -np.sign(surprise).fillna(0).astype(int)
    if control == "direction_flip":
        side = -side
    elif control == "forced_long":
        side = pd.Series(1, index=panel.index)
    rows: list[dict[str, Any]] = []
    reserved_until: pd.Timestamp | None = None
    for index in panel.index[onset]:
        decision = pd.Timestamp(panel.at[index, "decision_time"])
        entry = decision + pd.Timedelta(minutes=5)
        exit_ = entry + pd.Timedelta(hours=8)
        if reserved_until is not None and entry < reserved_until:
            continue
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_ <= end), None)
        if split is None:
            continue
        reserved_until = exit_
        rows.append({
            "candidate": prereg.POLICY_ID, "control": control, "split": split,
            "decision_time": decision, "feature_available_time": decision,
            "entry_time": entry, "exit_time": exit_, "side": int(side.at[index]),
            "funding_rate": float(used.at[index, "funding_rate"]),
            "premium_average": float(used.at[index, "premium_average"]),
            "implied_funding_proxy": float(used.at[index, "implied_funding_proxy"]),
            "funding_surprise": float(surprise.at[index]),
            "surprise_rank": float(surprise_rank.at[index]),
            "btc_variation": float(panel.at[index, "btc_variation"]),
            "variation_rank": float(panel.at[index, "variation_rank"]),
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def stats(clock: pd.DataFrame, split: str) -> dict[str, float | int]:
    subset = clock[clock.split.eq(split)]
    if subset.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs = int(subset.side.eq(1).sum())
    shorts = int(subset.side.eq(-1).sum())
    months = pd.to_datetime(subset.entry_time, utc=True).dt.strftime("%Y-%m").value_counts()
    return {
        "events": len(subset), "longs": longs, "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(subset),
        "max_month_share": int(months.max()) / len(subset),
    }


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVPIFSR preregistration drift")
    funding, premium, price = load_sources()
    panel = build_panel(funding, premium, price)
    primary = candidate_clock(panel)
    controls = {name: candidate_clock(panel, name) for name in CONTROLS}
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(panel, PANEL)
    _write_gzip_csv(primary, CLOCK)
    for name, control_clock in controls.items():
        _write_gzip_csv(control_clock, CONTROL_DIR / f"{name}.csv.gz")
    source_core = {
        "protocol_version": "hvpifsr_8_source_v1",
        "queries_sha256": {
            "funding": hashlib.sha256(FUNDING_QUERY.encode()).hexdigest(),
            "premium": hashlib.sha256(PREMIUM_QUERY.encode()).hexdigest(),
            "price": hashlib.sha256(PRICE_QUERY.encode()).hexdigest(),
        },
        "rows": {"funding": len(funding), "premium": len(premium), "price": len(price)},
        "output": {"path": str(PANEL), "sha256": sha(PANEL), "rows": len(panel)},
        "candidate_incidence_opened": False,
        "postentry_outcomes_opened": False,
    }
    manifest = {**source_core, "manifest_hash": prereg.canonical_hash(source_core)}
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    support = {split: stats(primary, split) for split in SPLITS}
    checks = {
        key: value
        for split, item in support.items()
        for key, value in (
            (f"{split}_minimum_events", item["events"] >= MINIMUM[split]),
            (f"{split}_side_balance", item["minority_side_share"] >= 0.2),
            (f"{split}_month_concentration", item["max_month_share"] <= 0.45),
        )
    }
    passed = all(checks.values())
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    core = {
        "protocol_version": "hvpifsr_8_source_support_v1",
        "policy_id": prereg.POLICY_ID,
        "preregistration": {
            "path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA,
            "manifest_hash": registration["manifest_hash"],
        },
        "source_manifest": {
            "path": str(MANIFEST), "sha256": sha(MANIFEST),
            "manifest_hash": manifest["manifest_hash"],
        },
        "completed_preentry_sources_opened": True,
        "candidate_incidence_opened": True,
        "postentry_return_pnl_execution_price_opened": False,
        "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "controls": {
            name: {
                "path": str(CONTROL_DIR / f"{name}.csv.gz"),
                "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"),
                "rows": len(control_clock), "promotion_authorized": False,
            }
            for name, control_clock in controls.items()
        },
        "support": support,
        "support_checks": checks,
        "support_passed": passed,
        "advance_to_gross9_novelty": passed,
        "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": prereg.canonical_hash(core)}
    RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    output = run()
    print(json.dumps({"passed": output["support_passed"], "support": output["support"]}, indent=2))
