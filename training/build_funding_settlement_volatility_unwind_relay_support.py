"""Build source-support clocks for FSVUR-6 without post-entry outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import build_options_crowding_deleveraging_relay_support_v4 as base
from training import build_options_led_intrahour_absorption_support as intrahour
from training import evaluate_options_led_volatility_expansion_premium_relay_economics_v5 as engine
from training import preregister_funding_settlement_volatility_unwind_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


CLOCK = Path("data/funding_settlement_volatility_unwind_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/funding_settlement_volatility_unwind_relay_controls_2023_2026")
RESULT = Path("results/funding_settlement_volatility_unwind_relay_support_2026-08-08.json")
PRICE_FILE = intrahour.PRICE_DIR / "btc_intrahour_path.csv.gz"
ENV_FILE = "/home/pakchu/rllm/.env"
SPLITS = base.SPLITS
MINIMUM_EVENTS = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_funding_extreme",
    "no_pre_return_extreme",
    "no_joint_vol_contraction",
    "no_settlement_reversal",
    "direction_flip",
)
ECONOMIC_OUTCOMES_AUTHORIZED = False
CLOCK_COLUMNS = (
    "candidate",
    "control",
    "split",
    "settlement_time",
    "decision_time",
    "feature_available_time",
    "entry_time",
    "exit_time",
    "side",
    "funding_rate",
    "prior_abs_funding_q60",
    "pre_settlement_return_8h",
    "prior_abs_pre_return_q60",
    "post_settlement_return_1h",
    "bvol_body",
    "dvol_body",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def postgres_funding(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    from sqlalchemy import create_engine, text
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(ENV_FILE)
    database = create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})
    query = text(
        "SELECT funding_time AS date,funding_rate FROM funding_rates_binance "
        "WHERE symbol=:symbol AND funding_time>=:start AND funding_time<:end ORDER BY funding_time"
    )
    with database.connect() as connection:
        frame = pd.read_sql_query(
            query,
            connection,
            params={"symbol": "BTCUSDT", "start": start.to_pydatetime(), "end": end.to_pydatetime()},
        )
    database.dispose()
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame["funding_rate"] = pd.to_numeric(frame["funding_rate"], errors="raise")
    return frame


def funding_source() -> pd.DataFrame:
    history_start = pd.Timestamp("2023-01-01", tz="UTC")
    boundary = pd.Timestamp("2024-01-01", tz="UTC")
    end = pd.Timestamp("2026-08-01", tz="UTC")
    frozen = engine.load_train_funding(history_start, boundary)[["date", "funding_rate"]]
    live = postgres_funding(boundary, end)
    funding = pd.concat([frozen, live], ignore_index=True).sort_values("date").reset_index(drop=True)
    if funding.empty or funding["date"].duplicated().any() or not funding["date"].is_monotonic_increasing:
        raise RuntimeError("funding settlement source clock invalid")
    if not np.isfinite(funding["funding_rate"]).all():
        raise RuntimeError("funding settlement source contains nonfinite rates")
    return funding


def features() -> pd.DataFrame:
    volatility = intrahour.features().copy()
    volatility = volatility.drop(columns=["funding_rate", "price_valid"], errors="ignore")
    price = pd.read_csv(PRICE_FILE, compression="gzip")
    price["decision_time"] = pd.to_datetime(price["decision_time"], utc=True, format="mixed")
    for column in ("hour_open", "hour_close"):
        price[column] = pd.to_numeric(price[column], errors="coerce")
    price["price_valid"] = price["source_valid"].astype(str).str.lower().eq("true")
    price = price.sort_values("decision_time").reset_index(drop=True)
    price["pre_open_8h"] = price["hour_open"].shift(8)
    price["pre_valid_8h"] = price["price_valid"].shift(8, fill_value=False)
    price["pre_settlement_return_8h"] = price["hour_open"] / price["pre_open_8h"] - 1.0
    price["post_settlement_return_1h"] = price["hour_close"] / price["hour_open"] - 1.0

    joined = volatility.merge(
        price[
            [
                "decision_time",
                "price_valid",
                "pre_valid_8h",
                "pre_settlement_return_8h",
                "post_settlement_return_1h",
            ]
        ],
        on="decision_time",
        validate="one_to_one",
    )
    joined["settlement_time"] = joined["decision_time"] - pd.Timedelta(hours=1)
    joined = joined.merge(
        funding_source().rename(columns={"date": "settlement_time"}),
        on="settlement_time",
        how="inner",
        validate="one_to_one",
    ).sort_values("settlement_time").reset_index(drop=True)
    joined["prior_abs_funding_q60"] = (
        joined["funding_rate"].abs().shift(1).rolling(270, min_periods=252).quantile(0.60)
    )
    joined["prior_abs_pre_return_q60"] = (
        joined["pre_settlement_return_8h"].abs().where(joined["price_valid"] & joined["pre_valid_8h"])
        .shift(1).rolling(270, min_periods=252).quantile(0.60)
    )
    source_columns = [
        "funding_rate",
        "pre_settlement_return_8h",
        "post_settlement_return_1h",
        "bvol_body",
        "dvol_body",
    ]
    joined["base_valid"] = (
        joined["base_valid"]
        & joined["price_valid"]
        & joined["pre_valid_8h"]
        & np.isfinite(joined[source_columns]).all(axis=1)
    )
    return joined


def build_clock(frame: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    funding_gate = frame["funding_rate"].ne(0)
    if control != "no_funding_extreme":
        funding_gate &= frame["prior_abs_funding_q60"].notna()
        funding_gate &= frame["funding_rate"].abs().ge(frame["prior_abs_funding_q60"])

    pre = frame["pre_settlement_return_8h"]
    pre_gate = pre.ne(0) & np.sign(pre).eq(np.sign(frame["funding_rate"]))
    if control != "no_pre_return_extreme":
        pre_gate &= frame["prior_abs_pre_return_q60"].notna()
        pre_gate &= pre.abs().ge(frame["prior_abs_pre_return_q60"])

    post = frame["post_settlement_return_1h"]
    reversal = post.ne(0)
    if control != "no_settlement_reversal":
        reversal &= np.sign(post).eq(-np.sign(pre))

    volatility_reset = pd.Series(True, index=frame.index)
    if control != "no_joint_vol_contraction":
        volatility_reset = frame["bvol_body"].lt(0) & frame["dvol_body"].lt(0)

    active = frame["base_valid"] & funding_gate & pre_gate & reversal & volatility_reset
    rows: list[dict[str, Any]] = []
    next_allowed: pd.Timestamp | None = None
    for index in frame.index[active]:
        settlement = pd.Timestamp(frame.at[index, "settlement_time"])
        decision = pd.Timestamp(frame.at[index, "decision_time"])
        entry = decision + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=6)
        if next_allowed is not None and entry < next_allowed:
            continue
        split = next(
            (name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end),
            None,
        )
        if split is None:
            continue
        side = int(np.sign(post.at[index]))
        if control == "direction_flip":
            side *= -1
        next_allowed = exit_time
        rows.append(
            {
                "candidate": "FSVUR-6",
                "control": control,
                "split": split,
                "settlement_time": settlement,
                "decision_time": decision,
                "feature_available_time": decision,
                "entry_time": entry,
                "exit_time": exit_time,
                "side": side,
                "funding_rate": float(frame.at[index, "funding_rate"]),
                "prior_abs_funding_q60": float(frame.at[index, "prior_abs_funding_q60"]),
                "pre_settlement_return_8h": float(pre.at[index]),
                "prior_abs_pre_return_q60": float(frame.at[index, "prior_abs_pre_return_q60"]),
                "post_settlement_return_1h": float(post.at[index]),
                "bvol_body": float(frame.at[index, "bvol_body"]),
                "dvol_body": float(frame.at[index, "dvol_body"]),
            }
        )
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def split_stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    subset = clock[clock["split"].eq(split)]
    if subset.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs = int(subset["side"].eq(1).sum())
    shorts = int(subset["side"].eq(-1).sum())
    months = subset["entry_time"].dt.strftime("%Y-%m").value_counts()
    return {
        "events": len(subset),
        "longs": longs,
        "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(subset),
        "max_month_share": int(months.max()) / len(subset),
    }


def run() -> dict[str, Any]:
    frame = features()
    primary = build_clock(frame)
    controls = {name: build_clock(frame, name) for name in CONTROLS}
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(primary, CLOCK)
    for name, control in controls.items():
        _write_gzip_csv(control, CONTROL_DIR / f"{name}.csv.gz")

    support = {name: split_stats(primary, name) for name in SPLITS}
    checks: dict[str, bool] = {}
    for name, stats in support.items():
        checks[f"{name}_minimum_events"] = stats["events"] >= MINIMUM_EVENTS[name]
        checks[f"{name}_side_balance"] = stats["minority_side_share"] >= 0.20
        checks[f"{name}_month_concentration"] = stats["max_month_share"] <= 0.45

    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    volatility_manifest = intrahour.NONPRICE_DIR / "manifest.json"
    price_manifest = intrahour.PRICE_DIR / "manifest.json"
    passed = all(checks.values())
    core = {
        "protocol_version": "fsvur_6_source_support_v1",
        "policy_id": "FSVUR-6",
        "preregistration": {
            "path": str(prereg.DEFAULT_OUTPUT),
            "sha256": sha256(prereg.DEFAULT_OUTPUT),
            "manifest_hash": registration["manifest_hash"],
        },
        "source_manifests": {
            "volatility": {"path": str(volatility_manifest), "sha256": sha256(volatility_manifest)},
            "completed_price": {"path": str(price_manifest), "sha256": sha256(price_manifest)},
            "train_funding": {"path": str(engine.TRAIN_FUNDING), "sha256": sha256(engine.TRAIN_FUNDING)},
            "later_funding": {"table": "funding_rates_binance", "symbol": "BTCUSDT"},
        },
        "completed_preentry_sources_opened": True,
        "postentry_return_pnl_execution_price_opened": False,
        "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha256(CLOCK), "rows": len(primary)},
        "controls": {
            name: {
                "path": str(CONTROL_DIR / f"{name}.csv.gz"),
                "sha256": sha256(CONTROL_DIR / f"{name}.csv.gz"),
                "rows": len(control),
            }
            for name, control in controls.items()
        },
        "support": support,
        "support_checks": checks,
        "support_passed": passed,
        "advance_to_gross9_novelty": passed,
        "advance_to_economic_outcomes": ECONOMIC_OUTCOMES_AUTHORIZED,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    report = run()
    print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2))
