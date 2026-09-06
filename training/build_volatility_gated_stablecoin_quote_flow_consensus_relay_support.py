"""Build source-support clocks for VGSQF-6 without post-entry outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import build_options_crowding_deleveraging_relay_support_v4 as volbase
from training import build_stablecoin_quote_flow_diffusion_support as flowbase
from training import preregister_volatility_gated_stablecoin_quote_flow_consensus_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


PANEL = Path(
    "data/binance_stablecoin_quote_flow_btc_2023_2026_aug/"
    "BTC_stablecoin_quote_flow_1h_2023-07-01_2026-07-31T23.csv.gz"
)
PANEL_SHA = "44374b9a2298ae4b64f0c1e7208665b1c08c8221045308694311123deae1c805"
SOURCE_MANIFEST = PANEL.parent / "build_manifest.json"
SOURCE_MANIFEST_SHA = "b9c64c3ce651934d9761a6d0731e814b2a92f5237b3040e11f794d7eb024a898"
CLOCK = Path("data/volatility_gated_stablecoin_quote_flow_consensus_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/volatility_gated_stablecoin_quote_flow_consensus_relay_controls_2023_2026")
RESULT = Path("results/volatility_gated_stablecoin_quote_flow_consensus_relay_support_2026-08-08.json")
SPLITS = volbase.SPLITS
MINIMUM_EVENTS = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_volatility_gate",
    "no_usdt_lag",
    "no_participation",
    "one_hour_stale_flow",
    "direction_flip",
)
ECONOMIC_OUTCOMES_AUTHORIZED = False
COLUMNS = (
    "candidate",
    "control",
    "split",
    "source_hour_start",
    "decision_time",
    "feature_available_time",
    "entry_time",
    "exit_time",
    "side",
    "z_usdt",
    "z_usdc",
    "z_fdusd",
    "alt_share",
    "prior_alt_share_q50",
    "bvol_close",
    "prior_bvol_q60",
    "dvol_close",
    "prior_dvol_q60",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def load_flow() -> pd.DataFrame:
    if sha(PANEL) != PANEL_SHA or sha(SOURCE_MANIFEST) != SOURCE_MANIFEST_SHA:
        raise RuntimeError("VGSQF spot-flow source drift")
    source = pd.read_csv(PANEL, compression="gzip")
    source["date"] = pd.to_datetime(source.date, utc=True, errors="raise")
    if (
        tuple(source.columns) != flowbase.SOURCE_COLUMNS
        or source[["date", "symbol"]].duplicated().any()
        or not source.source_complete.all()
    ):
        raise RuntimeError("VGSQF spot-flow schema invalid")
    return flowbase.derive_state(source)


def features() -> pd.DataFrame:
    state = load_flow()
    bvol, dvol, oi, funding = volbase.load_sources()
    vol = volbase.joined_features(bvol, dvol, oi, funding)[
        ["decision_time", "bvol_close", "dvol_close", "bvol_valid"]
    ].copy()
    for column in ("bvol_close", "dvol_close"):
        prefix = column.split("_")[0]
        vol[column] = pd.to_numeric(vol[column], errors="coerce")
        valid_level = vol[column].where(
            vol.bvol_valid & np.isfinite(vol[column]) & vol[column].gt(0)
        )
        vol[f"prior_{prefix}_q60"] = (
            valid_level.shift(1).rolling(720, min_periods=672).quantile(0.60)
        )
    frame = (
        state.merge(vol, on="decision_time", how="inner", validate="one_to_one")
        .sort_values("decision_time")
        .reset_index(drop=True)
    )
    frame["vol_valid"] = (
        frame.bvol_valid
        & np.isfinite(frame[["bvol_close", "dvol_close"]]).all(axis=1)
        & frame[["bvol_close", "dvol_close"]].gt(0).all(axis=1)
    )
    return frame


def conditions(frame: pd.DataFrame, control: str) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    flow = frame.shift(1) if control == "one_hour_stale_flow" else frame
    same_sign = (
        np.sign(flow.z_usdc).eq(np.sign(flow.z_fdusd))
        & flow.z_usdc.ne(0)
        & flow.z_fdusd.ne(0)
    )
    consensus = same_sign & flow.z_usdc.abs().ge(0.75) & flow.z_fdusd.abs().ge(0.75)
    side = np.sign(flow.z_usdc)
    usdt_lag = (
        pd.Series(True, index=frame.index)
        if control == "no_usdt_lag"
        else (side * flow.z_usdt).lt(0.50)
    )
    participation = (
        pd.Series(True, index=frame.index)
        if control == "no_participation"
        else flow.alt_share.ge(flow.prior_alt_share_q50)
    )
    volatile = (
        pd.Series(True, index=frame.index)
        if control == "no_volatility_gate"
        else frame.bvol_close.ge(frame.prior_bvol_q60)
        & frame.dvol_close.ge(frame.prior_dvol_q60)
    )
    previous = frame.shift(1)
    consecutive = frame.decision_time.diff().eq(pd.Timedelta(hours=1))
    valid = (
        frame.source_valid
        & previous.source_valid
        & frame.vol_valid
        & consecutive
        & frame.prior_bvol_q60.notna()
        & frame.prior_dvol_q60.notna()
    )
    return valid & consensus & usdt_lag & participation & volatile, side, flow


def clock(frame: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    active, side, flow = conditions(frame, control)
    onset = active & ~active.shift(1, fill_value=False)
    rows: list[dict[str, Any]] = []
    next_allowed: pd.Timestamp | None = None
    for index in frame.index[onset]:
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
        signal_side = int(side.at[index])
        if control == "direction_flip":
            signal_side = -signal_side
        next_allowed = exit_time
        source_row = flow.loc[index]
        rows.append(
            {
                "candidate": "VGSQF-6",
                "control": control,
                "split": split,
                "source_hour_start": source_row.source_hour_start,
                "decision_time": decision,
                "feature_available_time": decision,
                "entry_time": entry,
                "exit_time": exit_time,
                "side": signal_side,
                "z_usdt": float(source_row.z_usdt),
                "z_usdc": float(source_row.z_usdc),
                "z_fdusd": float(source_row.z_fdusd),
                "alt_share": float(source_row.alt_share),
                "prior_alt_share_q50": float(source_row.prior_alt_share_q50),
                "bvol_close": float(frame.at[index, "bvol_close"]),
                "prior_bvol_q60": float(frame.at[index, "prior_bvol_q60"]),
                "dvol_close": float(frame.at[index, "dvol_close"]),
                "prior_dvol_q60": float(frame.at[index, "prior_dvol_q60"]),
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def split_stats(clocks: pd.DataFrame, split: str) -> dict[str, int | float]:
    selected = clocks[clocks.split.eq(split)]
    if selected.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs = int(selected.side.eq(1).sum())
    shorts = int(selected.side.eq(-1).sum())
    monthly = selected.entry_time.dt.strftime("%Y-%m").value_counts()
    return {
        "events": len(selected),
        "longs": longs,
        "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(selected),
        "max_month_share": int(monthly.max()) / len(selected),
    }


def run() -> dict[str, Any]:
    frame = features()
    primary = clock(frame)
    controls = {name: clock(frame, name) for name in CONTROLS}
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(primary, CLOCK)
    for name, control_clocks in controls.items():
        _write_gzip_csv(control_clocks, CONTROL_DIR / f"{name}.csv.gz")

    support = {name: split_stats(primary, name) for name in SPLITS}
    checks: dict[str, bool] = {}
    for name, stats in support.items():
        checks[f"{name}_minimum_events"] = stats["events"] >= MINIMUM_EVENTS[name]
        checks[f"{name}_side_balance"] = stats["minority_side_share"] >= 0.20
        checks[f"{name}_month_concentration"] = stats["max_month_share"] <= 0.45
    preregistration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    volatility_manifest = volbase.SOURCE_DIR / "manifest.json"
    passed = all(checks.values())
    core = {
        "protocol_version": "vgsqf_6_source_support_v1",
        "policy_id": "VGSQF-6",
        "preregistration": {
            "path": str(prereg.DEFAULT_OUTPUT),
            "sha256": sha(prereg.DEFAULT_OUTPUT),
            "manifest_hash": preregistration["manifest_hash"],
        },
        "source_manifests": {
            "spot_flow": {"path": str(SOURCE_MANIFEST), "sha256": sha(SOURCE_MANIFEST)},
            "volatility": {"path": str(volatility_manifest), "sha256": sha(volatility_manifest)},
        },
        "completed_preentry_sources_opened": True,
        "postentry_return_pnl_execution_price_opened": False,
        "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "controls": {
            name: {
                "path": str(CONTROL_DIR / f"{name}.csv.gz"),
                "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"),
                "rows": len(control_clocks),
                "promotion_authorized": False,
            }
            for name, control_clocks in controls.items()
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
    output = run()
    print(json.dumps({"passed": output["support_passed"], "support": output["support"]}, indent=2))
