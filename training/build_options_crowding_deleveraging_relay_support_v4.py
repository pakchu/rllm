"""Build outcome-blind OCDR-12C clocks and source-support evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.build_binance_aggtrade_microstructure import _write_gzip_csv
from training import preregister_options_crowding_deleveraging_relay_v4 as prereg


SOURCE_DIR = Path("data/options_crowding_deleveraging_relay_sources_v4_2023_2026")
MANIFEST = SOURCE_DIR / "manifest.json"
OUTPUT_CLOCK = Path("data/options_crowding_deleveraging_relay_clocks_v4_2023_2026.csv.gz")
OUTPUT_CONTROLS = Path("data/options_crowding_deleveraging_relay_controls_v4_2023_2026")
OUTPUT_RESULT = Path("results/options_crowding_deleveraging_relay_support_v4_2026-08-08.json")
PREREG = prereg.DEFAULT_OUTPUT
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM_EVENTS = {"train": 16, "test": 24, "eval": 24, "final": 16}
CONTROLS = ("no_deribit_lead", "no_oi_tail", "no_funding_tail", "direction_flip")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def load_sources(source_dir: Path = SOURCE_DIR) -> tuple[pd.DataFrame, ...]:
    bvol = pd.read_csv(source_dir / "bvol_hourly.csv.gz", compression="gzip")
    dvol = pd.read_csv(source_dir / "dvol_hourly.csv.gz", compression="gzip")
    oi = pd.read_csv(source_dir / "open_interest_5m.csv.gz", compression="gzip")
    funding = pd.read_csv(source_dir / "funding.csv.gz", compression="gzip")
    return bvol, dvol, oi, funding


def joined_features(
    bvol: pd.DataFrame, dvol: pd.DataFrame, oi: pd.DataFrame, funding: pd.DataFrame
) -> pd.DataFrame:
    b = pd.DataFrame(
        {
            "decision_time": pd.to_datetime(bvol["feature_available_time_utc"], utc=True, format="mixed"),
            "bvol_open": pd.to_numeric(bvol["open"], errors="coerce"),
            "bvol_close": pd.to_numeric(bvol["close"], errors="coerce"),
            "bvol_valid": bvol["feature_valid"].astype(str).str.lower().eq("true"),
        }
    )
    d = pd.DataFrame(
        {
            "decision_time": pd.to_datetime(dvol["close_time"], utc=True, format="mixed"),
            "dvol_open": pd.to_numeric(dvol["open"], errors="coerce"),
            "dvol_close": pd.to_numeric(dvol["close"], errors="coerce"),
        }
    )
    joint = b.merge(d, on="decision_time", how="inner", validate="one_to_one")
    oi = oi.copy()
    oi["ts"] = pd.to_datetime(oi["ts"], utc=True, format="mixed")
    oi["sum_open_interest"] = pd.to_numeric(oi["sum_open_interest"], errors="coerce")
    oi = oi[oi["sum_open_interest"].gt(0)].sort_values("ts")
    current_query = joint[["decision_time"]].sort_values("decision_time")
    current = pd.merge_asof(
        current_query, oi[["ts", "sum_open_interest"]],
        left_on="decision_time", right_on="ts", direction="backward",
        tolerance=pd.Timedelta(minutes=5), allow_exact_matches=True,
    ).rename(columns={"ts": "oi_current_time", "sum_open_interest": "oi_current"})
    prior_query = current_query.assign(prior_target=current_query["decision_time"]-pd.Timedelta(hours=1))
    prior = pd.merge_asof(
        prior_query.sort_values("prior_target"), oi[["ts", "sum_open_interest"]],
        left_on="prior_target", right_on="ts", direction="backward",
        tolerance=pd.Timedelta(minutes=5), allow_exact_matches=True,
    ).sort_values("decision_time").rename(columns={"ts": "oi_prior_time", "sum_open_interest": "oi_prior"})
    joint = joint.merge(current, on="decision_time", validate="one_to_one").merge(
        prior[["decision_time", "oi_prior_time", "oi_prior"]], on="decision_time", validate="one_to_one"
    )
    joint["oi_change"] = joint["oi_current"] / joint["oi_prior"] - 1.0
    joint["oi_tail"] = (
        joint["oi_change"].where(np.isfinite(joint["oi_change"]))
        .shift(1)
        .rolling(720, min_periods=672)
        .quantile(0.75)
    )
    funding = funding.copy()
    funding["funding_time"] = pd.to_datetime(
        funding["funding_time"], utc=True, format="mixed"
    )
    funding["funding_rate"] = pd.to_numeric(funding["funding_rate"], errors="coerce")
    funding["funding_tail"] = (
        funding["funding_rate"].abs().shift(1).rolling(270, min_periods=252).quantile(0.75)
    )
    joint = pd.merge_asof(
        joint.sort_values("decision_time"),
        funding[["funding_time", "funding_rate", "funding_tail"]].sort_values("funding_time"),
        left_on="decision_time",
        right_on="funding_time",
        direction="backward",
        allow_exact_matches=True,
    )
    joint["bvol_body"] = (joint["bvol_close"] - joint["bvol_open"]) / joint["bvol_open"]
    joint["dvol_body"] = (joint["dvol_close"] - joint["dvol_open"]) / joint["dvol_open"]
    numeric = [
        "bvol_open", "bvol_close", "dvol_open", "dvol_close", "oi_current",
        "oi_prior", "oi_change", "funding_rate",
    ]
    joint["base_valid"] = (
        joint["bvol_valid"]
        & np.isfinite(joint[numeric]).all(axis=1)
        & joint[["bvol_open", "bvol_close", "dvol_open", "dvol_close", "oi_current", "oi_prior"]]
        .gt(0).all(axis=1)
        & joint["funding_rate"].ne(0)
    )
    return joint.sort_values("decision_time").reset_index(drop=True)


def build_clock(features: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    b = features["bvol_body"]
    d = features["dvol_body"]
    vol = b.gt(0) & d.gt(0) if control == "no_deribit_lead" else b.gt(0) & d.gt(b)
    oi_gate = features["oi_change"].gt(0)
    if control != "no_oi_tail":
        oi_gate &= features["oi_tail"].notna() & features["oi_change"].ge(features["oi_tail"])
    funding_gate = features["funding_rate"].ne(0)
    if control != "no_funding_tail":
        funding_gate &= features["funding_tail"].notna() & features["funding_rate"].abs().ge(features["funding_tail"])
    active = features["base_valid"] & vol & oi_gate & funding_gate
    onset = (
        active
        & ~active.shift(1, fill_value=False)
        & features["base_valid"].shift(1, fill_value=False)
        & features["decision_time"].diff().eq(pd.Timedelta(hours=1))
    )
    rows: list[dict[str, Any]] = []
    next_allowed: pd.Timestamp | None = None
    for index in features.index[onset]:
        decision = pd.Timestamp(features.at[index, "decision_time"])
        entry = decision + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=12)
        if next_allowed is not None and entry < next_allowed:
            continue
        split = next(
            (name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end),
            None,
        )
        if split is None:
            continue
        side = -int(np.sign(float(features.at[index, "funding_rate"])))
        if control == "direction_flip":
            side *= -1
        next_allowed = exit_time
        rows.append(
            {
                "candidate": "OCDR-12C",
                "control": control,
                "split": split,
                "decision_time": decision,
                "feature_available_time": decision,
                "entry_time": entry,
                "exit_time": exit_time,
                "side": side,
                "bvol_body": float(b.at[index]),
                "dvol_body": float(d.at[index]),
                "oi_change": float(features.at[index, "oi_change"]),
                "prior_oi_q75": float(features.at[index, "oi_tail"])
                if pd.notna(features.at[index, "oi_tail"]) else None,
                "funding_rate": float(features.at[index, "funding_rate"]),
                "prior_abs_funding_q75": float(features.at[index, "funding_tail"])
                if pd.notna(features.at[index, "funding_tail"]) else None,
            }
        )
    return pd.DataFrame(rows)


def split_stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    subset = clock[clock["split"].eq(split)]
    if subset.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    months = subset["entry_time"].dt.strftime("%Y-%m").value_counts()
    longs = int(subset["side"].eq(1).sum())
    shorts = int(subset["side"].eq(-1).sum())
    return {
        "events": len(subset),
        "longs": longs,
        "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(subset),
        "max_month_share": int(months.max()) / len(subset),
    }


def run() -> dict[str, Any]:
    preregistration = json.loads(PREREG.read_text())
    manifest = json.loads(MANIFEST.read_text())
    bvol, dvol, oi, funding = load_sources()
    features = joined_features(bvol, dvol, oi, funding)
    primary = build_clock(features)
    controls = {name: build_clock(features, name) for name in CONTROLS}
    OUTPUT_CLOCK.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_CONTROLS.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(primary, OUTPUT_CLOCK)
    for name, frame in controls.items():
        _write_gzip_csv(frame, OUTPUT_CONTROLS / f"{name}.csv.gz")
    support = {name: split_stats(primary, name) for name in SPLITS}
    checks: dict[str, bool] = {}
    for name, stats in support.items():
        checks[f"{name}_minimum_events"] = stats["events"] >= MINIMUM_EVENTS[name]
        checks[f"{name}_side_balance"] = stats["minority_side_share"] >= 0.20
        checks[f"{name}_month_concentration"] = stats["max_month_share"] <= 0.45
    core = {
        "protocol_version": "ocdr_12c_source_support_v1",
        "policy_id": "OCDR-12C",
        "preregistration": {"path": str(PREREG), "sha256": sha256(PREREG), "manifest_hash": preregistration["manifest_hash"]},
        "source_manifest": {"path": str(MANIFEST), "sha256": sha256(MANIFEST), "manifest_hash": manifest["manifest_hash"]},
        "btc_market_price_or_return_opened": False,
        "gross9_rows_opened": False,
        "candidate_incidence_opened": True,
        "joined_source_hours": len(features),
        "clock": {"path": str(OUTPUT_CLOCK), "sha256": sha256(OUTPUT_CLOCK), "rows": len(primary)},
        "controls": {name: {"path": str(OUTPUT_CONTROLS / f"{name}.csv.gz"), "sha256": sha256(OUTPUT_CONTROLS / f"{name}.csv.gz"), "rows": len(frame)} for name, frame in controls.items()},
        "support": support,
        "support_checks": checks,
        "support_passed": all(checks.values()),
        "gross9_novelty_status": "pending" if all(checks.values()) else "not_authorized",
        "advance_to_gross9_novelty": all(checks.values()),
        "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if all(checks.values()) else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    OUTPUT_RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    result = run()
    print(json.dumps({"support_passed": result["support_passed"], "support": result["support"]}, indent=2))
