"""Reject CVTT v1 if its preregistered prior-event requirement is infeasible."""
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.export_wikimedia_attention_source import sha256_file
from training.preregister_cross_venue_temporal_torsion_alpha import (
    DEFAULT_OUTPUT as DEFAULT_PREREGISTRATION,
    canonical_hash,
    policy_grid,
    validate_manifest as validate_preregistration,
)


DEFAULT_SOURCE_MANIFEST = (
    "results/cross_venue_temporal_torsion_source_manifest_2026-07-16.json"
)
DEFAULT_OUTPUT = "results/cross_venue_temporal_torsion_support_v1_2026-07-16.json"
ROLLING_WINDOW_BARS = 8_640
ROLLING_MINIMUM_ELIGIBLE = 2_016
QUARANTINE_FOLLOWING_BARS = 24


@dataclass(frozen=True)
class Config:
    source_manifest: str = DEFAULT_SOURCE_MANIFEST
    preregistration: str = DEFAULT_PREREGISTRATION
    output: str = DEFAULT_OUTPUT


def validate_hashed_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    core = {
        key: value
        for key, value in payload.items()
        if key not in {"manifest_hash", "created_at"}
    }
    if canonical_hash(core) != payload.get("manifest_hash"):
        raise RuntimeError(f"manifest hash mismatch: {path}")
    return payload


def resolve_manifest_output(source: dict[str, Any], name: str) -> Path:
    metadata = source["outputs"][name]
    path = Path(metadata["path"])
    if not path.exists() or sha256_file(path) != metadata["sha256"]:
        raise RuntimeError(f"CVTT frozen {name} output is missing or hash-mismatched")
    return path


def git_anchor(path: str | Path) -> dict[str, str]:
    candidate = Path(path)
    pathspec = str(candidate.resolve().relative_to(Path.cwd().resolve()))
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", pathspec],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError(f"support artifact is not clean: {pathspec}")
    commit = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", pathspec],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(commit) != 40:
        raise RuntimeError(f"support artifact is not committed: {pathspec}")
    return {"path": pathspec, "commit": commit, "sha256": sha256_file(candidate)}


def quarantine(unavailable: pd.Series) -> pd.Series:
    return unavailable.astype(bool).rolling(
        QUARANTINE_FOLLOWING_BARS + 1, min_periods=1
    ).max().astype(bool)


def route_support(frame: pd.DataFrame) -> dict[str, pd.Series]:
    spot_delay = frame["spot_return_time_centroid"] - frame[
        "spot_flow_time_centroid"
    ]
    um_delay = frame["um_return_time_centroid"] - frame["um_flow_time_centroid"]
    spot_side = np.sign(frame["spot_flow_fraction"]).fillna(0.0)
    um_side = np.sign(frame["um_flow_fraction"]).fillna(0.0)
    clean = frame["source_quarantined"].eq(0)
    spot_confirmed = (
        spot_side.ne(0)
        & (spot_side * frame["um_flow_fraction"]).gt(0)
        & (spot_side * frame["spot_log_return_5m"]).gt(0)
        & (spot_side * frame["um_log_return_5m"]).gt(0)
    )
    um_confirmed = (
        um_side.ne(0)
        & (um_side * frame["spot_flow_fraction"]).gt(0)
        & (um_side * frame["um_log_return_5m"]).gt(0)
        & (um_side * frame["spot_log_return_5m"]).gt(0)
    )
    return {
        "spot_preload_um_echo": clean & spot_delay.gt(0) & um_delay.lt(0) & spot_confirmed,
        "um_preload_spot_echo": clean & um_delay.gt(0) & spot_delay.lt(0) & um_confirmed,
    }


def prior_eligible_counts(
    eligible: pd.Series, *, window: int = ROLLING_WINDOW_BARS
) -> pd.Series:
    if window < 1:
        raise ValueError("rolling support window must be positive")
    return eligible.astype(np.int8).shift(1, fill_value=0).rolling(
        window, min_periods=1
    ).sum()


def source_quality(frame: pd.DataFrame) -> dict[str, Any]:
    unavailable = frame["source_quarantined"].eq(1)
    monthly = unavailable.groupby(frame["date"].dt.to_period("M")).mean()
    global_fraction = float(unavailable.mean())
    max_month_fraction = float(monthly.max())
    gates = {
        "global_missing_or_quarantined_fraction_max_0_01": global_fraction <= 0.01,
        "monthly_missing_or_quarantined_fraction_max_0_03": max_month_fraction <= 0.03,
    }
    return {
        "raw_unavailable_rows": int(frame["source_available"].eq(0).sum()),
        "unavailable_or_following_24_rows": int(unavailable.sum()),
        "global_fraction": global_fraction,
        "maximum_monthly_fraction": max_month_fraction,
        "maximum_month": str(monthly.idxmax()),
        "gates": gates,
        "pass": all(gates.values()),
    }


def load_features(source: dict[str, Any]) -> pd.DataFrame:
    columns = [
        "date",
        "source_available",
        "spot_flow_fraction",
        "um_flow_fraction",
        "spot_log_return_5m",
        "um_log_return_5m",
        "spot_flow_time_centroid",
        "um_flow_time_centroid",
        "spot_return_time_centroid",
        "um_return_time_centroid",
    ]
    frame = pd.read_csv(
        resolve_manifest_output(source, "features"),
        usecols=columns,
        parse_dates=["date"],
    )
    if frame.empty or frame["date"].max() >= pd.Timestamp("2023-01-01"):
        raise RuntimeError("CVTT support preflight refuses the sealed 2023 holdout")
    unavailable = frame["source_available"].eq(0)
    frame["source_quarantined"] = quarantine(unavailable).astype(np.int8)
    return frame


def run(cfg: Config) -> dict[str, Any]:
    prereg = json.loads(Path(cfg.preregistration).read_text())
    validate_preregistration(prereg)
    source = validate_hashed_manifest(cfg.source_manifest)
    if source.get("forward_trade_outcomes_opened") is not False:
        raise RuntimeError("CVTT support requires unopened forward outcomes")
    if source.get("preregistration", {}).get("manifest_hash") != prereg.get(
        "manifest_hash"
    ):
        raise RuntimeError("CVTT source and preregistration are not linked")

    frame = load_features(source)
    quality = source_quality(frame)
    route_masks = route_support(frame)
    routes: dict[str, Any] = {}
    for route, mask in route_masks.items():
        counts = prior_eligible_counts(mask)
        estimable = counts.ge(ROLLING_MINIMUM_ELIGIBLE)
        routes[route] = {
            "directionally_confirmed_crossed_clock_rows": int(mask.sum()),
            "maximum_strictly_prior_30d_eligible_rows": int(counts.max()),
            "rows_where_q95_is_estimable": int(estimable.sum()),
            "required_prior_eligible_rows": ROLLING_MINIMUM_ELIGIBLE,
            "pass": bool(estimable.any()),
        }
    support_pass = quality["pass"] and all(item["pass"] for item in routes.values())
    policy_support = [
        {
            "policy_id": policy.policy_id,
            "route": policy.route,
            "hold_bars": policy.hold_bars,
            "nonoverlap_events": 0,
            "pass": False,
            "reason": "preregistered q95 threshold is nowhere estimable",
        }
        for policy in policy_grid()
    ]
    if support_pass:
        raise RuntimeError("CVTT v1 support unexpectedly passed; full clock freeze required")

    core: dict[str, Any] = {
        "protocol_version": "cross_venue_temporal_torsion_support_v1",
        "phase": "support_only_before_forward_returns",
        "forward_trade_outcomes_opened": False,
        "source_columns_read": list(frame.columns),
        "explicitly_not_read": [
            "market open/high/low/close",
            "future return",
            "realized funding after signal",
            "sealed 2023 features",
        ],
        "implementation_contract": {
            "rolling_window_bars": ROLLING_WINDOW_BARS,
            "required_strictly_prior_eligible_rows": ROLLING_MINIMUM_ELIGIBLE,
            "source_quarantine_current_plus_following_bars": (
                QUARANTINE_FOLLOWING_BARS
            ),
            "thresholds_or_returns_computed": False,
            "short_circuit": (
                "route q95 cannot exist anywhere, so clocks and outcomes remain unopened"
            ),
        },
        "hashes": {
            "preregistration_file_sha256": sha256_file(cfg.preregistration),
            "preregistration_manifest_hash": prereg["manifest_hash"],
            "source_manifest_file_sha256": sha256_file(cfg.source_manifest),
            "source_manifest_hash": source["manifest_hash"],
        },
        "git_anchors": {
            "preregistration": git_anchor(cfg.preregistration),
            "source_freezer": git_anchor(
                "training/export_cross_venue_temporal_torsion_prefix.py"
            ),
            "support_checker": git_anchor(__file__),
        },
        "quality": quality,
        "route_support": routes,
        "policy_support": policy_support,
        "passing_policy_ids": [],
        "selection_may_open_forward_returns": False,
        "decision": "reject_v1_support_infeasible",
        "repair_boundary": (
            "a new version may repair event-count feasibility using support-only "
            "evidence, but v1 stays rejected and no return-dependent field may change"
        ),
        "sealed_holdout": ["2023-01-01", "2024-01-01"],
    }
    payload = {
        **core,
        "manifest_hash": canonical_hash(core),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = Config()
    for name, value in asdict(defaults).items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=type(value), default=value)
    return parser.parse_args()


def main() -> None:
    payload = run(Config(**vars(parse_args())))
    print(
        json.dumps(
            {
                "manifest_hash": payload["manifest_hash"],
                "quality": payload["quality"],
                "route_support": payload["route_support"],
                "decision": payload["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
