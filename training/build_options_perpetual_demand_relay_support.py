"""Build outcome-blind OPDR-24 clocks and support/novelty evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import freeze_options_perpetual_demand_relay_sources as frozen_sources  # noqa: E402
from training import preregister_options_perpetual_demand_relay as prereg  # noqa: E402
from training.build_binance_aggtrade_microstructure import _write_gzip_csv  # noqa: E402


PREREG_COMMIT = "c4b9c4f22d24783f8176897ec4159a5ae1f6e68c"
PREREGISTRATION = Path(prereg.DEFAULT_OUTPUT)
PREREGISTRATION_SHA256 = (
    "9673fe0fc0cc929514c730a56157f6ed409dd1063486c7df082c215e459ba696"
)
PREREGISTRATION_MANIFEST_HASH = (
    "c5f61217324c51faeb46324ff31906205e2fd71b84fbb1c39b067b2e4ce4cf6c"
)
SOURCE_FREEZE_PATH = Path(frozen_sources.Config.output)
SOURCE_FREEZE_SHA256 = (
    "5801b8b819f4951a141700a0249c9cd421ab88922931dc1336ec15de8d1c7883"
)
SOURCE_FREEZE_MANIFEST_HASH = (
    "43aa11881204627e779ae5e1e562f9e9ab50485a89f4b0eaa6337b934a07741c"
)
PREMIUM_SHA256 = (
    "7fbaae1f85482b9fc9e148af357c7315e4e7fc4b4e3ae36c31f27545109f8aa9"
)
PREMIUM_MANIFEST_SHA256 = (
    "821e84f2f03bf893a03d7904bf665b6fd7f6d38edd845d1a9c4eef384d1c1dd8"
)

DEFAULT_CLOCK = Path("data/options_perpetual_demand_relay_clocks_2023_2026.csv.gz")
DEFAULT_CONTROLS_DIR = Path("data/options_perpetual_demand_relay_controls_2023_2026")
DEFAULT_RESULT = Path("results/options_perpetual_demand_relay_support_2026-07-19.json")

SPLITS = {
    "train": (
        pd.Timestamp("2023-07-01", tz="UTC"),
        pd.Timestamp("2024-01-01", tz="UTC"),
    ),
    "test": (
        pd.Timestamp("2024-01-01", tz="UTC"),
        pd.Timestamp("2025-01-01", tz="UTC"),
    ),
    "eval": (
        pd.Timestamp("2025-01-01", tz="UTC"),
        pd.Timestamp("2026-01-01", tz="UTC"),
    ),
    "final": (
        pd.Timestamp("2026-01-01", tz="UTC"),
        pd.Timestamp("2026-07-01", tz="UTC"),
    ),
}
TRAIN_SUBPERIODS = {
    "2023_q3": (
        pd.Timestamp("2023-07-01", tz="UTC"),
        pd.Timestamp("2023-10-01", tz="UTC"),
    ),
    "2023_q4": (
        pd.Timestamp("2023-10-01", tz="UTC"),
        pd.Timestamp("2024-01-01", tz="UTC"),
    ),
}
CONTROL_NAMES = (
    "no_vol_disagreement",
    "no_premium_efficiency",
    "dvol_poor_mirror",
    "direction_flip",
    "extra_latency_1h",
    "deterministic_random_side",
)
PREMIUM_COLUMNS = (
    "date",
    "source_close_time",
    "feature_available_time",
    "source_valid",
    "premium_open",
    "premium_high",
    "premium_low",
    "premium_close",
)
CLOCK_COLUMNS = (
    "candidate",
    "control",
    "split",
    "decision_time",
    "feature_available_time",
    "entry_time",
    "exit_time",
    "side",
    "log_bvol_dvol_ratio",
    "premium_move_bp",
    "premium_path_range_bp",
    "premium_efficiency",
    "prior_ratio_q20",
    "prior_ratio_q80",
    "prior_move_abs_q80_bp",
    "prior_efficiency_q70",
)
COMPARATORS = {
    "old_dvol_price_follow": {
        "path": Path("data/opdr_old_dvol_price_follow_comparator_2023h2.csv.gz"),
        "sha256": "a9c9d1c8d32510e63e604dfdc8b9d079f7e7a4bc206fd0a0197cad8c65b03d3d",
        "time_column": "entry_time",
        "near_minutes": 60,
        "coverage_start": "2023-07-01T00:00:00Z",
        "coverage_end": "2024-01-01T00:00:00Z",
    },
    "PSR-30/6": {
        "path": Path("data/premium_snapback_recenter_clocks_2020_2026.csv.gz"),
        "sha256": "cb209ed35f9baa08cc2fb3dd5bd60b8e747b1408c09507b774ca275e0b2b2db6",
        "time_column": "entry_time",
        "near_minutes": 360,
        "coverage_start": "2023-07-01T00:00:00Z",
        "coverage_end": "2026-07-01T00:00:00Z",
    },
    "PCBR-12": {
        "path": Path("data/premium_compression_breakout_relay_clocks_2020_2026.csv.gz"),
        "sha256": "659fc1b6b6e3a20e60031ed1d50f51c8c7d2836956f911f62ad13e4152740cda",
        "time_column": "entry_time",
        "near_minutes": 360,
        "coverage_start": "2023-07-01T00:00:00Z",
        "coverage_end": "2026-07-01T00:00:00Z",
    },
    "CMSR-36": {
        "path": Path("data/coinm_next_maturity_shock_relay_clocks_2020_2023.csv.gz"),
        "sha256": "e81450d4e76ffd0ce2ae96edf97106f2f4c473da233be0db18dc2530c8da8e87",
        "time_column": "entry_time",
        "near_minutes": 360,
        "coverage_start": "2023-07-01T00:00:00Z",
        "coverage_end": "2024-01-01T00:00:00Z",
    },
}


@dataclass(frozen=True)
class Config:
    preregistration: str = str(PREREGISTRATION)
    source_freeze: str = str(SOURCE_FREEZE_PATH)
    bvol: str = frozen_sources.Config.bvol
    bvol_manifest: str = frozen_sources.Config.bvol_manifest
    dvol: str = frozen_sources.Config.dvol
    dvol_summary: str = frozen_sources.Config.dvol_summary
    premium: str = prereg.PREMIUM_PATH
    premium_manifest: str = prereg.PREMIUM_MANIFEST
    output_clock: str = str(DEFAULT_CLOCK)
    output_controls_dir: str = str(DEFAULT_CONTROLS_DIR)
    output_result: str = str(DEFAULT_RESULT)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_registration(cfg: Config) -> dict[str, Any]:
    if _sha256(cfg.preregistration) != PREREGISTRATION_SHA256:
        raise ValueError("OPDR-24 preregistration bytes changed")
    registration = json.loads(Path(cfg.preregistration).read_text(encoding="utf-8"))
    prereg.validate_manifest(registration, verify_sources=False)
    if registration.get("manifest_hash") != PREREGISTRATION_MANIFEST_HASH:
        raise ValueError("OPDR-24 preregistration manifest changed")
    if registration.get("outcomes_opened") is not False:
        raise ValueError("OPDR-24 preregistration opened outcomes")
    return registration


def _load_source_freeze(cfg: Config) -> dict[str, Any]:
    if _sha256(cfg.source_freeze) != SOURCE_FREEZE_SHA256:
        raise ValueError("OPDR-24 source-freeze bytes changed")
    report = json.loads(Path(cfg.source_freeze).read_text(encoding="utf-8"))
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    if report.get("manifest_hash") != _canonical_hash(core):
        raise ValueError("OPDR-24 source-freeze manifest is not self-consistent")
    if report.get("manifest_hash") != SOURCE_FREEZE_MANIFEST_HASH:
        raise ValueError("OPDR-24 source-freeze manifest changed")
    if report.get("outcomes_opened") is not False:
        raise ValueError("OPDR-24 source freeze opened outcomes")
    if report.get("ready_for_outcome_blind_support") is not True:
        raise ValueError("OPDR-24 source freeze is not support-ready")
    return report


def load_premium(cfg: Config) -> pd.DataFrame:
    """Load only premium-index path values and validate the sealed source."""

    if _sha256(cfg.premium) != PREMIUM_SHA256:
        raise ValueError("OPDR-24 premium source bytes changed")
    if _sha256(cfg.premium_manifest) != PREMIUM_MANIFEST_SHA256:
        raise ValueError("OPDR-24 premium manifest bytes changed")
    manifest = json.loads(Path(cfg.premium_manifest).read_text(encoding="utf-8"))
    protocol = manifest.get("protocol", {})
    if protocol.get("source_only") is not True:
        raise ValueError("OPDR-24 premium manifest is not source-only")
    if protocol.get("outcomes_opened") is not False:
        raise ValueError("OPDR-24 premium manifest opened outcomes")
    if protocol.get("btc_execution_prices_retained") is not False:
        raise ValueError("OPDR-24 premium source retained BTC execution prices")
    if protocol.get("returns_or_pnl_retained") is not False:
        raise ValueError("OPDR-24 premium source retained outcomes")
    frame = pd.read_csv(
        cfg.premium,
        compression="gzip",
        usecols=cast(Any, list(PREMIUM_COLUMNS)),
    ).loc[:, list(PREMIUM_COLUMNS)]
    for column in ("date", "source_close_time", "feature_available_time"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    expected = pd.date_range(
        "2020-01-01",
        "2026-07-01",
        freq="1min",
        inclusive="left",
        tz="UTC",
    )
    if not pd.DatetimeIndex(frame["date"]).equals(expected):
        raise ValueError("OPDR-24 premium one-minute grid changed")
    if not cast(pd.Series, frame["source_close_time"]).equals(
        cast(pd.Series, frame["date"])
        + pd.Timedelta(seconds=59, milliseconds=999)
    ):
        raise ValueError("OPDR-24 premium source close time changed")
    if not cast(pd.Series, frame["feature_available_time"]).equals(
        cast(pd.Series, frame["date"]) + pd.Timedelta(minutes=1, seconds=1)
    ):
        raise ValueError("OPDR-24 premium availability changed")
    valid = cast(pd.Series, frame["source_valid"]).astype(bool)
    numeric_columns = list(PREMIUM_COLUMNS[4:])
    numeric = frame.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.loc[valid].to_numpy(float)).all():
        raise ValueError("OPDR-24 valid premium row contains nonfinite values")
    if not bool(numeric.loc[~valid].isna().all().all()):
        raise ValueError("OPDR-24 invalid premium row retains values")
    if not bool(
        numeric.loc[valid, "premium_high"]
        .ge(numeric.loc[valid, ["premium_open", "premium_close"]].max(axis=1))
        .all()
    ):
        raise ValueError("OPDR-24 premium high envelope changed")
    if not bool(
        numeric.loc[valid, "premium_low"]
        .le(numeric.loc[valid, ["premium_open", "premium_close"]].min(axis=1))
        .all()
    ):
        raise ValueError("OPDR-24 premium low envelope changed")
    return frame


def aggregate_premium_hourly(source: pd.DataFrame) -> pd.DataFrame:
    """Aggregate exact `[T-1h,T)` premium minutes into a causal hourly path."""

    if source.empty or len(source) % 60:
        raise ValueError("OPDR-24 premium rows are not whole UTC hours")
    minute_times = pd.DatetimeIndex(source["date"])
    first = cast(pd.Timestamp, minute_times[0])
    if first.minute != 0 or first.second != 0:
        raise ValueError("OPDR-24 premium source is not hour-aligned")
    expected = pd.date_range(first, periods=len(source), freq="1min")
    if not minute_times.equals(expected):
        raise ValueError("OPDR-24 premium source is not contiguous")
    groups = np.arange(len(source), dtype=np.int64) // 60
    source_valid = cast(pd.Series, source["source_valid"]).astype(bool)
    numeric = {
        name: cast(pd.Series, pd.to_numeric(source[name], errors="coerce"))
        for name in PREMIUM_COLUMNS[4:]
    }
    finite = pd.Series(
        np.isfinite(
            np.column_stack([numeric[name].to_numpy(float) for name in PREMIUM_COLUMNS[4:]])
        ).all(axis=1),
        index=source.index,
    )
    row_valid = source_valid & finite
    open_ = numeric["premium_open"].groupby(groups).first()
    close = numeric["premium_close"].groupby(groups).last()
    path_range = (
        numeric["premium_high"] - numeric["premium_low"]
    ).groupby(groups).sum(min_count=60)
    move = close - open_
    complete = row_valid.astype(int).groupby(groups).sum().eq(60)
    valid = complete & path_range.gt(0.0)
    signal_time = cast(pd.Series, source["date"]).groupby(groups).first() + pd.Timedelta(hours=1)
    available = cast(pd.Series, source["feature_available_time"]).groupby(groups).last()
    frame = pd.DataFrame(
        {
            "signal_time": signal_time,
            "feature_available_time": available,
            "premium_valid": valid,
            "premium_move_bp": move * 10_000.0,
            "premium_path_range_bp": path_range * 10_000.0,
            "premium_efficiency": move.abs().div(path_range.where(path_range.gt(0.0))),
        }
    ).reset_index(drop=True)
    if not cast(pd.Series, frame["feature_available_time"]).equals(
        cast(pd.Series, frame["signal_time"]) + pd.Timedelta(seconds=1)
    ):
        raise ValueError("OPDR-24 aggregated premium availability changed")
    feature_columns = [
        "premium_move_bp",
        "premium_path_range_bp",
        "premium_efficiency",
    ]
    frame.loc[~cast(pd.Series, frame["premium_valid"]).astype(bool), feature_columns] = np.nan
    return frame


def build_joint_state(
    premium_hourly: pd.DataFrame,
    bvol: pd.DataFrame,
    dvol: pd.DataFrame,
) -> pd.DataFrame:
    """Join the three feature sources on the completed-hour decision boundary."""

    bvol_state = pd.DataFrame(
        {
            "signal_time": pd.to_datetime(
                bvol["feature_available_time_utc"], utc=True, errors="raise"
            ),
            "bvol_valid": cast(pd.Series, bvol["feature_valid"]).astype(bool),
            "bvol_close": pd.to_numeric(bvol["close"], errors="coerce"),
        }
    )
    dvol_state = pd.DataFrame(
        {
            "signal_time": pd.to_datetime(
                dvol["close_time"], utc=True, errors="raise"
            ),
            "dvol_close": pd.to_numeric(dvol["close"], errors="coerce"),
        }
    )
    end = max(end for _, end in SPLITS.values())
    premium_state = premium_hourly.loc[
        cast(pd.Series, premium_hourly["signal_time"]).lt(end)
    ].copy()
    bvol_state = bvol_state.loc[bvol_state["signal_time"].lt(end)].copy()
    dvol_state = dvol_state.loc[dvol_state["signal_time"].lt(end)].copy()
    frame = premium_state.merge(bvol_state, on="signal_time", how="inner", validate="one_to_one")
    frame = frame.merge(dvol_state, on="signal_time", how="inner", validate="one_to_one")
    frame = frame.sort_values("signal_time").reset_index(drop=True)
    expected = pd.date_range(
        "2023-06-20T01:00:00Z",
        "2026-07-01T00:00:00Z",
        freq="1h",
        inclusive="left",
    )
    if not pd.DatetimeIndex(frame["signal_time"]).equals(expected):
        raise ValueError("OPDR-24 joint hourly grid changed")
    dvol_finite_positive = np.isfinite(frame["dvol_close"].to_numpy(float)) & frame[
        "dvol_close"
    ].gt(0.0)
    bvol_finite_positive = np.isfinite(frame["bvol_close"].to_numpy(float)) & frame[
        "bvol_close"
    ].gt(0.0)
    joint_valid = (
        cast(pd.Series, frame["premium_valid"]).astype(bool)
        & cast(pd.Series, frame["bvol_valid"]).astype(bool)
        & pd.Series(dvol_finite_positive, index=frame.index)
        & pd.Series(bvol_finite_positive, index=frame.index)
    )
    frame["joint_valid"] = joint_valid
    ratio_values = cast(
        pd.Series,
        pd.Series(
            np.log(
                cast(pd.Series, frame["bvol_close"])
                .div(frame["dvol_close"])
                .to_numpy(float)
            ),
            index=frame.index,
        ),
    )
    frame["log_bvol_dvol_ratio"] = ratio_values.where(joint_valid)
    feature_columns = [
        "premium_move_bp",
        "premium_path_range_bp",
        "premium_efficiency",
    ]
    frame.loc[~joint_valid, feature_columns] = np.nan
    return frame


def derive_state(
    joint: pd.DataFrame,
    policy: prereg.Policy = prereg.Policy(),
) -> pd.DataFrame:
    """Derive singleton setup families with strictly-prior rolling thresholds."""

    valid = cast(pd.Series, joint["joint_valid"]).astype(bool)
    ratio = cast(pd.Series, joint["log_bvol_dvol_ratio"])
    move = cast(pd.Series, joint["premium_move_bp"])
    efficiency = cast(pd.Series, joint["premium_efficiency"])
    prior = {
        "prior_ratio_q20": ratio.where(valid)
        .shift(1)
        .rolling(policy.prior_window_hours, min_periods=policy.prior_min_periods_hours)
        .quantile(policy.bvol_dvol_ratio_low_quantile),
        "prior_ratio_q80": ratio.where(valid)
        .shift(1)
        .rolling(policy.prior_window_hours, min_periods=policy.prior_min_periods_hours)
        .quantile(1.0 - policy.bvol_dvol_ratio_low_quantile),
        "prior_move_abs_q80_bp": move.abs()
        .where(valid)
        .shift(1)
        .rolling(policy.prior_window_hours, min_periods=policy.prior_min_periods_hours)
        .quantile(policy.premium_move_abs_quantile),
        "prior_efficiency_q70": efficiency.where(valid)
        .shift(1)
        .rolling(policy.prior_window_hours, min_periods=policy.prior_min_periods_hours)
        .quantile(policy.premium_efficiency_quantile),
    }
    thresholds = pd.DataFrame(prior, index=joint.index)
    threshold_count = valid.astype(int).shift(1).rolling(policy.prior_window_hours).sum()
    base = valid & thresholds.notna().all(axis=1) & move.ne(0.0)
    displaced = move.abs().ge(thresholds["prior_move_abs_q80_bp"])
    efficient = efficiency.ge(thresholds["prior_efficiency_q70"])
    dvol_rich = ratio.le(thresholds["prior_ratio_q20"])
    dvol_poor = ratio.ge(thresholds["prior_ratio_q80"])
    families = {
        "primary_active": base & dvol_rich & displaced & efficient,
        "no_vol_disagreement_active": base & displaced & efficient,
        "no_premium_efficiency_active": base & dvol_rich & displaced,
        "dvol_poor_mirror_active": base & dvol_poor & displaced & efficient,
    }
    family_columns: dict[str, pd.Series] = {}
    for name, active in families.items():
        clean = active.fillna(False).astype(bool)
        family_columns[name] = clean
        family_columns[f"{name.removesuffix('_active')}_onset"] = (
            clean & ~clean.shift(1, fill_value=False)
        ).astype(bool)
    output = pd.DataFrame(
        {
            "decision_time": joint["signal_time"],
            "feature_available_time": joint["feature_available_time"],
            "side": np.sign(move.fillna(0.0)).astype(np.int8),
            "joint_valid": valid,
            "prior_valid_count": threshold_count,
            "log_bvol_dvol_ratio": ratio,
            "premium_move_bp": move,
            "premium_path_range_bp": joint["premium_path_range_bp"],
            "premium_efficiency": efficiency,
            **prior,
            **family_columns,
        }
    )
    return output


def _deterministic_side(decision_time: pd.Timestamp) -> int:
    digest = hashlib.sha256(
        f"{prereg.CANDIDATE}|{decision_time.isoformat()}".encode("utf-8")
    ).digest()
    return 1 if digest[0] & 1 else -1


def _split_for(decision_time: pd.Timestamp) -> tuple[str, pd.Timestamp, pd.Timestamp] | None:
    for split, (start, end) in SPLITS.items():
        split_start = cast(pd.Timestamp, start)
        split_end = cast(pd.Timestamp, end)
        if split_start <= decision_time < split_end:
            return split, split_start, split_end
    return None


def build_clocks(
    state: pd.DataFrame,
    *,
    control: str = "primary",
    policy: prereg.Policy = prereg.Policy(),
) -> pd.DataFrame:
    """Apply global non-overlap before split containment filtering."""

    active_column = {
        "primary": "primary_onset",
        "no_vol_disagreement": "no_vol_disagreement_onset",
        "no_premium_efficiency": "no_premium_efficiency_onset",
        "dvol_poor_mirror": "dvol_poor_mirror_onset",
        "direction_flip": "primary_onset",
        "extra_latency_1h": "primary_onset",
        "deterministic_random_side": "primary_onset",
    }[control]
    rows: list[dict[str, Any]] = []
    next_allowed = cast(
        pd.Timestamp,
        min(cast(pd.Timestamp, start) for start, _ in SPLITS.values()),
    )
    for position in np.flatnonzero(
        cast(pd.Series, state[active_column]).to_numpy(bool)
    ):
        row = state.iloc[int(position)]
        decision = cast(pd.Timestamp, pd.Timestamp(row["decision_time"]))
        split_info = _split_for(decision)
        if split_info is None:
            continue
        split, split_start, split_end = split_info
        entry = cast(
            pd.Timestamp,
            decision + pd.Timedelta(minutes=policy.entry_delay_minutes),
        )
        if control == "extra_latency_1h":
            entry = cast(pd.Timestamp, entry + pd.Timedelta(hours=1))
        exit_time = cast(
            pd.Timestamp, entry + pd.Timedelta(hours=policy.hold_hours)
        )
        if entry < next_allowed:
            continue
        feature_available = cast(
            pd.Timestamp, pd.Timestamp(row["feature_available_time"])
        )
        if feature_available >= entry:
            raise ValueError("OPDR-24 feature is unavailable before entry")
        next_allowed = exit_time
        if entry < split_start or exit_time > split_end:
            continue
        side = int(row["side"])
        if control == "direction_flip":
            side *= -1
        elif control == "deterministic_random_side":
            side = _deterministic_side(decision)
        if side not in {-1, 1}:
            raise ValueError("OPDR-24 active clock has invalid side")
        rows.append(
            {
                "candidate": prereg.CANDIDATE,
                "control": control,
                "split": split,
                "decision_time": decision,
                "feature_available_time": feature_available,
                "entry_time": entry,
                "exit_time": exit_time,
                "side": side,
                "log_bvol_dvol_ratio": float(row["log_bvol_dvol_ratio"]),
                "premium_move_bp": float(row["premium_move_bp"]),
                "premium_path_range_bp": float(row["premium_path_range_bp"]),
                "premium_efficiency": float(row["premium_efficiency"]),
                "prior_ratio_q20": float(row["prior_ratio_q20"]),
                "prior_ratio_q80": float(row["prior_ratio_q80"]),
                "prior_move_abs_q80_bp": float(row["prior_move_abs_q80_bp"]),
                "prior_efficiency_q70": float(row["prior_efficiency_q70"]),
            }
        )
    return pd.DataFrame(rows, columns=cast(Any, list(CLOCK_COLUMNS)))


def _support_stats(clocks: pd.DataFrame, split: str) -> dict[str, Any]:
    selected = clocks.loc[clocks["split"].eq(split)].copy()
    entry = pd.to_datetime(selected["entry_time"], utc=True, errors="raise")
    total = int(len(selected))
    months = entry.dt.strftime("%Y-%m").value_counts().sort_index()
    row: dict[str, Any] = {
        "events": total,
        "long": int(selected["side"].eq(1).sum()),
        "short": int(selected["side"].eq(-1).sum()),
        "long_share": float(selected["side"].eq(1).mean()) if total else 0.0,
        "short_share": float(selected["side"].eq(-1).mean()) if total else 0.0,
        "max_month_share": float(months.max() / total) if total else 1.0,
        "month_counts": {str(key): int(value) for key, value in months.items()},
    }
    if split == "train":
        row["subperiods"] = {
            name: int(entry.ge(start).mul(entry.lt(end)).sum())
            for name, (start, end) in TRAIN_SUBPERIODS.items()
        }
    return row


def _support_checks(
    support: dict[str, dict[str, Any]], registration: dict[str, Any]
) -> dict[str, bool]:
    gate = registration["support_gate"]
    checks: dict[str, bool] = {}
    for split, row in support.items():
        checks[f"{split}_events"] = (
            row["events"] >= gate["minimum_events"][split]
        )
        checks[f"{split}_side_balance"] = (
            min(row["long_share"], row["short_share"])
            >= gate["minimum_each_side_share"]
        )
        checks[f"{split}_month_concentration"] = (
            row["max_month_share"] <= gate["maximum_month_share"][split]
        )
    checks["2023_q3_events"] = (
        support["train"]["subperiods"]["2023_q3"]
        >= gate["minimum_events"]["2023_q3"]
    )
    checks["2023_q4_events"] = (
        support["train"]["subperiods"]["2023_q4"]
        >= gate["minimum_events"]["2023_q4"]
    )
    return checks


def _clock_times(path: Path, expected_hash: str, column: str) -> pd.DatetimeIndex:
    if _sha256(path) != expected_hash:
        raise ValueError(f"OPDR-24 comparator bytes changed: {path}")
    frame = pd.read_csv(path, compression="gzip", usecols=cast(Any, [column]))
    return pd.DatetimeIndex(
        pd.to_datetime(frame[column], utc=True, errors="raise")
        .drop_duplicates()
        .sort_values()
    )


def _novelty(
    primary: pd.DatetimeIndex,
    other: pd.DatetimeIndex,
    near_minutes: int,
    *,
    coverage_start: str,
    coverage_end: str,
) -> dict[str, Any]:
    """Measure primary containment on an explicit common coverage window."""

    left_full = primary.dropna().drop_duplicates().sort_values()
    right_full = other.dropna().drop_duplicates().sort_values()
    start = cast(pd.Timestamp, pd.Timestamp(coverage_start))
    end = cast(pd.Timestamp, pd.Timestamp(coverage_end))
    left = left_full[(left_full >= start) & (left_full < end)]
    right = right_full[(right_full >= start) & (right_full < end)]
    left_ns = set(left.view("int64").tolist())
    right_ns = set(right.view("int64").tolist())
    intersection = len(left_ns & right_ns)
    union = len(left_ns | right_ns)
    tolerance = int(pd.Timedelta(minutes=near_minutes).value)
    ordered = np.sort(np.fromiter(right_ns, dtype=np.int64))
    near = 0
    for value in left_ns:
        position = int(np.searchsorted(ordered, value))
        candidates = ordered[max(0, position - 1) : min(len(ordered), position + 1)]
        if len(candidates) and int(np.min(np.abs(candidates - value))) <= tolerance:
            near += 1
    return {
        "primary_full": int(len(left_full)),
        "other_full": int(len(right_full)),
        "shared_start": start.isoformat(),
        "shared_end": end.isoformat(),
        "primary_events": int(len(left_ns)),
        "other_events": int(len(right_ns)),
        "exact_intersection": int(intersection),
        "exact_jaccard": float(intersection / union) if union else 0.0,
        "near_minutes": int(near_minutes),
        "near_primary_events": int(near),
        "near_primary_share": float(near / len(left_ns)) if left_ns else 0.0,
    }


def build_result(
    cfg: Config,
    registration: dict[str, Any],
    source_freeze_report: dict[str, Any],
    premium: pd.DataFrame,
    premium_hourly: pd.DataFrame,
    joint: pd.DataFrame,
    state: pd.DataFrame,
    primary: pd.DataFrame,
    controls: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    support = {split: _support_stats(primary, split) for split in SPLITS}
    support_checks = _support_checks(support, registration)
    primary_times = pd.DatetimeIndex(
        pd.to_datetime(primary["entry_time"], utc=True, errors="raise")
    )
    novelty: dict[str, dict[str, Any]] = {}
    for name, spec in COMPARATORS.items():
        novelty[name] = _novelty(
            primary_times,
            _clock_times(
                cast(Path, spec["path"]),
                cast(str, spec["sha256"]),
                cast(str, spec["time_column"]),
            ),
            int(spec["near_minutes"]),
            coverage_start=str(spec["coverage_start"]),
            coverage_end=str(spec["coverage_end"]),
        )
    gate = registration["support_gate"]
    novelty_checks = {
        "old_dvol_price_follow_exact": (
            novelty["old_dvol_price_follow"]["exact_jaccard"]
            <= gate["old_price_follow_exact_entry_jaccard_max"]
        ),
        "old_dvol_price_follow_near": (
            novelty["old_dvol_price_follow"]["near_primary_share"]
            <= gate["old_price_follow_near_1h_containment_max"]
        ),
        "PSR-30/6_near": (
            novelty["PSR-30/6"]["near_primary_share"]
            <= gate["premium_family_near_6h_containment_max"]
        ),
        "PCBR-12_near": (
            novelty["PCBR-12"]["near_primary_share"]
            <= gate["premium_family_near_6h_containment_max"]
        ),
        "CMSR-36_near": (
            novelty["CMSR-36"]["near_primary_share"]
            <= gate["cmsr_near_6h_containment_max"]
        ),
    }
    all_checks = {**support_checks, **novelty_checks}
    core: dict[str, Any] = {
        "protocol_version": "options_perpetual_demand_relay_support_v1",
        "policy_id": prereg.CANDIDATE,
        "as_of_date": "2026-07-19",
        "outcomes_opened": False,
        "outcome_sources_opened": [],
        "preregistration": {
            "path": cfg.preregistration,
            "sha256": PREREGISTRATION_SHA256,
            "commit": PREREG_COMMIT,
            "manifest_hash": registration["manifest_hash"],
        },
        "source_freeze": {
            "path": cfg.source_freeze,
            "sha256": SOURCE_FREEZE_SHA256,
            "manifest_hash": source_freeze_report["manifest_hash"],
        },
        "sources": {
            "premium": {
                "path": cfg.premium,
                "sha256": PREMIUM_SHA256,
                "minute_rows": int(len(premium)),
                "hourly_rows": int(len(premium_hourly)),
                "hourly_valid_rows": int(premium_hourly["premium_valid"].sum()),
            },
            "bvol": {
                "path": cfg.bvol,
                "sha256": frozen_sources.BVOL_SHA256,
            },
            "dvol": {
                "path": cfg.dvol,
                "sha256": frozen_sources.DVOL_SHA256,
            },
            "joint_hourly_rows": int(len(joint)),
            "joint_valid_rows": int(joint["joint_valid"].sum()),
            "threshold_ready_rows": int(state["prior_ratio_q20"].notna().sum()),
            "btc_execution_rows_loaded": 0,
            "funding_rows_loaded": 0,
        },
        "clock": {
            "path": cfg.output_clock,
            "sha256": _sha256(cfg.output_clock),
            "rows": int(len(primary)),
            "forbidden_outcome_columns_present": False,
        },
        "controls": {
            name: {
                "path": str(Path(cfg.output_controls_dir) / f"{name}.csv.gz"),
                "sha256": _sha256(Path(cfg.output_controls_dir) / f"{name}.csv.gz"),
                "rows": int(len(frame)),
                "support": {
                    split: _support_stats(frame, split) for split in SPLITS
                },
            }
            for name, frame in controls.items()
        },
        "support": support,
        "support_checks": support_checks,
        "novelty": novelty,
        "novelty_checks": novelty_checks,
        "all_checks": all_checks,
        "support_passed": bool(all(all_checks.values())),
        "failed_checks": [name for name, passed in all_checks.items() if not passed],
        "advance_to_train_outcomes": bool(all(all_checks.values())),
        "sealed_outcome_windows": [
            "train_2023_h2",
            "test_2024",
            "eval_2025",
            "final_2026_h1",
        ],
        "implementation_sha256": _sha256(__file__),
    }
    return {**core, "manifest_hash": _canonical_hash(core)}


def run(cfg: Config = Config()) -> dict[str, Any]:
    registration = _load_registration(cfg)
    source_freeze_report = _load_source_freeze(cfg)
    premium = load_premium(cfg)
    calendar_start = pd.Timestamp("2023-06-20", tz="UTC")
    calendar_end = max(end for _, end in SPLITS.values())
    premium_slice = premium.loc[
        cast(pd.Series, premium["date"]).ge(calendar_start)
        & cast(pd.Series, premium["date"]).lt(calendar_end)
    ].copy()
    premium_hourly = aggregate_premium_hourly(premium_slice)
    source_cfg = frozen_sources.Config(
        preregistration=cfg.preregistration,
        bvol=cfg.bvol,
        bvol_manifest=cfg.bvol_manifest,
        dvol=cfg.dvol,
        dvol_summary=cfg.dvol_summary,
        output=cfg.source_freeze,
    )
    bvol, _ = frozen_sources.load_bvol(source_cfg)
    dvol, _ = frozen_sources.load_dvol(source_cfg)
    joint = build_joint_state(premium_hourly, bvol, dvol)
    state = derive_state(joint)
    primary = build_clocks(state)
    controls = {name: build_clocks(state, control=name) for name in CONTROL_NAMES}
    clock_path = Path(cfg.output_clock)
    clock_path.parent.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(primary, clock_path)
    controls_dir = Path(cfg.output_controls_dir)
    controls_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in controls.items():
        _write_gzip_csv(frame, controls_dir / f"{name}.csv.gz")
    result = build_result(
        cfg,
        registration,
        source_freeze_report,
        premium,
        premium_hourly,
        joint,
        state,
        primary,
        controls,
    )
    output = Path(cfg.output_result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", default=Config.preregistration)
    parser.add_argument("--source-freeze", default=Config.source_freeze)
    parser.add_argument("--bvol", default=Config.bvol)
    parser.add_argument("--bvol-manifest", default=Config.bvol_manifest)
    parser.add_argument("--dvol", default=Config.dvol)
    parser.add_argument("--dvol-summary", default=Config.dvol_summary)
    parser.add_argument("--premium", default=Config.premium)
    parser.add_argument("--premium-manifest", default=Config.premium_manifest)
    parser.add_argument("--output-clock", default=Config.output_clock)
    parser.add_argument("--output-controls-dir", default=Config.output_controls_dir)
    parser.add_argument("--output-result", default=Config.output_result)
    report = run(Config(**vars(parser.parse_args())))
    print(
        json.dumps(
            {
                "support": report["support"],
                "novelty": report["novelty"],
                "support_passed": report["support_passed"],
                "failed_checks": report["failed_checks"],
                "outcomes_opened": report["outcomes_opened"],
                "manifest_hash": report["manifest_hash"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
