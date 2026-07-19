"""Build outcome-blind SQFD-6 feature state, controls, and event clocks."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from training.build_binance_aggtrade_microstructure import _write_gzip_csv


def _utc_timestamp(value: str) -> pd.Timestamp:
    return cast(pd.Timestamp, pd.Timestamp(value))


PREREGISTRATION = Path(
    "results/stablecoin_quote_flow_diffusion_preregistration_2026-07-19.json"
)
DEFAULT_RESULT = Path("results/stablecoin_quote_flow_diffusion_support_2026-07-19.json")
DEFAULT_CLOCKS = Path("data/stablecoin_quote_flow_diffusion_clocks_2023_2026.csv.gz")
BUILDER_PATH = Path("training/build_stablecoin_quote_flow_diffusion_support.py")
SYMBOLS = ("BTCUSDT", "BTCUSDC", "BTCFDUSD")
SOURCE_COLUMNS = (
    "date",
    "symbol",
    "open_time_us",
    "close_time_us",
    "base_volume_btc",
    "trade_count",
    "taker_buy_base_btc",
    "taker_sell_base_btc",
    "signed_taker_flow_btc",
    "source_complete",
)
STATE_COLUMNS = (
    "source_hour_start",
    "decision_time",
    "feature_available_time",
    "source_valid",
    "z_usdt",
    "z_usdc",
    "z_fdusd",
    "alt_share",
    "prior_alt_share_q50",
    "min_alt_abs_z",
    "weighted_alt_z",
    "primary_side",
    "primary_onset",
    "no_alt_breadth_side",
    "no_alt_breadth_onset",
    "no_usdt_lag_side",
    "no_usdt_lag_onset",
    "no_participation_side",
    "no_participation_onset",
    "usdt_only_side",
    "usdt_only_onset",
)
CLOCK_COLUMNS = (
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
    "min_alt_abs_z",
    "weighted_alt_z",
)
SPLITS: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {
    "train": (
        _utc_timestamp("2023-07-01T00:00:00Z"),
        _utc_timestamp("2024-01-01T00:00:00Z"),
    ),
    "test": (
        _utc_timestamp("2024-01-01T00:00:00Z"),
        _utc_timestamp("2025-01-01T00:00:00Z"),
    ),
    "eval": (
        _utc_timestamp("2025-01-01T00:00:00Z"),
        _utc_timestamp("2026-01-01T00:00:00Z"),
    ),
    "final": (
        _utc_timestamp("2026-01-01T00:00:00Z"),
        _utc_timestamp("2026-07-01T00:00:00Z"),
    ),
}
SUPPORT_WINDOWS: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {
    "train": SPLITS["train"],
    "2023_q3": (
        _utc_timestamp("2023-07-01T00:00:00Z"),
        _utc_timestamp("2023-10-01T00:00:00Z"),
    ),
    "2023_q4": (
        _utc_timestamp("2023-10-01T00:00:00Z"),
        _utc_timestamp("2024-01-01T00:00:00Z"),
    ),
    "test": SPLITS["test"],
    "2024_h1": (
        _utc_timestamp("2024-01-01T00:00:00Z"),
        _utc_timestamp("2024-07-01T00:00:00Z"),
    ),
    "2024_h2": (
        _utc_timestamp("2024-07-01T00:00:00Z"),
        _utc_timestamp("2025-01-01T00:00:00Z"),
    ),
    "eval": SPLITS["eval"],
    "2025_h1": (
        _utc_timestamp("2025-01-01T00:00:00Z"),
        _utc_timestamp("2025-07-01T00:00:00Z"),
    ),
    "2025_h2": (
        _utc_timestamp("2025-07-01T00:00:00Z"),
        _utc_timestamp("2026-01-01T00:00:00Z"),
    ),
    "final": SPLITS["final"],
    "2026_q1": (
        _utc_timestamp("2026-01-01T00:00:00Z"),
        _utc_timestamp("2026-04-01T00:00:00Z"),
    ),
    "2026_q2": (
        _utc_timestamp("2026-04-01T00:00:00Z"),
        _utc_timestamp("2026-07-01T00:00:00Z"),
    ),
}
COMPARATOR_COVERAGE: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {
    "OPDR-24": (SPLITS["train"][0], SPLITS["final"][1]),
    "PCBR-12": (SPLITS["train"][0], SPLITS["final"][1]),
    "PSR-30/6": (SPLITS["train"][0], SPLITS["final"][1]),
    "FQPR-3": SPLITS["train"],
}


@dataclass(frozen=True)
class Policy:
    policy_id: str = "SQFD-6"
    prior_window_hours: int = 720
    prior_min_periods_hours: int = 672
    robust_center_quantile: float = 0.50
    robust_scale_lower_quantile: float = 0.25
    robust_scale_upper_quantile: float = 0.75
    normal_iqr_divisor: float = 1.349
    alternative_min_abs_z: float = 0.75
    usdt_lag_signed_z_max_exclusive: float = 0.50
    alternative_share_quantile: float = 0.50
    onset_only: bool = True
    entry_delay_minutes_after_hour_boundary: int = 5
    hold_hours: int = 6
    leverage: float = 0.5
    base_cost_notional_per_side: float = 0.0006
    stress_cost_notional_per_side: float = 0.0010


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


def load_preregistration(path: Path = PREREGISTRATION) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    if report.get("manifest_hash") != _canonical_hash(core):
        raise ValueError("SQFD preregistration hash mismatch")
    if report.get("outcomes_opened") is not False:
        raise ValueError("SQFD preregistration opened outcomes")
    if report.get("policy") != asdict(Policy()):
        raise ValueError("SQFD policy differs from the committed singleton")
    return report


def load_source(prereg: dict[str, Any]) -> pd.DataFrame:
    source = prereg["source_contract"]
    panel_path = Path(source["panel"])
    manifest_path = Path(source["manifest"])
    if _sha256(panel_path) != source["panel_sha256"]:
        raise ValueError("SQFD source panel hash mismatch")
    if _sha256(manifest_path) != source["manifest_sha256"]:
        raise ValueError("SQFD source manifest hash mismatch")
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("protocol", {}).get("outcomes_opened") is not False:
        raise ValueError("SQFD source manifest opened outcomes")
    frame = pd.read_csv(panel_path)
    if tuple(frame.columns) != SOURCE_COLUMNS:
        raise ValueError(f"SQFD source column drift: {frame.columns.tolist()}")
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="raise")
    if frame[["date", "symbol"]].duplicated().any():
        raise ValueError("SQFD source contains duplicate date-symbol rows")
    if set(frame["symbol"].unique()) != set(SYMBOLS):
        raise ValueError("SQFD source symbol basket changed")
    _validate_source_grid(frame, source["activation_utc"])
    if not bool(frame["source_complete"].all()):
        raise ValueError("SQFD source contains incomplete rows")
    numeric = frame.loc[:, SOURCE_COLUMNS[2:-1]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("SQFD source contains non-finite numeric rows")
    close = pd.to_datetime(frame["close_time_us"], unit="us", utc=True)
    boundary = frame["date"] + pd.Timedelta(hours=1)
    latency = boundary - close
    if not bool(
        latency.between(
            pd.Timedelta(microseconds=1), pd.Timedelta(milliseconds=1)
        ).all()
    ):
        raise ValueError("SQFD source close timestamps do not precede hour boundaries")
    return frame


def _validate_source_grid(
    frame: pd.DataFrame,
    activation_utc: dict[str, str],
) -> None:
    if set(activation_utc) != set(SYMBOLS):
        raise ValueError("SQFD source activation contract changed")
    for symbol in SYMBOLS:
        expected = pd.date_range(
            _utc_timestamp(activation_utc[symbol]),
            SPLITS["final"][1],
            freq="1h",
            inclusive="left",
        )
        actual = pd.DatetimeIndex(frame.loc[frame["symbol"].eq(symbol), "date"])
        if not actual.equals(expected):
            raise ValueError(f"SQFD source hourly grid changed: {symbol}")


def _strict_prior_quantile(
    values: pd.Series,
    *,
    quantile: float,
    policy: Policy,
) -> pd.Series:
    result = (
        values.shift(1)
        .rolling(
            window=policy.prior_window_hours,
            min_periods=policy.prior_min_periods_hours,
        )
        .quantile(quantile, interpolation="linear")
    )
    return cast(pd.Series, result)


def _sign(values: pd.Series) -> pd.Series:
    return pd.Series(
        np.sign(values.to_numpy(dtype=float)),
        index=values.index,
        dtype=float,
    )


def _onset(active: pd.Series) -> pd.Series:
    return active.fillna(False) & ~active.shift(1, fill_value=False)


def derive_state(source: pd.DataFrame, policy: Policy = Policy()) -> pd.DataFrame:
    source = source.copy()
    source["date"] = pd.to_datetime(source["date"], utc=True, errors="raise")
    volume = source.pivot(
        index="date", columns="symbol", values="base_volume_btc"
    ).sort_index()
    trades = source.pivot(
        index="date", columns="symbol", values="trade_count"
    ).sort_index()
    flow = source.pivot(
        index="date", columns="symbol", values="signed_taker_flow_btc"
    ).sort_index()
    if tuple(volume.columns) != tuple(sorted(SYMBOLS)):
        raise ValueError("SQFD pivoted symbol columns changed")
    valid = volume.gt(0.0).all(axis=1) & trades.gt(0.0).all(axis=1)
    imbalance = flow.divide(volume.where(volume.gt(0.0)))
    z = pd.DataFrame(index=imbalance.index, columns=imbalance.columns, dtype=float)
    for symbol in SYMBOLS:
        series = cast(pd.Series, imbalance[symbol])
        center = _strict_prior_quantile(
            series, quantile=policy.robust_center_quantile, policy=policy
        )
        lower = _strict_prior_quantile(
            series, quantile=policy.robust_scale_lower_quantile, policy=policy
        )
        upper = _strict_prior_quantile(
            series, quantile=policy.robust_scale_upper_quantile, policy=policy
        )
        scale = (upper - lower) / policy.normal_iqr_divisor
        z[symbol] = (series - center).divide(scale.where(scale.gt(0.0)))

    alt_volume = volume["BTCUSDC"] + volume["BTCFDUSD"]
    total_volume = volume.loc[:, list(SYMBOLS)].sum(axis=1)
    alt_share = alt_volume.divide(total_volume.where(total_volume.gt(0.0)))
    prior_alt_share = _strict_prior_quantile(
        alt_share, quantile=policy.alternative_share_quantile, policy=policy
    )
    alt_mean = (z["BTCUSDC"] + z["BTCFDUSD"]) / 2.0
    z_usdc = cast(pd.Series, z["BTCUSDC"])
    z_fdusd = cast(pd.Series, z["BTCFDUSD"])
    z_usdt = cast(pd.Series, z["BTCUSDT"])
    primary_side = _sign(cast(pd.Series, alt_mean))
    agreement = _sign(z_usdc).eq(_sign(z_fdusd)) & primary_side.ne(0.0)
    min_alt_abs_z = pd.concat([z_usdc.abs(), z_fdusd.abs()], axis=1).min(axis=1)
    weighted_alt_z = (volume["BTCUSDC"] * z_usdc + volume["BTCFDUSD"] * z_fdusd).divide(
        alt_volume.where(alt_volume.gt(0.0))
    )
    all_z_valid = z.loc[:, list(SYMBOLS)].notna().all(axis=1)
    strength = min_alt_abs_z.ge(policy.alternative_min_abs_z)
    usdt_lag = (primary_side * z_usdt).lt(policy.usdt_lag_signed_z_max_exclusive)
    participation = alt_share.ge(prior_alt_share)
    base_valid = valid & all_z_valid & prior_alt_share.notna()
    primary_active = base_valid & agreement & strength & usdt_lag & participation
    no_usdt_lag_active = base_valid & agreement & strength & participation
    no_participation_active = base_valid & agreement & strength & usdt_lag
    no_alt_side = _sign(cast(pd.Series, weighted_alt_z))
    no_alt_breadth_active = (
        base_valid
        & weighted_alt_z.abs().ge(policy.alternative_min_abs_z)
        & (no_alt_side * z_usdt).lt(policy.usdt_lag_signed_z_max_exclusive)
        & participation
    )
    usdt_side = _sign(z_usdt)
    usdt_only_active = (
        valid & z_usdt.notna() & z_usdt.abs().ge(policy.alternative_min_abs_z)
    )
    index = cast(pd.DatetimeIndex, volume.index)
    state = pd.DataFrame(
        {
            "source_hour_start": index,
            "decision_time": index + pd.Timedelta(hours=1),
            "feature_available_time": index + pd.Timedelta(hours=1),
            "source_valid": valid,
            "z_usdt": z["BTCUSDT"],
            "z_usdc": z["BTCUSDC"],
            "z_fdusd": z["BTCFDUSD"],
            "alt_share": alt_share,
            "prior_alt_share_q50": prior_alt_share,
            "min_alt_abs_z": min_alt_abs_z,
            "weighted_alt_z": weighted_alt_z,
            "primary_side": primary_side,
            "primary_onset": _onset(primary_active),
            "no_alt_breadth_side": no_alt_side,
            "no_alt_breadth_onset": _onset(no_alt_breadth_active),
            "no_usdt_lag_side": primary_side,
            "no_usdt_lag_onset": _onset(no_usdt_lag_active),
            "no_participation_side": primary_side,
            "no_participation_onset": _onset(no_participation_active),
            "usdt_only_side": usdt_side,
            "usdt_only_onset": _onset(usdt_only_active),
        }
    )
    return cast(pd.DataFrame, state.loc[:, list(STATE_COLUMNS)])


def _random_side(decision_time: pd.Timestamp) -> int:
    canonical = f"SQFD-6|{decision_time.strftime('%Y-%m-%dT%H:%M:%SZ')}"
    first_nibble = int(hashlib.sha256(canonical.encode("ascii")).hexdigest()[0], 16)
    return 1 if first_nibble % 2 == 0 else -1


def _reserve(
    state: pd.DataFrame,
    *,
    onset_column: str,
    side_column: str,
    policy: Policy,
) -> pd.DataFrame:
    selected = state.loc[state[onset_column].astype(bool)].copy()
    selected = selected.sort_values("decision_time", kind="mergesort")
    rows: list[dict[str, Any]] = []
    next_exit = _utc_timestamp("1970-01-01T00:00:00Z")
    for row in selected.to_dict("records"):
        decision = cast(pd.Timestamp, pd.Timestamp(row["decision_time"]))
        entry = cast(
            pd.Timestamp,
            decision
            + pd.Timedelta(minutes=policy.entry_delay_minutes_after_hour_boundary),
        )
        exit_time = cast(pd.Timestamp, entry + pd.Timedelta(hours=policy.hold_hours))
        if entry < next_exit:
            continue
        row["entry_time"] = entry
        row["exit_time"] = exit_time
        row["side"] = int(np.sign(float(row[side_column])))
        if row["side"] == 0:
            raise ValueError(f"SQFD onset has zero side: {onset_column}")
        rows.append(row)
        next_exit = exit_time
    return pd.DataFrame(rows)


def _assign_split(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.assign(split=pd.Series(dtype="string"))
    parts: list[pd.DataFrame] = []
    for name, (start, end) in SPLITS.items():
        contained = cast(
            pd.DataFrame,
            frame[
                frame["source_hour_start"].ge(start)
                & frame["entry_time"].ge(start)
                & frame["exit_time"].le(end)
            ].copy(),
        )
        contained["split"] = name
        parts.append(contained)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def build_clocks(state: pd.DataFrame, policy: Policy = Policy()) -> pd.DataFrame:
    independent = {
        "primary": ("primary_onset", "primary_side"),
        "no_alt_breadth": ("no_alt_breadth_onset", "no_alt_breadth_side"),
        "no_usdt_lag": ("no_usdt_lag_onset", "no_usdt_lag_side"),
        "no_participation": ("no_participation_onset", "no_participation_side"),
        "usdt_only": ("usdt_only_onset", "usdt_only_side"),
    }
    reserved: dict[str, pd.DataFrame] = {
        name: _reserve(
            state,
            onset_column=onset,
            side_column=side,
            policy=policy,
        )
        for name, (onset, side) in independent.items()
    }
    primary = reserved["primary"]
    direction_flip = primary.copy()
    direction_flip["side"] = -direction_flip["side"]
    random_side = primary.copy()
    random_side["side"] = [
        _random_side(cast(pd.Timestamp, pd.Timestamp(value)))
        for value in random_side["decision_time"]
    ]
    extra_latency = primary.copy()
    extra_latency["entry_time"] += pd.Timedelta(hours=1)
    extra_latency["exit_time"] += pd.Timedelta(hours=1)
    reserved.update(
        {
            "direction_flip": direction_flip,
            "deterministic_random_side": random_side,
            "extra_latency_1h": extra_latency,
        }
    )
    parts: list[pd.DataFrame] = []
    for control, frame in reserved.items():
        if frame.empty:
            continue
        contained = _assign_split(frame)
        if contained.empty:
            continue
        contained["candidate"] = policy.policy_id
        contained["control"] = control
        parts.append(cast(pd.DataFrame, contained.loc[:, list(CLOCK_COLUMNS)]))
    if not parts:
        return pd.DataFrame(
            {column: pd.Series(dtype="object") for column in CLOCK_COLUMNS}
        )
    clocks = pd.concat(parts, ignore_index=True)
    return clocks.sort_values(
        ["control", "entry_time", "side"], kind="mergesort"
    ).reset_index(drop=True)


def _summary(
    frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> dict[str, Any]:
    selected = cast(
        pd.DataFrame,
        frame[frame["source_hour_start"].ge(start) & frame["exit_time"].le(end)],
    )
    events = len(selected)
    side = cast(pd.Series, selected["side"])
    entry_time = cast(pd.Series, selected["entry_time"])
    exit_time = cast(pd.Series, selected["exit_time"])
    long_count = int(side.eq(1).sum())
    short_count = int(side.eq(-1).sum())
    month_share = (
        entry_time.dt.strftime("%Y-%m").value_counts(normalize=True)
        if events
        else pd.Series(dtype=float)
    )
    return {
        "events": int(events),
        "long": long_count,
        "short": short_count,
        "long_share": float(long_count / events) if events else 0.0,
        "short_share": float(short_count / events) if events else 0.0,
        "max_month_share": float(month_share.max()) if events else 0.0,
        "first_entry": cast(pd.Timestamp, entry_time.min()).isoformat()
        if events
        else None,
        "last_exit": cast(pd.Timestamp, exit_time.max()).isoformat()
        if events
        else None,
    }


def _nearest_share(
    left: pd.DatetimeIndex, right: pd.DatetimeIndex, hours: int
) -> float:
    if len(left) == 0 or len(right) == 0:
        return 0.0
    left_values = np.asarray(left.asi8, dtype=np.int64)
    right_values = np.sort(np.asarray(right.asi8, dtype=np.int64))
    positions = np.searchsorted(right_values, left_values)
    threshold = int(pd.Timedelta(hours=hours).value)
    matched = np.zeros(len(left_values), dtype=bool)
    for offset in (positions - 1, positions):
        valid = (offset >= 0) & (offset < len(right_values))
        distance = np.full(len(left_values), np.iinfo(np.int64).max, dtype=np.int64)
        distance[valid] = np.abs(left_values[valid] - right_values[offset[valid]])
        matched |= distance <= threshold
    return float(matched.mean())


def _novelty(
    primary: pd.DatetimeIndex,
    comparator: pd.DatetimeIndex,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    near_hours: int,
) -> dict[str, Any]:
    left = primary[(primary >= start) & (primary < end)]
    right = comparator[(comparator >= start) & (comparator < end)]
    intersection = left.intersection(right)
    union = left.union(right)
    left_near = _nearest_share(left, right, near_hours)
    right_near = _nearest_share(right, left, near_hours)
    return {
        "coverage": [start.isoformat(), end.isoformat()],
        "primary_events": int(len(left)),
        "comparator_events": int(len(right)),
        "exact_intersection": int(len(intersection)),
        "exact_jaccard": float(len(intersection) / len(union)) if len(union) else 0.0,
        "near_hours": int(near_hours),
        "primary_near_share": left_near,
        "comparator_near_share": right_near,
        "max_bidirectional_near_share": max(left_near, right_near),
    }


def _load_comparator(item: dict[str, Any]) -> pd.DatetimeIndex:
    path = Path(item["path"])
    if _sha256(path) != item["sha256"]:
        raise ValueError(f"SQFD comparator hash mismatch: {path}")
    frame = pd.read_csv(path)
    candidate = item["candidate"]
    if candidate == "OPDR-24":
        frame = cast(
            pd.DataFrame,
            frame[cast(pd.Series, frame["control"]).eq("primary")],
        )
    elif candidate == "PCBR-12":
        frame = cast(
            pd.DataFrame,
            frame[cast(pd.Series, frame["control"]).eq("primary")],
        )
    elif candidate == "PSR-30/6":
        frame = cast(
            pd.DataFrame,
            frame[cast(pd.Series, frame["candidate"]).eq("PSR-30/6")],
        )
    elif candidate == "FQPR-3":
        frame = cast(
            pd.DataFrame,
            frame[
                cast(pd.Series, frame["clock_name"]).eq("primary")
                & cast(pd.Series, frame["q"]).eq(0.65)
            ],
        )
    else:
        raise ValueError(f"unsupported SQFD comparator: {candidate}")
    return pd.DatetimeIndex(pd.to_datetime(frame["entry_time"], utc=True))


def _support_checks(
    primary: pd.DataFrame,
    summaries: dict[str, dict[str, Any]],
    novelty: dict[str, dict[str, Any]],
    prereg: dict[str, Any],
) -> tuple[dict[str, bool], list[str]]:
    del primary
    gate = prereg["support_gate"]
    checks: dict[str, bool] = {}
    for name, minimum in gate["minimum_events"].items():
        checks[f"{name}_events"] = summaries[name]["events"] >= int(minimum)
    for name in ("train", "test", "eval", "final"):
        minimum_share = float(gate["minimum_each_side_share"])
        checks[f"{name}_side_balance"] = (
            summaries[name]["long_share"] >= minimum_share
            and summaries[name]["short_share"] >= minimum_share
        )
        checks[f"{name}_month_concentration"] = summaries[name][
            "max_month_share"
        ] <= float(gate["maximum_month_share"][name])
    for candidate, row in novelty.items():
        checks[f"{candidate}_exact_jaccard"] = row["exact_jaccard"] <= float(
            gate["comparator_exact_entry_jaccard_max"]
        )
        checks[f"{candidate}_near_containment"] = row[
            "max_bidirectional_near_share"
        ] <= float(gate["comparator_near_6h_containment_max"])
    failures = [name for name, passed in checks.items() if not passed]
    return checks, failures


def build(
    *,
    preregistration: Path = PREREGISTRATION,
    result_path: Path = DEFAULT_RESULT,
    clocks_path: Path = DEFAULT_CLOCKS,
) -> dict[str, Any]:
    prereg = load_preregistration(preregistration)
    source = load_source(prereg)
    state = derive_state(source)
    clocks = build_clocks(state)
    primary = cast(
        pd.DataFrame,
        clocks[cast(pd.Series, clocks["control"]).eq("primary")].copy(),
    )
    summaries = {
        name: _summary(primary, start, end)
        for name, (start, end) in SUPPORT_WINDOWS.items()
    }
    primary_entries = pd.DatetimeIndex(primary["entry_time"])
    novelty: dict[str, dict[str, Any]] = {}
    for item in prereg["support_comparators"]["clocks"]:
        candidate = item["candidate"]
        comparator = _load_comparator(item)
        start, end = COMPARATOR_COVERAGE[candidate]
        novelty[candidate] = _novelty(
            primary_entries,
            comparator,
            start=start,
            end=end,
            near_hours=int(prereg["support_comparators"]["near_window_hours"]),
        )
    checks, failures = _support_checks(primary, summaries, novelty, prereg)
    clocks_path.parent.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(clocks, clocks_path)
    controls = {
        control: {
            split: _summary(
                group,
                SPLITS[split][0],
                SPLITS[split][1],
            )
            for split in SPLITS
        }
        for control, untyped_group in clocks.groupby("control", sort=True)
        for group in [cast(pd.DataFrame, untyped_group)]
    }
    core: dict[str, Any] = {
        "protocol_version": "stablecoin_quote_flow_diffusion_support_v1",
        "as_of_date": "2026-07-19",
        "candidate": Policy().policy_id,
        "preregistration": str(preregistration),
        "preregistration_sha256": _sha256(preregistration),
        "preregistration_manifest_hash": prereg["manifest_hash"],
        "builder": str(BUILDER_PATH),
        "builder_sha256": _sha256(BUILDER_PATH),
        "outcomes_opened": False,
        "outcome_sources_opened": [],
        "btc_execution_rows_loaded": 0,
        "funding_rows_loaded": 0,
        "source_rows_loaded": int(len(source)),
        "state_rows": int(len(state)),
        "source_panel_sha256": prereg["source_contract"]["panel_sha256"],
        "clock_path": str(clocks_path),
        "clock_sha256": _sha256(clocks_path),
        "clock_rows_all_controls": int(len(clocks)),
        "primary_clock_rows": int(len(primary)),
        "support": summaries,
        "control_support": controls,
        "novelty": novelty,
        "checks": checks,
        "failed_checks": failures,
        "support_passed": not failures,
        "advance_to_train_outcomes": not failures,
        "sealed_outcome_windows": [
            "train_2023_h2",
            "test_2024",
            "eval_2025",
            "final_2026_h1",
        ],
        "rejection_action": (
            "advance only to separately frozen train evaluator"
            if not failures
            else "reject SQFD-6 before opening execution OHLC or funding"
        ),
    }
    report = {**core, "manifest_hash": _canonical_hash(core)}
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", default=str(PREREGISTRATION))
    parser.add_argument("--result", default=str(DEFAULT_RESULT))
    parser.add_argument("--clocks", default=str(DEFAULT_CLOCKS))
    args = parser.parse_args()
    report = build(
        preregistration=Path(args.preregistration),
        result_path=Path(args.result),
        clocks_path=Path(args.clocks),
    )
    print(
        json.dumps(
            {
                "support_passed": report["support_passed"],
                "failed_checks": report["failed_checks"],
                "support": report["support"],
                "clock_sha256": report["clock_sha256"],
                "manifest_hash": report["manifest_hash"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
