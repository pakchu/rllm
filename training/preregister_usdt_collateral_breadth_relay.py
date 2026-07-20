"""Preregister the outcome-blind UCBR-12 direct-stablecoin support clock."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from training import build_binance_usdt_collateral_breadth_source as source_builder


REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_ID = "UCBR-12"
PROTOCOL_VERSION = "usdt_collateral_breadth_relay_v1"
SOURCE_PANEL = Path(
    "data/binance_usdt_collateral_breadth_2023/"
    "stablecoin_usdt_breadth_1h_2023-08-01T00_2023-12-31T23.csv.gz"
)
SOURCE_MANIFEST = Path("data/binance_usdt_collateral_breadth_2023/build_manifest.json")
SOURCE_PANEL_SHA256 = (
    "e96fae39c869f6db0dc30bccc5b2fa72f5e7f717c2528038afede18dd5b9892d"
)
SOURCE_MANIFEST_SHA256 = (
    "26e142b818306275d48690711b7adca00b43750041d104e5c27a65b355c424f2"
)
SDDR_CLOCK = Path("data/stablecoin_denominator_dislocation_clocks_2023.csv.gz")
SDDR_CLOCK_SHA256 = (
    "eaf2d6c187af9855e76474d2951fcdc12267174980a72649b73d068982ca8c69"
)
SDDR_SUPPORT = Path(
    "results/stablecoin_denominator_dislocation_support_2026-07-20.json"
)
SDDR_SUPPORT_SHA256 = (
    "1d7e8561963d903c5963bbd081c5cf0c9926dc9221f9a09d23fb565ab27f7bea"
)
SQFD_CLOCK = Path("data/stablecoin_quote_flow_diffusion_clocks_2023_2026.csv.gz")
SQFD_CLOCK_SHA256 = (
    "a81e144eea1e80ae5439fc66db1fad5bbd00cd9ac177e25142b5cfb5a07bcc5b"
)
SQFD_SUPPORT = Path("results/stablecoin_quote_flow_diffusion_support_2026-07-19.json")
SQFD_SUPPORT_SHA256 = (
    "07230e9e579f1b16e07712a022e572026b4fbfa17070e998970b3fd8ee21d4b5"
)
PREREGISTRATION_SOURCE = Path("training/preregister_usdt_collateral_breadth_relay.py")
PREREGISTRATION_DOCUMENT = Path(
    "docs/usdt-collateral-breadth-relay-preregistration-2026-07-20.md"
)
DEFAULT_OUTPUT = Path(
    "results/usdt_collateral_breadth_relay_preregistration_2026-07-20.json"
)
SYMBOLS = source_builder.DEFAULT_SYMBOLS
LOG_COLUMNS = source_builder.LOG_COLUMNS
VALID_COLUMNS = source_builder.VALID_COLUMNS
SOURCE_COLUMNS = source_builder.OUTPUT_COLUMNS
SOURCE_ONLY_CONTROLS = (
    "all_four",
    "leave_out_usdc",
    "leave_out_tusd",
    "leave_out_usdp",
    "leave_out_fdusd",
    "median_only",
    "stale_1h",
)
EVENT_COLUMNS = (
    "candidate",
    "control",
    "source_hour_start",
    "decision_time",
    "feature_available_time",
    "entry_time",
    "exit_time",
    "side",
    "source_sign",
    "valid_breadth",
    "agreeing_breadth",
    "consensus_strength",
    "median_z",
    "z_usdcusdt",
    "z_tusdusdt",
    "z_usdpusdt",
    "z_fdusdusdt",
)


@dataclass(frozen=True)
class Config:
    lookback_hours: int = 720
    minimum_history_hours: int = 672
    z_threshold: float = 1.25
    minimum_agreeing_issuers: int = 3
    hold_bars: int = 144
    bar_minutes: int = 5
    entry_delay_minutes: int = 5
    support_end_exclusive: str = "2024-01-01T00:00:00Z"
    minimum_events: int = 30
    minimum_events_per_full_signal_month: int = 5
    full_signal_months: tuple[str, ...] = (
        "2023-09",
        "2023-10",
        "2023-11",
        "2023-12",
    )
    minimum_side_share: float = 0.30
    maximum_month_share: float = 0.45
    maximum_exact_entry_jaccard: float = 0.10
    novelty_containment_hours: int = 6
    maximum_novelty_containment: float = 0.35


FROZEN_CONFIG = Config()


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(_resolve(path).read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_config(cfg: Config) -> None:
    if cfg != FROZEN_CONFIG:
        raise ValueError("UCBR-12 support configuration is frozen")
    if cfg.minimum_history_hours > cfg.lookback_hours:
        raise ValueError("minimum UCBR history cannot exceed lookback")
    if cfg.minimum_agreeing_issuers != 3:
        raise ValueError("UCBR-12 requires exactly three agreeing issuers")
    if cfg.hold_bars * cfg.bar_minutes != 12 * 60:
        raise ValueError("UCBR-12 hold must remain exactly twelve hours")


def _parse_bool(values: pd.Series, *, field: str) -> pd.Series:
    if values.dtype == bool:
        return values.astype(bool)
    parsed = values.astype("string").str.lower().map({"true": True, "false": False})
    if parsed.isna().any():
        raise ValueError(f"UCBR source {field} contains an unknown value")
    return parsed.astype(bool)


def load_source() -> tuple[pd.DataFrame, dict[str, Any]]:
    if sha256_file(SOURCE_PANEL) != SOURCE_PANEL_SHA256:
        raise ValueError("UCBR source panel hash mismatch")
    if sha256_file(SOURCE_MANIFEST) != SOURCE_MANIFEST_SHA256:
        raise ValueError("UCBR source manifest hash mismatch")
    manifest = json.loads(_resolve(SOURCE_MANIFEST).read_text(encoding="utf-8"))
    protocol = manifest.get("protocol", {})
    forbidden = (
        "outcomes_opened",
        "btc_prices_opened",
        "perpetual_ohlc_or_funding_opened",
        "future_returns_labels_or_pnl_opened",
        "post_2023_rows_requested",
        "raw_ohlc_retained",
        "volume_trade_count_or_taker_flow_retained",
    )
    if any(protocol.get(key) is not False for key in forbidden):
        raise ValueError("UCBR source manifest violates the outcome-blind contract")
    if protocol.get("direct_stablecoin_log_closes_retained") is not True:
        raise ValueError("UCBR source manifest lacks direct stablecoin log closes")
    if manifest.get("combined_sha256") != SOURCE_PANEL_SHA256:
        raise ValueError("UCBR manifest panel hash differs")

    frame = cast(
        pd.DataFrame,
        pd.read_csv(
            _resolve(SOURCE_PANEL),
            usecols=list(SOURCE_COLUMNS),
            parse_dates=["date", "source_available_at"],
        ),
    )
    expected = pd.date_range("2023-08-01", "2024-01-01", freq="1h", inclusive="left")
    if not frame["date"].equals(pd.Series(expected, name="date")):
        raise ValueError("UCBR source is not the exact frozen common-hour grid")
    if not frame["source_available_at"].equals(
        pd.Series(expected + pd.Timedelta("1h"), name="source_available_at")
    ):
        raise ValueError("UCBR source availability is not exact hour close")
    for column in (*VALID_COLUMNS, "source_complete"):
        frame[column] = _parse_bool(cast(pd.Series, frame[column]), field=column)
    observed_breadth = frame.loc[:, VALID_COLUMNS].sum(axis=1).astype("int8")
    if not frame["valid_breadth"].astype("int8").equals(observed_breadth):
        raise ValueError("UCBR source valid breadth disagrees with member flags")
    if not frame["source_complete"].equals(observed_breadth.ge(3)):
        raise ValueError("UCBR source completeness rule changed")
    if not frame["source_complete"].all():
        raise ValueError("UCBR frozen source contains fewer than three valid books")
    if not np.isfinite(frame.loc[:, LOG_COLUMNS].to_numpy(float)).all():
        raise ValueError("UCBR source contains non-finite direct prices")
    return frame, {
        "panel_sha256": SOURCE_PANEL_SHA256,
        "manifest_sha256": SOURCE_MANIFEST_SHA256,
        "rows": int(len(frame)),
        "first_date": str(frame["date"].iloc[0]),
        "last_date": str(frame["date"].iloc[-1]),
        "minimum_valid_breadth": int(frame["valid_breadth"].min()),
    }


def prior_quantile(
    values: pd.Series,
    *,
    quantile: float,
    cfg: Config,
) -> pd.Series:
    return cast(
        pd.Series,
        values.shift(1)
        .rolling(cfg.lookback_hours, min_periods=cfg.minimum_history_hours)
        .quantile(quantile, interpolation="linear"),
    )


def prior_robust_z(values: pd.Series, valid: pd.Series, cfg: Config) -> pd.DataFrame:
    admitted = values.where(valid.astype(bool))
    median = prior_quantile(admitted, quantile=0.50, cfg=cfg)
    lower = prior_quantile(admitted, quantile=0.25, cfg=cfg)
    upper = prior_quantile(admitted, quantile=0.75, cfg=cfg)
    scale = (upper - lower) / 1.349
    zscore = ((values - median) / scale).where(
        valid.astype(bool) & scale.gt(0.0) & np.isfinite(scale)
    )
    return pd.DataFrame(
        {"median": median, "lower": lower, "upper": upper, "scale": scale, "z": zscore}
    )


def _source_sign(
    z: pd.DataFrame,
    *,
    threshold: float,
    minimum: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    positive = z.ge(threshold).sum(axis=1).astype("int8")
    negative = z.le(-threshold).sum(axis=1).astype("int8")
    sign = pd.Series(
        np.where(positive.ge(minimum), 1, np.where(negative.ge(minimum), -1, 0)),
        index=z.index,
        dtype="int8",
    )
    agreeing = pd.concat([positive, negative], axis=1).max(axis=1).astype("int8")
    positive_strength = z.where(z.ge(threshold)).min(axis=1)
    negative_strength = z.where(z.le(-threshold)).abs().min(axis=1)
    strength = pd.Series(
        np.where(sign.eq(1), positive_strength, np.where(sign.eq(-1), negative_strength, np.nan)),
        index=z.index,
        dtype=float,
    )
    return sign, agreeing, strength


def _add_control(
    state: pd.DataFrame,
    *,
    name: str,
    source_sign: pd.Series,
    agreeing: pd.Series,
    strength: pd.Series,
) -> None:
    state[f"{name}_active"] = source_sign.ne(0)
    state[f"{name}_source_sign"] = source_sign.astype("int8")
    state[f"{name}_side"] = (-source_sign).astype("int8")
    state[f"{name}_agreeing_breadth"] = agreeing.astype("int8")
    state[f"{name}_consensus_strength"] = strength.astype(float)


def signal_states(frame: pd.DataFrame, cfg: Config = FROZEN_CONFIG) -> pd.DataFrame:
    _validate_config(cfg)
    z = pd.DataFrame(index=frame.index, dtype=float)
    for symbol, log_column, valid_column in zip(
        SYMBOLS, LOG_COLUMNS, VALID_COLUMNS, strict=True
    ):
        robust = prior_robust_z(
            cast(pd.Series, frame[log_column]),
            cast(pd.Series, frame[valid_column]),
            cfg,
        )
        z[symbol] = robust["z"]
    current_valid = frame.loc[:, VALID_COLUMNS].astype(bool)
    median_z = z.median(axis=1, skipna=True)
    z_breadth = z.notna().sum(axis=1).astype("int8")
    state = pd.DataFrame(
        {
            "z_usdcusdt": z["USDCUSDT"],
            "z_tusdusdt": z["TUSDUSDT"],
            "z_usdpusdt": z["USDPUSDT"],
            "z_fdusdusdt": z["FDUSDUSDT"],
            "valid_breadth": current_valid.sum(axis=1).astype("int8"),
            "median_z": median_z,
        }
    )

    sign, agreeing, strength = _source_sign(
        z,
        threshold=cfg.z_threshold,
        minimum=cfg.minimum_agreeing_issuers,
    )
    _add_control(
        state,
        name="primary",
        source_sign=sign,
        agreeing=agreeing,
        strength=strength,
    )
    all_sign, all_agreeing, all_strength = _source_sign(
        z,
        threshold=cfg.z_threshold,
        minimum=4,
    )
    _add_control(
        state,
        name="all_four",
        source_sign=all_sign,
        agreeing=all_agreeing,
        strength=all_strength,
    )
    exclusions = {
        "leave_out_usdc": "USDCUSDT",
        "leave_out_tusd": "TUSDUSDT",
        "leave_out_usdp": "USDPUSDT",
        "leave_out_fdusd": "FDUSDUSDT",
    }
    for name, excluded in exclusions.items():
        subset = z.drop(columns=excluded)
        sub_sign, sub_agreeing, sub_strength = _source_sign(
            subset,
            threshold=cfg.z_threshold,
            minimum=3,
        )
        _add_control(
            state,
            name=name,
            source_sign=sub_sign,
            agreeing=sub_agreeing,
            strength=sub_strength,
        )

    median_sign = np.sign(median_z).fillna(0.0).astype("int8")
    median_active = median_z.abs().ge(cfg.z_threshold) & z_breadth.ge(3)
    median_sign = median_sign.where(median_active, 0).astype("int8")
    _add_control(
        state,
        name="median_only",
        source_sign=median_sign,
        agreeing=z_breadth,
        strength=median_z.abs().where(median_active),
    )

    state["stale_1h_active"] = (
        state["primary_active"].shift(1, fill_value=False).astype(bool)
    )
    for suffix in ("source_sign", "side", "agreeing_breadth"):
        state[f"stale_1h_{suffix}"] = (
            state[f"primary_{suffix}"].shift(1, fill_value=0).astype("int8")
        )
    state["stale_1h_consensus_strength"] = state[
        "primary_consensus_strength"
    ].shift(1)
    return state


def onset(active: pd.Series) -> pd.Series:
    values = active.fillna(False).astype(bool)
    return values & ~values.shift(1, fill_value=False)


def schedule(
    frame: pd.DataFrame,
    states: pd.DataFrame,
    *,
    control: str,
    cfg: Config = FROZEN_CONFIG,
) -> pd.DataFrame:
    _validate_config(cfg)
    active_column = f"{control}_active"
    if active_column not in states:
        raise KeyError(control)
    triggers = onset(cast(pd.Series, states[active_column]))
    source_dates = pd.to_datetime(frame["date"], utc=True)
    available = pd.to_datetime(frame["source_available_at"], utc=True)
    end = pd.Timestamp(cfg.support_end_exclusive)
    hold = pd.Timedelta(minutes=cfg.hold_bars * cfg.bar_minutes)
    entry_delay = pd.Timedelta(minutes=cfg.entry_delay_minutes)
    rows: list[dict[str, Any]] = []
    next_allowed = pd.Timestamp.min.tz_localize("UTC")
    for position in np.flatnonzero(triggers.to_numpy(bool)):
        diagnostic_position = position - 1 if control == "stale_1h" else position
        if diagnostic_position < 0:
            raise ValueError("stale UCBR state has no source row")
        decision = cast(pd.Timestamp, available.iloc[position])
        feature_available = cast(pd.Timestamp, available.iloc[diagnostic_position])
        entry = decision + entry_delay
        exit_time = entry + hold
        if entry < next_allowed or exit_time > end:
            continue
        side = int(states.iloc[position][f"{control}_side"])
        source_sign = int(states.iloc[position][f"{control}_source_sign"])
        if side not in (-1, 1) or source_sign != -side:
            raise ValueError("UCBR scheduled direction is inconsistent")
        diagnostic = states.iloc[diagnostic_position]
        rows.append(
            {
                "candidate": POLICY_ID,
                "control": control,
                "source_hour_start": cast(
                    pd.Timestamp, source_dates.iloc[diagnostic_position]
                ),
                "decision_time": decision,
                "feature_available_time": feature_available,
                "entry_time": entry,
                "exit_time": exit_time,
                "side": side,
                "source_sign": source_sign,
                "valid_breadth": int(diagnostic["valid_breadth"]),
                "agreeing_breadth": int(
                    states.iloc[position][f"{control}_agreeing_breadth"]
                ),
                "consensus_strength": float(
                    states.iloc[position][f"{control}_consensus_strength"]
                ),
                "median_z": float(diagnostic["median_z"]),
                "z_usdcusdt": float(diagnostic["z_usdcusdt"]),
                "z_tusdusdt": float(diagnostic["z_tusdusdt"]),
                "z_usdpusdt": float(diagnostic["z_usdpusdt"]),
                "z_fdusdusdt": float(diagnostic["z_fdusdusdt"]),
            }
        )
        next_allowed = exit_time
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def _config_payload() -> dict[str, Any]:
    payload = asdict(FROZEN_CONFIG)
    payload["full_signal_months"] = list(FROZEN_CONFIG.full_signal_months)
    return payload


def preregistration_payload(output: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    _validate_config(FROZEN_CONFIG)
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "as_of_date": "2026-07-20",
        "candidate": POLICY_ID,
        "outcomes_opened": False,
        "outcome_sources_opened": False,
        "post_2023_source_rows_opened": False,
        "real_event_incidence_opened": False,
        "source": {
            "panel": str(SOURCE_PANEL),
            "panel_sha256": SOURCE_PANEL_SHA256,
            "manifest": str(SOURCE_MANIFEST),
            "manifest_sha256": SOURCE_MANIFEST_SHA256,
            "persisted_observables": list(SOURCE_COLUMNS),
            "raw_ohlc_or_flow_retained": False,
        },
        "policy": {
            "config": _config_payload(),
            "normalization": "per issuer: strictly-prior 720h median and IQR/1.349, min 672 valid hours",
            "primary": "at least three current-valid issuers have same-sign |z| >= 1.25; false-to-true onset only",
            "source_sign": "+1 means broad alternative-stablecoin strength / USDT weakness; -1 means USDT strength",
            "trade_side": "-source_sign: USDT weakness SHORT BTCUSDT, USDT strength LONG BTCUSDT",
            "availability": "completed source hour h is known at h+1h",
            "entry": "BTCUSDT USD-M perpetual at h+1h+5m open",
            "exit": "scheduled open after exactly 144 five-minute bars",
            "global_nonoverlap_per_clock": True,
        },
        "source_only_controls": list(SOURCE_ONLY_CONTROLS),
        "support_gate": {
            "minimum_events": FROZEN_CONFIG.minimum_events,
            "full_signal_months": list(FROZEN_CONFIG.full_signal_months),
            "minimum_events_per_full_signal_month": FROZEN_CONFIG.minimum_events_per_full_signal_month,
            "minimum_side_share": FROZEN_CONFIG.minimum_side_share,
            "maximum_month_share": FROZEN_CONFIG.maximum_month_share,
            "maximum_exact_entry_jaccard": FROZEN_CONFIG.maximum_exact_entry_jaccard,
            "novelty_containment_hours": FROZEN_CONFIG.novelty_containment_hours,
            "maximum_novelty_containment": FROZEN_CONFIG.maximum_novelty_containment,
            "stop_if_failed": True,
        },
        "support_comparators": [
            {
                "candidate": "SDDR-12",
                "clock": str(SDDR_CLOCK),
                "clock_sha256": SDDR_CLOCK_SHA256,
                "support": str(SDDR_SUPPORT),
                "support_sha256": SDDR_SUPPORT_SHA256,
                "controls": ["primary"],
            },
            {
                "candidate": "SQFD-6",
                "clock": str(SQFD_CLOCK),
                "clock_sha256": SQFD_CLOCK_SHA256,
                "support": str(SQFD_SUPPORT),
                "support_sha256": SQFD_SUPPORT_SHA256,
                "controls": ["primary", "no_usdt_lag", "no_participation"],
            },
        ],
        "later_outcome_contract": {
            "evaluator_must_be_committed_before_outcome": True,
            "sequential_stages": ["train_2023", "test_2024", "eval_2025", "final_2026h1"],
            "stop_on_first_failure": True,
            "leverage": 0.5,
            "base_cost_bp_per_notional_side": 6.0,
            "stress_cost_bp_per_notional_side": 10.0,
            "full_calendar_cagr": True,
            "strict_position_path_mdd": True,
            "exact_realized_funding": True,
            "minimum_trades_per_stage": 30,
            "absolute_return_positive": True,
            "predeclared_half_periods_positive": True,
            "primary_ratio_minimum": 3.0,
            "stress_ratio_minimum": 2.5,
            "strict_mdd_maximum_pct": 15.0,
            "mean_gross_move_minimum_bp": 25.0,
            "weekly_sign_flip_p_maximum": 0.10,
            "control_ratio_margin": 0.25,
            "mandatory_controls": [
                "direction_flip",
                "deterministic_random_side",
                "extra_latency_1h",
                "all_four",
                "leave_one_issuer_out",
                "median_only",
                "matched_btcusdt_12h_momentum_side",
                "matched_btcusdt_12h_reversion_side",
            ],
        },
        "rllm_boundary": "Gemma/RLLM is prohibited until the deterministic UCBR clock demonstrates gross edge above costs; it may later abstain or route risk but may not repair source timing, direction, breadth, or threshold",
        "preregistration_source": str(PREREGISTRATION_SOURCE),
        "preregistration_source_sha256": sha256_file(PREREGISTRATION_SOURCE),
        "preregistration_document": str(PREREGISTRATION_DOCUMENT),
        "preregistration_document_sha256": sha256_file(PREREGISTRATION_DOCUMENT),
        "output": str(output),
    }
    payload["manifest_hash"] = canonical_hash(payload)
    return payload


def write_preregistration(output: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    path = _resolve(output)
    expected = preregistration_payload(output)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != expected:
            raise FileExistsError("existing UCBR preregistration differs from frozen payload")
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(expected, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return expected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    print(json.dumps(write_preregistration(args.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
