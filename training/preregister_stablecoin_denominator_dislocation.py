"""Preregister the outcome-blind SDDR-12 support clock.

This module defines and freezes one stablecoin denominator-dislocation rule.
It may transform the frozen cross-price source and compare event timestamps to
frozen source-only clocks, but it never loads BTCUSDT perpetual OHLC, funding,
future returns, labels, PnL, or post-2023 source rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_ID = "SDDR-12"
PROTOCOL_VERSION = "stablecoin_denominator_dislocation_v1"
SOURCE_PANEL = Path(
    "data/binance_stablecoin_denominator_btc_2023/"
    "BTC_stablecoin_denominator_1h_2023-08-04T08_2023-12-31T23.csv.gz"
)
SOURCE_MANIFEST = Path(
    "data/binance_stablecoin_denominator_btc_2023/build_manifest.json"
)
SOURCE_PANEL_SHA256 = (
    "aab063f0f9d898d5cdafffb57f552244083cd93fe69a3c6ebaf97faf6e27b642"
)
SOURCE_MANIFEST_SHA256 = (
    "863e96b4325d051731c92852c6760986204a9df62f77ff0dd0e01ab08d8a15d3"
)
SQFD_CLOCK = Path("data/stablecoin_quote_flow_diffusion_clocks_2023_2026.csv.gz")
SQFD_CLOCK_SHA256 = (
    "a81e144eea1e80ae5439fc66db1fad5bbd00cd9ac177e25142b5cfb5a07bcc5b"
)
SQFD_SUPPORT = Path("results/stablecoin_quote_flow_diffusion_support_2026-07-19.json")
SQFD_SUPPORT_SHA256 = (
    "07230e9e579f1b16e07712a022e572026b4fbfa17070e998970b3fd8ee21d4b5"
)
PREREGISTRATION_SOURCE = Path(
    "training/preregister_stablecoin_denominator_dislocation.py"
)
PREREGISTRATION_DOCUMENT = Path(
    "docs/stablecoin-denominator-dislocation-preregistration-2026-07-20.md"
)
DEFAULT_OUTPUT = Path(
    "results/stablecoin_denominator_dislocation_preregistration_2026-07-20.json"
)

COMPARATOR_CONTROLS = ("primary", "no_usdt_lag", "no_participation")
SOURCE_COLUMNS = (
    "date",
    "source_available_at",
    "usdc_vs_usdt",
    "fdusd_vs_usdt",
    "alt_consensus",
    "alt_disagreement",
    "source_complete",
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
    "z_usdc",
    "z_fdusd",
    "min_abs_z",
    "prior_disagreement_q80",
    "alt_disagreement",
)


@dataclass(frozen=True)
class Config:
    lookback_hours: int = 720
    minimum_history_hours: int = 672
    z_threshold: float = 1.0
    disagreement_quantile: float = 0.80
    hold_bars: int = 12
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


def sha256_file(path: str | Path) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return hashlib.sha256(candidate.read_bytes()).hexdigest()


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
        raise ValueError("SDDR-12 support configuration is frozen")
    if cfg.minimum_history_hours > cfg.lookback_hours:
        raise ValueError("minimum SDDR history cannot exceed lookback")
    if cfg.hold_bars * cfg.bar_minutes != 60:
        raise ValueError("SDDR-12 hold must remain exactly one hour")


def _parse_complete(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.astype(bool)
    parsed = values.astype("string").str.lower().map({"true": True, "false": False})
    if parsed.isna().any():
        raise ValueError("SDDR source_complete contains an unknown value")
    return parsed.astype(bool)


def load_source() -> tuple[pd.DataFrame, dict[str, Any]]:
    if sha256_file(SOURCE_PANEL) != SOURCE_PANEL_SHA256:
        raise ValueError("SDDR source panel hash mismatch")
    if sha256_file(SOURCE_MANIFEST) != SOURCE_MANIFEST_SHA256:
        raise ValueError("SDDR source manifest hash mismatch")
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    protocol = manifest.get("protocol", {})
    forbidden = (
        "outcomes_opened",
        "perpetual_ohlc_or_funding_opened",
        "future_returns_labels_or_pnl_opened",
        "post_2023_rows_requested",
        "raw_btc_prices_retained",
        "flow_or_volume_fields_retained",
    )
    if any(protocol.get(key) is not False for key in forbidden):
        raise ValueError("SDDR source manifest violates the outcome-blind contract")
    if protocol.get("cross_quote_log_ratios_retained") is not True:
        raise ValueError("SDDR source manifest lacks cross-price ratios")
    if manifest.get("combined_sha256") != SOURCE_PANEL_SHA256:
        raise ValueError("SDDR manifest panel hash differs")

    frame = cast(
        pd.DataFrame,
        pd.read_csv(
            SOURCE_PANEL,
            usecols=list(SOURCE_COLUMNS),
            parse_dates=["date", "source_available_at"],
        ),
    )
    expected = pd.date_range(
        "2023-08-04 08:00", "2024-01-01", freq="1h", inclusive="left"
    )
    if not frame["date"].equals(pd.Series(expected, name="date")):
        raise ValueError("SDDR source is not the exact frozen common-hour grid")
    if not frame["source_available_at"].equals(
        pd.Series(expected + pd.Timedelta(hours=1), name="source_available_at")
    ):
        raise ValueError("SDDR source availability is not exact hour close")
    frame["source_complete"] = _parse_complete(frame["source_complete"])
    if not frame["source_complete"].all():
        raise ValueError("SDDR frozen source contains an incomplete common hour")
    feature_columns = list(SOURCE_COLUMNS[2:-1])
    if not np.isfinite(frame[feature_columns].to_numpy(float)).all():
        raise ValueError("SDDR source contains non-finite features")
    return frame, {
        "panel_sha256": SOURCE_PANEL_SHA256,
        "manifest_sha256": SOURCE_MANIFEST_SHA256,
        "rows": int(len(frame)),
        "first_date": str(frame["date"].iloc[0]),
        "last_date": str(frame["date"].iloc[-1]),
    }


def prior_quantile(
    values: pd.Series,
    *,
    quantile: float,
    window: int,
    minimum: int,
) -> pd.Series:
    """Return a rolling quantile whose newest admitted value is at t-1."""
    return cast(
        pd.Series,
        values.shift(1)
        .rolling(window, min_periods=minimum)
        .quantile(quantile, interpolation="linear"),
    )


def prior_robust_z(values: pd.Series, cfg: Config) -> pd.DataFrame:
    median = prior_quantile(
        values,
        quantile=0.50,
        window=cfg.lookback_hours,
        minimum=cfg.minimum_history_hours,
    )
    lower = prior_quantile(
        values,
        quantile=0.25,
        window=cfg.lookback_hours,
        minimum=cfg.minimum_history_hours,
    )
    upper = prior_quantile(
        values,
        quantile=0.75,
        window=cfg.lookback_hours,
        minimum=cfg.minimum_history_hours,
    )
    scale = (upper - lower) / 1.349
    zscore = ((values - median) / scale).where(scale.gt(0.0) & np.isfinite(scale))
    return pd.DataFrame(
        {"median": median, "lower": lower, "upper": upper, "scale": scale, "z": zscore}
    )


def signal_states(frame: pd.DataFrame, cfg: Config = FROZEN_CONFIG) -> pd.DataFrame:
    _validate_config(cfg)
    usdc = prior_robust_z(cast(pd.Series, frame["usdc_vs_usdt"]), cfg)
    fdusd = prior_robust_z(cast(pd.Series, frame["fdusd_vs_usdt"]), cfg)
    disagreement_threshold = prior_quantile(
        cast(pd.Series, frame["alt_disagreement"]),
        quantile=cfg.disagreement_quantile,
        window=cfg.lookback_hours,
        minimum=cfg.minimum_history_hours,
    )
    z_usdc = usdc["z"]
    z_fdusd = fdusd["z"]
    same_sign = np.sign(z_usdc) == np.sign(z_fdusd)
    min_abs_z = pd.concat([z_usdc.abs(), z_fdusd.abs()], axis=1).min(axis=1)
    coherent = cast(pd.Series, frame["alt_disagreement"]).le(disagreement_threshold)
    primary_active = (
        z_usdc.notna()
        & z_fdusd.notna()
        & same_sign
        & np.sign(z_usdc).ne(0.0)
        & min_abs_z.ge(cfg.z_threshold)
        & coherent
    )
    no_disagreement_active = (
        z_usdc.notna()
        & z_fdusd.notna()
        & same_sign
        & np.sign(z_usdc).ne(0.0)
        & min_abs_z.ge(cfg.z_threshold)
    )
    usdc_only_active = z_usdc.notna() & z_usdc.abs().ge(cfg.z_threshold)
    fdusd_only_active = z_fdusd.notna() & z_fdusd.abs().ge(cfg.z_threshold)
    side = np.sign((z_usdc + z_fdusd) / 2.0).fillna(0.0).astype(np.int8)
    stale_active = primary_active.shift(1, fill_value=False).astype(bool)
    stale_side = side.shift(1, fill_value=0).astype(np.int8)
    return pd.DataFrame(
        {
            "z_usdc": z_usdc,
            "z_fdusd": z_fdusd,
            "min_abs_z": min_abs_z,
            "prior_disagreement_q80": disagreement_threshold,
            "primary_active": primary_active,
            "primary_side": side,
            "no_disagreement_active": no_disagreement_active,
            "no_disagreement_side": side,
            "usdc_only_active": usdc_only_active,
            "usdc_only_side": np.sign(z_usdc).fillna(0.0).astype(np.int8),
            "fdusd_only_active": fdusd_only_active,
            "fdusd_only_side": np.sign(z_fdusd).fillna(0.0).astype(np.int8),
            "stale_1h_active": stale_active,
            "stale_1h_side": stale_side,
        }
    )


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
    side_column = f"{control}_side"
    if active_column not in states or side_column not in states:
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
        decision = cast(pd.Timestamp, available.iloc[position])
        entry = decision + entry_delay
        exit_time = entry + hold
        if entry < next_allowed or exit_time > end:
            continue
        side = int(states.iloc[position][side_column])
        if side not in (-1, 1):
            raise ValueError("SDDR scheduled side must be +/-1")
        rows.append(
            {
                "candidate": POLICY_ID,
                "control": control,
                "source_hour_start": cast(pd.Timestamp, source_dates.iloc[position]),
                "decision_time": decision,
                "feature_available_time": decision,
                "entry_time": entry,
                "exit_time": exit_time,
                "side": side,
                "z_usdc": float(states.iloc[position]["z_usdc"]),
                "z_fdusd": float(states.iloc[position]["z_fdusd"]),
                "min_abs_z": float(states.iloc[position]["min_abs_z"]),
                "prior_disagreement_q80": float(
                    states.iloc[position]["prior_disagreement_q80"]
                ),
                "alt_disagreement": float(frame.iloc[position]["alt_disagreement"]),
            }
        )
        next_allowed = exit_time
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def preregistration_payload(output: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    _validate_config(FROZEN_CONFIG)
    config_payload = asdict(FROZEN_CONFIG)
    config_payload["full_signal_months"] = list(FROZEN_CONFIG.full_signal_months)
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "as_of_date": "2026-07-20",
        "candidate": POLICY_ID,
        "outcomes_opened": False,
        "outcome_sources_opened": False,
        "post_2023_source_rows_opened": False,
        "source": {
            "panel": str(SOURCE_PANEL),
            "panel_sha256": SOURCE_PANEL_SHA256,
            "manifest": str(SOURCE_MANIFEST),
            "manifest_sha256": SOURCE_MANIFEST_SHA256,
            "persisted_observables": list(SOURCE_COLUMNS),
            "raw_btc_prices_retained": False,
        },
        "policy": {
            "config": config_payload,
            "normalization": "strictly-prior 720h median and IQR/1.349, min 672h",
            "primary": "same nonzero z sign; min(abs(z)) >= 1; current disagreement <= strictly-prior q80; false-to-true onset",
            "side": "sign(mean(z_usdc,z_fdusd)); rich USDT proxy LONG, cheap USDT proxy SHORT",
            "availability": "completed source hour h is known at h+1h",
            "entry": "BTCUSDT USD-M perpetual at h+1h+5m open",
            "exit": "scheduled open after exactly 12 five-minute bars",
            "global_nonoverlap": True,
        },
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
        "source_only_controls": [
            "no_disagreement",
            "usdc_only",
            "fdusd_only",
            "stale_1h",
        ],
        "support_comparators": {
            "clock": str(SQFD_CLOCK),
            "clock_sha256": SQFD_CLOCK_SHA256,
            "support": str(SQFD_SUPPORT),
            "support_sha256": SQFD_SUPPORT_SHA256,
            "controls": list(COMPARATOR_CONTROLS),
        },
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
            "primary_ratio_minimum": 3.0,
            "stress_ratio_minimum": 2.5,
            "strict_mdd_maximum_pct": 15.0,
            "mean_gross_move_minimum_bp": 20.0,
            "weekly_sign_flip_p_maximum": 0.10,
            "control_ratio_margin": 0.25,
            "mandatory_controls": [
                "direction_flip",
                "deterministic_random_side",
                "extra_latency_1h",
                "usdc_only",
                "fdusd_only",
                "no_disagreement",
                "matched_btcusdt_1h_momentum_side",
                "matched_btcusdt_1h_reversion_side",
            ],
        },
        "rllm_boundary": "Gemma/RLLM may be added only after the frozen deterministic base demonstrates positive gross edge above costs; model may abstain or route risk but may not repair this clock",
        "preregistration_source": str(PREREGISTRATION_SOURCE),
        "preregistration_source_sha256": sha256_file(PREREGISTRATION_SOURCE),
        "preregistration_document": str(PREREGISTRATION_DOCUMENT),
        "preregistration_document_sha256": sha256_file(PREREGISTRATION_DOCUMENT),
        "output": str(output),
    }
    payload["manifest_hash"] = canonical_hash(payload)
    return payload


def write_preregistration(output: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    path = Path(output)
    if path.exists():
        existing = json.loads(path.read_text())
        expected = preregistration_payload(path)
        if existing != expected:
            raise FileExistsError("existing SDDR preregistration differs from frozen payload")
        return existing
    payload = preregistration_payload(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    print(json.dumps(write_preregistration(args.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
