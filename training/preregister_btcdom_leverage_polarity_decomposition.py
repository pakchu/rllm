"""Preregister the outcome-blind DLPD-12 source clock and controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from training import build_binance_btcdom_premium_decomposition_source as source_builder


REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_ID = "DLPD-12"
PROTOCOL_VERSION = "btcdom_leverage_polarity_decomposition_v1"
SOURCE_PANEL = Path(
    "data/binance_btcdom_premium_decomposition_2021_2023/"
    "BTCUSDT_BTCDOMUSDT_premium_close_1h_2021-07-02_2023-12-31.csv.gz"
)
SOURCE_MANIFEST = Path(
    "data/binance_btcdom_premium_decomposition_2021_2023/build_manifest.json"
)
SOURCE_PANEL_SHA256 = (
    "75fb36b33810134746515e3ad99234e2a52f6f721551792788f6d3950ff5b1d9"
)
SOURCE_MANIFEST_SHA256 = (
    "885014743c299250c85cec42561db0dc99b09a60ecb1adfe893d8cac95651c05"
)
PREREGISTRATION_SOURCE = Path(
    "training/preregister_btcdom_leverage_polarity_decomposition.py"
)
PREREGISTRATION_DOCUMENT = Path(
    "docs/btcdom-leverage-polarity-decomposition-preregistration-2026-07-20.md"
)
DEFAULT_OUTPUT = Path(
    "results/btcdom_leverage_polarity_decomposition_preregistration_2026-07-20.json"
)
SOURCE_ONLY_CONTROLS = (
    "btc_only_tail",
    "dom_only_mirror",
    "same_sign",
    "stale_btc_1h",
    "stale_dom_1h",
)
CONTROLS = ("primary", *SOURCE_ONLY_CONTROLS)
EVENT_COLUMNS = (
    "candidate",
    "control",
    "split",
    "source_hour_start",
    "decision_time",
    "feature_available_time",
    "entry_time",
    "exit_time",
    "side",
    "btc_premium_z",
    "btcdom_premium_z",
)
COMPARATORS: tuple[dict[str, Any], ...] = (
    {
        "candidate": "PSR-30/6",
        "clock": "data/premium_snapback_recenter_clocks_2020_2026.csv.gz",
        "clock_sha256": "cb209ed35f9baa08cc2fb3dd5bd60b8e747b1408c09507b774ca275e0b2b2db6",
        "support": "results/premium_snapback_recenter_support_2026-07-19.json",
        "support_sha256": "f33708368b089dd588051971b8d17b4174aaac304ead7a30b07ebb3ee3520b4f",
        "format": "csv",
        "entry_field": "entry_time",
        "control": None,
    },
    {
        "candidate": "PCBR-12",
        "clock": "data/premium_compression_breakout_relay_clocks_2020_2026.csv.gz",
        "clock_sha256": "659fc1b6b6e3a20e60031ed1d50f51c8c7d2836956f911f62ad13e4152740cda",
        "support": "results/premium_compression_breakout_relay_support_2026-07-19.json",
        "support_sha256": "de41852acb7987685d31a799eddf56a7e59afa756f5435ce46e054ea72f83857",
        "format": "csv",
        "entry_field": "entry_time",
        "control": "primary",
    },
    {
        "candidate": "OPDR-24",
        "clock": "data/options_perpetual_demand_relay_clocks_2023_2026.csv.gz",
        "clock_sha256": "ceb79b206c3e1f6bf78b02cd2ace9a94f875ce930a704cc6e7a5a8b255021b99",
        "support": "results/options_perpetual_demand_relay_support_2026-07-19.json",
        "support_sha256": "d8a82c072c45a2e965b8e4d05383aa3cb7f39d92728aef54ccd51ad54a02b9f3",
        "format": "csv",
        "entry_field": "entry_time",
        "control": "primary",
    },
    {
        "candidate": "CLD-72",
        "clock": "results/cross_sectional_leadership_diffusion_event_clock_2026-07-18.json",
        "clock_sha256": "089ae3f854459a76bade4e3fd6682d1b1a9a6d600dc990a367840c179c0e623d",
        "support": "results/cross_sectional_leadership_diffusion_support_2026-07-18.json",
        "support_sha256": "e2e23be7504473edc0d5df44b5a25d2fa2ec6f82770206cf35bdf9ca66e020dc",
        "format": "json_events",
        "entry_field": "entry_date",
        "control": None,
    },
    {
        "candidate": "FCIR-12",
        "clock": "data/flow_centrality_incubation_relay_clocks_2023_2026.csv.gz",
        "clock_sha256": "d4bb6245f0bac34885e780e35ff1edb9b5cf2114dc3c13088ec19613ad8056ea",
        "support": "results/flow_centrality_incubation_relay_support_2026-07-19.json",
        "support_sha256": "2ecd30c6a4f8678207053522aabe7ef6bfbc24e3f5d29b4e52743c218e4d2e89",
        "format": "csv",
        "entry_field": "entry_time",
        "control": None,
    },
)


@dataclass(frozen=True)
class Config:
    lookback_hours: int = 720
    minimum_history_hours: int = 672
    absolute_z_threshold: float = 1.0
    hold_hours: int = 12
    entry_delay_minutes: int = 5
    support_years: tuple[int, ...] = (2022, 2023)
    minimum_events_per_year: int = 120
    minimum_events_per_quarter: int = 20
    minimum_side_share: float = 0.25
    maximum_month_share: float = 0.20
    novelty_year: int = 2023
    novelty_hours: int = 1
    maximum_exact_entry_jaccard: float = 0.10
    maximum_bidirectional_near_share: float = 0.35


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
        raise ValueError("DLPD-12 configuration is frozen")
    if cfg.minimum_history_hours > cfg.lookback_hours:
        raise ValueError("DLPD minimum history exceeds lookback")
    if cfg.hold_hours != 12 or cfg.entry_delay_minutes != 5:
        raise ValueError("DLPD execution clock changed")
    if cfg.support_years != (2022, 2023):
        raise ValueError("DLPD support years changed")


def _parse_bool(values: pd.Series, *, field: str) -> pd.Series:
    if values.dtype == bool:
        return values.astype(bool)
    parsed = values.astype("string").str.lower().map({"true": True, "false": False})
    if parsed.isna().any():
        raise ValueError(f"DLPD source {field} contains an unknown value")
    return parsed.astype(bool)


def load_source() -> tuple[pd.DataFrame, dict[str, Any]]:
    if sha256_file(SOURCE_PANEL) != SOURCE_PANEL_SHA256:
        raise ValueError("DLPD source panel hash mismatch")
    if sha256_file(SOURCE_MANIFEST) != SOURCE_MANIFEST_SHA256:
        raise ValueError("DLPD source manifest hash mismatch")
    manifest = json.loads(_resolve(SOURCE_MANIFEST).read_text(encoding="utf-8"))
    protocol = manifest.get("protocol", {})
    if protocol.get("source_only") is not True or protocol.get("outcomes_opened") is not False:
        raise ValueError("DLPD source manifest opened outcomes")
    forbidden_false = (
        "post_2023_rows_requested",
        "btc_or_btcdom_contract_ohlc_retained",
        "btc_or_btcdom_index_prices_retained",
        "funding_returns_labels_or_pnl_retained",
        "premium_ohlc_paths_retained",
        "missing_rows_zero_filled",
    )
    if any(protocol.get(key) is not False for key in forbidden_false):
        raise ValueError("DLPD source manifest violates its unopened boundary")
    if protocol.get("premium_closes_retained") is not True:
        raise ValueError("DLPD source manifest lost premium closes")
    if manifest.get("combined_sha256") != SOURCE_PANEL_SHA256:
        raise ValueError("DLPD manifest panel hash differs")

    frame = cast(
        pd.DataFrame,
        pd.read_csv(
            _resolve(SOURCE_PANEL),
            usecols=list(source_builder.OUTPUT_COLUMNS),
            parse_dates=["date", "source_close_time", "feature_available_time"],
        ),
    )
    expected = pd.date_range(source_builder.START, source_builder.END, freq="1h", inclusive="left")
    if not frame["date"].equals(pd.Series(expected, name="date")):
        raise ValueError("DLPD source grid changed")
    if not frame["source_close_time"].equals(
        pd.Series(
            expected + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1),
            name="source_close_time",
        )
    ):
        raise ValueError("DLPD source close times changed")
    if not frame["feature_available_time"].equals(
        pd.Series(
            expected + pd.Timedelta(hours=1, seconds=1),
            name="feature_available_time",
        )
    ):
        raise ValueError("DLPD source availability changed")
    for column in ("btcusdt_valid", "btcdomusdt_valid", "source_valid"):
        frame[column] = _parse_bool(cast(pd.Series, frame[column]), field=column)
    if not frame["source_valid"].equals(
        frame["btcusdt_valid"] & frame["btcdomusdt_valid"]
    ):
        raise ValueError("DLPD source pair-valid flag changed")
    return frame, {
        "panel_sha256": SOURCE_PANEL_SHA256,
        "manifest_sha256": SOURCE_MANIFEST_SHA256,
        "rows": int(len(frame)),
        "valid_rows": int(frame["source_valid"].sum()),
        "first_date": cast(pd.Timestamp, frame["date"].iloc[0]).isoformat(),
        "last_date": cast(pd.Timestamp, frame["date"].iloc[-1]).isoformat(),
    }


def prior_robust_z(values: pd.Series, valid: pd.Series, cfg: Config = FROZEN_CONFIG) -> pd.Series:
    admitted = values.where(valid.astype(bool))
    prior = admitted.shift(1).rolling(
        cfg.lookback_hours,
        min_periods=cfg.minimum_history_hours,
    )
    median = prior.median()
    scale = (prior.quantile(0.75) - prior.quantile(0.25)) / 1.349
    return cast(
        pd.Series,
        ((values - median) / scale).where(
            valid.astype(bool) & scale.gt(0.0) & np.isfinite(scale)
        ),
    )


def _attach_state(
    state: pd.DataFrame,
    *,
    control: str,
    btc_z: pd.Series,
    dom_z: pd.Series,
    active: pd.Series,
    side: pd.Series,
) -> None:
    state[f"{control}_active"] = active.fillna(False).astype(bool)
    state[f"{control}_side"] = side.where(active, 0).fillna(0).astype("int8")
    state[f"{control}_btc_z"] = btc_z.astype(float)
    state[f"{control}_dom_z"] = dom_z.astype(float)


def signal_states(frame: pd.DataFrame, cfg: Config = FROZEN_CONFIG) -> pd.DataFrame:
    _validate_config(cfg)
    pair_valid = cast(pd.Series, frame["source_valid"]).astype(bool)
    btc_z = prior_robust_z(
        cast(pd.Series, frame["btcusdt_premium_close"]),
        cast(pd.Series, frame["btcusdt_valid"]),
        cfg,
    )
    dom_z = prior_robust_z(
        cast(pd.Series, frame["btcdomusdt_premium_close"]),
        cast(pd.Series, frame["btcdomusdt_valid"]),
        cfg,
    )
    threshold = cfg.absolute_z_threshold
    btc_tail = btc_z.abs().ge(threshold)
    dom_tail = dom_z.abs().ge(threshold)
    btc_side = np.sign(btc_z).fillna(0).astype("int8")
    dom_mirror_side = (-np.sign(dom_z)).fillna(0).astype("int8")
    state = pd.DataFrame(index=frame.index)

    _attach_state(
        state,
        control="primary",
        btc_z=btc_z,
        dom_z=dom_z,
        active=pair_valid & btc_tail & dom_tail & btc_z.mul(dom_z).lt(0.0),
        side=btc_side,
    )
    _attach_state(
        state,
        control="btc_only_tail",
        btc_z=btc_z,
        dom_z=dom_z,
        active=pair_valid & btc_tail,
        side=btc_side,
    )
    _attach_state(
        state,
        control="dom_only_mirror",
        btc_z=btc_z,
        dom_z=dom_z,
        active=pair_valid & dom_tail,
        side=dom_mirror_side,
    )
    _attach_state(
        state,
        control="same_sign",
        btc_z=btc_z,
        dom_z=dom_z,
        active=pair_valid & btc_tail & dom_tail & btc_z.mul(dom_z).gt(0.0),
        side=btc_side,
    )

    stale_btc = btc_z.shift(1)
    stale_dom = dom_z.shift(1)
    _attach_state(
        state,
        control="stale_btc_1h",
        btc_z=stale_btc,
        dom_z=dom_z,
        active=(
            pair_valid
            & stale_btc.abs().ge(threshold)
            & dom_tail
            & stale_btc.mul(dom_z).lt(0.0)
        ),
        side=np.sign(stale_btc).fillna(0).astype("int8"),
    )
    _attach_state(
        state,
        control="stale_dom_1h",
        btc_z=btc_z,
        dom_z=stale_dom,
        active=(
            pair_valid
            & btc_tail
            & stale_dom.abs().ge(threshold)
            & btc_z.mul(stale_dom).lt(0.0)
        ),
        side=btc_side,
    )
    return state


def schedule(
    frame: pd.DataFrame,
    states: pd.DataFrame,
    *,
    control: str,
    year: int,
    cfg: Config = FROZEN_CONFIG,
) -> pd.DataFrame:
    _validate_config(cfg)
    if control not in CONTROLS:
        raise ValueError(f"unknown DLPD control: {control}")
    if year not in cfg.support_years:
        raise ValueError(f"year is outside DLPD source support: {year}")
    active = cast(pd.Series, states[f"{control}_active"]).astype(bool)
    onset = active & ~active.shift(1, fill_value=False)
    start = pd.Timestamp(f"{year}-01-01")
    end = pd.Timestamp(f"{year + 1}-01-01")
    next_entry = start
    rows: list[dict[str, Any]] = []
    for idx in frame.index[onset]:
        source_hour = cast(pd.Timestamp, frame.at[idx, "date"])
        decision = source_hour + pd.Timedelta(hours=1)
        feature_available = cast(pd.Timestamp, frame.at[idx, "feature_available_time"])
        entry = decision + pd.Timedelta(minutes=cfg.entry_delay_minutes)
        exit_time = entry + pd.Timedelta(hours=cfg.hold_hours)
        if entry < start or exit_time > end or entry < next_entry:
            continue
        side = int(states.at[idx, f"{control}_side"])
        if side not in (-1, 1):
            raise ValueError("DLPD active event has a non-directional side")
        if feature_available > entry:
            raise ValueError("DLPD feature becomes available after entry")
        rows.append(
            {
                "candidate": POLICY_ID,
                "control": control,
                "split": str(year),
                "source_hour_start": source_hour,
                "decision_time": decision,
                "feature_available_time": feature_available,
                "entry_time": entry,
                "exit_time": exit_time,
                "side": side,
                "btc_premium_z": float(states.at[idx, f"{control}_btc_z"]),
                "btcdom_premium_z": float(states.at[idx, f"{control}_dom_z"]),
            }
        )
        next_entry = exit_time
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def preregistration_payload(cfg: Config = FROZEN_CONFIG) -> dict[str, Any]:
    _validate_config(cfg)
    if sha256_file(SOURCE_PANEL) != SOURCE_PANEL_SHA256:
        raise ValueError("DLPD source panel changed before preregistration")
    if sha256_file(SOURCE_MANIFEST) != SOURCE_MANIFEST_SHA256:
        raise ValueError("DLPD source manifest changed before preregistration")
    for contract in COMPARATORS:
        if sha256_file(contract["clock"]) != contract["clock_sha256"]:
            raise ValueError(f"DLPD comparator clock changed: {contract['candidate']}")
        if sha256_file(contract["support"]) != contract["support_sha256"]:
            raise ValueError(f"DLPD comparator support changed: {contract['candidate']}")
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "as_of_date": "2026-07-20",
        "candidate": POLICY_ID,
        "source_panel": str(SOURCE_PANEL),
        "source_panel_sha256": SOURCE_PANEL_SHA256,
        "source_manifest": str(SOURCE_MANIFEST),
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "preregistration_source": str(PREREGISTRATION_SOURCE),
        "preregistration_source_sha256": sha256_file(PREREGISTRATION_SOURCE),
        "preregistration_document": str(PREREGISTRATION_DOCUMENT),
        "preregistration_document_sha256": sha256_file(PREREGISTRATION_DOCUMENT),
        "outcomes_opened": False,
        "outcome_sources_opened": False,
        "post_2023_source_rows_opened": False,
        "real_event_incidence_opened": False,
        "policy": {
            "config": {**asdict(cfg), "support_years": list(cfg.support_years)},
            "primary": "opposite extreme robust premium z-scores; side=BTC premium sign",
            "clock": "false-to-true onset; decision at hour close; entry +5m; fixed 12h hold",
            "normalization": "strictly prior 720-hour rolling median and IQR/1.349; minimum 672 valid",
        },
        "source_only_controls": list(SOURCE_ONLY_CONTROLS),
        "support_gate": {
            "years": list(cfg.support_years),
            "minimum_events_per_year": cfg.minimum_events_per_year,
            "minimum_events_per_quarter": cfg.minimum_events_per_quarter,
            "minimum_side_share": cfg.minimum_side_share,
            "maximum_month_share": cfg.maximum_month_share,
            "novelty_year": cfg.novelty_year,
            "novelty_hours": cfg.novelty_hours,
            "maximum_exact_entry_jaccard": cfg.maximum_exact_entry_jaccard,
            "maximum_bidirectional_near_share": cfg.maximum_bidirectional_near_share,
        },
        "support_comparators": [dict(item) for item in COMPARATORS],
        "conditional_outcome_gate": {
            "sequence": ["train_2022", "test_2023", "eval_2024_2025", "final_2026H1"],
            "each_opened_stage": {
                "absolute_return_positive": True,
                "cagr_to_strict_mdd_at_least": 3.0,
                "strict_mdd_at_most": 0.15,
                "ten_bp_stress_absolute_return_positive": True,
                "contained_subperiods_positive": True,
                "weekly_cluster_signflip_p_at_most": 0.10,
                "direction_flip_inferior": True,
            },
            "minimum_train_and_test_trades_each": 120,
            "component_controls_cannot_repair_primary": True,
            "source_support_is_not_profitability_evidence": True,
        },
    }
    payload["manifest_hash"] = canonical_hash(payload)
    return payload


def write_frozen_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = _resolve(path)
    encoded = (
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if target.exists() and target.read_bytes() != encoded:
        raise FileExistsError(f"existing frozen DLPD preregistration differs: {path}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(encoded)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    payload = preregistration_payload()
    write_frozen_json(args.output, payload)
    print(
        json.dumps(
            {
                "candidate": payload["candidate"],
                "output": args.output,
                "manifest_hash": payload["manifest_hash"],
                "outcomes_opened": payload["outcomes_opened"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
