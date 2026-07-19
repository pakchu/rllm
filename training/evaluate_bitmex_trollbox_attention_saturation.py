"""Frozen sequential evaluator for TBASR-24.

The preregistration and evaluator-freeze paths inspect only committed metadata,
semantic clocks, and source manifests.  They never parse a BTC OHLC or funding
row.  Outcome stages are opened strictly in order: train, then test.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd


POLICY_ID = "TBASR-24"
AS_OF_DATE = "2026-07-20"
SUPPORT_COMMIT = "e7cd2448176d29146fb501d054a759afc986e8fc"

EVALUATOR_SOURCE = Path(
    "training/evaluate_bitmex_trollbox_attention_saturation.py"
)
PREREGISTRATION = Path(
    "results/bitmex_trollbox_attention_saturation_preregistration_2026-07-20.json"
)
PREREGISTRATION_SHA256 = (
    "1db642127a01b5910267a9b986186e9fb1e7d31dccb81170df476461af669b21"
)
PREREGISTRATION_DOC = Path(
    "docs/bitmex-trollbox-attention-saturation-evaluator-preregistration-2026-07-20.md"
)
PREREGISTRATION_DOC_SHA256 = (
    "ff4a3c15128f6d3da89786b17168338751050fd52e8accf31f3fa6c7a8b5cd37"
)
EVALUATOR_FREEZE = Path(
    "results/bitmex_trollbox_attention_saturation_evaluator_freeze_2026-07-20.json"
)
SEMANTIC_SUPPORT = Path(
    "results/bitmex_trollbox_semantic_support_2026-07-20.json"
)
SEMANTIC_SUPPORT_SHA256 = (
    "2b89f710d59a5c0708d400541defb43d5e292f6d9bdedbe66d6bdcf614d09e94"
)
SEMANTIC_CLOCK = Path(
    "results/bitmex_trollbox_semantic_clock_2026-07-20.json"
)
SEMANTIC_CLOCK_SHA256 = (
    "af8687564614ec5a1cbd7a1438c908f687af7bd99ceede9539016e5c1b111bd4"
)
SEMANTIC_CLOCK_MANIFEST_HASH = (
    "fdcd9c7c376b18df2799acf24af04a421ca679e27009e6a539888defc7438aa8"
)
SEMANTIC_SUPPORT_RESULT_HASH = (
    "5996b7d7497d6bf5e96343f7ceca766363d58aa34280aea0fdb7b8653a8b1725"
)

MARKET_MANIFEST = Path(
    "data/binance_um_kline_reference_btc_2020_2023/build_manifest.json"
)
MARKET_MANIFEST_SHA256 = (
    "c04fbbd299cc748a6745c0ef030787da4d560833c744c81c98dd8840efc7913e"
)
MARKET_COMBINED = Path(
    "data/binance_um_kline_reference_btc_2020_2023/"
    "BTCUSDT_5m_2020-01-01_2023-12-31.csv.gz"
)
MARKET_COMBINED_SHA256 = (
    "e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d"
)
FUNDING = Path("data/binance_um_btcusdt_funding_marks_2020_2023.csv.gz")
FUNDING_SHA256 = (
    "3284bbb6bb67946acb673c6b67459543e217f752589e1d47b6c7c3b659f733e6"
)
FUNDING_MANIFEST = Path(
    "results/binance_um_btcusdt_funding_marks_2020_2023_manifest_2026-07-17.json"
)
FUNDING_MANIFEST_SHA256 = (
    "a0b2d27e1aa8cf2d9ab8cb659b598ee0a6d7bd25401c9e10ae92d1a74415845b"
)

STAGE_ORDER = ("train", "test")
STAGE_WINDOWS = {
    "train": (
        pd.Timestamp("2020-07-01T00:00:00Z"),
        pd.Timestamp("2022-01-01T00:00:00Z"),
    ),
    "test": (
        pd.Timestamp("2022-01-01T00:00:00Z"),
        pd.Timestamp("2023-01-01T00:00:00Z"),
    ),
}
SUBPERIOD_WINDOWS = {
    "train": {
        "2020_h2": (
            pd.Timestamp("2020-07-01T00:00:00Z"),
            pd.Timestamp("2021-01-01T00:00:00Z"),
        ),
        "2021_h1": (
            pd.Timestamp("2021-01-01T00:00:00Z"),
            pd.Timestamp("2021-07-01T00:00:00Z"),
        ),
        "2021_h2": (
            pd.Timestamp("2021-07-01T00:00:00Z"),
            pd.Timestamp("2022-01-01T00:00:00Z"),
        ),
    },
    "test": {
        "2022_h1": (
            pd.Timestamp("2022-01-01T00:00:00Z"),
            pd.Timestamp("2022-07-01T00:00:00Z"),
        ),
        "2022_h2": (
            pd.Timestamp("2022-07-01T00:00:00Z"),
            pd.Timestamp("2023-01-01T00:00:00Z"),
        ),
    },
}
STAGE_OUTPUTS = {
    "train": Path(
        "results/bitmex_trollbox_attention_saturation_train_2020h2_2021_2026-07-20.json"
    ),
    "test": Path(
        "results/bitmex_trollbox_attention_saturation_test_2022_2026-07-20.json"
    ),
}
STAGE_DOCS = {
    "train": Path(
        "docs/bitmex-trollbox-attention-saturation-train-result-2026-07-20.md"
    ),
    "test": Path(
        "docs/bitmex-trollbox-attention-saturation-test-result-2026-07-20.md"
    ),
}

SOURCE_START = pd.Timestamp("2020-01-01T00:00:00Z")
BAR = pd.Timedelta(minutes=5)
YEAR_SECONDS = 365.25 * 86_400.0
PRIMARY = "primary"
MECHANISM_CONTROLS = (
    "direction_flip",
    "deterministic_random_side",
    "semantic_alignment_ablation",
)
ALL_CONTROLS = (PRIMARY, *MECHANISM_CONTROLS)
GATE_CHECK_KEYS = (
    "absolute_return_positive",
    "cagr_to_strict_mdd_at_least_3",
    "strict_mdd_at_most_15pct",
    "minimum_trades",
    "minimum_longs",
    "minimum_shorts",
    "minimum_weekly_clusters",
    "weekly_cluster_signflip_p_at_most_10pct",
    "mean_gross_underlying_at_least_20bp",
    "each_half_year_absolute_return_positive",
    "stress_same_trade_count",
    "stress_absolute_return_positive",
    "stress_cagr_to_strict_mdd_at_least_2_5",
    "mechanism_control_margin_at_least_0_25",
)


@dataclass(frozen=True)
class EvaluationConfig:
    leverage: float = 1.0
    base_cost_notional_per_side: float = 0.0006
    stress_cost_notional_per_side: float = 0.0010
    hold_bars: int = 24
    displacement_bars: int = 12
    reference_shift_bars: int = 13
    reference_days: int = 28
    material_quantile: float = 0.90
    base_entry_delay_bars_after_observation: int = 1
    stress_extra_delay_bars: int = 1
    exact_cluster_max: int = 20
    cluster_draws: int = 20_000
    cluster_seed: int = 20_260_720

    @property
    def reference_bars(self) -> int:
        return self.reference_days * 24 * 12


def _timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp is pd.NaT:
        raise ValueError("TBASR-24 timestamp is NaT")
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return cast(pd.Timestamp, timestamp)


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


def _seal(core: dict[str, Any]) -> dict[str, Any]:
    return {**core, "manifest_hash": _canonical_hash(core)}


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"TBASR-24 expected JSON object: {path}")
    return payload


def _verify_manifest(
    payload: dict[str, Any],
    *,
    label: str,
    hash_key: str = "manifest_hash",
    ignored: tuple[str, ...] = (),
) -> None:
    core = {
        key: value
        for key, value in payload.items()
        if key not in {hash_key, *ignored}
    }
    if payload.get(hash_key) != _canonical_hash(core):
        raise ValueError(f"TBASR-24 {label} hash changed")


def _window_payload() -> dict[str, list[str]]:
    return {
        stage: [start.isoformat(), end.isoformat()]
        for stage, (start, end) in STAGE_WINDOWS.items()
    }


def _preregistration_core() -> dict[str, Any]:
    cfg = EvaluationConfig()
    return {
        "protocol_version": "bitmex_trollbox_attention_saturation_preregistration_v1",
        "candidate": POLICY_ID,
        "as_of_date": AS_OF_DATE,
        "support_commit": SUPPORT_COMMIT,
        "claim_boundary": (
            "candidate-level frozen sequence only; the branch is globally contaminated "
            "by prior BTC research"
        ),
        "hypothesis": (
            "an unusually broad Trollbox attention burst whose frozen Gemma2 crowd "
            "direction agrees with a completed material BTC displacement marks crowded "
            "short-horizon extrapolation that mean-reverts over the next two hours"
        ),
        "stage_windows": _window_payload(),
        "semantic_input": {
            "clock": str(SEMANTIC_CLOCK),
            "clock_sha256": SEMANTIC_CLOCK_SHA256,
            "clock_manifest_hash": SEMANTIC_CLOCK_MANIFEST_HASH,
            "support": str(SEMANTIC_SUPPORT),
            "support_sha256": SEMANTIC_SUPPORT_SHA256,
            "support_result_hash": SEMANTIC_SUPPORT_RESULT_HASH,
            "clear_labels": ["BULLISH", "BEARISH"],
            "unclear_action": "no trade",
        },
        "price_displacement": {
            "bar": "Binance BTCUSDT USD-M 5m OHLC",
            "formula": (
                "for final completed bar index i ending at observation_end, "
                "d_i=ln(close_i/open_{i-11}) over exactly 12 bars"
            ),
            "material_threshold": (
                "pandas rolling(window=8064,min_periods=8064).quantile(0.90,"
                "interpolation='linear').shift(13) applied to abs(d); bar timestamps "
                "are opens, so the newest reference final bar is i-13 and its close "
                "is target_start-5m, strictly before the target starts at i-11"
            ),
            "reference_days": cfg.reference_days,
            "reference_bars": cfg.reference_bars,
            "reference_shift_bars": cfg.reference_shift_bars,
            "quantile": cfg.material_quantile,
            "current_displacement_excluded": True,
            "reference_endpoint_window": (
                "[target_start-28d,target_start) at 5m endpoints; latest endpoint "
                "target_start-5m"
            ),
            "direction_alignment": {
                "BULLISH": "d_i > 0",
                "BEARISH": "d_i < 0",
            },
            "parameter_search": [],
        },
        "policy": {
            "side": "frozen semantic contrarian_side after displacement alignment",
            "leverage": cfg.leverage,
            "entry": (
                "open at observation_end+5m; the intervening 5m bar must complete "
                "after the chat observation"
            ),
            "exit": "open exactly 24 bars / 2h after entry",
            "split_end": "base and stress exits must be strictly before split end",
            "overlap": (
                "stable chronological greedy selection; accept only if entry >= prior "
                "accepted exit"
            ),
            "base_cost_notional_per_side": cfg.base_cost_notional_per_side,
            "stress": {
                "cost_notional_per_side": cfg.stress_cost_notional_per_side,
                "entry_and_exit_delay_bars": cfg.stress_extra_delay_bars,
                "same_qualified_event_identities": True,
            },
        },
        "controls": {
            "direction_flip": "same primary clocks, opposite side",
            "deterministic_random_side": (
                "same primary clocks; SHA256(TBASR-24|random-side|UTC entry), "
                "first hexadecimal nibble parity"
            ),
            "semantic_alignment_ablation": (
                "all clear material-displacement events without requiring semantic/price "
                "direction agreement; retain semantic contrarian side and apply its own "
                "frozen greedy non-overlap"
            ),
        },
        "strict_accounting": {
            "funding_boundary": (
                "interior exact-time symmetric; exact entry/exit credits dropped and "
                "debits retained; every visited settlement mark updates strict path"
            ),
            "mdd": (
                "global/pre-entry HWM; entry fee; every held 5m favorable then adverse "
                "OHLC; funding settlement marks; virtual adverse-mark exit fee; actual "
                "exit fee"
            ),
            "cagr": "full declared split calendar including all warm-up and idle cash",
        },
        "outcome_gate": {
            "minimum_trades": {"train": 80, "test": 40},
            "minimum_trades_each_side": {"train": 20, "test": 10},
            "minimum_weekly_clusters": {"train": 40, "test": 25},
            "absolute_return_positive": True,
            "cagr_to_strict_mdd_min": 3.0,
            "strict_mdd_max_pct": 15.0,
            "weekly_cluster_signflip_p_max": 0.10,
            "mean_gross_underlying_move_bp_min": 20.0,
            "each_half_year_absolute_return_positive": True,
            "stress_absolute_return_positive": True,
            "stress_cagr_to_strict_mdd_min": 2.5,
            "mechanism_control_margin_min": 0.25,
            "statistical_test_contract": {
                "cluster_key": "UTC entry timestamp ISO year/week",
                "cluster_value": "sum of net account trade returns",
                "observed": "absolute mean net account trade return",
                "null": (
                    "independent Rademacher sign per weekly sum; signed total divided "
                    "by trade count; compare abs(null)>=observed-1e-15"
                ),
                "exact_cluster_max": cfg.exact_cluster_max,
                "draws": cfg.cluster_draws,
                "seed": cfg.cluster_seed,
                "empty_p_value": 1.0,
            },
        },
        "sequential_opening": {
            "order": list(STAGE_ORDER),
            "test_rows_must_not_be_parsed_until_train_passes": True,
            "stop_on_first_failure": True,
            "post_failure_parameter_repair": False,
        },
        "market_or_funding_rows_parsed": 0,
        "strategy_outcomes_calculated": False,
        "mutable_parameters": [],
    }


def write_preregistration(
    output_path: str | Path = PREREGISTRATION,
) -> dict[str, Any]:
    report = _seal(_preregistration_core())
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
    return report


def verify_preregistration(
    path: str | Path = PREREGISTRATION,
) -> dict[str, Any]:
    payload = _load_json(path)
    _verify_manifest(payload, label="preregistration")
    expected = _seal(_preregistration_core())
    if payload != expected:
        raise ValueError("TBASR-24 preregistration contract drifted")
    return payload


def _load_semantic_events() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    if _sha256(SEMANTIC_SUPPORT) != SEMANTIC_SUPPORT_SHA256:
        raise ValueError("TBASR-24 semantic support bytes changed")
    if _sha256(SEMANTIC_CLOCK) != SEMANTIC_CLOCK_SHA256:
        raise ValueError("TBASR-24 semantic clock bytes changed")
    support = _load_json(SEMANTIC_SUPPORT)
    support_core = {
        key: value
        for key, value in support.items()
        if key not in {"result_hash", "created_at"}
    }
    if support.get("result_hash") != _canonical_hash(support_core):
        raise ValueError("TBASR-24 semantic support result hash changed")
    if support.get("result_hash") != SEMANTIC_SUPPORT_RESULT_HASH:
        raise ValueError("TBASR-24 semantic support identity changed")
    if support.get("market_or_outcomes_opened") is not False:
        raise ValueError("TBASR-24 semantic support opened market outcomes")
    if support.get("support_gate", {}).get("passed") is not True:
        raise ValueError("TBASR-24 semantic support did not pass")

    clock = _load_json(SEMANTIC_CLOCK)
    _verify_manifest(clock, label="semantic clock", ignored=("created_at",))
    if clock.get("manifest_hash") != SEMANTIC_CLOCK_MANIFEST_HASH:
        raise ValueError("TBASR-24 semantic clock identity changed")
    if clock.get("support_result_hash") != SEMANTIC_SUPPORT_RESULT_HASH:
        raise ValueError("TBASR-24 semantic clock support binding changed")
    if clock.get("market_or_outcomes_opened") is not False:
        raise ValueError("TBASR-24 semantic clock opened market outcomes")
    events = clock.get("events")
    if not isinstance(events, list) or len(events) != 5_417:
        raise ValueError("TBASR-24 semantic event count changed")
    frame = pd.DataFrame(events)
    expected_columns = {
        "observation_start",
        "observation_end",
        "entry_earliest",
        "exit_time",
        "crowd_label",
        "contrarian_side",
        "bullish_participants",
        "bearish_participants",
        "unclear_participants",
        "selected_participants",
        "selected_messages",
        "meta_instruction_guarded_messages",
    }
    if set(frame.columns) != expected_columns:
        raise ValueError("TBASR-24 semantic event schema changed")
    for column in (
        "observation_start",
        "observation_end",
        "entry_earliest",
        "exit_time",
    ):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    if not cast(pd.Series, frame["observation_start"]).is_monotonic_increasing:
        raise ValueError("TBASR-24 semantic events are not chronological")
    if not bool(
        frame["observation_end"].sub(frame["observation_start"]).eq(BAR).all()
    ):
        raise ValueError("TBASR-24 semantic observation length changed")
    if not bool(
        frame["entry_earliest"].sub(frame["observation_end"]).eq(BAR).all()
    ):
        raise ValueError("TBASR-24 semantic entry latency changed")
    if not bool(
        frame["exit_time"]
        .sub(frame["entry_earliest"])
        .eq(EvaluationConfig().hold_bars * BAR)
        .all()
    ):
        raise ValueError("TBASR-24 semantic hold changed")
    expected_side = {"BULLISH": -1, "BEARISH": 1, "UNCLEAR": 0}
    if not bool(
        frame.apply(
            lambda row: int(row["contrarian_side"])
            == expected_side[str(row["crowd_label"])],
            axis=1,
        ).all()
    ):
        raise ValueError("TBASR-24 semantic side changed")
    return frame, support, clock


def _load_market_manifest() -> dict[str, Any]:
    if _sha256(MARKET_MANIFEST) != MARKET_MANIFEST_SHA256:
        raise ValueError("TBASR-24 market manifest bytes changed")
    payload = _load_json(MARKET_MANIFEST)
    protocol = payload.get("protocol", {})
    if protocol.get("outcomes_opened") is not False:
        raise ValueError("TBASR-24 market source manifest opened outcomes")
    if protocol.get("archive_checksums_verified") is not True:
        raise ValueError("TBASR-24 market archives lack official checksum verification")
    if payload.get("combined_output") != str(MARKET_COMBINED):
        raise ValueError("TBASR-24 combined market path changed")
    if payload.get("combined_sha256") != MARKET_COMBINED_SHA256:
        raise ValueError("TBASR-24 combined market identity changed")
    if payload.get("first_date") != "2020-01-01 00:00:00":
        raise ValueError("TBASR-24 market start changed")
    if payload.get("last_date") != "2023-12-31 23:55:00":
        raise ValueError("TBASR-24 market end changed")
    months = payload.get("months")
    if not isinstance(months, list) or len(months) != 48:
        raise ValueError("TBASR-24 market month contract changed")
    return payload


def _load_funding_manifest() -> dict[str, Any]:
    if _sha256(FUNDING_MANIFEST) != FUNDING_MANIFEST_SHA256:
        raise ValueError("TBASR-24 funding manifest bytes changed")
    payload = _load_json(FUNDING_MANIFEST)
    _verify_manifest(payload, label="funding manifest", ignored=("created_at",))
    if payload.get("outcomes_opened") is not False:
        raise ValueError("TBASR-24 funding manifest opened outcomes")
    data = payload.get("data", {})
    if data.get("path") != str(FUNDING) or data.get("sha256") != FUNDING_SHA256:
        raise ValueError("TBASR-24 funding source identity changed")
    if payload.get("mapping", {}).get("funding_time") != (
        "exact returned fundingTime retained"
    ):
        raise ValueError("TBASR-24 exact funding-time contract changed")
    return payload


def _month_contracts(stage: str) -> list[dict[str, Any]]:
    if stage not in STAGE_WINDOWS:
        raise ValueError(f"TBASR-24 unknown stage: {stage}")
    market_manifest = _load_market_manifest()
    end = STAGE_WINDOWS[stage][1]
    contracts: list[dict[str, Any]] = []
    for item in market_manifest["months"]:
        month_start = _timestamp(f"{item['month']}-01T00:00:00Z")
        if SOURCE_START <= month_start < end:
            contracts.append(
                {
                    "month": item["month"],
                    "path": item["output"],
                    "sha256": item["output_sha256"],
                    "rows": int(item["rows"]),
                    "first_date": item["first_date"],
                    "last_date": item["last_date"],
                }
            )
    expected = 24 if stage == "train" else 36
    if len(contracts) != expected:
        raise ValueError(f"TBASR-24 {stage} month count changed")
    return contracts


def _source_contracts() -> dict[str, Any]:
    market_manifest = _load_market_manifest()
    funding_manifest = _load_funding_manifest()
    return {
        "market_manifest": {
            "path": str(MARKET_MANIFEST),
            "sha256": MARKET_MANIFEST_SHA256,
            "combined_path": str(MARKET_COMBINED),
            "combined_sha256": MARKET_COMBINED_SHA256,
            "official_archive_checksums_verified": True,
            "rows": int(market_manifest["rows"]),
        },
        "funding_manifest": {
            "path": str(FUNDING_MANIFEST),
            "sha256": FUNDING_MANIFEST_SHA256,
            "data_path": str(FUNDING),
            "data_sha256": FUNDING_SHA256,
            "rows": int(funding_manifest["data"]["rows"]),
            "mark_contract": (
                "exact fundingTime/rate; official containing 8h mark-price open as "
                "the frozen settlement-mark proxy"
            ),
        },
        "stage_market_months": {
            stage: _month_contracts(stage) for stage in STAGE_ORDER
        },
    }


def _expected_static_inputs() -> dict[str, str]:
    return {
        str(PREREGISTRATION): PREREGISTRATION_SHA256,
        str(PREREGISTRATION_DOC): PREREGISTRATION_DOC_SHA256,
        str(SEMANTIC_SUPPORT): SEMANTIC_SUPPORT_SHA256,
        str(SEMANTIC_CLOCK): SEMANTIC_CLOCK_SHA256,
        str(MARKET_MANIFEST): MARKET_MANIFEST_SHA256,
        str(FUNDING_MANIFEST): FUNDING_MANIFEST_SHA256,
    }


def freeze_evaluator(
    output_path: str | Path = EVALUATOR_FREEZE,
) -> dict[str, Any]:
    prereg = verify_preregistration()
    semantic, support, clock = _load_semantic_events()
    source_contracts = _source_contracts()
    clear = cast(pd.DataFrame, semantic.loc[semantic["contrarian_side"].ne(0)].copy())
    event_counts: dict[str, dict[str, int]] = {}
    for stage, (start, end) in STAGE_WINDOWS.items():
        in_window = clear["observation_start"].ge(start) & clear[
            "observation_start"
        ].lt(end)
        event_counts[stage] = {
            "clear_semantic_events": int(in_window.sum()),
        }
    static_inputs = _expected_static_inputs()
    for name, expected in static_inputs.items():
        if _sha256(name) != expected:
            raise ValueError(f"TBASR-24 frozen static input changed: {name}")
    core: dict[str, Any] = {
        "protocol_version": "bitmex_trollbox_attention_saturation_evaluator_freeze_v1",
        "candidate": POLICY_ID,
        "as_of_date": AS_OF_DATE,
        "support_commit": SUPPORT_COMMIT,
        "preregistration_manifest_hash": prereg["manifest_hash"],
        "semantic_support_result_hash": support["result_hash"],
        "semantic_clock_manifest_hash": clock["manifest_hash"],
        "evaluator_source": str(EVALUATOR_SOURCE),
        "evaluator_source_sha256": _sha256(EVALUATOR_SOURCE),
        "evaluation_config": asdict(EvaluationConfig()),
        "stage_windows": _window_payload(),
        "subperiod_windows": {
            stage: {
                name: [start.isoformat(), end.isoformat()]
                for name, (start, end) in windows.items()
            }
            for stage, windows in SUBPERIOD_WINDOWS.items()
        },
        "event_counts_before_market": event_counts,
        "source_contracts": source_contracts,
        "static_inputs": static_inputs,
        "controls": list(MECHANISM_CONTROLS),
        "outcome_gate": prereg["outcome_gate"],
        "strict_accounting": prereg["strict_accounting"],
        "opened_windows": [],
        "sealed_windows": list(STAGE_ORDER),
        "execution_ohlc_rows_parsed_during_freeze": 0,
        "funding_rows_parsed_during_freeze": 0,
        "price_conditioned_schedules_built_during_freeze": False,
        "execution_data_bytes_hashed_during_freeze": False,
        "simulation_run_during_freeze": False,
        "strategy_outcomes_calculated": False,
        "mutable_parameters": [],
    }
    report = _seal(core)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
    return report


def verify_evaluator_freeze(
    path: str | Path = EVALUATOR_FREEZE,
) -> dict[str, Any]:
    payload = _load_json(path)
    _verify_manifest(payload, label="evaluator freeze")
    if payload.get("evaluator_source_sha256") != _sha256(EVALUATOR_SOURCE):
        raise ValueError("TBASR-24 evaluator source changed after freeze")
    if payload.get("evaluation_config") != asdict(EvaluationConfig()):
        raise ValueError("TBASR-24 evaluator configuration changed after freeze")
    if payload.get("opened_windows") != []:
        raise ValueError("TBASR-24 evaluator freeze already opened a window")
    if payload.get("sealed_windows") != list(STAGE_ORDER):
        raise ValueError("TBASR-24 evaluator stage seal changed")
    if payload.get("mutable_parameters") != []:
        raise ValueError("TBASR-24 evaluator retained mutable parameters")
    if payload.get("execution_ohlc_rows_parsed_during_freeze") != 0:
        raise ValueError("TBASR-24 evaluator freeze parsed OHLC")
    if payload.get("funding_rows_parsed_during_freeze") != 0:
        raise ValueError("TBASR-24 evaluator freeze parsed funding")
    if payload.get("price_conditioned_schedules_built_during_freeze") is not False:
        raise ValueError("TBASR-24 evaluator freeze built a price schedule")
    if payload.get("execution_data_bytes_hashed_during_freeze") is not False:
        raise ValueError("TBASR-24 evaluator freeze hashed execution data")
    if payload.get("simulation_run_during_freeze") is not False:
        raise ValueError("TBASR-24 evaluator freeze simulated outcomes")
    prereg = verify_preregistration()
    if payload.get("preregistration_manifest_hash") != prereg["manifest_hash"]:
        raise ValueError("TBASR-24 evaluator preregistration binding changed")
    expected_static_inputs = _expected_static_inputs()
    if payload.get("static_inputs") != expected_static_inputs:
        raise ValueError("TBASR-24 evaluator static-input contract changed")
    for name, expected in expected_static_inputs.items():
        if _sha256(name) != expected:
            raise ValueError(f"TBASR-24 frozen static input changed: {name}")
    if payload.get("source_contracts") != _source_contracts():
        raise ValueError("TBASR-24 evaluator source contract changed")
    return payload


def _parse_market_months(
    contracts: list[dict[str, Any]],
    *,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    wanted = ("date", "open", "high", "low", "close")
    rows: list[tuple[Any, ...]] = []
    line_hash = hashlib.sha256()
    verified_months: list[dict[str, Any]] = []
    for contract in contracts:
        path = Path(contract["path"])
        actual_sha = _sha256(path)
        if actual_sha != contract["sha256"]:
            raise ValueError(f"TBASR-24 market month changed: {contract['month']}")
        month_rows = 0
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            header = handle.readline().rstrip("\r\n").split(",")
            positions = {column: header.index(column) for column in wanted}
            if positions["date"] != 0:
                raise ValueError("TBASR-24 market timestamp is not first")
            for line in handle:
                fields = line.rstrip("\r\n").split(",")
                timestamp = _timestamp(fields[positions["date"]])
                if timestamp >= end:
                    raise ValueError("TBASR-24 market month crossed the stage end")
                rows.append(
                    (
                        timestamp,
                        float(fields[positions["open"]]),
                        float(fields[positions["high"]]),
                        float(fields[positions["low"]]),
                        float(fields[positions["close"]]),
                    )
                )
                line_hash.update(line.encode("utf-8"))
                month_rows += 1
        if month_rows != int(contract["rows"]):
            raise ValueError(f"TBASR-24 market month row count changed: {contract['month']}")
        verified_months.append(
            {
                "month": contract["month"],
                "path": str(path),
                "sha256": actual_sha,
                "rows": month_rows,
            }
        )
    frame = pd.DataFrame(rows, columns=pd.Index(wanted))
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="raise")
    actual = pd.DatetimeIndex(frame["date"])
    expected = pd.date_range(SOURCE_START, end, freq="5min", inclusive="left")
    if not actual.equals(expected):
        raise ValueError("TBASR-24 market window is not the exact 5m grid")
    values = frame[["open", "high", "low", "close"]].to_numpy(float)
    if not np.isfinite(values).all() or bool((values <= 0.0).any()):
        raise ValueError("TBASR-24 market contains invalid prices")
    opening, high, low, close = values.T
    if (
        bool((high < np.maximum(opening, close)).any())
        or bool((low > np.minimum(opening, close)).any())
        or bool((high < low).any())
    ):
        raise ValueError("TBASR-24 market violates OHLC invariants")
    return frame, {
        "rows": int(len(frame)),
        "first_timestamp": _timestamp(frame["date"].min()).isoformat(),
        "last_timestamp": _timestamp(frame["date"].max()).isoformat(),
        "parsed_line_sha256": line_hash.hexdigest(),
        "verified_months": verified_months,
        "rows_at_or_after_stage_end_parsed": 0,
    }


def _parse_funding_prefix(
    path: str | Path,
    *,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    wanted = ("funding_time_utc", "symbol", "funding_rate", "settlement_mark_price")
    rows: list[tuple[Any, ...]] = []
    line_hash = hashlib.sha256()
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        header = handle.readline().rstrip("\r\n").split(",")
        positions = {column: header.index(column) for column in wanted}
        for line in handle:
            fields = line.rstrip("\r\n").split(",")
            timestamp = _timestamp(fields[positions["funding_time_utc"]])
            if timestamp >= end:
                break
            if timestamp < SOURCE_START:
                continue
            rows.append(
                (
                    timestamp,
                    fields[positions["symbol"]],
                    float(fields[positions["funding_rate"]]),
                    float(fields[positions["settlement_mark_price"]]),
                )
            )
            line_hash.update(line.encode("utf-8"))
    frame = pd.DataFrame(
        rows,
        columns=pd.Index(
            ["funding_time", "symbol", "funding_rate", "settlement_mark_price"]
        ),
    )
    frame["funding_time"] = pd.to_datetime(
        frame["funding_time"], utc=True, errors="raise"
    )
    actual = pd.DatetimeIndex(frame["funding_time"])
    if not actual.is_unique or not actual.is_monotonic_increasing:
        raise ValueError("TBASR-24 funding timestamps are invalid")
    expected = pd.date_range(SOURCE_START, end, freq="8h", inclusive="left")
    step_ns = int(pd.Timedelta(hours=8).value)
    floor_ns = (np.asarray(actual.asi8, dtype=np.int64) // step_ns) * step_ns
    expected_ns = np.asarray(expected.asi8, dtype=np.int64)
    if not np.array_equal(floor_ns, expected_ns):
        raise ValueError("TBASR-24 funding prefix is not the exact 8h grid")
    offsets_ms = (np.asarray(actual.asi8, dtype=np.int64) - expected_ns) / 1_000_000.0
    if np.abs(offsets_ms).max(initial=0.0) > 60_000.0:
        raise ValueError("TBASR-24 funding timestamp offset exceeds one minute")
    if not bool(frame["symbol"].eq("BTCUSDT").all()):
        raise ValueError("TBASR-24 funding symbol changed")
    values = frame[["funding_rate", "settlement_mark_price"]].to_numpy(float)
    if not np.isfinite(values).all() or bool(
        frame["settlement_mark_price"].le(0.0).any()
    ):
        raise ValueError("TBASR-24 funding contains invalid values")
    return frame, {
        "rows": int(len(frame)),
        "first_timestamp": _timestamp(frame["funding_time"].min()).isoformat(),
        "last_timestamp": _timestamp(frame["funding_time"].max()).isoformat(),
        "parsed_line_sha256": line_hash.hexdigest(),
        "maximum_absolute_grid_offset_ms": float(np.abs(offsets_ms).max(initial=0.0)),
        "rows_at_or_after_stage_end_parsed": 0,
    }


def _verified_prior_reports(
    stage: str,
    *,
    freeze_hash: str,
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    prereg = verify_preregistration()
    for prior in STAGE_ORDER[: STAGE_ORDER.index(stage)]:
        payload = _load_json(STAGE_OUTPUTS[prior])
        _verify_manifest(payload, label=f"stored {prior}")
        if payload.get("protocol_version") != (
            "bitmex_trollbox_attention_saturation_stage_v1"
        ):
            raise ValueError(f"TBASR-24 {prior} report protocol changed")
        if payload.get("candidate") != POLICY_ID or payload.get("stage") != prior:
            raise ValueError(f"TBASR-24 {prior} report identity changed")
        gate_checks = payload.get("gate_checks")
        if not isinstance(gate_checks, dict) or set(gate_checks) != set(
            GATE_CHECK_KEYS
        ):
            raise ValueError(f"TBASR-24 {prior} gate evidence changed")
        all_gates_passed = all(value is True for value in gate_checks.values())
        if payload.get("stage_passed") is not all_gates_passed:
            raise ValueError(f"TBASR-24 {prior} pass flag contradicts gate evidence")
        if not all_gates_passed:
            raise ValueError(f"TBASR-24 {prior} did not pass; {stage} remains sealed")
        if payload.get("parameter_search_performed") is not False:
            raise ValueError(f"TBASR-24 {prior} performed a parameter search")
        if payload.get("post_failure_repair_performed") is not False:
            raise ValueError(f"TBASR-24 {prior} performed post-failure repair")
        prior_index = STAGE_ORDER.index(prior)
        if payload.get("opened_windows") != list(STAGE_ORDER[: prior_index + 1]):
            raise ValueError(f"TBASR-24 {prior} opened an unexpected window")
        if payload.get("sealed_windows") != list(STAGE_ORDER[prior_index + 1 :]):
            raise ValueError(f"TBASR-24 {prior} stage seal changed")
        if payload.get("evaluator_freeze_manifest_hash") != freeze_hash:
            raise ValueError(f"TBASR-24 {prior} froze another evaluator")
        if payload.get("evaluator_source_sha256") != _sha256(EVALUATOR_SOURCE):
            raise ValueError(f"TBASR-24 {prior} evaluator source changed")
        if payload.get("preregistration_manifest_hash") != prereg["manifest_hash"]:
            raise ValueError(f"TBASR-24 {prior} preregistration binding changed")
        reports.append(payload)
    return reports


def load_execution_window(
    stage: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if stage not in STAGE_ORDER:
        raise ValueError(f"TBASR-24 unknown execution stage: {stage}")
    freeze = verify_evaluator_freeze()
    _verified_prior_reports(stage, freeze_hash=str(freeze["manifest_hash"]))
    end = STAGE_WINDOWS[stage][1]
    contracts = freeze["source_contracts"]["stage_market_months"][stage]
    market, market_diagnostics = _parse_market_months(contracts, end=end)
    funding, funding_diagnostics = _parse_funding_prefix(FUNDING, end=end)
    # Full-container hashes reveal no decoded value.  They bind the parsed
    # prefix to the already-public, checksum-frozen source container.
    if _sha256(MARKET_COMBINED) != MARKET_COMBINED_SHA256:
        raise ValueError("TBASR-24 combined market bytes changed")
    if _sha256(FUNDING) != FUNDING_SHA256:
        raise ValueError("TBASR-24 funding bytes changed")
    return market, funding, {
        "stage": stage,
        "decoded_window": [SOURCE_START.isoformat(), end.isoformat()],
        "market": market_diagnostics,
        "funding": funding_diagnostics,
        "combined_market_sha256_verified": MARKET_COMBINED_SHA256,
        "funding_sha256_verified": FUNDING_SHA256,
        "future_container_bytes_hashed_for_identity_only": True,
        "future_rows_decoded": 0,
    }


def _displacement_arrays(
    market: pd.DataFrame,
    *,
    cfg: EvaluationConfig = EvaluationConfig(),
) -> tuple[np.ndarray, np.ndarray]:
    if cfg != EvaluationConfig():
        raise ValueError("TBASR-24 evaluation configuration is frozen")
    opening = market["open"].to_numpy(float)
    close = market["close"].to_numpy(float)
    displacement = np.full(len(market), np.nan, dtype=float)
    displacement[cfg.displacement_bars - 1 :] = np.log(
        close[cfg.displacement_bars - 1 :]
        / opening[: -(cfg.displacement_bars - 1)]
    )
    absolute = pd.Series(np.abs(displacement), dtype=float)
    threshold = (
        absolute.rolling(
            window=cfg.reference_bars,
            min_periods=cfg.reference_bars,
        )
        .quantile(cfg.material_quantile, interpolation="linear")
        .shift(cfg.reference_shift_bars)
        .to_numpy(float)
    )
    return displacement, threshold


def _deterministic_random_side(entry_time: Any) -> int:
    timestamp = _timestamp(entry_time).strftime("%Y-%m-%dT%H:%M:%SZ")
    nibble = int(
        hashlib.sha256(
            f"{POLICY_ID}|random-side|{timestamp}".encode("ascii")
        ).hexdigest()[0],
        16,
    )
    return 1 if nibble % 2 == 0 else -1


def _schedule_hash(frame: pd.DataFrame) -> str:
    rows = [
        {
            "control": str(row["control"]),
            "split": str(row["split"]),
            "observation_end": _timestamp(row["observation_end"]).isoformat(),
            "feature_available_time": _timestamp(
                row["feature_available_time"]
            ).isoformat(),
            "entry_time": _timestamp(row["entry_time"]).isoformat(),
            "exit_time": _timestamp(row["exit_time"]).isoformat(),
            "crowd_label": str(row["crowd_label"]),
            "side": int(row["side"]),
            "displacement_log_return": float(row["displacement_log_return"]),
            "material_threshold_abs_log_return": float(
                row["material_threshold_abs_log_return"]
            ),
        }
        for row in frame.to_dict(orient="records")
    ]
    return _canonical_hash(rows)


def _greedy_nonoverlap(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    ordered = frame.sort_values(
        ["entry_time", "observation_end"], kind="mergesort"
    ).reset_index(drop=True)
    accepted: list[dict[str, Any]] = []
    prior_exit: pd.Timestamp | None = None
    skipped = 0
    for row in ordered.to_dict(orient="records"):
        entry = _timestamp(row["entry_time"])
        exit_time = _timestamp(row["exit_time"])
        if prior_exit is not None and entry < prior_exit:
            skipped += 1
            continue
        accepted.append(row)
        prior_exit = exit_time
    return pd.DataFrame(accepted, columns=ordered.columns), skipped


def build_stage_schedules(
    market: pd.DataFrame,
    stage: str,
    *,
    cfg: EvaluationConfig = EvaluationConfig(),
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, Any]]:
    if cfg != EvaluationConfig():
        raise ValueError("TBASR-24 evaluation configuration is frozen")
    if stage not in STAGE_WINDOWS:
        raise ValueError(f"TBASR-24 unknown stage: {stage}")
    events, _, _ = _load_semantic_events()
    start, end = STAGE_WINDOWS[stage]
    clear = cast(
        pd.DataFrame,
        events.loc[
            events["crowd_label"].isin(("BULLISH", "BEARISH"))
            & events["observation_start"].ge(start)
            & events["observation_start"].lt(end)
        ].copy(),
    )
    displacement, threshold = _displacement_arrays(market, cfg=cfg)
    positions = {
        _timestamp(value): index
        for index, value in enumerate(cast(pd.Series, market["date"]))
    }
    records: list[dict[str, Any]] = []
    excluded_split_boundary = 0
    for event in clear.to_dict(orient="records"):
        observation_end = _timestamp(event["observation_end"])
        final_bar_time = observation_end - BAR
        final_position = positions.get(final_bar_time)
        if final_position is None:
            raise ValueError("TBASR-24 event final bar is absent from market grid")
        first_position = final_position - cfg.displacement_bars + 1
        if first_position < 0:
            raise ValueError("TBASR-24 displacement starts before source")
        latest_reference_final = final_position - cfg.reference_shift_bars
        first_reference_final = latest_reference_final - cfg.reference_bars + 1
        if first_reference_final < 0:
            raise ValueError("TBASR-24 event lacks the complete 28-day reference")
        target_start = _timestamp(market.iloc[first_position]["date"])
        reference_endpoint_start = (
            _timestamp(market.iloc[first_reference_final]["date"]) + BAR
        )
        reference_endpoint_end = (
            _timestamp(market.iloc[latest_reference_final]["date"]) + BAR
        )
        if reference_endpoint_start != target_start - pd.Timedelta(
            days=cfg.reference_days
        ):
            raise ValueError("TBASR-24 reference start indexing changed")
        if reference_endpoint_end != target_start - BAR:
            raise ValueError("TBASR-24 reference end touches the target move")
        move = float(displacement[final_position])
        material_threshold = float(threshold[final_position])
        if not np.isfinite(move) or not np.isfinite(material_threshold):
            raise ValueError("TBASR-24 event lacks frozen displacement reference")
        entry_time = _timestamp(event["entry_earliest"])
        exit_time = entry_time + cfg.hold_bars * BAR
        if exit_time != _timestamp(event["exit_time"]):
            raise ValueError("TBASR-24 semantic exit clock changed")
        stress_entry = entry_time + cfg.stress_extra_delay_bars * BAR
        stress_exit = stress_entry + cfg.hold_bars * BAR
        if entry_time < start or stress_exit >= end:
            excluded_split_boundary += 1
            continue
        crowd_label = str(event["crowd_label"])
        side = int(event["contrarian_side"])
        aligned = (crowd_label == "BULLISH" and move > 0.0) or (
            crowd_label == "BEARISH" and move < 0.0
        )
        records.append(
            {
                "candidate": POLICY_ID,
                "control": PRIMARY,
                "split": stage,
                "observation_start": _timestamp(event["observation_start"]),
                "observation_end": observation_end,
                "feature_available_time": observation_end,
                "displacement_start": _timestamp(
                    market.iloc[first_position]["date"]
                ),
                "displacement_end": observation_end,
                "reference_endpoint_start": reference_endpoint_start,
                "reference_endpoint_end": reference_endpoint_end,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "stress_entry_time": stress_entry,
                "stress_exit_time": stress_exit,
                "crowd_label": crowd_label,
                "side": side,
                "displacement_log_return": move,
                "material_threshold_abs_log_return": material_threshold,
                "material": abs(move) >= material_threshold,
                "semantic_displacement_aligned": aligned,
            }
        )
    universe = pd.DataFrame(records)
    if universe.empty:
        raise ValueError(f"TBASR-24 {stage} clear event universe is empty")
    material = cast(pd.DataFrame, universe.loc[universe["material"]].copy())
    qualified = cast(
        pd.DataFrame,
        material.loc[material["semantic_displacement_aligned"]].copy(),
    )
    primary, primary_skipped = _greedy_nonoverlap(qualified)
    primary["control"] = PRIMARY

    direction_flip = primary.copy()
    direction_flip["control"] = "direction_flip"
    direction_flip["side"] = -direction_flip["side"].astype(int)

    random_side = primary.copy()
    random_side["control"] = "deterministic_random_side"
    random_side["side"] = random_side["entry_time"].map(
        _deterministic_random_side
    )

    alignment_ablation, ablation_skipped = _greedy_nonoverlap(material)
    alignment_ablation["control"] = "semantic_alignment_ablation"

    stress = primary.copy()
    stress["control"] = "primary_stress_delayed"
    stress["entry_time"] = stress["stress_entry_time"]
    stress["exit_time"] = stress["stress_exit_time"]

    schedules = {
        PRIMARY: primary.reset_index(drop=True),
        "direction_flip": direction_flip.reset_index(drop=True),
        "deterministic_random_side": random_side.reset_index(drop=True),
        "semantic_alignment_ablation": alignment_ablation.reset_index(drop=True),
        "stress": stress.reset_index(drop=True),
    }
    for name, schedule in schedules.items():
        if len(schedule) > 1 and not bool(
            schedule["entry_time"]
            .iloc[1:]
            .ge(schedule["exit_time"].iloc[:-1].to_numpy())
            .all()
        ):
            raise ValueError(f"TBASR-24 {name} schedule overlaps")
        if not bool(
            schedule["exit_time"]
            .sub(schedule["entry_time"])
            .eq(cfg.hold_bars * BAR)
            .all()
        ):
            raise ValueError(f"TBASR-24 {name} hold changed")
    if not schedules[PRIMARY][["observation_end"]].equals(
        schedules["direction_flip"][["observation_end"]]
    ):
        raise ValueError("TBASR-24 direction flip changed primary clocks")
    if not schedules[PRIMARY][["observation_end"]].equals(
        schedules["deterministic_random_side"][["observation_end"]]
    ):
        raise ValueError("TBASR-24 random side changed primary clocks")
    if not schedules[PRIMARY][["observation_end"]].equals(
        schedules["stress"][["observation_end"]]
    ):
        raise ValueError("TBASR-24 stress changed primary event identities")
    incidence = {
        "clear_semantic_events": int(len(clear)),
        "boundary_exclusions": int(excluded_split_boundary),
        "reference_ready_events": int(len(universe)),
        "material_events": int(len(material)),
        "aligned_material_events_before_overlap": int(len(qualified)),
        "primary_events": int(len(primary)),
        "primary_overlaps_skipped": int(primary_skipped),
        "alignment_ablation_events": int(len(alignment_ablation)),
        "alignment_ablation_overlaps_skipped": int(ablation_skipped),
        "primary_longs": int(primary["side"].eq(1).sum()),
        "primary_shorts": int(primary["side"].eq(-1).sum()),
    }
    return schedules, universe, incidence


def weekly_cluster_signflip_two_sided(
    trades: pd.DataFrame,
    *,
    cfg: EvaluationConfig = EvaluationConfig(),
) -> dict[str, Any]:
    if trades.empty:
        return {
            "p_value_two_sided": 1.0,
            "cluster_count": 0,
            "method": "empty",
            "draws": 0,
            "seed": cfg.cluster_seed,
            "observed_abs_mean_net_return": 0.0,
            "weekly_net_return_sums": {},
            "largest_absolute_week_share": 0.0,
        }
    entry = pd.to_datetime(trades["entry_time"], utc=True, errors="raise")
    iso = entry.dt.isocalendar()
    keys = iso.year.astype(str) + "-W" + iso.week.astype(str).str.zfill(2)
    weekly = cast(pd.Series, trades["net_return"]).groupby(keys).sum()
    values = weekly.to_numpy(float)
    observed = abs(float(cast(pd.Series, trades["net_return"]).mean()))
    clusters = len(values)
    exceed = 0
    if clusters <= cfg.exact_cluster_max:
        total = 1 << clusters
        bit_positions = np.arange(clusters, dtype=np.uint64)
        for begin in range(0, total, 50_000):
            indices = np.arange(begin, min(total, begin + 50_000), dtype=np.uint64)
            bits = (indices[:, None] >> bit_positions[None, :]) & 1
            signs = 1.0 - 2.0 * bits.astype(float)
            null = np.abs((signs @ values) / float(len(trades)))
            exceed += int(np.count_nonzero(null >= observed - 1e-15))
        p_value = float(exceed / total)
        method = "exact"
        draws = total
    else:
        generator = np.random.default_rng(cfg.cluster_seed)
        remaining = cfg.cluster_draws
        while remaining:
            batch = min(10_000, remaining)
            signs = generator.choice((-1.0, 1.0), size=(batch, clusters))
            null = np.abs((signs @ values) / float(len(trades)))
            exceed += int(np.count_nonzero(null >= observed - 1e-15))
            remaining -= batch
        p_value = float((1 + exceed) / (cfg.cluster_draws + 1))
        method = "monte_carlo"
        draws = cfg.cluster_draws
    return {
        "p_value_two_sided": p_value,
        "cluster_count": int(clusters),
        "method": method,
        "draws": int(draws),
        "seed": int(cfg.cluster_seed),
        "observed_abs_mean_net_return": observed,
        "weekly_net_return_sums": {
            str(label): float(value) for label, value in weekly.items()
        },
        "largest_absolute_week_share": float(
            np.abs(values).max(initial=0.0)
            / max(float(np.abs(values).sum()), 1e-15)
        ),
    }


def _ratio(cagr: float, strict_mdd: float) -> float:
    if strict_mdd > 0.0:
        return cagr / strict_mdd
    if cagr > 0.0:
        return float("inf")
    if cagr < 0.0:
        return float("-inf")
    return 0.0


def simulate_strict(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    clocks: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cost_rate_per_side: float,
    cfg: EvaluationConfig = EvaluationConfig(),
) -> dict[str, Any]:
    if cfg != EvaluationConfig():
        raise ValueError("TBASR-24 evaluation configuration is frozen")
    if not 0.0 <= cost_rate_per_side < 0.1 or end <= start:
        raise ValueError("TBASR-24 simulation window or cost is invalid")
    positions = {
        _timestamp(value): index
        for index, value in enumerate(cast(pd.Series, market["date"]))
    }
    funding_times = cast(pd.Series, funding["funding_time"])
    realized_equity = 1.0
    high_water_mark = 1.0
    maximum_drawdown = 0.0
    records: list[dict[str, Any]] = []
    previous_exit: pd.Timestamp | None = None

    def update_path(value: float) -> None:
        nonlocal high_water_mark, maximum_drawdown
        if not np.isfinite(value):
            raise ValueError("TBASR-24 strict equity path is non-finite")
        high_water_mark = max(high_water_mark, value)
        maximum_drawdown = max(
            maximum_drawdown,
            1.0 - value / max(high_water_mark, 1e-15),
        )

    for clock in clocks.to_dict(orient="records"):
        entry_time = _timestamp(clock["entry_time"])
        exit_time = _timestamp(clock["exit_time"])
        if entry_time < start or exit_time > end:
            raise ValueError("TBASR-24 clock crosses the simulation window")
        if previous_exit is not None and entry_time < previous_exit:
            raise ValueError("TBASR-24 simulation schedule overlaps")
        previous_exit = exit_time
        entry_position = positions.get(entry_time)
        exit_position = positions.get(exit_time)
        if entry_position is None or exit_position is None:
            raise ValueError("TBASR-24 clock is absent from the market grid")
        if exit_position - entry_position != cfg.hold_bars:
            raise ValueError("TBASR-24 hold is not exactly 24 bars / 2 hours")
        side = int(clock["side"])
        if side not in (-1, 1):
            raise ValueError("TBASR-24 side must be -1 or 1")

        entry_price = float(market.iloc[entry_position]["open"])
        exit_price = float(market.iloc[exit_position]["open"])
        pre_entry_equity = realized_equity
        quantity = pre_entry_equity * cfg.leverage / entry_price
        entry_fee = quantity * entry_price * cost_rate_per_side
        cash = pre_entry_equity - entry_fee
        update_path(cash)

        included_funding = cast(
            pd.DataFrame,
            funding.loc[
                funding_times.ge(entry_time) & funding_times.le(exit_time)
            ].copy(),
        )
        next_funding = 0
        funding_cash = 0.0
        applied_funding_events = 0
        dropped_boundary_credits = 0
        visited_funding_events = 0

        def apply_funding_through(upper: pd.Timestamp) -> None:
            nonlocal cash
            nonlocal funding_cash
            nonlocal next_funding
            nonlocal applied_funding_events
            nonlocal dropped_boundary_credits
            nonlocal visited_funding_events
            while next_funding < len(included_funding):
                event = included_funding.iloc[next_funding]
                event_time = _timestamp(event["funding_time"])
                if event_time > upper:
                    break
                settlement_mark = float(event["settlement_mark_price"])
                visited_funding_events += 1
                cash_flow = (
                    -side * quantity * settlement_mark * float(event["funding_rate"])
                )
                boundary = event_time in (entry_time, exit_time)
                if boundary and cash_flow > 0.0:
                    dropped_boundary_credits += 1
                else:
                    cash += cash_flow
                    funding_cash += cash_flow
                    applied_funding_events += 1
                marked = cash + side * quantity * (settlement_mark - entry_price)
                virtual_exit_fee = quantity * settlement_mark * cost_rate_per_side
                update_path(marked - virtual_exit_fee)
                next_funding += 1

        for position in range(entry_position, exit_position):
            bar = market.iloc[position]
            bar_time = _timestamp(bar["date"])
            bar_end = bar_time + BAR - pd.Timedelta(1, unit="ns")
            apply_funding_through(bar_end)
            favorable_price = float(bar["high"] if side > 0 else bar["low"])
            favorable_equity = cash + side * quantity * (
                favorable_price - entry_price
            )
            update_path(favorable_equity)
            adverse_price = float(bar["low"] if side > 0 else bar["high"])
            adverse_equity = cash + side * quantity * (adverse_price - entry_price)
            virtual_exit_fee = quantity * adverse_price * cost_rate_per_side
            update_path(adverse_equity - virtual_exit_fee)

        apply_funding_through(exit_time)
        if next_funding != len(included_funding):
            raise ValueError("TBASR-24 funding event remained after exit")
        gross_pnl = side * quantity * (exit_price - entry_price)
        exit_fee = quantity * exit_price * cost_rate_per_side
        realized_equity = cash + gross_pnl - exit_fee
        update_path(realized_equity)
        net_return = realized_equity / pre_entry_equity - 1.0
        records.append(
            {
                "control": str(clock.get("control", PRIMARY)),
                "split": str(clock.get("split", "synthetic")),
                "observation_end": (
                    _timestamp(clock["observation_end"]).isoformat()
                    if "observation_end" in clock
                    else None
                ),
                "entry_time": entry_time.isoformat(),
                "exit_time": exit_time.isoformat(),
                "side": side,
                "crowd_label": clock.get("crowd_label"),
                "displacement_log_return": clock.get("displacement_log_return"),
                "material_threshold_abs_log_return": clock.get(
                    "material_threshold_abs_log_return"
                ),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "bars_held": cfg.hold_bars,
                "pre_entry_equity": pre_entry_equity,
                "quantity_btc": quantity,
                "entry_fee": entry_fee,
                "exit_fee": exit_fee,
                "funding_cash": funding_cash,
                "funding_events": applied_funding_events,
                "visited_funding_events": visited_funding_events,
                "dropped_boundary_funding_credits": dropped_boundary_credits,
                "gross_underlying_bp": side
                * (exit_price / entry_price - 1.0)
                * 10_000.0,
                "gross_pnl": gross_pnl,
                "net_return": net_return,
                "post_exit_equity": realized_equity,
            }
        )

    years = (end - start).total_seconds() / YEAR_SECONDS
    if years <= 0.0:
        raise ValueError("TBASR-24 evaluation window has no duration")
    absolute_return = realized_equity - 1.0
    cagr = realized_equity ** (1.0 / years) - 1.0 if realized_equity > 0.0 else -1.0
    trade_frame = pd.DataFrame(records)
    significance = weekly_cluster_signflip_two_sided(trade_frame, cfg=cfg)
    mean_gross = (
        float(cast(pd.Series, trade_frame["gross_underlying_bp"]).mean())
        if len(trade_frame)
        else 0.0
    )
    return {
        "period_start": start.isoformat(),
        "period_end": end.isoformat(),
        "calendar_years": float(years),
        "absolute_return_pct": float(absolute_return * 100.0),
        "cagr_pct": float(cagr * 100.0),
        "strict_mdd_pct": float(maximum_drawdown * 100.0),
        "cagr_to_strict_mdd": float(_ratio(cagr, maximum_drawdown)),
        "trades": int(len(records)),
        "longs": int(sum(int(row["side"]) == 1 for row in records)),
        "shorts": int(sum(int(row["side"]) == -1 for row in records)),
        "ending_equity": float(realized_equity),
        "mean_gross_underlying_bp": mean_gross,
        "weekly_cluster_signflip": significance,
        "trade_details": records,
    }


def _headline(metrics: dict[str, Any]) -> dict[str, Any]:
    significance = metrics["weekly_cluster_signflip"]
    return {
        "absolute_return_pct": metrics["absolute_return_pct"],
        "cagr_pct": metrics["cagr_pct"],
        "strict_mdd_pct": metrics["strict_mdd_pct"],
        "cagr_to_strict_mdd": metrics["cagr_to_strict_mdd"],
        "trades": metrics["trades"],
        "longs": metrics["longs"],
        "shorts": metrics["shorts"],
        "mean_gross_underlying_bp": metrics["mean_gross_underlying_bp"],
        "weekly_cluster_signflip_p": significance["p_value_two_sided"],
        "weekly_clusters": significance["cluster_count"],
        "weekly_test_method": significance["method"],
        "largest_absolute_week_share": significance[
            "largest_absolute_week_share"
        ],
    }


def _simulate_subperiod(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cfg: EvaluationConfig,
) -> dict[str, Any]:
    selected = cast(
        pd.DataFrame,
        schedule.loc[
            schedule["entry_time"].ge(start) & schedule["exit_time"].le(end)
        ].copy(),
    )
    return simulate_strict(
        market,
        funding,
        selected,
        start=start,
        end=end,
        cost_rate_per_side=cfg.base_cost_notional_per_side,
        cfg=cfg,
    )


def _stage_gates(
    stage: str,
    base: dict[str, Any],
    stress: dict[str, Any],
    subperiods: dict[str, dict[str, Any]],
    controls: dict[str, dict[str, Any]],
    prereg: dict[str, Any],
) -> tuple[dict[str, bool], dict[str, float], float]:
    gate = prereg["outcome_gate"]
    primary_ratio = float(base["cagr_to_strict_mdd"])
    control_ratios = {
        name: float(controls[name]["cagr_to_strict_mdd"])
        for name in MECHANISM_CONTROLS
    }
    if any(not np.isfinite(ratio) for ratio in control_ratios.values()):
        minimum_margin = float("-inf")
    else:
        minimum_margin = min(
            primary_ratio - ratio for ratio in control_ratios.values()
        )
    checks = {
        "absolute_return_positive": base["absolute_return_pct"] > 0.0,
        "cagr_to_strict_mdd_at_least_3": primary_ratio
        >= float(gate["cagr_to_strict_mdd_min"]),
        "strict_mdd_at_most_15pct": base["strict_mdd_pct"]
        <= float(gate["strict_mdd_max_pct"]),
        "minimum_trades": base["trades"] >= int(gate["minimum_trades"][stage]),
        "minimum_longs": base["longs"]
        >= int(gate["minimum_trades_each_side"][stage]),
        "minimum_shorts": base["shorts"]
        >= int(gate["minimum_trades_each_side"][stage]),
        "minimum_weekly_clusters": base["weekly_cluster_signflip"]["cluster_count"]
        >= int(gate["minimum_weekly_clusters"][stage]),
        "weekly_cluster_signflip_p_at_most_10pct": base[
            "weekly_cluster_signflip"
        ]["p_value_two_sided"]
        <= float(gate["weekly_cluster_signflip_p_max"]),
        "mean_gross_underlying_at_least_20bp": base[
            "mean_gross_underlying_bp"
        ]
        >= float(gate["mean_gross_underlying_move_bp_min"]),
        "each_half_year_absolute_return_positive": all(
            item["absolute_return_pct"] > 0.0 for item in subperiods.values()
        ),
        "stress_same_trade_count": stress["trades"] == base["trades"],
        "stress_absolute_return_positive": stress["absolute_return_pct"] > 0.0,
        "stress_cagr_to_strict_mdd_at_least_2_5": stress[
            "cagr_to_strict_mdd"
        ]
        >= float(gate["stress_cagr_to_strict_mdd_min"]),
        "mechanism_control_margin_at_least_0_25": minimum_margin
        >= float(gate["mechanism_control_margin_min"]),
    }
    if tuple(checks) != GATE_CHECK_KEYS:
        raise ValueError("TBASR-24 frozen gate implementation drifted")
    return checks, control_ratios, minimum_margin


def _build_stage_report(stage: str) -> dict[str, Any]:
    if stage not in STAGE_ORDER:
        raise ValueError(f"TBASR-24 unknown stage: {stage}")
    freeze = verify_evaluator_freeze()
    prior = _verified_prior_reports(stage, freeze_hash=str(freeze["manifest_hash"]))
    prereg = verify_preregistration()
    cfg = EvaluationConfig()
    market, funding, source_diagnostics = load_execution_window(stage)
    schedules, universe, incidence = build_stage_schedules(
        market, stage, cfg=cfg
    )
    start, end = STAGE_WINDOWS[stage]
    base = simulate_strict(
        market,
        funding,
        schedules[PRIMARY],
        start=start,
        end=end,
        cost_rate_per_side=cfg.base_cost_notional_per_side,
        cfg=cfg,
    )
    stress = simulate_strict(
        market,
        funding,
        schedules["stress"],
        start=start,
        end=end,
        cost_rate_per_side=cfg.stress_cost_notional_per_side,
        cfg=cfg,
    )
    control_metrics = {
        name: simulate_strict(
            market,
            funding,
            schedules[name],
            start=start,
            end=end,
            cost_rate_per_side=cfg.base_cost_notional_per_side,
            cfg=cfg,
        )
        for name in MECHANISM_CONTROLS
    }
    subperiod_metrics = {
        name: _simulate_subperiod(
            market,
            funding,
            schedules[PRIMARY],
            start=period_start,
            end=period_end,
            cfg=cfg,
        )
        for name, (period_start, period_end) in SUBPERIOD_WINDOWS[stage].items()
    }
    checks, control_ratios, minimum_margin = _stage_gates(
        stage,
        base,
        stress,
        subperiod_metrics,
        control_metrics,
        prereg,
    )
    stage_index = STAGE_ORDER.index(stage)
    absolute_subperiod_returns = np.abs(
        np.asarray(
            [
                float(metrics["absolute_return_pct"])
                for metrics in subperiod_metrics.values()
            ],
            dtype=float,
        )
    )
    core: dict[str, Any] = {
        "protocol_version": "bitmex_trollbox_attention_saturation_stage_v1",
        "candidate": POLICY_ID,
        "stage": stage,
        "stage_passed": bool(all(checks.values())),
        "evaluator_freeze_manifest_hash": freeze["manifest_hash"],
        "evaluator_source_sha256": _sha256(EVALUATOR_SOURCE),
        "preregistration_manifest_hash": prereg["manifest_hash"],
        "opened_windows": list(STAGE_ORDER[: stage_index + 1]),
        "sealed_windows": list(STAGE_ORDER[stage_index + 1 :]),
        "prior_passed_reports": [item["manifest_hash"] for item in prior],
        "period": [start.isoformat(), end.isoformat()],
        "source_diagnostics": source_diagnostics,
        "incidence": incidence,
        "universe_rows": int(len(universe)),
        "schedule_records": {
            name: {
                "events": int(len(schedule)),
                "schedule_hash": _schedule_hash(schedule),
            }
            for name, schedule in schedules.items()
        },
        "base_headline": _headline(base),
        "stress_headline": _headline(stress),
        "subperiod_headlines": {
            name: _headline(metrics) for name, metrics in subperiod_metrics.items()
        },
        "control_headlines": {
            name: _headline(metrics) for name, metrics in control_metrics.items()
        },
        "control_cagr_to_strict_mdd": control_ratios,
        "minimum_mechanism_control_margin": minimum_margin,
        "largest_absolute_subperiod_return_share": float(
            absolute_subperiod_returns.max(initial=0.0)
            / max(float(absolute_subperiod_returns.sum()), 1e-15)
        ),
        "gate_checks": checks,
        "base_metrics": base,
        "stress_metrics": stress,
        "subperiod_metrics": subperiod_metrics,
        "control_metrics": control_metrics,
        "parameter_search_performed": False,
        "post_failure_repair_performed": False,
    }
    return _seal(core)


def _render_stage_doc(report: dict[str, Any]) -> str:
    stage = report["stage"]
    base = report["base_headline"]
    stress = report["stress_headline"]
    status = "PASS" if report["stage_passed"] else "REJECT"
    failed = [name for name, passed in report["gate_checks"].items() if not passed]
    rows = [
        f"# TBASR-24 {stage} result — {status}",
        "",
        "This report was produced by the committed, write-once evaluator contract.",
        "No threshold, hold, leverage, cost, or semantic prompt was selected from this outcome.",
        "",
        "| run | absolute return | CAGR | strict MDD | CAGR/MDD | trades | long | short |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| base | {base['absolute_return_pct']:.4f}% | {base['cagr_pct']:.4f}% | "
            f"{base['strict_mdd_pct']:.4f}% | {base['cagr_to_strict_mdd']:.4f} | "
            f"{base['trades']} | {base['longs']} | {base['shorts']} |"
        ),
        (
            f"| delayed stress | {stress['absolute_return_pct']:.4f}% | "
            f"{stress['cagr_pct']:.4f}% | {stress['strict_mdd_pct']:.4f}% | "
            f"{stress['cagr_to_strict_mdd']:.4f} | {stress['trades']} | "
            f"{stress['longs']} | {stress['shorts']} |"
        ),
        "",
        f"- weekly clustered sign-flip p: `{base['weekly_cluster_signflip_p']}`;",
        f"- weekly clusters: `{base['weekly_clusters']}`;",
        f"- mean gross underlying move: `{base['mean_gross_underlying_bp']:.4f} bp`;",
        f"- minimum mechanism-control ratio margin: `{report['minimum_mechanism_control_margin']}`;",
        f"- failed gates: `{failed}`.",
        "",
    ]
    if not report["stage_passed"]:
        rows.extend(
            [
                "The candidate is rejected at this frozen stage. Later windows remain sealed,",
                "and this candidate may not be repaired on the observed outcome.",
                "",
            ]
        )
    return "\n".join(rows)


def run_stage(stage: str) -> dict[str, Any]:
    report = _build_stage_report(stage)
    output = STAGE_OUTPUTS[stage]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
    document = STAGE_DOCS[stage]
    document.parent.mkdir(parents=True, exist_ok=True)
    with document.open("x", encoding="utf-8") as handle:
        handle.write(_render_stage_doc(report))
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--write-preregistration",
        action="store_true",
        help="write the deterministic outcome-blind preregistration",
    )
    action.add_argument(
        "--freeze",
        action="store_true",
        help="freeze evaluator/source contracts without parsing market rows",
    )
    action.add_argument(
        "--stage",
        choices=STAGE_ORDER,
        help="open and evaluate exactly one sequential outcome stage",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.write_preregistration:
        report = write_preregistration()
    elif args.freeze:
        report = freeze_evaluator()
    else:
        report = run_stage(str(args.stage))
    summary = {
        "candidate": report["candidate"],
        "protocol_version": report["protocol_version"],
        "manifest_hash": report["manifest_hash"],
    }
    if "stage" in report:
        summary["stage"] = report["stage"]
        summary["stage_passed"] = report["stage_passed"]
        summary["base_headline"] = report["base_headline"]
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
