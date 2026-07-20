"""Strict sequential pre-2024 evaluator for frozen BAFR-24F.

The evaluator has three irreversible stages: evaluator freeze, 2020-2022
train, and 2023 sealed selection. Train physically skips values before its
window and stops before parsing any 2023 market or funding value. Selection
remains sealed until the write-once train artifact exactly replays and passes
every preregistered gate.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gzip
import hashlib
from itertools import product
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from training import build_binance_aggressor_frustration_support as support_builder


CANDIDATE = "BAFR-24F"
SUPPORT_COMMIT = "080e7ae84b1fda16212593963df60cd01f679ff8"
NOVELTY_COMMIT = "0b33aec4885da47fe168d16926a6c8d621488ee7"

MECHANISM_DOCUMENT = Path(
    "docs/binance-aggressor-frustration-reversal-source-mechanism-decision-"
    "2026-07-20.md"
)
MECHANISM_DOCUMENT_SHA256 = (
    "c9f0405fd0a33a7a7b0b59c9f81955dc97ffa335b886a34e1fd2224741c3777d"
)
SUPPORT_SOURCE = Path("training/build_binance_aggressor_frustration_support.py")
SUPPORT_SOURCE_SHA256 = (
    "8fc46eb0d1a05e038028b36871588d574afece3d3328abb1c6a3e2f46b4ab0b2"
)
SUPPORT_DOCUMENT = Path(
    "docs/binance-aggressor-frustration-reversal-support-freeze-2026-07-20.md"
)
SUPPORT_DOCUMENT_SHA256 = (
    "d6fe2643ba90317359cce10a90a294ce684f16aafcf448e84a9c19c8d95bc62e"
)
SUPPORT_RESULT = Path("results/binance_aggressor_frustration_support_2026-07-20.json")
SUPPORT_RESULT_SHA256 = (
    "cf6edad6a4eb46c6630dbb5008c88da1ddd39f9ac5c1606785be02f2b323fb62"
)
PRIMARY_CLOCK = Path("results/binance_aggressor_frustration_clock_2026-07-20.csv")
PRIMARY_CLOCK_SHA256 = (
    "f3b816a76decce31136ed23d22f043eb8e80ef1b8697b869241b060062f01747"
)
NOVELTY_SOURCE = Path("training/compare_binance_aggressor_frustration_novelty.py")
NOVELTY_SOURCE_SHA256 = (
    "f08c8ee61e28e3fff9799cf0f319a5cb3879003e348a9737dfaf8c96634acaaf"
)
NOVELTY_DOCUMENT = Path(
    "docs/binance-aggressor-frustration-reversal-novelty-freeze-2026-07-20.md"
)
NOVELTY_DOCUMENT_SHA256 = (
    "eacc7ed2eac66125f0f3a84793a653abea24650750fa0d7df872ddd67bab0573"
)
NOVELTY_RESULT = Path("results/binance_aggressor_frustration_novelty_2026-07-20.json")
NOVELTY_RESULT_SHA256 = (
    "38ab5dbb1b36f14e32a4d7a09d94c37b84eaec5d1b75bbc5ef576660e05e3028"
)
PREREGISTRATION = Path(
    "docs/binance-aggressor-frustration-reversal-evaluator-preregistration-"
    "2026-07-20.md"
)
PREREGISTRATION_SHA256 = (
    "2f9cccb640e29082fd9033db79fbf07c8706d47257c2c845250bec4ce2040979"
)

FEATURE_DATA = Path(
    "data/binance_um_aggressor_frustration_btc_2020_2023/"
    "BTCUSDT_aggressor_frustration_5m_2020-01-01_2023-12-31.csv.gz"
)
FEATURE_DATA_SHA256 = (
    "e46dc9a4f5e4d4a93bc260d40c0a599ccd0e609d5cb8ebf438c716f7272f7275"
)
FEATURE_MANIFEST = Path(
    "data/binance_um_aggressor_frustration_btc_2020_2023/build_manifest.json"
)
FEATURE_MANIFEST_SHA256 = (
    "9fa1025c90fb8ad1729f2278236a73e94b0d20bcf9b79178610306cf3b85a28b"
)
MARKET_DATA = Path(
    "data/binance_um_kline_reference_btc_2020_2023/"
    "BTCUSDT_5m_2020-01-01_2023-12-31.csv.gz"
)
MARKET_DATA_SHA256 = (
    "e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d"
)
MARKET_MANIFEST = Path("data/binance_um_kline_reference_btc_2020_2023/build_manifest.json")
MARKET_MANIFEST_SHA256 = (
    "c04fbbd299cc748a6745c0ef030787da4d560833c744c81c98dd8840efc7913e"
)
FUNDING_DATA = Path("data/binance_um_btcusdt_funding_marks_2020_2023.csv.gz")
FUNDING_DATA_SHA256 = (
    "3284bbb6bb67946acb673c6b67459543e217f752589e1d47b6c7c3b659f733e6"
)
FUNDING_MANIFEST = Path(
    "results/binance_um_btcusdt_funding_marks_2020_2023_manifest_2026-07-17.json"
)
FUNDING_MANIFEST_SHA256 = (
    "a0b2d27e1aa8cf2d9ab8cb659b598ee0a6d7bd25401c9e10ae92d1a74415845b"
)
COMPLETED_BAR_DATA = Path(
    "data/binance_um_aggtrade_microstructure_btc_2020_2023/"
    "BTCUSDT_aggtrade_5m_2020-01-01_2023-12-31.csv.gz"
)
COMPLETED_BAR_DATA_SHA256 = (
    "c2bb0e6742f8cdc4e13315e7f0a13d6ab9cd536fb40d9cb4484b7a6ba30131cf"
)
COMPLETED_BAR_MANIFEST = Path(
    "data/binance_um_aggtrade_microstructure_btc_2020_2023/build_manifest.json"
)
COMPLETED_BAR_MANIFEST_SHA256 = (
    "6eec40460a6146c58994e52f1af9ace4eecc0c085887d97af5ef17c30b9f7e73"
)

EVALUATION_SOURCE = Path("training/evaluate_binance_aggressor_frustration_pre2024.py")
FREEZE_SOURCE = Path("training/freeze_binance_aggressor_frustration_evaluator.py")
EVALUATION_TEST = Path("tests/test_evaluate_binance_aggressor_frustration_pre2024.py")
FREEZE_TEST = Path("tests/test_freeze_binance_aggressor_frustration_evaluator.py")
EVALUATION_FREEZE = Path(
    "results/binance_aggressor_frustration_evaluator_freeze_2026-07-20.json"
)
TRAIN_OUTPUT = Path(
    "results/binance_aggressor_frustration_train_2020_2022_2026-07-20.json"
)
SELECTION_OUTPUT = Path(
    "results/binance_aggressor_frustration_selection_2023_2026-07-20.json"
)

BAR = pd.Timedelta(minutes=5)
TRAIN_START = pd.Timestamp("2020-01-01")
TRAIN_END = pd.Timestamp("2023-01-01")
SELECTION_END = pd.Timestamp("2024-01-01")
WINDOWS: dict[str, tuple[str, str]] = {
    "train": ("2020-01-01", "2023-01-01"),
    "train_2020": ("2020-01-01", "2021-01-01"),
    "train_2021": ("2021-01-01", "2022-01-01"),
    "train_2022": ("2022-01-01", "2023-01-01"),
    "selection_2023": ("2023-01-01", "2024-01-01"),
    "selection_2023_h1": ("2023-01-01", "2023-07-01"),
    "selection_2023_h2": ("2023-07-01", "2024-01-01"),
}
STAGE_WINDOWS: dict[str, tuple[str, tuple[str, ...]]] = {
    "train": ("train", ("train_2020", "train_2021", "train_2022")),
    "selection": (
        "selection_2023",
        ("selection_2023_h1", "selection_2023_h2"),
    ),
}

POLICY_NAMES = (
    "primary",
    "direction_flip",
    "aggressor_flow_only",
    "tick_direction_only",
    "strict_nonzero_tick_only",
    "carried_zero_tick_only",
    "completed_bar_rejection",
    "stale_1h",
    "stale_24h",
)
MECHANISM_REJECTION_CONTROLS = (
    "aggressor_flow_only",
    "tick_direction_only",
    "strict_nonzero_tick_only",
    "carried_zero_tick_only",
    "completed_bar_rejection",
)
SUPERIORITY_CONTROLS = tuple(name for name in POLICY_NAMES if name != "primary")
CONTROL_SEMANTICS = {
    "primary": "frozen BAFR score, side, next-open entry, and 24-bar hold",
    "direction_flip": "primary clock with every side multiplied by -1",
    "aggressor_flow_only": (
        "score=-signed_quote_notional/quote_notional; own prior-clean q90 clock"
    ),
    "tick_direction_only": (
        "score=tick_notional_imbalance; own prior-clean q90 clock"
    ),
    "strict_nonzero_tick_only": (
        "score=(strict_sell_frustrated_notional-strict_buy_frustrated_notional)"
        "/quote_notional; own prior-clean q90 clock"
    ),
    "carried_zero_tick_only": (
        "score=(carried_sell_frustrated_notional-carried_buy_frustrated_notional)"
        "/quote_notional; own prior-clean q90 clock"
    ),
    "completed_bar_rejection": (
        "score=-signed_quote_notional/quote_notional only when flow sign opposes "
        "completed micro_log_return; otherwise zero; own prior-clean q90 clock"
    ),
    "stale_1h": "primary signal, entry, exit, and side shifted by exactly 12 bars",
    "stale_24h": "primary signal, entry, exit, and side shifted by exactly 288 bars",
}


@dataclass(frozen=True)
class EvaluationConfig:
    leverage: float = 0.5
    base_cost_notional_per_side: float = 0.0006
    stress_cost_notional_per_side: float = 0.0010
    hold_bars: int = 24
    score_quantile: float = 0.90
    baseline_clean_observations: int = 8_640
    baseline_minimum_observations: int = 2_016
    post_gap_quarantine_bars: int = 24
    stale_1h_bars: int = 12
    stale_24h_bars: int = 288
    cluster_permutations: int = 100_000
    cluster_seed: int = 20_260_720
    minimum_mean_gross_underlying_bp: float = 24.0
    minimum_cagr_to_strict_mdd: float = 3.0
    stress_minimum_cagr_to_strict_mdd: float = 2.5
    maximum_strict_mdd_pct: float = 15.0
    maximum_weekly_cluster_p: float = 0.10
    minimum_control_ratio_margin: float = 0.25
    minimum_train_trades: int = 500
    minimum_selection_trades: int = 150
    minimum_train_split_trades: int = 100
    minimum_selection_split_trades: int = 60
    minimum_train_trades_each_side: int = 100
    minimum_selection_trades_each_side: int = 40
    minimum_train_weekly_clusters: int = 26
    minimum_selection_weekly_clusters: int = 12


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def seal_result(core: dict[str, Any]) -> dict[str, Any]:
    return {**core, "result_hash": canonical_hash(core)}


def validate_result_hash(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "result_hash"}
    if canonical_hash(core) != payload.get("result_hash"):
        raise ValueError("BAFR-24F result hash mismatch")


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _require_hash(path: Path, expected: str) -> None:
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"frozen BAFR-24F dependency changed: {path}")


def _naive_utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="raise").dt.tz_convert(None)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    output = num.divide(den.where(den.gt(0.0)))
    return output.where(np.isfinite(output), np.nan)


def direct_feature_control_scores(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Return formula-frozen controls using only the completed signal bar."""
    quote = frame["quote_notional"]
    return {
        "aggressor_flow_only": -_safe_divide(frame["signed_quote_notional"], quote),
        "tick_direction_only": pd.to_numeric(
            frame["tick_notional_imbalance"], errors="coerce"
        ),
        "strict_nonzero_tick_only": _safe_divide(
            frame["strict_sell_frustrated_notional"]
            - frame["strict_buy_frustrated_notional"],
            quote,
        ),
        "carried_zero_tick_only": _safe_divide(
            frame["carried_sell_frustrated_notional"]
            - frame["carried_buy_frustrated_notional"],
            quote,
        ),
    }


def completed_bar_rejection_score(frame: pd.DataFrame) -> pd.Series:
    """Return the pre-entry completed-bar flow/rejection control score."""
    flow = _safe_divide(frame["signed_quote_notional"], frame["quote_notional"])
    micro_return = pd.to_numeric(frame["micro_log_return"], errors="coerce")
    return (-flow).where(flow.mul(micro_return).lt(0.0), 0.0)


def _normalize_clock(clock: pd.DataFrame, *, name: str) -> pd.DataFrame:
    required = {"signal_date", "entry_date", "exit_date", "side"}
    missing = required.difference(clock.columns)
    if missing:
        raise ValueError(f"BAFR-24F {name} clock lacks columns: {sorted(missing)}")
    normalized = clock.loc[
        :, ["signal_date", "entry_date", "exit_date", "side"]
    ].copy()
    for column in ("signal_date", "entry_date", "exit_date"):
        normalized[column] = _naive_utc(normalized[column])
    normalized["side"] = pd.to_numeric(normalized["side"], errors="raise").astype(np.int8)
    normalized["control"] = name
    return normalized[
        ["control", "side", "signal_date", "entry_date", "exit_date"]
    ].reset_index(drop=True)


def _clock_hash(clock: pd.DataFrame) -> str:
    name = str(clock["control"].iloc[0])
    if not clock["control"].eq(name).all():
        raise ValueError("BAFR-24F clock mixes control identities")
    normalized = _normalize_clock(clock, name=name)
    rows = [
        {
            "control": str(row.control),
            "side": int(row.side),
            "signal_date": pd.Timestamp(row.signal_date).isoformat(),
            "entry_date": pd.Timestamp(row.entry_date).isoformat(),
            "exit_date": pd.Timestamp(row.exit_date).isoformat(),
        }
        for row in normalized.itertuples(index=False)
    ]
    return canonical_hash(rows)


def _validate_clock(clock: pd.DataFrame, *, name: str, cfg: EvaluationConfig) -> None:
    if clock.empty:
        raise ValueError(f"BAFR-24F {name} clock is empty")
    if not clock["entry_date"].is_monotonic_increasing:
        raise ValueError(f"BAFR-24F {name} clock is not sorted")
    if clock["entry_date"].duplicated().any():
        raise ValueError(f"BAFR-24F {name} clock has duplicate entries")
    if not clock["side"].isin((-1, 1)).all():
        raise ValueError(f"BAFR-24F {name} clock has an invalid side")
    if not clock["control"].eq(name).all():
        raise ValueError(f"BAFR-24F {name} clock identity changed")
    if not (clock["entry_date"] - clock["signal_date"]).eq(BAR).all():
        raise ValueError(f"BAFR-24F {name} is not next-open")
    if not (clock["exit_date"] - clock["entry_date"]).eq(BAR * cfg.hold_bars).all():
        raise ValueError(f"BAFR-24F {name} hold changed")
    if clock["signal_date"].min() < TRAIN_START or clock["exit_date"].max() >= SELECTION_END:
        raise ValueError(f"BAFR-24F {name} crossed the frozen source interval")
    for column in ("signal_date", "entry_date", "exit_date"):
        epoch_seconds = clock[column].astype("int64") // 1_000_000_000
        if not np.equal(epoch_seconds % 300, 0).all():
            raise ValueError(f"BAFR-24F {name} is not five-minute aligned")
    if len(clock) > 1:
        current = clock["entry_date"].iloc[1:].reset_index(drop=True)
        previous_exit = clock["exit_date"].iloc[:-1].reset_index(drop=True)
        if not current.ge(previous_exit).all():
            raise ValueError(f"BAFR-24F {name} clock overlaps")


def _build_score_clock(
    dates: pd.Series,
    score: pd.Series,
    clean: pd.Series,
    *,
    name: str,
    cfg: EvaluationConfig,
) -> pd.DataFrame:
    numeric_score = pd.to_numeric(score, errors="coerce")
    valid_clean = clean.astype(bool) & numeric_score.notna() & np.isfinite(numeric_score)
    threshold = support_builder.prior_clean_quantile(
        numeric_score.abs(),
        valid_clean,
        quantile=cfg.score_quantile,
        window=cfg.baseline_clean_observations,
        min_periods=cfg.baseline_minimum_observations,
    )
    raw = valid_clean & numeric_score.ne(0.0) & numeric_score.abs().ge(threshold)
    quarantined = ~clean.astype(bool).to_numpy(bool)
    sides = np.sign(numeric_score.fillna(0.0)).to_numpy(np.int8)
    rows: list[dict[str, Any]] = []
    previous_exit = -1
    for signal_position in np.flatnonzero(raw.to_numpy(bool)):
        entry_position = int(signal_position + 1)
        exit_position = int(entry_position + cfg.hold_bars)
        if exit_position >= len(dates):
            continue
        if entry_position < previous_exit:
            continue
        if quarantined[signal_position : exit_position + 1].any():
            continue
        if pd.Timestamp(dates.iloc[exit_position]) >= SELECTION_END:
            continue
        rows.append(
            {
                "control": name,
                "side": int(sides[signal_position]),
                "signal_date": dates.iloc[signal_position],
                "entry_date": dates.iloc[entry_position],
                "exit_date": dates.iloc[exit_position],
            }
        )
        previous_exit = exit_position
    return pd.DataFrame(
        rows,
        columns=["control", "side", "signal_date", "entry_date", "exit_date"],
    )


def _shift_clock(
    primary: pd.DataFrame,
    *,
    name: str,
    bars: int,
) -> pd.DataFrame:
    shifted = primary.copy()
    delta = BAR * bars
    for column in ("signal_date", "entry_date", "exit_date"):
        shifted[column] += delta
    shifted["control"] = name
    shifted = shifted.loc[shifted["exit_date"].lt(SELECTION_END)]
    return shifted.reset_index(drop=True)


def _source_gap_days(manifest: dict[str, Any]) -> list[str]:
    return sorted(
        {
            str(archive["date"])
            for month in manifest.get("months", [])
            for archive in month.get("archives", [])
            if int(archive.get("state_reset_count", 0)) > 0
        }
    )


def _verify_admission_chain() -> tuple[dict[str, Any], dict[str, Any]]:
    frozen = (
        (MECHANISM_DOCUMENT, MECHANISM_DOCUMENT_SHA256),
        (SUPPORT_SOURCE, SUPPORT_SOURCE_SHA256),
        (SUPPORT_DOCUMENT, SUPPORT_DOCUMENT_SHA256),
        (SUPPORT_RESULT, SUPPORT_RESULT_SHA256),
        (PRIMARY_CLOCK, PRIMARY_CLOCK_SHA256),
        (NOVELTY_SOURCE, NOVELTY_SOURCE_SHA256),
        (NOVELTY_DOCUMENT, NOVELTY_DOCUMENT_SHA256),
        (NOVELTY_RESULT, NOVELTY_RESULT_SHA256),
        (PREREGISTRATION, PREREGISTRATION_SHA256),
        (FEATURE_DATA, FEATURE_DATA_SHA256),
        (FEATURE_MANIFEST, FEATURE_MANIFEST_SHA256),
        (MARKET_DATA, MARKET_DATA_SHA256),
        (MARKET_MANIFEST, MARKET_MANIFEST_SHA256),
        (FUNDING_DATA, FUNDING_DATA_SHA256),
        (FUNDING_MANIFEST, FUNDING_MANIFEST_SHA256),
        (COMPLETED_BAR_DATA, COMPLETED_BAR_DATA_SHA256),
        (COMPLETED_BAR_MANIFEST, COMPLETED_BAR_MANIFEST_SHA256),
    )
    for path, expected in frozen:
        _require_hash(path, expected)

    support = _read_json(SUPPORT_RESULT)
    if (
        support.get("candidate") != CANDIDATE
        or support.get("outcomes_opened") is not False
        or support.get("passed") is not True
        or support.get("clock", {}).get("sha256") != PRIMARY_CLOCK_SHA256
        or support.get("source", {}).get("market_columns_loaded") != ["date"]
        or support.get("source", {}).get("price_or_outcome_columns_loaded") != []
    ):
        raise ValueError("BAFR-24F support admission contract changed")

    novelty = _read_json(NOVELTY_RESULT)
    if (
        novelty.get("candidate") != CANDIDATE
        or novelty.get("outcomes_opened") is not False
        or novelty.get("passed") is not True
        or novelty.get("next_stage") != "freeze_outcome_evaluator"
    ):
        raise ValueError("BAFR-24F novelty admission contract changed")
    comparisons = novelty.get("comparisons")
    if not isinstance(comparisons, list) or not comparisons or not all(
        item.get("passes") is True for item in comparisons
    ):
        raise ValueError("BAFR-24F novelty comparisons no longer pass")
    return support, novelty


def _verified_source_manifests() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    feature = _read_json(FEATURE_MANIFEST)
    market = _read_json(MARKET_MANIFEST)
    completed = _read_json(COMPLETED_BAR_MANIFEST)
    if feature.get("protocol", {}).get("outcomes_opened") is not False:
        raise ValueError("BAFR feature source opened outcomes")
    if feature.get("combined_sha256") != FEATURE_DATA_SHA256:
        raise ValueError("BAFR feature manifest data hash changed")
    if market.get("protocol", {}).get("outcomes_opened") is not False:
        raise ValueError("BAFR market source lacks unopened provenance")
    if market.get("combined_sha256") != MARKET_DATA_SHA256:
        raise ValueError("BAFR market manifest data hash changed")
    if completed.get("protocol", {}).get("outcomes_opened") is not False:
        raise ValueError("BAFR completed-bar predictor source opened outcomes")
    if completed.get("combined_sha256") != COMPLETED_BAR_DATA_SHA256:
        raise ValueError("BAFR completed-bar manifest data hash changed")
    return feature, market, completed


def build_control_clocks(
    cfg: EvaluationConfig | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    frozen_cfg = EvaluationConfig() if cfg is None else cfg
    if frozen_cfg != EvaluationConfig():
        raise ValueError("BAFR-24F evaluation parameters are frozen")
    support, novelty = _verify_admission_chain()
    feature_manifest, market_manifest, completed_manifest = _verified_source_manifests()

    market_dates = pd.read_csv(
        MARKET_DATA,
        compression="gzip",
        usecols=["date"],
    )
    market_dates["date"] = _naive_utc(market_dates["date"])
    if (
        market_dates.empty
        or market_dates["date"].duplicated().any()
        or not market_dates["date"].is_monotonic_increasing
        or not market_dates["date"].diff().dropna().eq(BAR).all()
        or market_dates["date"].iloc[0] != TRAIN_START
        or market_dates["date"].iloc[-1] != SELECTION_END - BAR
        or len(market_dates) != int(market_manifest.get("rows", -1))
    ):
        raise ValueError("BAFR-24F market timestamp grid changed")

    feature_columns = [
        "date",
        "quote_notional",
        "classified_quote_notional",
        "signed_quote_notional",
        "up_tick_notional",
        "down_tick_notional",
        "strict_buy_frustrated_notional",
        "strict_sell_frustrated_notional",
        "carried_buy_frustrated_notional",
        "carried_sell_frustrated_notional",
        "frustration_score",
        "tick_notional_imbalance",
    ]
    features = pd.read_csv(
        FEATURE_DATA,
        compression="gzip",
        usecols=feature_columns,
    )
    features["date"] = _naive_utc(features["date"])
    if (
        features["date"].duplicated().any()
        or not features["date"].is_monotonic_increasing
        or len(features) != int(feature_manifest.get("rows", -1))
    ):
        raise ValueError("BAFR-24F feature predictor rows changed")
    frame = market_dates.merge(features, on="date", how="left", validate="one_to_one")
    available = frame[feature_columns[1:]].notna().all(axis=1)
    gap_days = _source_gap_days(feature_manifest)
    gap_day = frame["date"].dt.strftime("%Y-%m-%d").isin(gap_days)
    quarantined = support_builder.quarantine_mask(
        available,
        gap_day,
        post_gap_bars=frozen_cfg.post_gap_quarantine_bars,
    )
    clean = ~quarantined

    quote = frame["quote_notional"]
    classified = frame["classified_quote_notional"]
    expected_primary = _safe_divide(
        frame["strict_sell_frustrated_notional"]
        + frame["carried_sell_frustrated_notional"]
        - frame["strict_buy_frustrated_notional"]
        - frame["carried_buy_frustrated_notional"],
        quote,
    ).fillna(0.0)
    expected_tick = _safe_divide(
        frame["up_tick_notional"] - frame["down_tick_notional"], classified
    ).fillna(0.0)
    if not np.allclose(
        frame.loc[available, "frustration_score"].to_numpy(float),
        expected_primary.loc[available].to_numpy(float),
        rtol=1e-9,
        atol=1e-9,
    ):
        raise ValueError("BAFR-24F primary score identity changed")
    if not np.allclose(
        frame.loc[available, "tick_notional_imbalance"].to_numpy(float),
        expected_tick.loc[available].to_numpy(float),
        rtol=1e-9,
        atol=1e-9,
    ):
        raise ValueError("BAFR-24F tick score identity changed")

    primary = _normalize_clock(pd.read_csv(PRIMARY_CLOCK), name="primary")
    controls: dict[str, pd.DataFrame] = {"primary": primary}
    direction_flip = primary.copy()
    direction_flip["side"] *= -1
    direction_flip["control"] = "direction_flip"
    controls["direction_flip"] = direction_flip

    score_map = direct_feature_control_scores(frame)
    for name, score in score_map.items():
        controls[name] = _build_score_clock(
            frame["date"], score, clean, name=name, cfg=frozen_cfg
        )

    completed_columns = [
        "date",
        "quote_notional",
        "signed_quote_notional",
        "micro_log_return",
    ]
    completed = pd.read_csv(
        COMPLETED_BAR_DATA,
        compression="gzip",
        usecols=completed_columns,
    )
    completed["date"] = _naive_utc(completed["date"])
    completed_source_rows = len(completed)
    if (
        completed["date"].duplicated().any()
        or not completed["date"].is_monotonic_increasing
        or len(completed) != int(completed_manifest.get("rows", -1))
    ):
        raise ValueError("BAFR-24F completed-bar predictor rows changed")
    completed = market_dates.merge(
        completed, on="date", how="left", validate="one_to_one"
    )
    completed_available = completed[completed_columns[1:]].notna().all(axis=1)
    missing_completed_quarantine = (
        (~completed_available)
        .astype(np.int8)
        .rolling(frozen_cfg.post_gap_quarantine_bars + 1, min_periods=1)
        .max()
        .astype(bool)
    )
    completed_clean = clean & ~missing_completed_quarantine
    completed_score = completed_bar_rejection_score(completed)
    controls["completed_bar_rejection"] = _build_score_clock(
        completed["date"],
        completed_score,
        completed_clean,
        name="completed_bar_rejection",
        cfg=frozen_cfg,
    )
    controls["stale_1h"] = _shift_clock(
        primary, name="stale_1h", bars=frozen_cfg.stale_1h_bars
    )
    controls["stale_24h"] = _shift_clock(
        primary, name="stale_24h", bars=frozen_cfg.stale_24h_bars
    )

    if tuple(controls) != POLICY_NAMES:
        raise ValueError("BAFR-24F control order changed")
    for name, clock in controls.items():
        _validate_clock(clock, name=name, cfg=frozen_cfg)
    if len(primary) != int(support.get("clock", {}).get("rows", -1)):
        raise ValueError("BAFR-24F primary clock count changed")

    metadata = {
        "support_commit": SUPPORT_COMMIT,
        "novelty_commit": NOVELTY_COMMIT,
        "support_result_sha256": SUPPORT_RESULT_SHA256,
        "novelty_result_sha256": NOVELTY_RESULT_SHA256,
        "feature_predictor_rows_parsed": int(len(features)),
        "feature_predictor_columns_parsed": feature_columns,
        "completed_bar_predictor_rows_parsed": int(completed_source_rows),
        "completed_bar_predictor_columns_parsed": completed_columns,
        "official_market_columns_parsed": ["date"],
        "official_market_value_rows_parsed": 0,
        "funding_value_rows_parsed": 0,
        "post_entry_outcome_rows_loaded": 0,
        "strategy_outcomes_calculated": False,
        "source_gap_days": gap_days,
        "novelty_comparison_count": int(len(novelty["comparisons"])),
    }
    return controls, metadata


def _validate_funding_manifest() -> dict[str, Any]:
    _require_hash(FUNDING_MANIFEST, FUNDING_MANIFEST_SHA256)
    manifest = _read_json(FUNDING_MANIFEST)
    if manifest.get("outcomes_opened") is not False:
        raise ValueError("BAFR-24F funding source lacks unopened provenance")
    if manifest.get("strategy_outcomes_calculated") != []:
        raise ValueError("BAFR-24F funding source calculated a strategy outcome")
    if manifest.get("data", {}).get("sha256") != FUNDING_DATA_SHA256:
        raise ValueError("BAFR-24F funding manifest data hash changed")
    if manifest.get("quality", {}).get("events") != manifest.get("data", {}).get("rows"):
        raise ValueError("BAFR-24F funding event count differs from manifest")
    maximum_error = manifest.get("quality", {}).get(
        "maximum_proxy_funding_cash_error_bp_notional", float("inf")
    )
    allowed_error = manifest.get("mapping", {}).get(
        "maximum_allowed_proxy_funding_cash_error_bp_notional", -1.0
    )
    if maximum_error > allowed_error:
        raise ValueError("BAFR-24F settlement-mark proxy error exceeds frozen limit")
    if (
        pd.Timestamp(manifest.get("selection_end_exclusive")) != SELECTION_END
        or manifest.get("config", {}).get("interval") != "8h"
    ):
        raise ValueError("BAFR-24F funding interval contract changed")
    return manifest


def _scan_market_timestamps(path: Path) -> dict[str, Any]:
    counts = {"train": 0, "selection": 0}
    total = 0
    first: str | None = None
    last: str | None = None
    previous: pd.Timestamp | None = None
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        header = handle.readline().rstrip("\r\n").split(",")
        if not header or header[0] != "date":
            raise ValueError("BAFR-24F market date must be the first physical column")
        for line in handle:
            date_text = line.split(",", 1)[0]
            timestamp = pd.Timestamp(date_text)
            if previous is not None and timestamp - previous != BAR:
                raise ValueError("BAFR-24F market timestamps are not a complete grid")
            first = date_text if first is None else first
            last = date_text
            previous = timestamp
            total += 1
            if TRAIN_START <= timestamp < TRAIN_END:
                counts["train"] += 1
            elif TRAIN_END <= timestamp < SELECTION_END:
                counts["selection"] += 1
            else:
                raise ValueError("BAFR-24F market timestamp crossed frozen coverage")
    expected_last = str(SELECTION_END - BAR)
    if first != str(TRAIN_START) or last != expected_last:
        raise ValueError("BAFR-24F market source lacks exact pre-2024 coverage")
    return {
        "timestamp_column": "date",
        "timestamp_rows_scanned": int(total),
        "value_rows_parsed": 0,
        "first_timestamp": first,
        "last_timestamp": last,
        "audited_eof_last_timestamp": last,
        "window_value_row_counts": counts,
    }


def _scan_funding_timestamps(path: Path) -> dict[str, Any]:
    counts = {"train": 0, "selection": 0}
    total = 0
    first: int | None = None
    last: int | None = None
    train_start_ms = int(TRAIN_START.timestamp() * 1_000)
    train_end_ms = int(TRAIN_END.timestamp() * 1_000)
    selection_end_ms = int(SELECTION_END.timestamp() * 1_000)
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        header = handle.readline().rstrip("\r\n").split(",")
        if not header or header[0] != "funding_time_ms":
            raise ValueError("BAFR-24F funding time must be the first physical column")
        for line in handle:
            timestamp_ms = int(line.split(",", 1)[0])
            if last is not None and timestamp_ms <= last:
                raise ValueError("BAFR-24F funding timestamps are not strictly increasing")
            first = timestamp_ms if first is None else first
            last = timestamp_ms
            total += 1
            if train_start_ms <= timestamp_ms < train_end_ms:
                counts["train"] += 1
            elif train_end_ms <= timestamp_ms < selection_end_ms:
                counts["selection"] += 1
            else:
                raise ValueError("BAFR-24F funding timestamp crossed frozen coverage")
    return {
        "timestamp_column": "funding_time_ms",
        "timestamp_rows_scanned": int(total),
        "value_rows_parsed": 0,
        "first_timestamp_ms": first,
        "last_timestamp_ms": last,
        "audited_eof_last_timestamp_ms": last,
        "window_value_row_counts": counts,
    }


def scan_outcome_boundaries() -> dict[str, Any]:
    _require_hash(MARKET_DATA, MARKET_DATA_SHA256)
    _require_hash(MARKET_MANIFEST, MARKET_MANIFEST_SHA256)
    _require_hash(FUNDING_DATA, FUNDING_DATA_SHA256)
    manifest = _validate_funding_manifest()
    market_manifest = _read_json(MARKET_MANIFEST)
    market = _scan_market_timestamps(MARKET_DATA)
    funding = _scan_funding_timestamps(FUNDING_DATA)
    if market["timestamp_rows_scanned"] != market_manifest.get("rows"):
        raise ValueError("BAFR-24F market timestamp count differs from manifest")
    if funding["timestamp_rows_scanned"] != manifest.get("data", {}).get("rows"):
        raise ValueError("BAFR-24F funding timestamp count differs from manifest")
    if funding["last_timestamp_ms"] != manifest.get("data", {}).get(
        "last_funding_time_ms"
    ):
        raise ValueError("BAFR-24F funding terminal timestamp differs from manifest")
    expected_funding_terminal = int(
        (SELECTION_END - pd.Timedelta(hours=8)).timestamp() * 1_000
    )
    if funding["last_timestamp_ms"] != expected_funding_terminal:
        raise ValueError("BAFR-24F funding source lacks final 2023 settlement")
    return {"market": market, "funding": funding}


def verify_evaluation_freeze(
    cfg: EvaluationConfig | None = None,
) -> dict[str, Any]:
    frozen_cfg = EvaluationConfig() if cfg is None else cfg
    if frozen_cfg != EvaluationConfig():
        raise ValueError("BAFR-24F evaluation parameters are frozen")
    if not EVALUATION_FREEZE.is_file():
        raise ValueError("BAFR-24F evaluator freeze is missing")
    payload = _read_json(EVALUATION_FREEZE)
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if canonical_hash(core) != payload.get("manifest_hash"):
        raise ValueError("BAFR-24F evaluator freeze manifest hash mismatch")
    if payload.get("outcomes_opened") is not False or payload.get("opened_windows") != []:
        raise ValueError("BAFR-24F evaluator was not frozen before outcomes")
    if payload.get("evaluation_source") != str(EVALUATION_SOURCE):
        raise ValueError("BAFR-24F evaluator source path changed")
    if payload.get("evaluation_source_sha256") != sha256_file(EVALUATION_SOURCE):
        raise ValueError("BAFR-24F evaluator differs from pre-outcome freeze")
    if payload.get("freeze_source") != str(FREEZE_SOURCE):
        raise ValueError("BAFR-24F freeze source path changed")
    if payload.get("freeze_source_sha256") != sha256_file(FREEZE_SOURCE):
        raise ValueError("BAFR-24F freeze tool differs from pre-outcome freeze")
    if payload.get("evaluation_test") != str(EVALUATION_TEST):
        raise ValueError("BAFR-24F evaluator-test path changed")
    if payload.get("evaluation_test_sha256") != sha256_file(EVALUATION_TEST):
        raise ValueError("BAFR-24F evaluator tests differ from pre-outcome freeze")
    if payload.get("freeze_test") != str(FREEZE_TEST):
        raise ValueError("BAFR-24F freeze-test path changed")
    if payload.get("freeze_test_sha256") != sha256_file(FREEZE_TEST):
        raise ValueError("BAFR-24F freeze tests differ from pre-outcome freeze")
    if payload.get("preregistration_sha256") != PREREGISTRATION_SHA256:
        raise ValueError("BAFR-24F preregistration changed")
    if payload.get("sealed_windows") != [
        "train_2020_2022",
        "selection_2023",
        "2024",
        "2025",
        "2026_ytd",
    ]:
        raise ValueError("BAFR-24F sealed windows changed")
    if payload.get("mutable_parameters") != []:
        raise ValueError("BAFR-24F freeze permits mutable parameters")
    if payload.get("market_value_rows_parsed_during_freeze") != 0:
        raise ValueError("BAFR-24F freeze parsed market values")
    if payload.get("funding_value_rows_parsed_during_freeze") != 0:
        raise ValueError("BAFR-24F freeze parsed funding values")
    if payload.get("execution_simulation_run_during_freeze") is not False:
        raise ValueError("BAFR-24F freeze simulated execution")
    if payload.get("evaluation_config") != asdict(frozen_cfg):
        raise ValueError("BAFR-24F evaluator configuration changed")
    if payload.get("policy_names") != list(POLICY_NAMES):
        raise ValueError("BAFR-24F policy set changed")
    if payload.get("control_semantics") != CONTROL_SEMANTICS:
        raise ValueError("BAFR-24F control semantics changed")
    controls, predictor_metadata = build_control_clocks(frozen_cfg)
    actual_hashes = {name: _clock_hash(clock) for name, clock in controls.items()}
    actual_counts = {name: int(len(clock)) for name, clock in controls.items()}
    if payload.get("control_clock_hashes") != actual_hashes:
        raise ValueError("BAFR-24F control clock differs from pre-outcome freeze")
    if payload.get("control_clock_counts") != actual_counts:
        raise ValueError("BAFR-24F control count differs from pre-outcome freeze")
    if payload.get("predictor_boundary") != predictor_metadata:
        raise ValueError("BAFR-24F predictor boundary differs from freeze")
    if payload.get("outcome_boundaries") != scan_outcome_boundaries():
        raise ValueError("BAFR-24F outcome timestamp boundaries changed")
    return payload


def _parse_market_window(
    path: Path,
    *,
    start: str,
    end: str,
    audited_eof_last_timestamp: str | None = None,
) -> pd.DataFrame:
    rows: list[tuple[str, float, float, float, float]] = []
    end_boundary_seen = False
    last_timestamp: str | None = None
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        header = handle.readline().rstrip("\r\n").split(",")
        required = ["date", "open", "high", "low", "close"]
        positions = {column: header.index(column) for column in required}
        if positions["date"] != 0:
            raise ValueError("BAFR-24F market date must be the first physical column")
        for line in handle:
            date_text = line.split(",", 1)[0]
            last_timestamp = date_text
            if date_text < start:
                continue
            if date_text >= end:
                end_boundary_seen = True
                break
            fields = next(csv.reader([line]))
            rows.append(
                (
                    date_text,
                    float(fields[positions["open"]]),
                    float(fields[positions["high"]]),
                    float(fields[positions["low"]]),
                    float(fields[positions["close"]]),
                )
            )
    if not end_boundary_seen and last_timestamp != audited_eof_last_timestamp:
        raise ValueError(
            f"BAFR-24F market source did not reach a physical or audited boundary {end}"
        )
    return pd.DataFrame(rows, columns=["date", "open", "high", "low", "close"])


def _parse_funding_window(
    path: Path,
    *,
    start: str,
    end: str,
    audited_eof_last_timestamp_ms: int | None = None,
) -> pd.DataFrame:
    start_ms = int(pd.Timestamp(start).timestamp() * 1_000)
    end_ms = int(pd.Timestamp(end).timestamp() * 1_000)
    rows: list[tuple[int, str, str, str, str, str, str]] = []
    end_boundary_seen = False
    last_timestamp_ms: int | None = None
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        header = handle.readline().rstrip("\r\n").split(",")
        required = [
            "funding_time_ms",
            "funding_time_utc",
            "symbol",
            "funding_rate",
            "settlement_mark_price",
            "funding_time_offset_ms",
            "mark_source",
        ]
        positions = {column: header.index(column) for column in required}
        if positions["funding_time_ms"] != 0:
            raise ValueError("BAFR-24F funding timestamp must be first")
        for line in handle:
            timestamp_ms = int(line.split(",", 1)[0])
            last_timestamp_ms = timestamp_ms
            if timestamp_ms < start_ms:
                continue
            if timestamp_ms >= end_ms:
                end_boundary_seen = True
                break
            fields = next(csv.reader([line]))
            rows.append(
                (
                    timestamp_ms,
                    fields[positions["funding_time_utc"]],
                    fields[positions["symbol"]],
                    fields[positions["funding_rate"]],
                    fields[positions["settlement_mark_price"]],
                    fields[positions["funding_time_offset_ms"]],
                    fields[positions["mark_source"]],
                )
            )
    if not end_boundary_seen and last_timestamp_ms != audited_eof_last_timestamp_ms:
        raise ValueError(
            f"BAFR-24F funding source did not reach a physical or audited boundary {end}"
        )
    return pd.DataFrame(
        rows,
        columns=[
            "funding_time_ms",
            "funding_time_utc",
            "symbol",
            "funding_rate",
            "settlement_mark_price",
            "funding_time_offset_ms",
            "mark_source",
        ],
    )


def load_market_window(
    stage: str, freeze: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    full_window, _ = STAGE_WINDOWS[stage]
    start, end = WINDOWS[full_window]
    _require_hash(MARKET_DATA, MARKET_DATA_SHA256)
    audited = freeze["outcome_boundaries"]["market"]["audited_eof_last_timestamp"]
    market = _parse_market_window(
        MARKET_DATA,
        start=start,
        end=end,
        audited_eof_last_timestamp=audited if end == "2024-01-01" else None,
    )
    expected_rows = freeze["outcome_boundaries"]["market"][
        "window_value_row_counts"
    ][stage]
    if len(market) != expected_rows:
        raise ValueError("BAFR-24F market window row count differs from freeze")
    market["date"] = _naive_utc(market["date"])
    if (
        market.empty
        or market["date"].iloc[0] != pd.Timestamp(start)
        or market["date"].iloc[-1] != pd.Timestamp(end) - BAR
        or market["date"].duplicated().any()
        or not market["date"].is_monotonic_increasing
        or not market["date"].diff().dropna().eq(BAR).all()
    ):
        raise ValueError("BAFR-24F market value window is invalid")
    prices = market[["open", "high", "low", "close"]].to_numpy(float)
    if not np.isfinite(prices).all() or (prices <= 0.0).any():
        raise ValueError("BAFR-24F market contains invalid prices")
    opens, highs, lows, closes = (
        market[column].to_numpy(float) for column in ("open", "high", "low", "close")
    )
    if (
        (highs < np.maximum(opens, closes)).any()
        or (lows > np.minimum(opens, closes)).any()
        or (highs < lows).any()
    ):
        raise ValueError("BAFR-24F market violates OHLC invariants")
    return market, {
        "sha256": MARKET_DATA_SHA256,
        "rows": int(len(market)),
        "columns_parsed": ["date", "open", "high", "low", "close"],
        "physical_value_window": f"{start} <= date < {end}",
        "values_before_start_parsed": 0,
        "values_at_or_after_end_parsed": 0,
        "first_date": str(market["date"].iloc[0]),
        "last_date": str(market["date"].iloc[-1]),
    }


def load_funding_window(
    stage: str, freeze: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = _validate_funding_manifest()
    full_window, _ = STAGE_WINDOWS[stage]
    start, end = WINDOWS[full_window]
    _require_hash(FUNDING_DATA, FUNDING_DATA_SHA256)
    audited = freeze["outcome_boundaries"]["funding"][
        "audited_eof_last_timestamp_ms"
    ]
    funding = _parse_funding_window(
        FUNDING_DATA,
        start=start,
        end=end,
        audited_eof_last_timestamp_ms=audited if end == "2024-01-01" else None,
    )
    expected_rows = freeze["outcome_boundaries"]["funding"][
        "window_value_row_counts"
    ][stage]
    if len(funding) != expected_rows:
        raise ValueError("BAFR-24F funding window row count differs from freeze")
    funding["funding_time_ms"] = pd.to_numeric(
        funding["funding_time_ms"], errors="raise"
    ).astype(np.int64)
    utc = _naive_utc(funding["funding_time_utc"])
    epoch = pd.to_datetime(
        funding["funding_time_ms"], unit="ms", utc=True, errors="raise"
    ).dt.tz_convert(None)
    if not utc.equals(epoch):
        raise ValueError("BAFR-24F funding timestamps disagree")
    if funding["funding_time_ms"].duplicated().any() or not funding[
        "funding_time_ms"
    ].is_monotonic_increasing:
        raise ValueError("BAFR-24F funding timestamps are invalid")
    if not funding["symbol"].eq("BTCUSDT").all():
        raise ValueError("BAFR-24F funding contains another symbol")
    if not funding["mark_source"].eq("binance_8h_mark_price_kline_open").all():
        raise ValueError("BAFR-24F funding uses another settlement-mark proxy")
    offsets = pd.to_numeric(
        funding["funding_time_offset_ms"], errors="raise"
    ).to_numpy(np.int64)
    maximum_offset = int(manifest["mapping"]["maximum_allowed_timestamp_offset_ms"])
    if (offsets < 0).any() or (offsets > maximum_offset).any():
        raise ValueError("BAFR-24F funding timestamps exceed mark tolerance")
    rates = pd.to_numeric(funding["funding_rate"], errors="raise").to_numpy(float)
    marks = pd.to_numeric(
        funding["settlement_mark_price"], errors="raise"
    ).to_numpy(float)
    if not np.isfinite(rates).all() or not np.isfinite(marks).all() or (marks <= 0.0).any():
        raise ValueError("BAFR-24F funding values are invalid")
    normalized = pd.DataFrame(
        {
            "funding_time_ms": funding["funding_time_ms"].to_numpy(np.int64),
            "funding_time": utc,
            "funding_rate": rates,
            "settlement_mark_price": marks,
        }
    )
    return normalized, {
        "manifest_sha256": FUNDING_MANIFEST_SHA256,
        "data_sha256": FUNDING_DATA_SHA256,
        "rows": int(len(normalized)),
        "physical_value_window": f"{start} <= funding_time < {end}",
        "values_before_start_parsed": 0,
        "values_at_or_after_end_parsed": 0,
        "first_funding_time": (
            str(normalized["funding_time"].iloc[0]) if len(normalized) else None
        ),
        "last_funding_time": (
            str(normalized["funding_time"].iloc[-1]) if len(normalized) else None
        ),
        "mark_semantics": "frozen official 8h mark-price-kline-open proxy",
    }


def _slice_schedule(schedule: pd.DataFrame, *, start: str, end: str) -> pd.DataFrame:
    start_timestamp = pd.Timestamp(start)
    end_timestamp = pd.Timestamp(end)
    inside = (
        schedule["entry_date"].ge(start_timestamp)
        & schedule["entry_date"].lt(end_timestamp)
        & schedule["exit_date"].gt(start_timestamp)
        & schedule["exit_date"].lt(end_timestamp)
    )
    return schedule.loc[inside].reset_index(drop=True)


def attach_market_positions(schedule: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    attached = schedule.copy()
    positions = pd.Series(np.arange(len(market), dtype=np.int64), index=market["date"])
    for label in ("entry", "exit"):
        mapped = attached[f"{label}_date"].map(positions)
        if mapped.isna().any():
            missing = attached.loc[mapped.isna(), f"{label}_date"].head().tolist()
            raise ValueError(f"BAFR-24F {label} timestamps missing from market: {missing}")
        attached[f"{label}_position"] = mapped.astype(np.int64)
    if len(attached) and not (
        attached["exit_position"] - attached["entry_position"]
    ).eq(EvaluationConfig().hold_bars).all():
        raise ValueError("BAFR-24F market positions violate the frozen hold")
    return attached


def weekly_cluster_sign_flip(
    values: Iterable[float],
    entry_dates: Iterable[pd.Timestamp | str],
    *,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    returns = np.asarray(list(values), dtype=float)
    dates = pd.to_datetime(list(entry_dates))
    if len(returns) == 0 or len(returns) != len(dates):
        return {
            "p_value_one_sided": 1.0,
            "observed_mean_return": 0.0,
            "cluster_count": 0,
            "method": "empty",
            "permutations": 0,
            "seed": int(seed),
        }
    frame = pd.DataFrame({"week": dates.to_period("W-SUN"), "return": returns})
    clusters = frame.groupby("week", sort=True)["return"].sum().to_numpy(float)
    observed = float(np.sum(clusters) / len(returns))
    if len(clusters) <= 18:
        outcomes = np.fromiter(
            (
                np.dot(signs, clusters) / len(returns)
                for signs in product((-1.0, 1.0), repeat=len(clusters))
            ),
            dtype=float,
        )
        p_value = float(np.mean(outcomes >= observed - 1e-15))
        method = "exact"
        completed = int(len(outcomes))
    else:
        if permutations <= 0:
            raise ValueError("BAFR-24F cluster permutation count must be positive")
        rng = np.random.default_rng(seed)
        exceedances = 0
        completed = 0
        while completed < permutations:
            batch = min(10_000, permutations - completed)
            signs = rng.integers(0, 2, size=(batch, len(clusters)), dtype=np.int8)
            randomized = (signs.astype(float) * 2.0 - 1.0).dot(clusters) / len(returns)
            exceedances += int(np.count_nonzero(randomized >= observed - 1e-15))
            completed += batch
        p_value = float((1 + exceedances) / (permutations + 1))
        method = "monte_carlo"
    return {
        "p_value_one_sided": p_value,
        "observed_mean_return": observed,
        "cluster_count": int(len(clusters)),
        "method": method,
        "permutations": completed,
        "seed": int(seed),
    }


def _trade_statistics(values: list[float]) -> dict[str, Any]:
    count = len(values)
    if not count:
        return {
            "n_trades": 0,
            "mean_trade_return_pct": 0.0,
            "std_trade_return_pct": 0.0,
            "t_stat_like": 0.0,
            "ci95_mean_trade_return_pct": [0.0, 0.0],
        }
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if count > 1 else 0.0
    standard_error = std / math.sqrt(count)
    return {
        "n_trades": count,
        "mean_trade_return_pct": mean * 100.0,
        "std_trade_return_pct": std * 100.0,
        "t_stat_like": mean / standard_error if standard_error > 0.0 else 0.0,
        "ci95_mean_trade_return_pct": [
            (mean - 1.96 * standard_error) * 100.0,
            (mean + 1.96 * standard_error) * 100.0,
        ],
    }


def simulate_schedule(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    start: str,
    end: str,
    cost_notional_per_side: float,
    cfg: EvaluationConfig,
    compute_cluster: bool,
) -> dict[str, Any]:
    if cfg != EvaluationConfig():
        raise ValueError("BAFR-24F simulation configuration is frozen")
    start_timestamp = pd.Timestamp(start)
    end_timestamp = pd.Timestamp(end)
    if start_timestamp >= end_timestamp or cfg.leverage <= 0.0:
        raise ValueError("BAFR-24F simulation parameters are invalid")
    if not 0.0 <= cost_notional_per_side < 0.1:
        raise ValueError("BAFR-24F execution cost is invalid")
    opens = market["open"].to_numpy(float)
    highs = market["high"].to_numpy(float)
    lows = market["low"].to_numpy(float)
    dates = market["date"]
    market_ms = (dates.astype("int64") // 1_000_000).to_numpy(np.int64)
    funding_times = funding["funding_time_ms"].to_numpy(np.int64)
    funding_rates = funding["funding_rate"].to_numpy(float)
    funding_marks = funding["settlement_mark_price"].to_numpy(float)

    equity = 1.0
    high_water_mark = 1.0
    strict_mdd = 0.0
    previous_exit = -1
    trade_returns: list[float] = []
    gross_returns: list[float] = []
    entry_dates: list[pd.Timestamp] = []
    sides: list[int] = []
    funding_settlement_count = 0
    applied_funding_count = 0
    dropped_boundary_funding_credits = 0
    trades_with_funding = 0
    funding_cash_sum = 0.0
    entry_equity_sum = 0.0

    def update_path(value: float) -> None:
        nonlocal high_water_mark, strict_mdd
        if not np.isfinite(value):
            raise ValueError("BAFR-24F strict equity path is non-finite")
        floored = max(0.0, float(value))
        high_water_mark = max(high_water_mark, floored)
        strict_mdd = max(
            strict_mdd,
            1.0 - floored / max(high_water_mark, 1e-15),
        )

    for row in schedule.itertuples(index=False):
        entry_position = int(row.entry_position)
        exit_position = int(row.exit_position)
        side = int(row.side)
        if side not in (-1, 1):
            raise ValueError("BAFR-24F side must be long or short")
        if not 0 <= entry_position < exit_position < len(market):
            raise ValueError("BAFR-24F scheduled positions are invalid")
        if exit_position - entry_position != cfg.hold_bars:
            raise ValueError("BAFR-24F scheduled hold changed")
        if entry_position < previous_exit:
            raise ValueError("BAFR-24F schedules overlap")
        entry_time = pd.Timestamp(row.entry_date)
        exit_time = pd.Timestamp(row.exit_date)
        if entry_time != dates.iloc[entry_position] or exit_time != dates.iloc[exit_position]:
            raise ValueError("BAFR-24F schedule timestamp differs from market")
        if not (
            start_timestamp <= entry_time < end_timestamp
            and start_timestamp < exit_time < end_timestamp
        ):
            raise ValueError("BAFR-24F trade crosses a simulation split")

        entry_price = float(opens[entry_position])
        exit_price = float(opens[exit_position])
        held_high = float(np.max(highs[entry_position:exit_position]))
        held_low = float(np.min(lows[entry_position:exit_position]))
        if min(entry_price, exit_price, held_high, held_low) <= 0.0:
            raise ValueError("BAFR-24F scheduled trade has an invalid price")

        entry_equity = equity
        quantity = entry_equity * cfg.leverage / entry_price
        entry_fee = quantity * entry_price * cost_notional_per_side
        entry_equity_sum += entry_equity
        update_path(entry_equity - entry_fee)

        entry_ms = int(market_ms[entry_position])
        exit_ms = int(market_ms[exit_position])
        left = int(np.searchsorted(funding_times, entry_ms, side="left"))
        right = int(np.searchsorted(funding_times, exit_ms, side="right"))
        event_times = funding_times[left:right]
        rates = funding_rates[left:right]
        marks = funding_marks[left:right]
        contributions = -side * quantity * marks * rates
        boundary = (event_times == entry_ms) | (event_times == exit_ms)
        dropped_credit = boundary & (contributions > 0.0)
        applied = ~dropped_credit
        applied_contributions = contributions[applied]
        funding_cash = float(np.sum(applied_contributions, dtype=float))
        funding_credit = float(
            np.sum(np.maximum(applied_contributions, 0.0), dtype=float)
        )
        funding_debit = float(
            np.sum(np.minimum(applied_contributions, 0.0), dtype=float)
        )

        favorable_price = held_high if side > 0 else held_low
        adverse_price = held_low if side > 0 else held_high
        favorable_equity = (
            entry_equity
            - entry_fee
            + side * quantity * (favorable_price - entry_price)
            + funding_credit
        )
        update_path(favorable_equity)
        adverse_exit_fee = quantity * adverse_price * cost_notional_per_side
        adverse_equity = (
            entry_equity
            - entry_fee
            + side * quantity * (adverse_price - entry_price)
            + funding_credit
            + funding_debit
            - adverse_exit_fee
        )
        update_path(adverse_equity)

        gross_return = side * (exit_price / entry_price - 1.0)
        exit_fee = quantity * exit_price * cost_notional_per_side
        realized_equity = (
            entry_equity
            - entry_fee
            + side * quantity * (exit_price - entry_price)
            + funding_cash
            - exit_fee
        )
        equity = max(0.0, float(realized_equity))
        update_path(equity)

        trade_returns.append(equity / entry_equity - 1.0 if entry_equity > 0.0 else -1.0)
        gross_returns.append(gross_return)
        entry_dates.append(entry_time)
        sides.append(side)
        funding_settlement_count += int(len(contributions))
        applied_funding_count += int(np.count_nonzero(applied))
        dropped_boundary_funding_credits += int(np.count_nonzero(dropped_credit))
        trades_with_funding += int(len(contributions) > 0)
        funding_cash_sum += funding_cash
        previous_exit = exit_position

    years = (end_timestamp - start_timestamp).total_seconds() / (365.25 * 86_400.0)
    absolute_return = (equity - 1.0) * 100.0
    cagr = (equity ** (1.0 / years) - 1.0) * 100.0 if equity > 0.0 else -100.0
    strict_mdd_pct = strict_mdd * 100.0
    if strict_mdd_pct > 1e-12:
        ratio = float(cagr / strict_mdd_pct)
        zero_mdd_ratio_cap_applied = False
    elif cagr > 0.0:
        ratio = 1.0e12
        zero_mdd_ratio_cap_applied = True
    else:
        ratio = 0.0
        zero_mdd_ratio_cap_applied = False
    cluster = (
        weekly_cluster_sign_flip(
            trade_returns,
            entry_dates,
            permutations=cfg.cluster_permutations,
            seed=cfg.cluster_seed,
        )
        if compute_cluster
        else None
    )
    return {
        "absolute_return_pct": float(absolute_return),
        "cagr_pct": float(cagr),
        "strict_mdd_pct": float(strict_mdd_pct),
        "cagr_to_strict_mdd": ratio,
        "zero_mdd_ratio_cap_applied": zero_mdd_ratio_cap_applied,
        "trade_count": int(len(sides)),
        "long_count": int(sum(side > 0 for side in sides)),
        "short_count": int(sum(side < 0 for side in sides)),
        "wall_clock_years": float(years),
        "mean_gross_underlying_move_bp": (
            float(np.mean(gross_returns) * 10_000.0) if gross_returns else 0.0
        ),
        "funding_settlement_count": int(funding_settlement_count),
        "applied_funding_settlement_count": int(applied_funding_count),
        "dropped_boundary_funding_credits": int(dropped_boundary_funding_credits),
        "trades_with_funding": int(trades_with_funding),
        "total_funding_cash_pct_of_entry_equity_sum": float(
            100.0 * funding_cash_sum / entry_equity_sum
            if entry_equity_sum > 0.0
            else 0.0
        ),
        "execution_cost_notional_per_side_bp": float(
            cost_notional_per_side * 10_000.0
        ),
        "leverage": float(cfg.leverage),
        "trade_statistics": _trade_statistics(trade_returns),
        "weekly_cluster_sign_flip": cluster,
    }


def _evaluate_policy_stage(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    clock: pd.DataFrame,
    *,
    stage: str,
    cfg: EvaluationConfig,
    compute_cluster: bool,
) -> dict[str, Any]:
    full_name, split_names = STAGE_WINDOWS[stage]
    full_start, full_end = WINDOWS[full_name]
    calendar_clock = clock.loc[
        clock["entry_date"].ge(pd.Timestamp(full_start))
        & clock["entry_date"].lt(pd.Timestamp(full_end))
    ]
    full_clock = _slice_schedule(clock, start=full_start, end=full_end)
    attached = attach_market_positions(full_clock, market)
    base = simulate_schedule(
        market,
        funding,
        attached,
        start=full_start,
        end=full_end,
        cost_notional_per_side=cfg.base_cost_notional_per_side,
        cfg=cfg,
        compute_cluster=compute_cluster,
    )
    stress = simulate_schedule(
        market,
        funding,
        attached,
        start=full_start,
        end=full_end,
        cost_notional_per_side=cfg.stress_cost_notional_per_side,
        cfg=cfg,
        compute_cluster=False,
    )
    splits: dict[str, Any] = {}
    for name in split_names:
        start, end = WINDOWS[name]
        split = _slice_schedule(attached, start=start, end=end)
        splits[name] = simulate_schedule(
            market,
            funding,
            split,
            start=start,
            end=end,
            cost_notional_per_side=cfg.base_cost_notional_per_side,
            cfg=cfg,
            compute_cluster=False,
        )
    sides: dict[str, Any] = {}
    for label, side in (("long_only", 1), ("short_only", -1)):
        side_clock = attached.loc[attached["side"].eq(side)].reset_index(drop=True)
        sides[label] = simulate_schedule(
            market,
            funding,
            side_clock,
            start=full_start,
            end=full_end,
            cost_notional_per_side=cfg.base_cost_notional_per_side,
            cfg=cfg,
            compute_cluster=False,
        )
    return {
        "support_calendar_event_count": int(len(calendar_clock)),
        "fully_contained_trade_count": int(len(attached)),
        "boundary_crossing_events_excluded": int(len(calendar_clock) - len(attached)),
        "base_6bp": base,
        "stress_10bp": stress,
        "splits_base_6bp": splits,
        "side_contributions_base_6bp": sides,
    }


def _stage_minimums(stage: str, cfg: EvaluationConfig) -> dict[str, int]:
    if stage == "train":
        return {
            "trades": cfg.minimum_train_trades,
            "split_trades": cfg.minimum_train_split_trades,
            "side_trades": cfg.minimum_train_trades_each_side,
            "weekly_clusters": cfg.minimum_train_weekly_clusters,
        }
    if stage == "selection":
        return {
            "trades": cfg.minimum_selection_trades,
            "split_trades": cfg.minimum_selection_split_trades,
            "side_trades": cfg.minimum_selection_trades_each_side,
            "weekly_clusters": cfg.minimum_selection_weekly_clusters,
        }
    raise ValueError(f"unknown BAFR-24F stage: {stage}")


def performance_gate_failures(
    policy_result: dict[str, Any],
    *,
    stage: str,
    cfg: EvaluationConfig,
) -> list[str]:
    base = policy_result["base_6bp"]
    stress = policy_result["stress_10bp"]
    minimums = _stage_minimums(stage, cfg)
    failures: list[str] = []
    if base["absolute_return_pct"] <= 0.0:
        failures.append(f"{stage}: non-positive absolute return")
    if base["cagr_to_strict_mdd"] < cfg.minimum_cagr_to_strict_mdd:
        failures.append(f"{stage}: CAGR/strict-MDD below 3")
    if base["strict_mdd_pct"] > cfg.maximum_strict_mdd_pct:
        failures.append(f"{stage}: strict MDD above 15%")
    if base["zero_mdd_ratio_cap_applied"]:
        failures.append(f"{stage}: base ratio used zero-MDD cap")
    if base["trade_count"] < minimums["trades"]:
        failures.append(f"{stage}: too few fully-contained trades")
    if base["long_count"] < minimums["side_trades"]:
        failures.append(f"{stage}: too few long trades")
    if base["short_count"] < minimums["side_trades"]:
        failures.append(f"{stage}: too few short trades")
    if stress["trade_count"] != base["trade_count"]:
        failures.append(f"{stage}: stress trade count differs from base")
    if stress["absolute_return_pct"] <= 0.0:
        failures.append(f"{stage}: 10bp stress non-positive")
    if stress["cagr_to_strict_mdd"] < cfg.stress_minimum_cagr_to_strict_mdd:
        failures.append(f"{stage}: 10bp stress CAGR/strict-MDD below 2.5")
    if stress["zero_mdd_ratio_cap_applied"]:
        failures.append(f"{stage}: stress ratio used zero-MDD cap")
    if base["mean_gross_underlying_move_bp"] < cfg.minimum_mean_gross_underlying_bp:
        failures.append(f"{stage}: mean gross edge below 24 bp")
    cluster = base["weekly_cluster_sign_flip"]
    if cluster is None:
        failures.append(f"{stage}: weekly-cluster test missing")
    else:
        if cluster["cluster_count"] < minimums["weekly_clusters"]:
            failures.append(f"{stage}: too few weekly clusters")
        if cluster["p_value_one_sided"] > cfg.maximum_weekly_cluster_p:
            failures.append(f"{stage}: weekly-cluster p-value above 0.10")
    for label, metrics in policy_result["side_contributions_base_6bp"].items():
        if metrics["absolute_return_pct"] <= 0.0:
            failures.append(f"{stage}: {label} contribution non-positive")
    for name, metrics in policy_result["splits_base_6bp"].items():
        if metrics["absolute_return_pct"] <= 0.0:
            failures.append(f"{name}: non-positive absolute return")
        if metrics["trade_count"] < minimums["split_trades"]:
            failures.append(f"{name}: too few fully-contained trades")
    return failures


def qualification(
    policy_results: dict[str, dict[str, Any]],
    *,
    stage: str,
    cfg: EvaluationConfig,
) -> dict[str, Any]:
    primary_failures = performance_gate_failures(
        policy_results["primary"], stage=stage, cfg=cfg
    )
    mechanism_failures = {
        name: performance_gate_failures(policy_results[name], stage=stage, cfg=cfg)
        for name in MECHANISM_REJECTION_CONTROLS
    }
    passing_mechanism_controls = [
        name for name, failures in mechanism_failures.items() if not failures
    ]
    primary_ratio = float(
        policy_results["primary"]["base_6bp"]["cagr_to_strict_mdd"]
    )
    control_ratios = {
        name: float(policy_results[name]["base_6bp"]["cagr_to_strict_mdd"])
        for name in SUPERIORITY_CONTROLS
    }
    ratio_margins = {
        name: primary_ratio - ratio for name, ratio in control_ratios.items()
    }
    invalid_ratio = (
        not np.isfinite(primary_ratio)
        or policy_results["primary"]["base_6bp"]["zero_mdd_ratio_cap_applied"]
        or any(
            not np.isfinite(control_ratios[name])
            or policy_results[name]["base_6bp"]["zero_mdd_ratio_cap_applied"]
            for name in SUPERIORITY_CONTROLS
        )
    )
    minimum_margin = None if invalid_ratio else min(ratio_margins.values())
    superiority_failures = []
    if minimum_margin is None or minimum_margin < cfg.minimum_control_ratio_margin:
        superiority_failures.append(
            f"{stage}: minimum primary-control CAGR/MDD margin below 0.25"
        )
    failures = [*primary_failures, *superiority_failures]
    failures.extend(
        f"mechanism-null control independently passed every gate: {name}"
        for name in passing_mechanism_controls
    )
    return {
        "qualifies": not failures,
        "stage": stage,
        "scope": "frozen pre-2024 performance and mechanism-falsification gates",
        "final_promotion_allowed": False,
        "failures": failures,
        "primary_performance_gate_failures": primary_failures,
        "mechanism_control_gate_failures": mechanism_failures,
        "passing_mechanism_controls": passing_mechanism_controls,
        "control_cagr_to_strict_mdd": control_ratios,
        "primary_minus_control_ratio_margins": ratio_margins,
        "minimum_primary_control_ratio_margin": minimum_margin,
        "superiority_gate_failures": superiority_failures,
        "direction_and_stale_controls_are_not_replacement_mechanisms": True,
    }


def _headline(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: metrics[key]
        for key in (
            "absolute_return_pct",
            "cagr_pct",
            "strict_mdd_pct",
            "cagr_to_strict_mdd",
            "mean_gross_underlying_move_bp",
            "trade_count",
            "long_count",
            "short_count",
        )
    }


def _compute_stage_report(
    cfg: EvaluationConfig,
    freeze: dict[str, Any],
    controls: dict[str, pd.DataFrame],
    *,
    stage: str,
    created_at: str,
    train_parent: dict[str, Any] | None = None,
) -> dict[str, Any]:
    market, market_source = load_market_window(stage, freeze)
    funding, funding_source = load_funding_window(stage, freeze)
    cluster_controls = {"primary", *MECHANISM_REJECTION_CONTROLS}
    policy_results = {
        name: _evaluate_policy_stage(
            market,
            funding,
            clock,
            stage=stage,
            cfg=cfg,
            compute_cluster=name in cluster_controls,
        )
        for name, clock in controls.items()
    }
    verdict = qualification(policy_results, stage=stage, cfg=cfg)
    if stage == "train":
        opened_windows = ["train_2020_2022"]
        decision = "open_selection_2023" if verdict["qualifies"] else "reject_before_selection"
        selection_opened = False
    else:
        opened_windows = ["train_2020_2022", "selection_2023"]
        decision = (
            "freeze_forward_and_portfolio_validation"
            if verdict["qualifies"]
            else "reject_before_forward_validation"
        )
        selection_opened = True
    core: dict[str, Any] = {
        "schema_version": 1,
        "created_at": created_at,
        "candidate": CANDIDATE,
        "protocol": {
            "name": "BAFR-24F strict sequential pre-2024 evaluation",
            "stage": stage,
            "opened_windows": opened_windows,
            "selection_2023_opened": selection_opened,
            "forward_windows_opened": False,
            "full_calendar_cagr_including_idle_cash": True,
            "next_open_execution": True,
            "funding_interval": "entry_time <= funding_time <= exit_time",
            "funding_boundary": (
                "exact-boundary debits included; exact-boundary credits dropped"
            ),
            "funding_notional": (
                "fixed entry quantity times exact realized rate and frozen "
                "settlement-mark proxy"
            ),
            "strict_mdd": (
                "global/pre-entry HWM; entry cost; all favorable held OHLC plus "
                "funding credits; all adverse held OHLC plus funding debits and "
                "hypothetical exit cost; realized exit and exit cost"
            ),
            "controls_cannot_replace_primary": True,
            "absolute_return_always_reported": True,
        },
        "evaluation_config": asdict(cfg),
        "evaluation_freeze_sha256": sha256_file(EVALUATION_FREEZE),
        "evaluation_freeze_manifest_hash": freeze["manifest_hash"],
        "control_clock_hashes": {
            name: _clock_hash(clock) for name, clock in controls.items()
        },
        "control_clock_counts": {
            name: int(len(clock)) for name, clock in controls.items()
        },
        "source": {
            "support_result_sha256": SUPPORT_RESULT_SHA256,
            "novelty_result_sha256": NOVELTY_RESULT_SHA256,
            "primary_clock_sha256": PRIMARY_CLOCK_SHA256,
            "market": market_source,
            "funding": funding_source,
        },
        "policies": policy_results,
        "qualification": verdict,
        "decision": decision,
    }
    if train_parent is not None:
        core["train_parent"] = train_parent
        core["protocol"]["train_replayed_before_selection"] = True
    return seal_result(core)


def _write_result_once(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    validate_result_hash(payload)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )


def evaluate_train(cfg: EvaluationConfig | None = None) -> dict[str, Any]:
    frozen_cfg = EvaluationConfig() if cfg is None else cfg
    freeze = verify_evaluation_freeze(frozen_cfg)
    if TRAIN_OUTPUT.exists():
        raise FileExistsError("BAFR-24F train result is write-once")
    if SELECTION_OUTPUT.exists():
        raise ValueError("BAFR-24F selection result exists before train")
    controls, _ = build_control_clocks(frozen_cfg)
    report = _compute_stage_report(
        frozen_cfg,
        freeze,
        controls,
        stage="train",
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    _write_result_once(TRAIN_OUTPUT, report)
    return report


def _verify_passing_train_result(
    cfg: EvaluationConfig,
    freeze: dict[str, Any],
    controls: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    if not TRAIN_OUTPUT.is_file():
        raise PermissionError("2023 selection remains sealed: train artifact is missing")
    train = _read_json(TRAIN_OUTPUT)
    validate_result_hash(train)
    if train.get("schema_version") != 1 or train.get("candidate") != CANDIDATE:
        raise ValueError("BAFR-24F train result identity changed")
    if train.get("evaluation_config") != asdict(cfg):
        raise ValueError("BAFR-24F train config differs from freeze")
    if train.get("evaluation_freeze_sha256") != sha256_file(EVALUATION_FREEZE):
        raise ValueError("BAFR-24F train belongs to another evaluator freeze")
    protocol = train.get("protocol", {})
    if protocol.get("opened_windows") != ["train_2020_2022"]:
        raise ValueError("BAFR-24F train opened an unexpected window")
    if protocol.get("selection_2023_opened") is not False:
        raise ValueError("BAFR-24F train already opened selection outcomes")
    if train.get("control_clock_hashes") != freeze["control_clock_hashes"]:
        raise ValueError("BAFR-24F train used another control clock")
    verdict = train.get("qualification", {})
    if verdict.get("qualifies") is not True or verdict.get("failures") != []:
        raise PermissionError("2023 selection remains sealed because train failed")
    if train.get("decision") != "open_selection_2023":
        raise PermissionError("BAFR-24F train did not authorize selection")
    created_at = train.get("created_at")
    if not isinstance(created_at, str) or not created_at:
        raise ValueError("BAFR-24F train lacks a creation timestamp")
    replay = _compute_stage_report(
        cfg,
        freeze,
        controls,
        stage="train",
        created_at=created_at,
    )
    if replay != train:
        raise ValueError("BAFR-24F train artifact does not exactly replay")
    return train


def evaluate_selection(cfg: EvaluationConfig | None = None) -> dict[str, Any]:
    frozen_cfg = EvaluationConfig() if cfg is None else cfg
    freeze = verify_evaluation_freeze(frozen_cfg)
    if SELECTION_OUTPUT.exists():
        raise FileExistsError("BAFR-24F selection result is write-once")
    controls, _ = build_control_clocks(frozen_cfg)
    train = _verify_passing_train_result(frozen_cfg, freeze, controls)
    train_parent = {
        "path": str(TRAIN_OUTPUT),
        "sha256": sha256_file(TRAIN_OUTPUT),
        "result_hash": train["result_hash"],
    }
    report = _compute_stage_report(
        frozen_cfg,
        freeze,
        controls,
        stage="selection",
        created_at=datetime.now(timezone.utc).isoformat(),
        train_parent=train_parent,
    )
    _write_result_once(SELECTION_OUTPUT, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("train", "selection"), required=True)
    args = parser.parse_args()
    result = evaluate_train() if args.stage == "train" else evaluate_selection()
    full_name, _ = STAGE_WINDOWS[args.stage]
    print(
        json.dumps(
            {
                "stage": args.stage,
                "qualification": result["qualification"],
                "decision": result["decision"],
                "primary": _headline(result["policies"]["primary"]["base_6bp"]),
                "primary_stress_10bp": _headline(
                    result["policies"]["primary"]["stress_10bp"]
                ),
                "primary_splits": {
                    name: _headline(metrics)
                    for name, metrics in result["policies"]["primary"][
                        "splits_base_6bp"
                    ].items()
                },
                "primary_sides": {
                    name: _headline(metrics)
                    for name, metrics in result["policies"]["primary"][
                        "side_contributions_base_6bp"
                    ].items()
                },
                "controls": {
                    name: _headline(policy["base_6bp"])
                    for name, policy in result["policies"].items()
                    if name != "primary"
                },
                "full_window": full_name,
                "output": str(
                    TRAIN_OUTPUT if args.stage == "train" else SELECTION_OUTPUT
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
