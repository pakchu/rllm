"""Sequential strict economic evaluator for frozen QLCD-288 clocks.

The evaluator freeze reads the source-only clock but parses no BTC execution
OHLC or funding row. Phase one opens train (2020-2022) and selection (2023)
sequentially. Test, eval, and recent-report data require a separately committed
phase-two evaluator after both phase-one stages pass.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import evaluate_stablecoin_quote_flow_diffusion as strict_source  # noqa: E402
from training import (  # noqa: E402
    evaluate_quantity_lattice_cohort_disagreement_support as support_source,
)
from training import preregister_quantity_lattice_cohort_disagreement as prereg  # noqa: E402
from training.build_six_alt_price_free_flow_panel import (  # noqa: E402
    deterministic_gzip_csv,
)


BAR = cast(pd.Timedelta, pd.Timedelta(minutes=5))
YEAR_SECONDS = 365.25 * 86_400.0
POLICY_ID = prereg.Policy().policy_id
SUPPORT_COMMIT = "71bc605fa1ee345d26983e029f98fedafaa726ac"

PREREGISTRATION = Path(prereg.DEFAULT_OUTPUT)
SUPPORT_RESULT = Path(
    "results/quantity_lattice_cohort_disagreement_support_2026-07-20.json"
)
PRIMARY_CLOCK = Path(
    "data/quantity_lattice_cohort_disagreement_clock_2020_2023.csv.gz"
)
SOURCE_ACCESS_SEAL = Path(
    "results/quantity_lattice_cohort_disagreement_source_access_seal_2026-07-20.json"
)
EVALUATOR_SOURCE = Path(
    "training/evaluate_quantity_lattice_cohort_disagreement.py"
)
EVALUATOR_FREEZE = Path(
    "results/quantity_lattice_cohort_disagreement_evaluator_freeze_2026-07-20.json"
)
EVALUATION_CLOCK = Path(
    "data/quantity_lattice_cohort_disagreement_evaluation_clocks_2020_2023.csv.gz"
)

LEGACY_MARKET = Path(
    "data/binance_um_kline_reference_btc_2020_2023/"
    "BTCUSDT_5m_2020-01-01_2023-12-31.csv.gz"
)
LEGACY_MARKET_SHA256 = (
    "e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d"
)
LEGACY_MARKET_MANIFEST = Path(
    "data/binance_um_kline_reference_btc_2020_2023/build_manifest.json"
)
LEGACY_MARKET_MANIFEST_SHA256 = (
    "c04fbbd299cc748a6745c0ef030787da4d560833c744c81c98dd8840efc7913e"
)
LEGACY_FUNDING = Path("data/binance_um_btcusdt_funding_marks_2020_2023.csv.gz")
LEGACY_FUNDING_SHA256 = (
    "3284bbb6bb67946acb673c6b67459543e217f752589e1d47b6c7c3b659f733e6"
)
LEGACY_FUNDING_MANIFEST = Path(
    "results/binance_um_btcusdt_funding_marks_2020_2023_manifest_2026-07-17.json"
)
LEGACY_FUNDING_MANIFEST_SHA256 = (
    "a0b2d27e1aa8cf2d9ab8cb659b598ee0a6d7bd25401c9e10ae92d1a74415845b"
)

STATIC_INPUT_SHA256: dict[str, str] = {
    str(PREREGISTRATION): (
        "eb7920891cb2c9c5753f08e5a3ebfd3c3d39de28fdb2a245fc5b6d978c0f84d9"
    ),
    str(SUPPORT_RESULT): (
        "d5b5f2e59fe2f8d8df775a9ee7a05da0bab2898af210d6e724669d9781efe640"
    ),
    str(PRIMARY_CLOCK): (
        "ed882ac8a28f1f0b2b7ad7bf3d2de1f37b175cde63b20d4d1c7a290f3eb89bec"
    ),
    str(SOURCE_ACCESS_SEAL): (
        "cade903a3d15349903c3e16853a23a092b36a293cb46ceb7b0c5514737aca834"
    ),
    "training/preregister_quantity_lattice_cohort_disagreement.py": (
        "565022325caa7d4cf167475a9eb0ca154cb809c777d5dbc906244a5d348ce012"
    ),
    "training/evaluate_quantity_lattice_cohort_disagreement_support.py": (
        "a800441c6f5a6aafa08a41323ca6d03a933c7567845d83f4b8112e46727af8a6"
    ),
    "docs/quantity-lattice-cohort-disagreement-preregistration-2026-07-20.md": (
        "b28db48c10eccb6d5464d0b377dc4f18df5921e4c06e41c8b3922efada7429c6"
    ),
    "docs/quantity-lattice-cohort-support-pass-2026-07-20.md": (
        "397f8ba82a8973ae19a56309b96f8dcf9cfb80b1a4e481ddaca20ec6db4bc8b0"
    ),
    str(LEGACY_MARKET_MANIFEST): LEGACY_MARKET_MANIFEST_SHA256,
    str(LEGACY_FUNDING_MANIFEST): LEGACY_FUNDING_MANIFEST_SHA256,
    str(strict_source.EVALUATOR_SOURCE): (
        "0ea59a107f05777ba91ab1c8fc5900e724ba48ec6ce647a42c34c34422222e3b"
    ),
}

STAGE_ORDER = ("train", "selection", "test", "eval", "recent")
PHASE_ONE = ("train", "selection")


def _utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp is pd.NaT:
        raise ValueError("QLCD-288 timestamp is NaT")
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return cast(pd.Timestamp, timestamp)


STAGE_WINDOWS: dict[str, tuple[pd.Timestamp, pd.Timestamp | None]] = {
    "train": (_utc("2020-01-01"), _utc("2023-01-01")),
    "selection": (_utc("2023-01-01"), _utc("2024-01-01")),
    "test": (_utc("2024-01-01"), _utc("2025-01-01")),
    "eval": (_utc("2025-01-01"), _utc("2026-01-01")),
    "recent": (_utc("2026-01-01"), None),
}
SUBPERIOD_WINDOWS: dict[str, dict[str, tuple[pd.Timestamp, pd.Timestamp]]] = {
    "train": {
        "2020": (_utc("2020-01-01"), _utc("2021-01-01")),
        "2021": (_utc("2021-01-01"), _utc("2022-01-01")),
        "2022": (_utc("2022-01-01"), _utc("2023-01-01")),
    },
    "selection": {
        "2023_h1": (_utc("2023-01-01"), _utc("2023-07-01")),
        "2023_h2": (_utc("2023-07-01"), _utc("2024-01-01")),
    },
}
STAGE_OUTPUTS = {
    "train": Path(
        "results/quantity_lattice_cohort_disagreement_train_2020_2022_2026-07-20.json"
    ),
    "selection": Path(
        "results/quantity_lattice_cohort_disagreement_selection_2023_2026-07-20.json"
    ),
    "test": Path(
        "results/quantity_lattice_cohort_disagreement_test_2024_2026-07-20.json"
    ),
    "eval": Path(
        "results/quantity_lattice_cohort_disagreement_eval_2025_2026-07-20.json"
    ),
    "recent": Path(
        "results/quantity_lattice_cohort_disagreement_recent_2026_2026-07-20.json"
    ),
}
STAGE_DOCS = {
    stage: Path(f"docs/quantity-lattice-cohort-disagreement-{stage}-result-2026-07-20.md")
    for stage in STAGE_ORDER
}
STAGE_SOURCE_MANIFESTS = {
    stage: Path(
        "results/"
        f"quantity_lattice_cohort_disagreement_{stage}_execution_source_2026-07-20.json"
    )
    for stage in PHASE_ONE
}
STAGE_SOURCE_DIRS = {
    stage: Path("data/quantity_lattice_cohort_disagreement_execution") / stage
    for stage in PHASE_ONE
}
CLOCK_COLUMNS = (
    "decision_time",
    "entry_time",
    "exit_time",
    "side",
    "score",
    "threshold",
)
CONTROL_ORDER = (
    "primary",
    "exact_side_flip",
    "medium_vs_fine",
    "remove_opposition",
    "all_quantity_imbalance",
    "stale_one_hour",
    "stale_twenty_four_hours",
)
EVALUATION_CLOCK_COLUMNS = ("control", *CLOCK_COLUMNS)


@dataclass(frozen=True)
class EvaluationConfig:
    leverage: float = 0.5
    base_cost_notional_per_side: float = 0.0006
    stress_cost_notional_per_side: float = 0.0010
    hold_bars: int = 288
    exact_cluster_max: int = 20
    cluster_draws: int = 20_000
    cluster_seed: int = 20_260_720


FROZEN_CONFIG = EvaluationConfig()
FROZEN_GATES: dict[str, Any] = {
    "base_absolute_return_positive": True,
    "base_cagr_to_strict_mdd_min": 3.0,
    "strict_mdd_pct_max": 15.0,
    "stress_absolute_return_positive": True,
    "stress_cagr_to_strict_mdd_min": 2.5,
    "mean_gross_underlying_bp_min": 24.0,
    "weekly_cluster_signflip_p_strict_max": 0.10,
    "train_each_year_absolute_return_positive": True,
    "selection_each_half_absolute_return_positive": True,
}
FUTURE_PHASE_CONTRACT: dict[str, Any] = {
    "current_evaluator_approval": "train_and_selection_only",
    "phase_two_stages": ["test_2024", "eval_2025", "recent_report_2026_plus"],
    "signal_source": (
        "official Binance USD-M BTCUSDT daily aggTrades archives with each "
        "archive bound to its official CHECKSUM file"
    ),
    "continuity_anchor": {
        "source": str(support_source.DEFAULT_SOURCE),
        "source_sha256": support_source.EXPECTED_SOURCE_SHA256,
        "clock": str(PRIMARY_CLOCK),
        "clock_sha256": STATIC_INPUT_SHA256[str(PRIMARY_CLOCK)],
    },
    "source_failure_rules": {
        "checksum_missing_or_mismatch": "abort phase two; never quarantine-and-continue",
        "parse_or_quantity_precision_failure": "abort phase two",
        "intra_day_id_gap_overlap_regression_or_exact_duplicate": (
            "quarantine the complete incident UTC day and following 24 five-minute bars"
        ),
        "cross_day_id_discontinuity": (
            "quarantine both adjacent UTC days and the 24 bars following the later day"
        ),
        "missing_five_minute_bucket": (
            "treat as verified empty only when official USD-M kline volume and trade "
            "count are both zero; otherwise quarantine the complete UTC day and next 24 bars"
        ),
        "manual_exception_after_phase_one": False,
    },
    "clock_replay": (
        "concatenate the hash-bound 2020-2023 source with future source, preserve the "
        "strictly-prior 8640-row q99.75 baseline without reset, replay the global "
        "non-overlap scheduler without reset, and require the full pre-2024 primary "
        "clock to match its frozen bytes before admitting any future event"
    ),
    "stage_membership": (
        "admit only positions fully contained in the declared stage; freeze every "
        "cross-boundary exclusion identity before that stage's outcomes"
    ),
    "execution_source": (
        "official checksum-bound BTCUSDT USD-M 5m klines and exact funding events/marks; "
        "physically isolate each stage before numeric parsing; reject symlinks and stale caches"
    ),
    "recent_end": (
        "latest completed UTC day fixed in the phase-two freeze before any 2026 outcome opens"
    ),
    "phase_two_must_be_committed_before_2024_outcomes": True,
    "mutable_rules_after_phase_one": [],
}


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
        raise ValueError(f"QLCD-288 expected a JSON object: {path}")
    return payload


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")


def _verify_manifest(payload: dict[str, Any], *, label: str) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != _canonical_hash(core):
        raise ValueError(f"QLCD-288 {label} manifest hash changed")


def _write_once_bytes(path: str | Path, payload: bytes) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise FileExistsError(f"QLCD-288 artifact is write-once: {output}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _write_pair_once(
    first_path: str | Path,
    first_payload: bytes,
    second_path: str | Path,
    second_payload: bytes,
) -> None:
    paths = (Path(first_path), Path(second_path))
    if any(path.exists() for path in paths):
        raise FileExistsError("QLCD-288 result pair is write-once")
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    temporaries: list[Path] = []
    linked: list[Path] = []
    try:
        for path, payload in zip(paths, (first_payload, second_payload), strict=True):
            with tempfile.NamedTemporaryFile(
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                temporaries.append(temporary)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        for temporary, path in zip(temporaries, paths, strict=True):
            try:
                os.link(temporary, path)
            except FileExistsError as error:
                raise FileExistsError("QLCD-288 result pair is write-once") from error
            linked.append(path)
    except Exception:
        for path in linked:
            path.unlink(missing_ok=True)
        raise
    finally:
        for temporary in temporaries:
            temporary.unlink(missing_ok=True)


def _expected_preregistered_protocol() -> dict[str, Any]:
    return {
        "sequential_stages": [
            ["train", "2020-01-01", "2023-01-01"],
            ["selection", "2023-01-01", "2024-01-01"],
            ["test", "2024-01-01", "2025-01-01"],
            ["eval", "2025-01-01", "2026-01-01"],
            ["recent_report", "2026-01-01", None],
        ],
        "base_cost_bp_per_side": 6.0,
        "stress_cost_bp_per_side": 10.0,
        "full_calendar_cagr": True,
        "strict_held_path_mdd": True,
        "base_absolute_return_positive": True,
        "cagr_to_strict_mdd_min": 3.0,
        "strict_mdd_pct_max": 15.0,
        "stress_absolute_return_positive": True,
        "stress_cagr_to_strict_mdd_min": 2.5,
        "mean_gross_underlying_bp_min": 24.0,
        "weekly_cluster_signflip_p_max": 0.1,
        "train_each_year_positive": True,
        "selection_each_half_positive": True,
        "stop_on_first_failure": True,
    }


def _verify_static_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    for path, expected in STATIC_INPUT_SHA256.items():
        if _sha256(path) != expected:
            raise ValueError(f"QLCD-288 frozen input changed: {path}")
    registration = _load_json(PREREGISTRATION)
    prereg.validate_manifest(registration)
    if registration.get("manifest_hash") != (
        "9fd76b3dd9fd0d900689684c9d6b1d2c57ede9877eec73979b3ff11d29f59a16"
    ):
        raise ValueError("QLCD-288 preregistration manifest changed")
    if registration.get("later_economic_protocol") != _expected_preregistered_protocol():
        raise ValueError("QLCD-288 preregistered economic protocol changed")
    if registration.get("outcomes_opened") is not False:
        raise ValueError("QLCD-288 preregistration opened outcomes")

    support = _load_json(SUPPORT_RESULT)
    if support.get("decision") != "PASS_SUPPORT":
        raise ValueError("QLCD-288 source support did not pass")
    if support.get("outcomes_opened") is not False:
        raise ValueError("QLCD-288 support opened outcomes")
    if support.get("post_entry_market_rows_read") != 0:
        raise ValueError("QLCD-288 support opened market rows")
    if support.get("funding_rows_read") != 0:
        raise ValueError("QLCD-288 support opened funding rows")
    if support.get("clock_sha256") != STATIC_INPUT_SHA256[str(PRIMARY_CLOCK)]:
        raise ValueError("QLCD-288 support clock binding changed")

    seal = _load_json(SOURCE_ACCESS_SEAL)
    if seal.get("source_rows_parsed") != 0 or seal.get("outcomes_opened") is not False:
        raise ValueError("QLCD-288 source-access boundary changed")

    market_manifest = _load_json(LEGACY_MARKET_MANIFEST)
    if market_manifest.get("combined_sha256") != LEGACY_MARKET_SHA256:
        raise ValueError("QLCD-288 market manifest binding changed")
    if market_manifest.get("protocol", {}).get("archive_checksums_verified") is not True:
        raise ValueError("QLCD-288 market archives were not checksum verified")
    funding_manifest = _load_json(LEGACY_FUNDING_MANIFEST)
    if funding_manifest.get("data", {}).get("sha256") != LEGACY_FUNDING_SHA256:
        raise ValueError("QLCD-288 funding manifest binding changed")
    if funding_manifest.get("outcomes_opened") is not False:
        raise ValueError("QLCD-288 funding source manifest opened outcomes")
    return registration, support


def _schedule_hash(frame: pd.DataFrame) -> str:
    records = [
        {
            "decision_time": _utc(row["decision_time"]).isoformat(),
            "entry_time": _utc(row["entry_time"]).isoformat(),
            "exit_time": _utc(row["exit_time"]).isoformat(),
            "side": int(row["side"]),
            "score_hex": float(row["score"]).hex(),
            "threshold_hex": float(row["threshold"]).hex(),
        }
        for row in frame.to_dict(orient="records")
    ]
    return _canonical_hash(records)


def load_schedules(
    *,
    clock_path: str | Path = PRIMARY_CLOCK,
    expected_clock_sha256: str | None = None,
) -> pd.DataFrame:
    path = Path(clock_path)
    expected_hash = expected_clock_sha256 or STATIC_INPUT_SHA256[str(PRIMARY_CLOCK)]
    if _sha256(path) != expected_hash:
        raise ValueError("QLCD-288 frozen clock bytes changed")
    frame = pd.read_csv(path)
    if tuple(frame.columns) != CLOCK_COLUMNS:
        raise ValueError("QLCD-288 frozen clock schema changed")
    for column in ("decision_time", "entry_time", "exit_time"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    numeric = frame[["score", "threshold"]].to_numpy(float)
    if not np.isfinite(numeric).all() or bool((numeric <= 0.0).any()):
        raise ValueError("QLCD-288 score or threshold is invalid")
    if not bool(frame["score"].ge(frame["threshold"]).all()):
        raise ValueError("QLCD-288 clock violates its frozen threshold")
    if not bool(frame["side"].isin((-1, 1)).all()):
        raise ValueError("QLCD-288 clock side is invalid")
    if not bool(
        frame["entry_time"].sub(frame["decision_time"]).eq(pd.Timedelta(minutes=5)).all()
    ):
        raise ValueError("QLCD-288 entry delay changed")
    if not bool(
        frame["exit_time"].sub(frame["entry_time"]).eq(pd.Timedelta(hours=24)).all()
    ):
        raise ValueError("QLCD-288 hold changed")
    entries = cast(pd.Series, frame["entry_time"]).reset_index(drop=True)
    exits = cast(pd.Series, frame["exit_time"]).reset_index(drop=True)
    if not entries.is_monotonic_increasing or entries.duplicated().any():
        raise ValueError("QLCD-288 clock entries are invalid")
    if len(frame) > 1 and not bool(entries.iloc[1:].ge(exits.iloc[:-1].to_numpy()).all()):
        raise ValueError("QLCD-288 frozen schedule overlaps")
    return frame


def _control_clock(
    frame: pd.DataFrame,
    *,
    score: pd.Series,
    side: pd.Series,
    structural_eligibility: pd.Series,
) -> pd.DataFrame:
    registration = support_source.load_preregistration()
    policy = cast(dict[str, Any], registration["policy"])
    clean = cast(pd.Series, frame["source_complete"]).astype(bool)
    threshold = support_source.lagged_threshold(
        score,
        clean,
        window=int(policy["baseline_bars"]),
        minimum=int(policy["baseline_min_periods"]),
        quantile=float(policy["score_quantile"]),
    )
    eligible = (
        clean
        & structural_eligibility.astype(bool)
        & side.astype(int).ne(0)
        & score.astype(float).gt(0.0)
        & threshold.notna()
        & score.astype(float).ge(threshold)
    )
    dates = pd.to_datetime(frame["date"], utc=True, errors="raise")
    delay = int(policy["execution_delay_bars"])
    hold = int(policy["hold_bars"])
    next_free = 0
    rows: list[dict[str, Any]] = []
    for position in np.flatnonzero(eligible.to_numpy(bool)):
        entry_position = int(position) + delay
        exit_position = entry_position + hold
        if entry_position < next_free or exit_position >= len(frame):
            continue
        entry_time = dates.iloc[entry_position]
        exit_time = dates.iloc[exit_position]
        if entry_time >= support_source.SOURCE_END or exit_time >= support_source.SOURCE_END:
            continue
        rows.append(
            {
                "decision_time": dates.iloc[int(position)] + BAR,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "side": int(side.iloc[int(position)]),
                "score": float(score.iloc[int(position)]),
                "threshold": float(threshold.iloc[int(position)]),
            }
        )
        next_free = exit_position
    return pd.DataFrame(rows, columns=pd.Index(CLOCK_COLUMNS))


def _load_outcome_blind_feature_source() -> pd.DataFrame:
    registration = support_source.load_preregistration()
    cfg = support_source.SupportConfig()
    support_source.load_source_access_seal(cfg, registration)
    frame, _, source_hash, manifest_hash = support_source.load_source(
        cfg.source,
        cfg.source_manifest,
        registration,
    )
    if source_hash != support_source.EXPECTED_SOURCE_SHA256:
        raise ValueError("QLCD-288 outcome-blind feature source changed")
    if manifest_hash != support_source.EXPECTED_SOURCE_MANIFEST_SHA256:
        raise ValueError("QLCD-288 outcome-blind feature manifest changed")
    return frame


def derive_evaluation_clocks() -> pd.DataFrame:
    primary = load_schedules().copy()
    parts: list[pd.DataFrame] = []

    def add(control: str, clock: pd.DataFrame) -> None:
        checked = clock.copy()
        checked.insert(0, "control", control)
        parts.append(checked.loc[:, list(EVALUATION_CLOCK_COLUMNS)])

    add("primary", primary)
    flipped = primary.copy()
    flipped["side"] = -flipped["side"].astype(int)
    add("exact_side_flip", flipped)

    source = _load_outcome_blind_feature_source()
    clean_count = source["agg_trade_count"].astype(float).ge(
        prereg.Policy().minimum_bar_agg_trade_count
    )
    fine_count = source["fine_event_count"].astype(float).ge(
        prereg.Policy().minimum_fine_event_count
    )
    total_quantity = source["total_quantity_mbtc"].astype(float)
    medium_quantity = source["medium_quantity_mbtc"].astype(float)
    medium_signed = source["medium_signed_quantity_mbtc"].astype(float)
    fine_quantity = source["fine_quantity_mbtc"].astype(float)
    fine_signed = source["fine_signed_quantity_mbtc"].astype(float)
    medium_share = np.divide(
        medium_quantity,
        total_quantity,
        out=np.zeros(len(source), dtype=float),
        where=total_quantity.to_numpy() != 0.0,
    )
    medium_coherence = np.divide(
        np.abs(medium_signed.to_numpy()),
        medium_quantity.to_numpy(),
        out=np.zeros(len(source), dtype=float),
        where=medium_quantity.to_numpy() != 0.0,
    )
    fine_signed_share = np.divide(
        fine_signed.to_numpy(),
        fine_quantity.to_numpy(),
        out=np.zeros(len(source), dtype=float),
        where=fine_quantity.to_numpy() != 0.0,
    )
    medium_side = pd.Series(
        np.sign(medium_signed.to_numpy()).astype(np.int8),
        index=source.index,
    )
    medium_opposition = np.clip(
        -medium_side.to_numpy(float) * fine_signed_share,
        0.0,
        1.0,
    )
    medium_score = pd.Series(
        medium_share * medium_coherence * medium_opposition,
        index=source.index,
    )
    medium_structural = (
        clean_count
        & source["medium_event_count"].astype(float).ge(
            prereg.Policy().minimum_coarse_event_count
        )
        & fine_count
        & pd.Series(medium_opposition, index=source.index).gt(0.0)
    )
    add(
        "medium_vs_fine",
        _control_clock(
            source,
            score=medium_score,
            side=medium_side,
            structural_eligibility=medium_structural,
        ),
    )

    coarse_side = cast(pd.Series, source["coarse_side"]).astype(int)
    no_opposition_score = (
        source["coarse_quantity_share"].astype(float)
        * source["coarse_coherence"].astype(float)
    )
    no_opposition_structural = (
        clean_count
        & source["coarse_event_count"].astype(float).ge(
            prereg.Policy().minimum_coarse_event_count
        )
        & fine_count
    )
    add(
        "remove_opposition",
        _control_clock(
            source,
            score=no_opposition_score,
            side=coarse_side,
            structural_eligibility=no_opposition_structural,
        ),
    )

    total_signed = source["total_signed_quantity_mbtc"].astype(float)
    total_side = pd.Series(
        np.sign(total_signed.to_numpy()).astype(np.int8),
        index=source.index,
    )
    total_score = pd.Series(
        np.divide(
            np.abs(total_signed.to_numpy()),
            total_quantity.to_numpy(),
            out=np.zeros(len(source), dtype=float),
            where=total_quantity.to_numpy() != 0.0,
        ),
        index=source.index,
    )
    add(
        "all_quantity_imbalance",
        _control_clock(
            source,
            score=total_score,
            side=total_side,
            structural_eligibility=clean_count & total_quantity.gt(0.0),
        ),
    )

    for control, bars in (("stale_one_hour", 12), ("stale_twenty_four_hours", 288)):
        stale = primary.copy()
        shift = BAR * bars
        for column in ("decision_time", "entry_time", "exit_time"):
            stale[column] = stale[column] + shift
        stale = cast(
            pd.DataFrame,
            stale.loc[stale["exit_time"].lt(support_source.SOURCE_END)].copy(),
        )
        add(control, stale)

    combined = pd.concat(parts, ignore_index=True)
    order = {name: index for index, name in enumerate(CONTROL_ORDER)}
    combined["_order"] = cast(pd.Series, combined["control"]).map(
        lambda value: order[str(value)]
    )
    return (
        combined.sort_values(["_order", "entry_time"], kind="mergesort")
        .drop(columns="_order")
        .reset_index(drop=True)
        .loc[:, list(EVALUATION_CLOCK_COLUMNS)]
    )


def _write_evaluation_clocks(frame: pd.DataFrame, path: str | Path) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        deterministic_gzip_csv(frame, temporary)
        rebuilt = temporary.read_bytes()
        try:
            os.link(temporary, output)
        except FileExistsError:
            if output.read_bytes() != rebuilt:
                raise RuntimeError("refusing to overwrite frozen QLCD evaluation clocks")
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256(output)


def load_control_schedules(
    *,
    clock_path: str | Path = EVALUATION_CLOCK,
    expected_clock_sha256: str | None = None,
) -> dict[str, pd.DataFrame]:
    path = Path(clock_path)
    if expected_clock_sha256 is not None and _sha256(path) != expected_clock_sha256:
        raise ValueError("QLCD-288 evaluation-clock bytes changed")
    frame = pd.read_csv(path)
    if tuple(frame.columns) != EVALUATION_CLOCK_COLUMNS:
        raise ValueError("QLCD-288 evaluation-clock schema changed")
    if set(frame["control"].astype(str)) != set(CONTROL_ORDER):
        raise ValueError("QLCD-288 falsification control family changed")
    for column in ("decision_time", "entry_time", "exit_time"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    schedules: dict[str, pd.DataFrame] = {}
    for control in CONTROL_ORDER:
        schedule = cast(
            pd.DataFrame,
            frame.loc[frame["control"].eq(control), list(CLOCK_COLUMNS)].copy(),
        ).sort_values("entry_time", kind="mergesort").reset_index(drop=True)
        if not bool(schedule["side"].isin((-1, 1)).all()):
            raise ValueError(f"QLCD-288 {control} side is invalid")
        if not bool(
            schedule["entry_time"]
            .sub(schedule["decision_time"])
            .eq(pd.Timedelta(minutes=5))
            .all()
        ):
            raise ValueError(f"QLCD-288 {control} decision latency changed")
        if not bool(
            schedule["exit_time"]
            .sub(schedule["entry_time"])
            .eq(pd.Timedelta(hours=24))
            .all()
        ):
            raise ValueError(f"QLCD-288 {control} hold changed")
        entries = schedule["entry_time"].reset_index(drop=True)
        exits = schedule["exit_time"].reset_index(drop=True)
        if len(schedule) > 1 and not bool(
            entries.iloc[1:].ge(exits.iloc[:-1].to_numpy()).all()
        ):
            raise ValueError(f"QLCD-288 {control} schedule overlaps")
        schedules[control] = schedule
    primary = schedules["primary"].reset_index(drop=True)
    frozen_primary = load_schedules().reset_index(drop=True)
    pd.testing.assert_frame_equal(primary, frozen_primary, check_exact=True)
    flip = schedules["exact_side_flip"].reset_index(drop=True)
    if not flip[["decision_time", "entry_time", "exit_time"]].equals(
        primary[["decision_time", "entry_time", "exit_time"]]
    ) or not bool(flip["side"].eq(-primary["side"]).all()):
        raise ValueError("QLCD-288 exact-side-flip semantics changed")
    for control, bars in (("stale_one_hour", 12), ("stale_twenty_four_hours", 288)):
        expected = primary.copy()
        shift = BAR * bars
        for column in ("decision_time", "entry_time", "exit_time"):
            expected[column] = expected[column] + shift
        expected = expected.loc[
            expected["exit_time"].lt(support_source.SOURCE_END)
        ].reset_index(drop=True)
        pd.testing.assert_frame_equal(
            schedules[control].reset_index(drop=True),
            expected,
            check_exact=True,
        )
    return schedules


def _window_schedule(frame: pd.DataFrame, stage: str) -> pd.DataFrame:
    if stage not in PHASE_ONE:
        raise RuntimeError("QLCD-288 test/eval/recent signal source remains phase-two sealed")
    start, end = STAGE_WINDOWS[stage]
    assert end is not None
    selected = cast(
        pd.DataFrame,
        frame.loc[frame["entry_time"].ge(start) & frame["exit_time"].le(end)].copy(),
    )
    return selected.sort_values("entry_time", kind="mergesort").reset_index(drop=True)


def derive_phase_one_schedules() -> dict[str, pd.DataFrame]:
    frame = load_schedules()
    return {stage: _window_schedule(frame, stage) for stage in PHASE_ONE}


def _stage_exit_boundary_required(
    schedules: Mapping[str, pd.DataFrame],
    stage: str,
) -> bool:
    end = STAGE_WINDOWS[stage][1]
    assert end is not None
    return any(
        bool(_window_schedule(schedule, stage)["exit_time"].eq(end).any())
        for schedule in schedules.values()
    )


def _stage_source_spec(
    schedules: Mapping[str, pd.DataFrame],
    stage: str,
) -> dict[str, Any]:
    start, end = STAGE_WINDOWS[stage]
    assert end is not None
    return {
        "stage": stage,
        "required_manifest": str(STAGE_SOURCE_MANIFESTS[stage]),
        "required_protocol_version": (
            "quantity_lattice_cohort_disagreement_execution_source_v1"
        ),
        "physical_window": [start.isoformat(), end.isoformat()],
        "physical_rows_limited_to_window": True,
        "exit_boundary_required": _stage_exit_boundary_required(schedules, stage),
        "strategy_outcomes_calculated": False,
    }


def _cross_boundary_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    boundary = STAGE_WINDOWS["train"][1]
    assert boundary is not None
    crossing = cast(
        pd.DataFrame,
        frame.loc[
            frame["entry_time"].lt(boundary) & frame["exit_time"].gt(boundary)
        ].copy(),
    )
    return [
        {
            "decision_time": _utc(row["decision_time"]).isoformat(),
            "entry_time": _utc(row["entry_time"]).isoformat(),
            "exit_time": _utc(row["exit_time"]).isoformat(),
            "side": int(row["side"]),
            "score_hex": float(row["score"]).hex(),
            "threshold_hex": float(row["threshold"]).hex(),
        }
        for row in crossing.to_dict(orient="records")
    ]


def freeze_evaluator(
    output_path: str | Path = EVALUATOR_FREEZE,
    *,
    evaluation_clock_path: str | Path = EVALUATION_CLOCK,
) -> dict[str, Any]:
    output = Path(output_path)
    if output.exists():
        raise FileExistsError("QLCD-288 evaluator freeze is write-once")
    if any(path.exists() for path in STAGE_OUTPUTS.values()):
        raise RuntimeError("QLCD-288 cannot freeze after an outcome stage exists")
    registration, support = _verify_static_inputs()
    primary_clock = load_schedules()
    evaluation_clock_sha = _write_evaluation_clocks(
        derive_evaluation_clocks(),
        evaluation_clock_path,
    )
    controls = load_control_schedules(
        clock_path=evaluation_clock_path,
        expected_clock_sha256=evaluation_clock_sha,
    )
    stage_records = {
        control: {
            stage: {
                "events": int(len(schedule)),
                "schedule_hash": _schedule_hash(schedule),
                "first_entry": _utc(schedule["entry_time"].min()).isoformat(),
                "last_exit": _utc(schedule["exit_time"].max()).isoformat(),
            }
            for stage in PHASE_ONE
            for schedule in [_window_schedule(control_clock, stage)]
        }
        for control, control_clock in controls.items()
    }
    primary_crossing = _cross_boundary_records(primary_clock)
    core = {
        "protocol_version": "quantity_lattice_cohort_disagreement_evaluator_v1",
        "candidate": POLICY_ID,
        "as_of_date": "2026-07-20",
        "support_commit": SUPPORT_COMMIT,
        "preregistration_manifest_hash": registration["manifest_hash"],
        "preregistration_file_sha256": STATIC_INPUT_SHA256[str(PREREGISTRATION)],
        "support_result_sha256": STATIC_INPUT_SHA256[str(SUPPORT_RESULT)],
        "support_decision": support["decision"],
        "source_clock": str(PRIMARY_CLOCK),
        "source_clock_sha256": STATIC_INPUT_SHA256[str(PRIMARY_CLOCK)],
        "evaluation_clock": str(evaluation_clock_path),
        "evaluation_clock_sha256": evaluation_clock_sha,
        "falsification_controls": list(CONTROL_ORDER[1:]),
        "falsification_controls_are_mandatory_report_only": True,
        "falsification_controls_cannot_repair_primary": True,
        "evaluator_source": str(EVALUATOR_SOURCE),
        "evaluator_source_sha256": _sha256(EVALUATOR_SOURCE),
        "strict_source_dependency": str(strict_source.EVALUATOR_SOURCE),
        "strict_source_dependency_sha256": STATIC_INPUT_SHA256[
            str(strict_source.EVALUATOR_SOURCE)
        ],
        "evaluation_config": asdict(FROZEN_CONFIG),
        "economic_gates": dict(FROZEN_GATES),
        "static_inputs": dict(STATIC_INPUT_SHA256),
        "stage_order": list(STAGE_ORDER),
        "phase_scope": list(PHASE_ONE),
        "stage_windows": {
            stage: [start.isoformat(), end.isoformat() if end is not None else None]
            for stage, (start, end) in STAGE_WINDOWS.items()
        },
        "subperiod_windows": {
            stage: {
                name: [start.isoformat(), end.isoformat()]
                for name, (start, end) in windows.items()
            }
            for stage, windows in SUBPERIOD_WINDOWS.items()
        },
        "stage_schedule_records": stage_records,
        "primary_cross_stage_exclusions": primary_crossing,
        "primary_cross_stage_exclusion_hash": _canonical_hash(primary_crossing),
        "stage_membership_contract": (
            "positions must be fully contained in a declared stage; every crossing "
            "position is excluded without reassignment, truncation, or outcome access"
        ),
        "execution_source_specs": {
            stage: _stage_source_spec(controls, stage) for stage in PHASE_ONE
        },
        "legacy_container_contract": {
            "market": {
                "path": str(LEGACY_MARKET),
                "sha256": LEGACY_MARKET_SHA256,
                "manifest": str(LEGACY_MARKET_MANIFEST),
                "manifest_sha256": LEGACY_MARKET_MANIFEST_SHA256,
            },
            "funding": {
                "path": str(LEGACY_FUNDING),
                "sha256": LEGACY_FUNDING_SHA256,
                "manifest": str(LEGACY_FUNDING_MANIFEST),
                "manifest_sha256": LEGACY_FUNDING_MANIFEST_SHA256,
            },
        },
        "strict_accounting": {
            "entry_exit": "5m open at frozen clock entry and exit; exact 288-bar hold",
            "cost": "notional cost on entry and exit at 6bp/side base or 10bp/side stress",
            "funding_boundary": (
                "interior symmetric; exact entry/exit funding credits dropped and debits retained"
            ),
            "mdd": (
                "global pre-entry HWM; costs; exact funding marks; every held 5m "
                "favorable-then-adverse OHLC path"
            ),
            "cagr": "full declared stage calendar including idle cash",
            "subperiod": "only trades fully contained in each frozen year or half",
            "mean_gross": "unlevered side-signed entry-open to exit-open move in bp",
            "weekly_signflip": (
                "nominal frozen two-sided ISO-entry-week clustered randomization "
                "diagnostic, not a standalone discovery p-value; exact through 20 "
                "clusters, otherwise 20,000 Monte Carlo draws with seed 20260720"
            ),
        },
        "opened_windows": [],
        "sealed_windows": list(STAGE_ORDER),
        "execution_ohlc_rows_parsed_during_freeze": 0,
        "funding_rows_parsed_during_freeze": 0,
        "execution_outcome_identity_hashes_predeclared": True,
        "execution_outcome_data_bytes_hashed_during_freeze": False,
        "execution_outcome_rows_opened_during_freeze": False,
        "simulation_run_during_freeze": False,
        "phase_two_required_after_selection_pass": True,
        "post_2023_access_supported": False,
        "future_clock_contract": (
            "a separately committed phase-two evaluator must rebuild exact QLCD features "
            "from official checksum-verified aggTrades with prior-baseline and global "
            "non-overlap continuity before opening 2024+ outcomes"
        ),
        "future_phase_contract": FUTURE_PHASE_CONTRACT,
        "mutable_parameters": [],
    }
    report = _seal(core)
    _write_once_bytes(output, _json_bytes(report))
    return report


def verify_evaluator_freeze(path: str | Path = EVALUATOR_FREEZE) -> dict[str, Any]:
    payload = _load_json(path)
    _verify_manifest(payload, label="evaluator freeze")
    if payload.get("protocol_version") != (
        "quantity_lattice_cohort_disagreement_evaluator_v1"
    ):
        raise ValueError("QLCD-288 evaluator protocol changed")
    if payload.get("candidate") != POLICY_ID:
        raise ValueError("QLCD-288 evaluator identity changed")
    if payload.get("evaluator_source_sha256") != _sha256(EVALUATOR_SOURCE):
        raise ValueError("QLCD-288 evaluator source changed after freeze")
    if payload.get("evaluation_config") != asdict(FROZEN_CONFIG):
        raise ValueError("QLCD-288 evaluator configuration changed")
    if payload.get("economic_gates") != FROZEN_GATES:
        raise ValueError("QLCD-288 economic gates changed")
    if payload.get("opened_windows") != [] or payload.get("mutable_parameters") != []:
        raise ValueError("QLCD-288 evaluator is not sealed")
    if payload.get("sealed_windows") != list(STAGE_ORDER):
        raise ValueError("QLCD-288 evaluator stage seal changed")
    if payload.get("phase_scope") != list(PHASE_ONE):
        raise ValueError("QLCD-288 phase-one scope changed")
    if payload.get("falsification_controls") != list(CONTROL_ORDER[1:]):
        raise ValueError("QLCD-288 falsification controls changed")
    if payload.get("falsification_controls_are_mandatory_report_only") is not True:
        raise ValueError("QLCD-288 falsification claim boundary changed")
    if payload.get("future_phase_contract") != FUTURE_PHASE_CONTRACT:
        raise ValueError("QLCD-288 future-phase source contract changed")
    if payload.get("post_2023_access_supported") is not False:
        raise ValueError("QLCD-288 evaluator opened phase-two access")
    if payload.get("execution_ohlc_rows_parsed_during_freeze") != 0:
        raise ValueError("QLCD-288 freeze parsed market rows")
    if payload.get("funding_rows_parsed_during_freeze") != 0:
        raise ValueError("QLCD-288 freeze parsed funding rows")
    if payload.get("simulation_run_during_freeze") is not False:
        raise ValueError("QLCD-288 freeze simulated outcomes")
    registration, support = _verify_static_inputs()
    if payload.get("preregistration_manifest_hash") != registration["manifest_hash"]:
        raise ValueError("QLCD-288 freeze binds another preregistration")
    if payload.get("support_decision") != support["decision"]:
        raise ValueError("QLCD-288 freeze binds another support decision")
    controls = load_control_schedules(
        clock_path=cast(str, payload["evaluation_clock"]),
        expected_clock_sha256=cast(str, payload["evaluation_clock_sha256"]),
    )
    for control, control_clock in controls.items():
        for stage in PHASE_ONE:
            schedule = _window_schedule(control_clock, stage)
            record = payload["stage_schedule_records"][control][stage]
            if record.get("events") != len(schedule):
                raise ValueError(f"QLCD-288 {control}/{stage} event count changed")
            if record.get("schedule_hash") != _schedule_hash(schedule):
                raise ValueError(f"QLCD-288 {control}/{stage} schedule changed")
    crossing = _cross_boundary_records(controls["primary"])
    if payload.get("primary_cross_stage_exclusions") != crossing:
        raise ValueError("QLCD-288 cross-stage exclusion identities changed")
    if payload.get("primary_cross_stage_exclusion_hash") != _canonical_hash(crossing):
        raise ValueError("QLCD-288 cross-stage exclusion hash changed")
    return payload


def _verified_prior_reports(
    stage: str,
    *,
    freeze_hash: str,
) -> list[dict[str, Any]]:
    if stage not in STAGE_ORDER:
        raise ValueError(f"QLCD-288 unknown stage: {stage}")
    reports: list[dict[str, Any]] = []
    for prior in STAGE_ORDER[: STAGE_ORDER.index(stage)]:
        payload = _load_json(STAGE_OUTPUTS[prior])
        _verify_manifest(payload, label=f"stored {prior}")
        if payload.get("stage") != prior or payload.get("stage_passed") is not True:
            raise ValueError(f"QLCD-288 {prior} did not pass; {stage} remains sealed")
        index = STAGE_ORDER.index(prior)
        if payload.get("opened_windows") != list(STAGE_ORDER[: index + 1]):
            raise ValueError(f"QLCD-288 {prior} opened an unexpected window")
        if payload.get("sealed_windows") != list(STAGE_ORDER[index + 1 :]):
            raise ValueError(f"QLCD-288 {prior} stage seal changed")
        if payload.get("evaluator_freeze_manifest_hash") != freeze_hash:
            raise ValueError(f"QLCD-288 {prior} froze another evaluator")
        if payload.get("evaluator_source_sha256") != _sha256(EVALUATOR_SOURCE):
            raise ValueError(f"QLCD-288 {prior} evaluator source changed")
        reports.append(payload)
    return reports


def _slice_gzip_csv(
    source: str | Path,
    output: str | Path,
    *,
    timestamp_column: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    expected_rows: int,
    include_end_boundary: bool = False,
) -> dict[str, Any]:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"QLCD-288 stage source is write-once: {output_path}")
    rows = 0
    prior_rows_skipped = 0
    first: pd.Timestamp | None = None
    last: pd.Timestamp | None = None
    if expected_rows < 1:
        raise ValueError("QLCD-288 expected stage-source rows must be positive")
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        with gzip.open(source, "rt", encoding="utf-8", newline="") as input_handle:
            header_line = input_handle.readline()
            header = header_line.rstrip("\r\n").split(",")
            if timestamp_column not in header:
                raise ValueError("QLCD-288 stage source timestamp column is absent")
            timestamp_index = header.index(timestamp_column)
            with temporary.open("wb") as raw:
                with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
                    compressed.write(header_line.encode("utf-8"))
                    for line in input_handle:
                        fields = line.rstrip("\r\n").split(",")
                        timestamp = _utc(fields[timestamp_index])
                        if timestamp < start:
                            prior_rows_skipped += 1
                            continue
                        if timestamp > end or (
                            timestamp == end and not include_end_boundary
                        ):
                            raise ValueError(
                                "QLCD-288 stage source ended before its expected row count"
                            )
                        if last is not None and timestamp <= last:
                            raise ValueError(
                                "QLCD-288 source timestamps are not strictly increasing"
                            )
                        compressed.write(line.encode("utf-8"))
                        first = timestamp if first is None else first
                        last = timestamp
                        rows += 1
                        if rows == expected_rows:
                            break
        if rows != expected_rows or first is None or last is None:
            raise ValueError("QLCD-288 stage source row count changed")
        try:
            os.link(temporary, output_path)
        except FileExistsError as error:
            raise FileExistsError(
                f"QLCD-288 stage source is write-once: {output_path}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(output_path),
        "sha256": _sha256(output_path),
        "rows": rows,
        "prior_rows_skipped_by_timestamp_only": prior_rows_skipped,
        "first_timestamp": first.isoformat(),
        "last_timestamp": last.isoformat(),
        "first_excluded_row_read": False,
        "numeric_rows_parsed_during_slice": 0,
        "post_stage_numeric_rows_parsed": 0,
    }


def _expected_stage_rows(stage: str, *, include_end_boundary: bool) -> tuple[int, int]:
    if stage not in PHASE_ONE:
        raise RuntimeError("QLCD-288 phase-two row contract remains sealed")
    start, end = STAGE_WINDOWS[stage]
    assert end is not None
    market_rows = len(
        pd.date_range(
            start,
            end,
            freq="5min",
            inclusive="both" if include_end_boundary else "left",
        )
    )
    funding_rows = len(pd.date_range(start, end, freq="8h", inclusive="left"))
    if include_end_boundary:
        funding_rows += 1
    return market_rows, funding_rows


def _rebuild_stage_source_identity(
    stage: str,
    *,
    include_end_boundary: bool,
) -> dict[str, dict[str, Any]]:
    start, end = STAGE_WINDOWS[stage]
    assert end is not None
    market_rows, funding_rows = _expected_stage_rows(
        stage,
        include_end_boundary=include_end_boundary,
    )
    with tempfile.TemporaryDirectory(prefix=f"qlcd-{stage}-identity-") as directory:
        root = Path(directory)
        market = _slice_gzip_csv(
            LEGACY_MARKET,
            root / "market.csv.gz",
            timestamp_column="date",
            start=start,
            end=end,
            expected_rows=market_rows,
            include_end_boundary=include_end_boundary,
        )
        funding = _slice_gzip_csv(
            LEGACY_FUNDING,
            root / "funding.csv.gz",
            timestamp_column="funding_time_utc",
            start=start,
            end=end,
            expected_rows=funding_rows,
            include_end_boundary=include_end_boundary,
        )
    return {"market": market, "funding": funding}


def prepare_stage_source(stage: str) -> dict[str, Any]:
    if stage not in PHASE_ONE:
        raise RuntimeError("QLCD-288 test/eval/recent execution sources remain phase-two sealed")
    freeze = verify_evaluator_freeze()
    _verified_prior_reports(stage, freeze_hash=cast(str, freeze["manifest_hash"]))
    manifest_path = STAGE_SOURCE_MANIFESTS[stage]
    directory = STAGE_SOURCE_DIRS[stage]
    market_path = directory / "BTCUSDT_5m.csv.gz"
    funding_path = directory / "BTCUSDT_funding_marks.csv.gz"
    if manifest_path.exists() or market_path.exists() or funding_path.exists():
        raise FileExistsError(f"QLCD-288 {stage} stage source is write-once")
    spec = cast(dict[str, Any], freeze["execution_source_specs"][stage])
    start, end = STAGE_WINDOWS[stage]
    assert end is not None
    include_end = bool(spec["exit_boundary_required"])
    market_rows, funding_rows = _expected_stage_rows(
        stage,
        include_end_boundary=include_end,
    )
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.glob(".prepare-*")):
        raise FileExistsError(f"QLCD-288 {stage} has an interrupted preparation artifact")
    linked: list[Path] = []
    with tempfile.TemporaryDirectory(dir=directory, prefix=".prepare-") as temporary_dir:
        root = Path(temporary_dir)
        pending_market = root / "BTCUSDT_5m.csv.gz"
        pending_funding = root / "BTCUSDT_funding_marks.csv.gz"
        market = _slice_gzip_csv(
            LEGACY_MARKET,
            pending_market,
            timestamp_column="date",
            start=start,
            end=end,
            expected_rows=market_rows,
            include_end_boundary=include_end,
        )
        funding = _slice_gzip_csv(
            LEGACY_FUNDING,
            pending_funding,
            timestamp_column="funding_time_utc",
            start=start,
            end=end,
            expected_rows=funding_rows,
            include_end_boundary=include_end,
        )
        parsed_market, market_diagnostics = strict_source._parse_market_window(
            pending_market,
            start,
            end,
            require_exact_physical_window=True,
            include_end_boundary=include_end,
        )
        parsed_funding, funding_diagnostics = strict_source._parse_funding_window(
            pending_funding,
            start,
            end,
            require_exact_physical_window=True,
            include_end_boundary=include_end,
        )
        if len(parsed_market) != market["rows"] or len(parsed_funding) != funding["rows"]:
            raise ValueError("QLCD-288 stage-source validation count changed")
        market["path"] = str(market_path)
        funding["path"] = str(funding_path)
        core = {
            "protocol_version": spec["required_protocol_version"],
            "candidate": POLICY_ID,
            "stage": stage,
            "evaluator_freeze_manifest_hash": freeze["manifest_hash"],
            "physical_window": spec["physical_window"],
            "physical_rows_limited_to_window": True,
            "exit_boundary_required": include_end,
            "strategy_outcomes_calculated": False,
            "official_manifest_hashes_verified": True,
            "post_stage_numeric_rows_parsed": 0,
            "parent_market": freeze["legacy_container_contract"]["market"],
            "parent_funding": freeze["legacy_container_contract"]["funding"],
            "market": {**market, "diagnostics": market_diagnostics},
            "funding": {**funding, "diagnostics": funding_diagnostics},
        }
        report = _seal(core)
        try:
            os.link(pending_market, market_path)
            linked.append(market_path)
            os.link(pending_funding, funding_path)
            linked.append(funding_path)
            _write_once_bytes(manifest_path, _json_bytes(report))
        except Exception:
            for path in linked:
                path.unlink(missing_ok=True)
            raise
    return report


def _load_stage_source(stage: str, *, freeze: Mapping[str, Any]) -> dict[str, Any]:
    if stage not in PHASE_ONE:
        raise RuntimeError("QLCD-288 phase-two execution source remains sealed")
    spec = cast(dict[str, Any], freeze["execution_source_specs"][stage])
    payload = _load_json(STAGE_SOURCE_MANIFESTS[stage])
    _verify_manifest(payload, label=f"{stage} execution source")
    if payload.get("protocol_version") != spec["required_protocol_version"]:
        raise ValueError(f"QLCD-288 {stage} source protocol changed")
    if payload.get("candidate") != POLICY_ID or payload.get("stage") != stage:
        raise ValueError(f"QLCD-288 {stage} source identity changed")
    for key in (
        "physical_window",
        "physical_rows_limited_to_window",
        "exit_boundary_required",
        "strategy_outcomes_calculated",
    ):
        if payload.get(key) != spec[key]:
            raise ValueError(f"QLCD-288 {stage} source {key} changed")
    if payload.get("evaluator_freeze_manifest_hash") != freeze["manifest_hash"]:
        raise ValueError(f"QLCD-288 {stage} source froze another evaluator")
    if payload.get("official_manifest_hashes_verified") is not True:
        raise ValueError(f"QLCD-288 {stage} source lacks manifest verification")
    if payload.get("post_stage_numeric_rows_parsed") != 0:
        raise ValueError(f"QLCD-288 {stage} parsed a future numeric row")
    if payload.get("parent_market") != freeze["legacy_container_contract"]["market"]:
        raise ValueError(f"QLCD-288 {stage} parent market binding changed")
    if payload.get("parent_funding") != freeze["legacy_container_contract"]["funding"]:
        raise ValueError(f"QLCD-288 {stage} parent funding binding changed")
    expected_paths = {
        "market": STAGE_SOURCE_DIRS[stage] / "BTCUSDT_5m.csv.gz",
        "funding": STAGE_SOURCE_DIRS[stage] / "BTCUSDT_funding_marks.csv.gz",
    }
    for name, expected in expected_paths.items():
        item = payload.get(name)
        if not isinstance(item, dict) or item.get("path") != str(expected):
            raise ValueError(f"QLCD-288 {stage} {name} path changed")
        if _sha256(expected) != item.get("sha256"):
            raise ValueError(f"QLCD-288 {stage} {name} bytes changed")
    rebuilt = _rebuild_stage_source_identity(
        stage,
        include_end_boundary=bool(spec["exit_boundary_required"]),
    )
    for name in ("market", "funding"):
        item = cast(dict[str, Any], payload[name])
        if item.get("sha256") != rebuilt[name]["sha256"]:
            raise ValueError(f"QLCD-288 {stage} {name} is not the frozen parent slice")
        if int(item.get("rows", -1)) != int(rebuilt[name]["rows"]):
            raise ValueError(f"QLCD-288 {stage} {name} row count changed")
    return payload


def load_execution_window(
    stage: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if stage not in PHASE_ONE:
        raise RuntimeError("QLCD-288 test/eval/recent execution outcomes remain phase-two sealed")
    freeze = verify_evaluator_freeze()
    _verified_prior_reports(stage, freeze_hash=cast(str, freeze["manifest_hash"]))
    contract = _load_stage_source(stage, freeze=freeze)
    start, end = STAGE_WINDOWS[stage]
    assert end is not None
    market_item = cast(dict[str, Any], contract["market"])
    funding_item = cast(dict[str, Any], contract["funding"])
    include_end = bool(contract["exit_boundary_required"])
    market, market_diagnostics = strict_source._parse_market_window(
        market_item["path"],
        start,
        end,
        require_exact_physical_window=True,
        include_end_boundary=include_end,
    )
    funding, funding_diagnostics = strict_source._parse_funding_window(
        funding_item["path"],
        start,
        end,
        require_exact_physical_window=True,
        include_end_boundary=include_end,
    )
    if _sha256(market_item["path"]) != market_item["sha256"]:
        raise ValueError(f"QLCD-288 {stage} market bytes changed")
    if _sha256(funding_item["path"]) != funding_item["sha256"]:
        raise ValueError(f"QLCD-288 {stage} funding bytes changed")
    return market, funding, {
        "stage": stage,
        "physical_window": [start.isoformat(), end.isoformat()],
        "market": market_diagnostics,
        "funding": funding_diagnostics,
        "market_sha256": market_item["sha256"],
        "funding_sha256": funding_item["sha256"],
        "execution_source_manifest": {
            "path": str(STAGE_SOURCE_MANIFESTS[stage]),
            "manifest_hash": contract["manifest_hash"],
        },
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
    cfg: EvaluationConfig = FROZEN_CONFIG,
) -> dict[str, Any]:
    if cfg != FROZEN_CONFIG:
        raise ValueError("QLCD-288 evaluation configuration is frozen")
    if not 0.0 <= cost_rate_per_side < 0.1 or end <= start:
        raise ValueError("QLCD-288 simulation window or cost is invalid")
    dates = pd.to_datetime(market["date"], utc=True, errors="raise")
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("QLCD-288 market timestamps are invalid")
    positions = {_utc(value): index for index, value in enumerate(dates)}
    funding_times = pd.to_datetime(funding["funding_time"], utc=True, errors="raise")
    if funding_times.duplicated().any() or not funding_times.is_monotonic_increasing:
        raise ValueError("QLCD-288 funding timestamps are invalid")

    realized_equity = 1.0
    high_water_mark = 1.0
    maximum_drawdown = 0.0
    records: list[dict[str, Any]] = []
    previous_exit: pd.Timestamp | None = None

    def update_path(value: float) -> None:
        nonlocal high_water_mark, maximum_drawdown
        if not np.isfinite(value):
            raise ValueError("QLCD-288 strict equity path is non-finite")
        high_water_mark = max(high_water_mark, value)
        maximum_drawdown = max(
            maximum_drawdown,
            1.0 - value / max(high_water_mark, 1e-15),
        )

    for clock in clocks.to_dict(orient="records"):
        entry_time = _utc(clock["entry_time"])
        exit_time = _utc(clock["exit_time"])
        if entry_time < start or exit_time > end:
            raise ValueError("QLCD-288 clock crosses the simulation window")
        if previous_exit is not None and entry_time < previous_exit:
            raise ValueError("QLCD-288 simulation schedule overlaps")
        previous_exit = exit_time
        entry_position = positions.get(entry_time)
        exit_position = positions.get(exit_time)
        if entry_position is None or exit_position is None:
            raise ValueError("QLCD-288 clock is absent from the market grid")
        if exit_position - entry_position != cfg.hold_bars:
            raise ValueError("QLCD-288 hold is not exactly 288 bars / 24 hours")
        side = int(clock["side"])
        if side not in (-1, 1):
            raise ValueError("QLCD-288 side must be -1 or 1")

        entry_price = float(market.iloc[entry_position]["open"])
        exit_price = float(market.iloc[exit_position]["open"])
        if not np.isfinite([entry_price, exit_price]).all() or min(entry_price, exit_price) <= 0:
            raise ValueError("QLCD-288 entry or exit price is invalid")
        pre_entry_equity = realized_equity
        if pre_entry_equity <= 0.0:
            raise ValueError("QLCD-288 equity became non-positive")
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
            nonlocal cash, funding_cash, next_funding
            nonlocal applied_funding_events, dropped_boundary_credits
            nonlocal visited_funding_events
            while next_funding < len(included_funding):
                event = included_funding.iloc[next_funding]
                event_time = _utc(event["funding_time"])
                if event_time > upper:
                    break
                settlement_mark = float(event["settlement_mark_price"])
                funding_rate = float(event["funding_rate"])
                if not np.isfinite([settlement_mark, funding_rate]).all() or settlement_mark <= 0:
                    raise ValueError("QLCD-288 funding row is invalid")
                visited_funding_events += 1
                cash_flow = -side * quantity * settlement_mark * funding_rate
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
            bar_time = _utc(bar["date"])
            bar_end = _utc(bar_time + BAR - pd.Timedelta(1, unit="ns"))
            apply_funding_through(bar_end)
            high = float(bar["high"])
            low = float(bar["low"])
            if not np.isfinite([high, low]).all() or high < low or low <= 0.0:
                raise ValueError("QLCD-288 held OHLC path is invalid")
            favorable = high if side > 0 else low
            update_path(cash + side * quantity * (favorable - entry_price))
            adverse = low if side > 0 else high
            adverse_equity = cash + side * quantity * (adverse - entry_price)
            update_path(adverse_equity - quantity * adverse * cost_rate_per_side)
        apply_funding_through(exit_time)
        if next_funding != len(included_funding):
            raise ValueError("QLCD-288 funding event was not visited before exit")
        gross_pnl = side * quantity * (exit_price - entry_price)
        exit_fee = quantity * exit_price * cost_rate_per_side
        realized_equity = cash + gross_pnl - exit_fee
        update_path(realized_equity)
        records.append(
            {
                "entry_time": entry_time.isoformat(),
                "exit_time": exit_time.isoformat(),
                "side": side,
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
                "net_return": realized_equity / pre_entry_equity - 1.0,
                "post_exit_equity": realized_equity,
            }
        )

    years = (end - start).total_seconds() / YEAR_SECONDS
    absolute_return = realized_equity - 1.0
    cagr = realized_equity ** (1.0 / years) - 1.0 if realized_equity > 0.0 else -1.0
    trades = pd.DataFrame(records)
    statistical_cfg = strict_source.EvaluationConfig(
        leverage=cfg.leverage,
        base_cost_notional_per_side=cfg.base_cost_notional_per_side,
        stress_cost_notional_per_side=cfg.stress_cost_notional_per_side,
        hold_bars=cfg.hold_bars,
        exact_cluster_max=cfg.exact_cluster_max,
        cluster_draws=cfg.cluster_draws,
        cluster_seed=cfg.cluster_seed,
    )
    significance = strict_source.weekly_cluster_signflip_two_sided(
        trades,
        cfg=statistical_cfg,
    )
    mean_gross = (
        float(cast(pd.Series, trades["gross_underlying_bp"]).mean())
        if len(trades)
        else 0.0
    )
    return {
        "period_start": start.isoformat(),
        "period_end": end.isoformat(),
        "calendar_years": years,
        "absolute_return_pct": absolute_return * 100.0,
        "cagr_pct": cagr * 100.0,
        "strict_mdd_pct": maximum_drawdown * 100.0,
        "cagr_to_strict_mdd": _ratio(cagr, maximum_drawdown),
        "trades": len(records),
        "longs": sum(int(row["side"]) == 1 for row in records),
        "shorts": sum(int(row["side"]) == -1 for row in records),
        "ending_equity": realized_equity,
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
    }


def _simulate_subperiod(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
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
        cost_rate_per_side=FROZEN_CONFIG.base_cost_notional_per_side,
    )


def _stage_gates(
    stage: str,
    base: Mapping[str, Any],
    stress: Mapping[str, Any],
    subperiods: Mapping[str, Mapping[str, Any]],
) -> dict[str, bool]:
    if stage not in PHASE_ONE:
        raise RuntimeError("QLCD-288 phase-two gates remain sealed")
    checks = {
        "base_absolute_return_positive": float(base["absolute_return_pct"]) > 0.0,
        "base_cagr_to_strict_mdd_at_least_3": float(
            base["cagr_to_strict_mdd"]
        )
        >= float(FROZEN_GATES["base_cagr_to_strict_mdd_min"]),
        "strict_mdd_at_most_15pct": float(base["strict_mdd_pct"])
        <= float(FROZEN_GATES["strict_mdd_pct_max"]),
        "stress_absolute_return_positive": float(stress["absolute_return_pct"]) > 0.0,
        "stress_cagr_to_strict_mdd_at_least_2_5": float(
            stress["cagr_to_strict_mdd"]
        )
        >= float(FROZEN_GATES["stress_cagr_to_strict_mdd_min"]),
        "mean_gross_underlying_at_least_24bp": float(
            base["mean_gross_underlying_bp"]
        )
        >= float(FROZEN_GATES["mean_gross_underlying_bp_min"]),
        "weekly_cluster_signflip_p_strictly_below_10pct": float(
            base["weekly_cluster_signflip"]["p_value_two_sided"]
        )
        < float(FROZEN_GATES["weekly_cluster_signflip_p_strict_max"]),
    }
    if stage == "train":
        checks["each_train_year_absolute_return_positive"] = all(
            float(item["absolute_return_pct"]) > 0.0 for item in subperiods.values()
        ) and set(subperiods) == set(SUBPERIOD_WINDOWS["train"])
    elif stage == "selection":
        checks["each_selection_half_absolute_return_positive"] = all(
            float(item["absolute_return_pct"]) > 0.0 for item in subperiods.values()
        ) and set(subperiods) == set(SUBPERIOD_WINDOWS["selection"])
    return checks


def _build_stage_report(stage: str) -> dict[str, Any]:
    if stage not in PHASE_ONE:
        raise RuntimeError("QLCD-288 phase-two evaluator is not frozen")
    freeze = verify_evaluator_freeze()
    _verify_static_inputs()
    prior = _verified_prior_reports(stage, freeze_hash=cast(str, freeze["manifest_hash"]))
    schedules = load_control_schedules(
        clock_path=cast(str, freeze["evaluation_clock"]),
        expected_clock_sha256=cast(str, freeze["evaluation_clock_sha256"]),
    )
    schedule = _window_schedule(schedules["primary"], stage)
    market, funding, diagnostics = load_execution_window(stage)
    start, end = STAGE_WINDOWS[stage]
    assert end is not None
    base = simulate_strict(
        market,
        funding,
        schedule,
        start=start,
        end=end,
        cost_rate_per_side=FROZEN_CONFIG.base_cost_notional_per_side,
    )
    stress = simulate_strict(
        market,
        funding,
        schedule,
        start=start,
        end=end,
        cost_rate_per_side=FROZEN_CONFIG.stress_cost_notional_per_side,
    )
    subperiods = {
        name: _simulate_subperiod(
            market,
            funding,
            schedule,
            start=sub_start,
            end=sub_end,
        )
        for name, (sub_start, sub_end) in SUBPERIOD_WINDOWS[stage].items()
    }
    controls = {
        control: simulate_strict(
            market,
            funding,
            _window_schedule(control_schedule, stage),
            start=start,
            end=end,
            cost_rate_per_side=FROZEN_CONFIG.base_cost_notional_per_side,
        )
        for control, control_schedule in schedules.items()
        if control != "primary"
    }
    gates = _stage_gates(stage, base, stress, subperiods)
    passed = all(gates.values())
    index = STAGE_ORDER.index(stage)
    core = {
        "protocol_version": "quantity_lattice_cohort_disagreement_stage_v1",
        "candidate": POLICY_ID,
        "stage": stage,
        "evaluator_freeze_manifest_hash": freeze["manifest_hash"],
        "evaluator_source_sha256": freeze["evaluator_source_sha256"],
        "verified_prior_stage_manifest_hashes": {
            row["stage"]: row["manifest_hash"] for row in prior
        },
        "config": asdict(FROZEN_CONFIG),
        "execution_diagnostics": diagnostics,
        "primary": {
            "metrics": base,
            "headline": _headline(base),
            "stress_metrics": stress,
            "stress_headline": _headline(stress),
            "contained_subperiod_metrics": subperiods,
            "contained_subperiod_headlines": {
                name: _headline(item) for name, item in subperiods.items()
            },
        },
        "falsification_controls": {
            name: {"metrics": item, "headline": _headline(item)}
            for name, item in controls.items()
        },
        "falsification_controls_are_mandatory_report_only": True,
        "falsification_controls_cannot_repair_primary": True,
        "gates": gates,
        "failed_gates": [name for name, value in gates.items() if not value],
        "stage_passed": passed,
        "no_parameter_selection_or_repair": True,
        "opened_windows": list(STAGE_ORDER[: index + 1]),
        "sealed_windows": list(STAGE_ORDER[index + 1 :]),
        "disposition": (
            "ADVANCE_TO_SELECTION"
            if stage == "train" and passed
            else "QUALIFIED_FOR_PHASE_TWO_FREEZE"
            if stage == "selection" and passed
            else "REJECT_NO_REPAIR"
        ),
    }
    return _seal(core)


def _metric_row(label: str, item: Mapping[str, Any]) -> str:
    return (
        f"| {label} | {float(item['absolute_return_pct']):.2f}% | "
        f"{float(item['cagr_pct']):.2f}% | {float(item['strict_mdd_pct']):.2f}% | "
        f"{float(item['cagr_to_strict_mdd']):.2f} | {int(item['trades'])} | "
        f"{int(item['longs'])}/{int(item['shorts'])} | "
        f"{float(item['mean_gross_underlying_bp']):.2f}bp | "
        f"{float(item['weekly_cluster_signflip_p']):.4f} |"
    )


def render_stage_doc(report: dict[str, Any]) -> str:
    stage = str(report["stage"])
    primary = cast(dict[str, Any], report["primary"])
    headline = cast(dict[str, Any], primary["headline"])
    stress = cast(dict[str, Any], primary["stress_headline"])
    subperiods = cast(dict[str, dict[str, Any]], primary["contained_subperiod_headlines"])
    controls = cast(
        dict[str, dict[str, Any]],
        report["falsification_controls"],
    )
    verdict = (
        report["disposition"] if report["stage_passed"] else "REJECT_NO_REPAIR"
    )
    lines = [
        f"# QLCD-288 {stage} economic result — 2026-07-20",
        "",
        "## Verdict",
        "",
        f"**{verdict}.** The frozen phase-one evaluator was applied without parameter repair.",
        "",
        "## Strict metrics",
        "",
        "| Slice | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | L/S | Mean gross | Weekly nominal p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        _metric_row("base 6bp/side", headline),
        _metric_row("stress 10bp/side", stress),
    ]
    for name, item in subperiods.items():
        lines.append(_metric_row(name, item))
    lines.extend(["", "## Mandatory report-only falsification controls", ""])
    for name in CONTROL_ORDER[1:]:
        lines.append(_metric_row(name, controls[name]["headline"]))
    lines.extend(
        [
            "",
            "These controls were frozen before outcomes and are always reported. The preregistration set no control-margin gate, so they are diagnostic rather than promotion gates; none may repair a failed primary.",
            "The weekly sign-flip value is a frozen nominal clustered randomization diagnostic used only as one preregistered gate; it is not presented as a standalone discovery p-value or multiple-search-adjusted inference.",
        ]
    )
    lines.extend(["", "## Frozen gates", ""])
    for name, passed in cast(dict[str, bool], report["gates"]).items():
        lines.append(f"- `{name}`: **{'pass' if passed else 'fail'}**")
    lines.extend(
        [
            "",
            "## Accounting and boundary",
            "",
            "- Absolute return and CAGR include the full declared calendar, including idle cash.",
            "- Strict MDD keeps the global pre-entry high-water mark and marks every held bar favorable then adverse after costs and funding.",
            "- Entry/exit use the frozen five-minute opens; exposure is 0.5x and hold is exactly 288 bars.",
            "- Exact entry/exit funding credits are dropped while debits are retained.",
            "- Test, eval, and recent-report sources remain sealed unless both phase-one stages pass and a phase-two evaluator is committed first.",
            "- No threshold, direction, delay, hold, cost, split, or gate may repair a failed result.",
            "",
            "## Artifact binding",
            "",
            f"- evaluator freeze manifest: `{report['evaluator_freeze_manifest_hash']}`",
            f"- evaluator source SHA-256: `{report['evaluator_source_sha256']}`",
            f"- stage manifest: `{report['manifest_hash']}`",
            "",
        ]
    )
    return "\n".join(lines)


def evaluate_stage(stage: str) -> dict[str, Any]:
    if stage not in PHASE_ONE:
        raise RuntimeError("QLCD-288 phase-two evaluator is not frozen")
    output = STAGE_OUTPUTS[stage]
    document = STAGE_DOCS[stage]
    if output.exists() or document.exists():
        raise FileExistsError(f"QLCD-288 {stage} result is write-once")
    report = _build_stage_report(stage)
    _write_pair_once(
        output,
        _json_bytes(report),
        document,
        render_stage_doc(report).encode("utf-8"),
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--verify-freeze", action="store_true")
    group.add_argument("--prepare-stage-source", choices=PHASE_ONE)
    group.add_argument("--stage", choices=PHASE_ONE)
    args = parser.parse_args()
    if args.freeze:
        report = freeze_evaluator()
    elif args.verify_freeze:
        report = verify_evaluator_freeze()
    elif args.prepare_stage_source:
        report = prepare_stage_source(args.prepare_stage_source)
    else:
        report = evaluate_stage(args.stage)
    summary = {
        key: report[key]
        for key in (
            "protocol_version",
            "candidate",
            "stage",
            "stage_passed",
            "disposition",
            "manifest_hash",
        )
        if key in report
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
