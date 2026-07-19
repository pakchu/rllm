"""Freeze and run text-free source support for TBASR-24.

This stage reads only privacy-preserving five-minute attention aggregates. It
does not open message text, LLM labels, prices, funding, execution bars,
returns, PnL, or any row whose causal availability time is 2023 or later.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


POLICY_ID = "TBASR-24"
ATTENTION_SOURCE = Path(
    "data/bitmex_trollbox_attention_5m_2020_2022.csv.gz"
)
SOURCE_MANIFEST = Path(
    "results/bitmex_trollbox_attention_source_manifest_2026-07-20.json"
)
SOURCE_DECISION = Path(
    "docs/bitmex-trollbox-attention-saturation-mechanism-decision-2026-07-20.md"
)
SOURCE_DECISION_SHA256 = (
    "ae8f7f63e2fa07feeb66fc3d825f20ad00ff715840f7ae199a6a181a78969639"
)
SOURCE_DOWNLOADER = Path("training/download_bitmex_trollbox_attention.py")
SOURCE_DOWNLOADER_SHA256 = (
    "da327d04c065df2e8117a96356fc9b48698a90b6821d1a88e0155218f19457af"
)
PREREGISTRATION_DOCUMENT = Path(
    "docs/bitmex-trollbox-attention-support-preregistration-2026-07-20.md"
)
PREREGISTRATION_SOURCE = Path(
    "training/preregister_bitmex_trollbox_attention_support.py"
)


@dataclass(frozen=True)
class Config:
    support_output: str = (
        "results/bitmex_trollbox_attention_support_2026-07-20.json"
    )
    attention_clock_output: str = (
        "results/bitmex_trollbox_attention_clock_2026-07-20.json"
    )
    lookback_weeks: int = 8
    minimum_prior_slots: int = 8
    message_count_quantile: float = 0.98
    participant_count_quantile: float = 0.95
    minimum_messages: int = 5
    minimum_participants: int = 3
    maximum_participant_share: float = 0.50
    cooldown_bars: int = 12
    latency_bars: int = 1
    hold_bars: int = 24
    eligibility_start: str = "2020-07-01"
    selection_end_exclusive: str = "2023-01-01"
    minimum_total: int = 240
    minimum_train_2020h2_2021: int = 150
    minimum_train_2020h2: int = 40
    minimum_train_2021: int = 80
    minimum_test_2022: int = 90
    minimum_each_test_half: int = 35
    minimum_each_quarter: int = 10
    minimum_active_weeks: int = 100
    minimum_train_active_weeks: int = 65
    minimum_test_active_weeks: int = 35
    maximum_quarter_share: float = 0.18


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_config(cfg: Config) -> None:
    expected = Config(
        support_output=cfg.support_output,
        attention_clock_output=cfg.attention_clock_output,
    )
    if cfg != expected:
        raise ValueError("TBASR-24 attention support configuration is frozen")
    for path, expected_sha in {
        SOURCE_DECISION: SOURCE_DECISION_SHA256,
        SOURCE_DOWNLOADER: SOURCE_DOWNLOADER_SHA256,
    }.items():
        if sha256_file(path) != expected_sha:
            raise ValueError(f"TBASR frozen source anchor mismatch: {path}")


def _manifest_core(manifest: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key != "manifest_hash"}


def _utc_naive(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp
    return timestamp.tz_convert("UTC").tz_localize(None)


def read_attention_aggregate(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    opener = gzip.open if source.suffix == ".gz" else Path.open
    with opener(source, "rt", encoding="utf-8", newline="") as handle:
        fieldnames = next(csv.reader([handle.readline()]))
    expected = [
        "date",
        "message_count",
        "unique_participant_count",
        "maximum_participant_share",
        "character_count",
    ]
    if fieldnames != expected:
        raise ValueError("TBASR attention aggregate columns mismatch")
    selected = expected[:-1]
    frame = pd.read_csv(source, usecols=selected)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise", utc=True).dt.tz_convert(None)
    for column in ("message_count", "unique_participant_count"):
        values = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(values.to_numpy(float)).all():
            raise ValueError(f"non-finite {column}")
        if values.lt(0).any() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"invalid integer count in {column}")
        frame[column] = values.astype(np.int64)
    shares = pd.to_numeric(frame["maximum_participant_share"], errors="raise")
    if not np.isfinite(shares.to_numpy(float)).all() or not shares.between(0.0, 1.0).all():
        raise ValueError("invalid maximum participant share")
    frame["maximum_participant_share"] = shares.astype(float)
    if frame["date"].duplicated().any() or not frame["date"].is_monotonic_increasing:
        raise RuntimeError("TBASR attention aggregate clock is not unique/ordered")
    if frame["unique_participant_count"].gt(frame["message_count"]).any():
        raise RuntimeError("TBASR participants exceed messages")
    zero = frame["message_count"].eq(0)
    if frame.loc[zero, "unique_participant_count"].ne(0).any():
        raise RuntimeError("TBASR zero-message bar has participants")
    if frame.loc[zero, "maximum_participant_share"].ne(0.0).any():
        raise RuntimeError("TBASR zero-message bar has participant share")
    nonzero = ~zero
    if frame.loc[nonzero, "maximum_participant_share"].le(0.0).any():
        raise RuntimeError("TBASR active bar lacks participant share")
    frame.attrs["character_count_loaded"] = False
    frame.attrs["message_text_rows_loaded"] = 0
    frame.attrs["market_rows_loaded"] = 0
    return frame


def load_attention_source() -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    if manifest.get("protocol_version") != "bitmex_trollbox_attention_source_v1":
        raise RuntimeError("TBASR attention source manifest version mismatch")
    if canonical_hash(_manifest_core(manifest)) != manifest.get("manifest_hash"):
        raise RuntimeError("TBASR attention source manifest hash mismatch")
    expected_config = {
        "page_dir": "data/bitmex_trollbox_english_2020_2022_pages",
        "aggregate_output": str(ATTENTION_SOURCE),
        "state_output": (
            "data/bitmex_trollbox_english_2020_2022_download_state.json"
        ),
        "manifest_output": str(SOURCE_MANIFEST),
        "start_cursor": 0,
        "end_exclusive": "2023-01-01",
        "channel_id": 1,
        "page_size": 500,
        "request_pause_sec": 0.25,
        "timeout_sec": 30.0,
        "maximum_retries": 8,
        "participant_salt_label": "TBASR-24-private-participant-v1",
    }
    if manifest.get("config") != expected_config:
        raise RuntimeError("TBASR attention source request contract mismatch")
    source_audit = manifest.get("source_audit", {})
    if not source_audit.get("chronological_ids"):
        raise RuntimeError("TBASR source IDs are not chronological")
    if not source_audit.get("availability_timestamps_monotonic"):
        raise RuntimeError("TBASR source availability clock is not monotonic")
    if source_audit.get("end_exclusive") != "2023-01-01":
        raise RuntimeError("TBASR source cutoff mismatch")
    aggregate = manifest.get("aggregate", {})
    if aggregate.get("path") != str(ATTENTION_SOURCE):
        raise RuntimeError("TBASR aggregate path mismatch")
    if sha256_file(ATTENTION_SOURCE) != aggregate.get("sha256"):
        raise RuntimeError("TBASR aggregate hash mismatch")

    frame = read_attention_aggregate(ATTENTION_SOURCE)
    if len(frame) != int(aggregate.get("rows", -1)):
        raise RuntimeError("TBASR aggregate row count mismatch")
    expected_grid = pd.date_range(
        _utc_naive(str(aggregate["start"])),
        pd.Timestamp("2022-12-31 23:55:00"),
        freq="5min",
    )
    if not frame["date"].equals(pd.Series(expected_grid, name="date")):
        raise RuntimeError("TBASR aggregate is not the complete frozen 5m grid")
    if _utc_naive(str(aggregate["end"])) != expected_grid[-1]:
        raise RuntimeError("TBASR aggregate manifest end mismatch")
    return frame, manifest


def slot_of_week(dates: pd.Series) -> pd.Series:
    return (
        dates.dt.dayofweek * 24 * 12
        + dates.dt.hour * 12
        + dates.dt.minute // 5
    ).astype(np.int16)


def strictly_prior_slot_quantile(
    values: pd.Series,
    slots: pd.Series,
    *,
    lookback: int,
    minimum: int,
    quantile: float,
) -> pd.Series:
    return values.groupby(slots, sort=False).transform(
        lambda series: series.shift(1)
        .rolling(lookback, min_periods=minimum)
        .quantile(quantile)
    )


def _greedy_cooldown(
    dates: pd.Series,
    candidates: pd.Series,
    cooldown_bars: int,
) -> pd.Series:
    selected = pd.Series(False, index=dates.index)
    last: pd.Timestamp | None = None
    separation = pd.Timedelta(minutes=5 * cooldown_bars)
    for index in dates.index[candidates]:
        timestamp = dates.loc[index]
        if last is None or timestamp >= last + separation:
            selected.loc[index] = True
            last = timestamp
    return selected


def build_attention_panel(frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    panel = frame.sort_values("date", ignore_index=True).copy()
    panel["slot_of_week"] = slot_of_week(panel["date"])
    panel["message_threshold"] = strictly_prior_slot_quantile(
        panel["message_count"],
        panel["slot_of_week"],
        lookback=cfg.lookback_weeks,
        minimum=cfg.minimum_prior_slots,
        quantile=cfg.message_count_quantile,
    )
    panel["participant_threshold"] = strictly_prior_slot_quantile(
        panel["unique_participant_count"],
        panel["slot_of_week"],
        lookback=cfg.lookback_weeks,
        minimum=cfg.minimum_prior_slots,
        quantile=cfg.participant_count_quantile,
    )
    panel["thresholds_ready"] = panel[
        ["message_threshold", "participant_threshold"]
    ].notna().all(axis=1)
    panel["eligible"] = panel["date"].ge(pd.Timestamp(cfg.eligibility_start)) & panel[
        "date"
    ].lt(pd.Timestamp(cfg.selection_end_exclusive))
    panel["raw_candidate"] = (
        panel["eligible"]
        & panel["thresholds_ready"]
        & panel["message_count"].ge(cfg.minimum_messages)
        & panel["unique_participant_count"].ge(cfg.minimum_participants)
        & panel["maximum_participant_share"].le(cfg.maximum_participant_share)
        & panel["message_count"].ge(panel["message_threshold"])
        & panel["unique_participant_count"].ge(panel["participant_threshold"])
    )
    panel["candidate"] = _greedy_cooldown(
        panel["date"], panel["raw_candidate"], cfg.cooldown_bars
    )
    panel["observation_end"] = panel["date"] + pd.Timedelta(minutes=5)
    panel["entry_earliest"] = panel["observation_end"] + pd.Timedelta(
        minutes=5 * cfg.latency_bars
    )
    panel["exit_time"] = panel["entry_earliest"] + pd.Timedelta(
        minutes=5 * cfg.hold_bars
    )
    return panel


def support_summary(schedule: pd.DataFrame, cfg: Config) -> dict[str, Any]:
    dates = schedule["date"]
    train = dates.lt(pd.Timestamp("2022-01-01"))
    test = dates.ge(pd.Timestamp("2022-01-01"))
    test_h1 = test & dates.dt.month.le(6)
    test_h2 = test & dates.dt.month.ge(7)
    counts = {
        "total_2020h2_2022": int(len(schedule)),
        "train_2020h2_2021": int(train.sum()),
        "train_2020h2": int((dates.dt.year.eq(2020) & train).sum()),
        "train_2021": int(dates.dt.year.eq(2021).sum()),
        "test_2022": int(test.sum()),
        "test_2022_h1": int(test_h1.sum()),
        "test_2022_h2": int(test_h2.sum()),
    }
    quarters = dates.dt.to_period("Q").astype(str)
    quarter_counts = {
        key: int(value)
        for key, value in quarters.value_counts().sort_index().items()
    }
    expected_quarters = [
        "2020Q3",
        "2020Q4",
        "2021Q1",
        "2021Q2",
        "2021Q3",
        "2021Q4",
        "2022Q1",
        "2022Q2",
        "2022Q3",
        "2022Q4",
    ]
    week = dates.dt.to_period("W-SUN").astype(str)
    active_weeks = {
        "all": int(week.nunique()),
        "train": int(week[train].nunique()),
        "test": int(week[test].nunique()),
    }
    maximum_quarter_share = (
        max(quarter_counts.values()) / len(schedule) if len(schedule) else 1.0
    )
    checks = {
        "total": counts["total_2020h2_2022"] >= cfg.minimum_total,
        "train_total": counts["train_2020h2_2021"]
        >= cfg.minimum_train_2020h2_2021,
        "train_2020h2": counts["train_2020h2"] >= cfg.minimum_train_2020h2,
        "train_2021": counts["train_2021"] >= cfg.minimum_train_2021,
        "test_total": counts["test_2022"] >= cfg.minimum_test_2022,
        "test_h1": counts["test_2022_h1"] >= cfg.minimum_each_test_half,
        "test_h2": counts["test_2022_h2"] >= cfg.minimum_each_test_half,
        "each_quarter": all(
            quarter_counts.get(quarter, 0) >= cfg.minimum_each_quarter
            for quarter in expected_quarters
        ),
        "active_weeks": active_weeks["all"] >= cfg.minimum_active_weeks,
        "train_active_weeks": active_weeks["train"]
        >= cfg.minimum_train_active_weeks,
        "test_active_weeks": active_weeks["test"]
        >= cfg.minimum_test_active_weeks,
        "quarter_concentration": maximum_quarter_share
        <= cfg.maximum_quarter_share,
    }
    return {
        "counts": counts,
        "quarter_counts": quarter_counts,
        "expected_quarters": expected_quarters,
        "active_weeks": active_weeks,
        "maximum_quarter_share": float(maximum_quarter_share),
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def attention_records(schedule: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in schedule[
        ["date", "observation_end", "entry_earliest", "exit_time"]
    ].to_dict(orient="records"):
        records.append(
            {
                "observation_start": str(row["date"]),
                "observation_end": str(row["observation_end"]),
                "entry_earliest": str(row["entry_earliest"]),
                "exit_time": str(row["exit_time"]),
            }
        )
    return records


def attention_clock_hash(
    events: list[dict[str, Any]],
    *,
    cfg: Config,
    protocol_hash: str,
    source_manifest_hash: str,
    source_sha256: str,
) -> str:
    return canonical_hash(
        {
            "policy_id": POLICY_ID,
            "events": events,
            "config": asdict(cfg),
            "protocol_hash": protocol_hash,
            "source_manifest_hash": source_manifest_hash,
            "source_sha256": source_sha256,
        }
    )


def protocol(
    cfg: Config,
    source_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    aggregate = source_manifest.get("aggregate", {}) if source_manifest else {}
    return {
        "policy_id": POLICY_ID,
        "stage": "attention_only_source_support",
        "outcomes_opened": False,
        "message_semantics_opened": False,
        "source": {
            "official_rest": "https://docs.bitmex.com/api-explorer/chat-get",
            "official_websocket": "https://www.bitmex.com/app/wsAPI",
            "channel_id": 1,
            "availability_clock": (
                "cumulative max raw date in increasing chat-ID order"
            ),
            "aggregate": str(ATTENTION_SOURCE),
            "aggregate_sha256": aggregate.get(
                "sha256", "pending_outcome_blind_download"
            ),
            "manifest": str(SOURCE_MANIFEST),
            "manifest_hash": (
                source_manifest.get("manifest_hash")
                if source_manifest
                else "pending_outcome_blind_download"
            ),
            "message_text_loaded": False,
            "character_count_loaded": False,
            "market_price_loaded": False,
            "funding_loaded": False,
            "post_decision_outcome_loaded": False,
        },
        "feature": {
            "reference": (
                f"same 5m slot-of-week over {cfg.lookback_weeks} strictly earlier "
                "weeks; current slot excluded"
            ),
            "message_quantile": cfg.message_count_quantile,
            "participant_quantile": cfg.participant_count_quantile,
            "minimum_prior_slots": cfg.minimum_prior_slots,
            "minimum_messages": cfg.minimum_messages,
            "minimum_participants": cfg.minimum_participants,
            "maximum_participant_share": cfg.maximum_participant_share,
            "quantile_interpolation": "pandas linear",
            "cooldown": "greedy first event, 12 five-minute bars inclusive separation",
            "threshold_grid": False,
        },
        "clock": {
            "observation": "one completed causal-availability 5m bar",
            "decision": "observation end; no semantic label yet",
            "earliest_later_entry": (
                f"{cfg.latency_bars} additional complete five-minute bar later"
            ),
            "frozen_later_hold_bars": cfg.hold_bars,
        },
        "support_gate": {
            key: value
            for key, value in asdict(cfg).items()
            if key.startswith("minimum_") or key == "maximum_quarter_share"
        },
        "failure_action": (
            "reject before message semantics or outcomes; no threshold, lookback, "
            "cooldown, calendar, or support repair"
        ),
        "next_stage_if_passed": (
            "freeze exact small Gemma2 model revision, prompt, decoding, participant "
            "cap, synthetic semantic controls, and directional-support gate before "
            "opening message labels"
        ),
        "frozen_artifacts": {
            "source_decision": str(SOURCE_DECISION),
            "source_decision_sha256": SOURCE_DECISION_SHA256,
            "source_downloader": str(SOURCE_DOWNLOADER),
            "source_downloader_sha256": SOURCE_DOWNLOADER_SHA256,
            "preregistration_document": str(PREREGISTRATION_DOCUMENT),
            "preregistration_document_sha256": sha256_file(
                PREREGISTRATION_DOCUMENT
            ),
            "preregistration_source": str(PREREGISTRATION_SOURCE),
            "preregistration_source_sha256": sha256_file(PREREGISTRATION_SOURCE),
        },
        "research_history_boundary": (
            "candidate-level freeze only; no complete TBASR incidence, semantics, "
            "or post-event BTC outcome informed this singleton"
        ),
    }


def run_support(cfg: Config) -> tuple[dict[str, Any], dict[str, Any] | None]:
    _validate_config(cfg)
    frame, source_manifest = load_attention_source()
    panel = build_attention_panel(frame, cfg)
    schedule = panel.loc[panel["candidate"]].reset_index(drop=True)
    summary = support_summary(schedule, cfg)
    events = attention_records(schedule)
    protocol_payload = protocol(cfg, source_manifest)
    protocol_hash = canonical_hash(protocol_payload)
    source_sha = str(source_manifest["aggregate"]["sha256"])
    source_manifest_hash = str(source_manifest["manifest_hash"])
    clock_hash = attention_clock_hash(
        events,
        cfg=cfg,
        protocol_hash=protocol_hash,
        source_manifest_hash=source_manifest_hash,
        source_sha256=source_sha,
    )
    core = {
        "protocol_version": "bitmex_trollbox_attention_support_v1",
        "protocol": protocol_payload,
        "protocol_hash": protocol_hash,
        "outcomes_opened": False,
        "message_semantics_opened": False,
        "source_loaded": True,
        "source_audit": {
            "aggregate_rows_parsed": int(len(frame)),
            "character_count_loaded": bool(frame.attrs["character_count_loaded"]),
            "message_text_rows_loaded": int(frame.attrs["message_text_rows_loaded"]),
            "market_rows_loaded": int(frame.attrs["market_rows_loaded"]),
            "rows_at_or_after_2023_loaded": 0,
        },
        "window_support": {
            "eligible_bars": int(panel["eligible"].sum()),
            "threshold_ready_eligible_bars": int(
                (panel["eligible"] & panel["thresholds_ready"]).sum()
            ),
            "raw_candidate_bars": int(panel["raw_candidate"].sum()),
            "cooldown_selected_bars": int(len(schedule)),
        },
        "support_gate": summary,
        "attention_clock_hash": clock_hash,
        "attention_clock_written": bool(summary["passed"]),
        "sealed": [
            "all message text and semantic labels",
            "all BTC price and post-event outcomes",
            "2023",
            "2024",
            "2025",
            "2026_ytd",
        ],
        "failure_action": None if summary["passed"] else protocol_payload["failure_action"],
    }
    result = {
        **core,
        "result_hash": canonical_hash(core),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    result_path = Path(cfg.support_output)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    clock: dict[str, Any] | None = None
    if summary["passed"]:
        clock_core = {
            "protocol_version": "bitmex_trollbox_attention_clock_v1",
            "policy_id": POLICY_ID,
            "outcomes_opened": False,
            "message_semantics_opened": False,
            "support_result_hash": result["result_hash"],
            "protocol_hash": protocol_hash,
            "config": asdict(cfg),
            "source_manifest_hash": source_manifest_hash,
            "source_sha256": source_sha,
            "attention_clock_hash": clock_hash,
            "events": events,
        }
        clock = {
            **clock_core,
            "manifest_hash": canonical_hash(clock_core),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        clock_path = Path(cfg.attention_clock_output)
        clock_path.parent.mkdir(parents=True, exist_ok=True)
        clock_path.write_text(json.dumps(clock, indent=2, ensure_ascii=False) + "\n")
    else:
        Path(cfg.attention_clock_output).unlink(missing_ok=True)
    return result, clock


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-output", default=Config.support_output)
    parser.add_argument(
        "--attention-clock-output", default=Config.attention_clock_output
    )
    return Config(**vars(parser.parse_args()))


def main() -> None:
    result, _ = run_support(parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
