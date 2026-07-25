from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from training import bctp_stage_sources as stages


def _canonical_hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _write_gzip(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            gz.write("".join(lines).encode("utf-8"))


def _market_line(ts: pd.Timestamp) -> str:
    stamp = ts.strftime("%Y-%m-%d %H:%M:%S")
    return f"{stamp},100,101,99,100.5,1,100.5,10,0.4,40.2\n"


def _funding_line(ts: pd.Timestamp) -> str:
    ms = int(ts.timestamp() * 1000)
    stamp = ts.strftime("%Y-%m-%d %H:%M:%S")
    return f"{ms},{stamp},BTCUSDT,0.0001,100.5,{ms},{stamp},0,mark_price_kline_open\n"


def _parents(tmp_path: Path, *, market_rows: list[str] | None = None, funding_rows: list[str] | None = None) -> tuple[Path, Path]:
    spec = stages.STAGE_SPECS["2020"]
    market_header = ",".join(stages.MARKET_COLUMNS) + "\n"
    funding_header = ",".join(stages.FUNDING_COLUMNS) + "\n"
    if market_rows is None:
        market_rows = [
            _market_line(ts)
            for ts in pd.date_range(spec.start, periods=spec.market_rows, freq="5min", tz="UTC")
        ]
    if funding_rows is None:
        funding_rows = [
            _funding_line(ts)
            for ts in pd.date_range(spec.start, periods=spec.funding_rows, freq="8h", tz="UTC")
        ]
    market = tmp_path / "parent_market.csv.gz"
    funding = tmp_path / "parent_funding.csv.gz"
    _write_gzip(market, [market_header, *market_rows])
    _write_gzip(funding, [funding_header, *funding_rows])
    return market, funding


def _schedule(tmp_path: Path, stage: str = "2021") -> Path:
    base_rows = [
        {
            "policy_id": policy_id,
            "sequence_id": f"{policy_id}-base",
            "entry_time": f"{stage}-01-01T00:00:00Z",
            "target": "TARGET_FLAT",
        }
        for policy_id in stages.freeze.FAMILY_IDS
    ]
    delayed_rows = [
        {
            "policy_id": policy_id,
            "sequence_id": f"{policy_id}-delay",
            "entry_time": f"{stage}-01-01T00:05:00Z",
            "target": "TARGET_FLAT",
        }
        for policy_id in stages.PROMOTABLE_PRIMARY_IDS
    ]

    def write_schedule(name: str, rows: list[dict]) -> dict:
        frame = pd.DataFrame(rows, columns=stages.SCHEDULE_COLUMNS)
        path = tmp_path / f"{name}_{stage}.csv.gz"
        _write_gzip(
            path,
            [frame.to_csv(index=False, lineterminator="\n")],
        )
        return {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "frame_hash": stages._schedule_frame_hash(frame),
            "rows": len(frame),
            "columns": list(stages.SCHEDULE_COLUMNS),
        }

    core = {
        "protocol_version": stages.SCHEDULE_MANIFEST_PROTOCOL,
        "target_stage": stage,
        "stage": stage,
        "evaluator_manifest_hash": stages.EXPECTED_EVALUATOR_MANIFEST_HASH,
        "family_ids": list(stages.freeze.FAMILY_IDS),
        "promotable_primary_ids": list(stages.PROMOTABLE_PRIMARY_IDS),
        "base_schedules": write_schedule("base", base_rows),
        "delayed_primary_schedules": write_schedule(
            "delayed",
            delayed_rows,
        ),
        "stress_reuses_base_target_sequences": True,
        "strategy_outcomes_calculated": False,
        "outcome_payload_opened": False,
        "market_or_funding_payload_opened": False,
        "market_or_funding_payload_bytes_hashed": False,
    }
    payload = {**core, "manifest_hash": _canonical_hash(core)}
    path = tmp_path / f"schedule_{stage}.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def test_prepare_stage_source_does_not_read_first_excluded_poison_line(tmp_path: Path) -> None:
    spec = stages.STAGE_SPECS["2020"]
    poison_market = "2021-01-01 00:00:00,not-a-number,POISON\n"
    poison_funding = "1700000000000,not-a-date,POISON\n"
    market, funding = _parents(
        tmp_path,
        market_rows=[
            *[_market_line(ts) for ts in pd.date_range(spec.start, periods=spec.market_rows, freq="5min", tz="UTC")],
            poison_market,
        ],
        funding_rows=[
            *[_funding_line(ts) for ts in pd.date_range(spec.start, periods=spec.funding_rows, freq="8h", tz="UTC")],
            poison_funding,
        ],
    )

    manifest = stages.prepare_stage_source(
        "2020",
        market_parent=market,
        funding_parent=funding,
        output_root=tmp_path / "out",
        allow_synthetic_parents=True,
    )

    assert manifest["market"]["rows_copied"] == spec.market_rows
    assert manifest["funding"]["rows_copied"] == spec.funding_rows
    assert manifest["market"]["stopped_after_expected_count_without_first_future_row"] is True
    assert manifest["funding"]["post_stage_numeric_rows_parsed"] == 0
    market_df, funding_df, diagnostics = stages.load_stage_source("2020", output_root=tmp_path / "out")
    assert len(market_df) == spec.market_rows
    assert len(funding_df) == spec.funding_rows
    assert diagnostics["strategy_outcomes_calculated"] is False


def test_schedule_gate_precedes_parent_open_for_post_2020(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="target-schedule"):
        stages.prepare_stage_source(
            "2021",
            market_parent=tmp_path / "missing_market.csv.gz",
            funding_parent=tmp_path / "missing_funding.csv.gz",
            output_root=tmp_path / "out",
        )


def test_exact_grid_rejection_happens_after_physical_copy(tmp_path: Path) -> None:
    spec = stages.STAGE_SPECS["2020"]
    market_rows = [_market_line(ts) for ts in pd.date_range(spec.start, periods=spec.market_rows, freq="5min", tz="UTC")]
    # Keep exact row count but shift the second row off the 5-minute grid.
    market_rows[1] = _market_line(pd.Timestamp("2020-01-01T00:06:00Z"))
    market, funding = _parents(tmp_path, market_rows=market_rows)

    with pytest.raises(ValueError, match="grid"):
        stages.prepare_stage_source(
            "2020",
            market_parent=market,
            funding_parent=funding,
            output_root=tmp_path / "out",
            allow_synthetic_parents=True,
        )
    assert not (
        tmp_path / "out" / "2020" / "bctp_market_2020.csv.gz"
    ).exists()
    assert not (tmp_path / "out" / "2020" / "source_manifest.json").exists()
    good_market, good_funding = _parents(tmp_path / "good")
    recovered = stages.prepare_stage_source(
        "2020",
        market_parent=good_market,
        funding_parent=good_funding,
        output_root=tmp_path / "out",
        allow_synthetic_parents=True,
    )
    assert recovered["stage"] == "2020"


def test_schema_and_value_rejections_are_closed(tmp_path: Path) -> None:
    spec = stages.STAGE_SPECS["2020"]
    bad_market = tmp_path / "bad_market.csv.gz"
    funding = _parents(tmp_path)[1]
    _write_gzip(
        bad_market,
        ["date,open,high,low,close\n", *[_market_line(ts) for ts in pd.date_range(spec.start, periods=spec.market_rows, freq="5min", tz="UTC")]],
    )
    with pytest.raises(ValueError, match="schema"):
        stages.prepare_stage_source(
            "2020",
            market_parent=bad_market,
            funding_parent=funding,
            output_root=tmp_path / "schema",
            allow_synthetic_parents=True,
        )

    market_rows = [_market_line(ts) for ts in pd.date_range(spec.start, periods=spec.market_rows, freq="5min", tz="UTC")]
    market_rows[10] = "2020-01-01 00:50:00,100,99,101,100.5,1,100.5,10,0.4,40.2\n"
    market, funding = _parents(tmp_path / "bad_ohlc", market_rows=market_rows)
    with pytest.raises(ValueError, match="OHLC"):
        stages.prepare_stage_source(
            "2020",
            market_parent=market,
            funding_parent=funding,
            output_root=tmp_path / "ohlc",
            allow_synthetic_parents=True,
        )


def test_deterministic_gzip_and_write_once_existing_manifest(tmp_path: Path) -> None:
    market, funding = _parents(tmp_path)
    first = stages.prepare_stage_source(
        "2020",
        market_parent=market,
        funding_parent=funding,
        output_root=tmp_path / "out1",
        allow_synthetic_parents=True,
    )
    second = stages.prepare_stage_source(
        "2020",
        market_parent=market,
        funding_parent=funding,
        output_root=tmp_path / "out2",
        allow_synthetic_parents=True,
    )
    again = stages.prepare_stage_source(
        "2020",
        market_parent=market,
        funding_parent=funding,
        output_root=tmp_path / "out1",
        allow_synthetic_parents=True,
    )

    assert (
        first["market"]["decoded_lines_sha256"]
        == second["market"]["decoded_lines_sha256"]
        == again["market"]["decoded_lines_sha256"]
    )
    assert (
        first["funding"]["decoded_lines_sha256"]
        == second["funding"]["decoded_lines_sha256"]
    )
    assert first["market"]["gzip_sha256"] == second["market"][
        "gzip_sha256"
    ]
    with gzip.open(
        first["market"]["path"],
        "rb",
    ) as handle:
        assert hashlib.sha256(handle.read()).hexdigest() == first[
            "market"
        ]["decoded_lines_sha256"]
    assert first["manifest_hash"] == again["manifest_hash"]
    assert first["strategy_outcomes_calculated"] is False
    assert first["post_stage_numeric_rows_parsed"] == 0
    assert first["market_or_funding_parent_payload_bytes_hashed"] is False


def test_write_once_drift_and_orphan_guards(tmp_path: Path) -> None:
    market, funding = _parents(tmp_path)
    out = tmp_path / "out"
    stages.prepare_stage_source(
        "2020",
        market_parent=market,
        funding_parent=funding,
        output_root=out,
        allow_synthetic_parents=True,
    )
    with gzip.open(out / "2020" / "bctp_market_2020.csv.gz", "at", encoding="utf-8") as handle:
        handle.write(_market_line(pd.Timestamp("2020-12-31T23:55:00Z")))
    with pytest.raises(ValueError, match="drift|row count|grid"):
        stages.prepare_stage_source(
            "2020",
            market_parent=market,
            funding_parent=funding,
            output_root=out,
            allow_synthetic_parents=True,
        )

    orphan_root = tmp_path / "orphan" / "2020"
    orphan_root.mkdir(parents=True)
    (orphan_root / "bctp_market_2020.csv.gz").write_bytes(b"orphan")
    with pytest.raises(RuntimeError, match="orphaned"):
        stages.prepare_stage_source(
            "2020",
            market_parent=market,
            funding_parent=funding,
            output_root=tmp_path / "orphan",
            allow_synthetic_parents=True,
        )


def test_post_2020_valid_schedule_allows_copy_without_real_payload_hash(tmp_path: Path) -> None:
    spec = stages.STAGE_SPECS["2021"]
    market_header = ",".join(stages.MARKET_COLUMNS) + "\n"
    funding_header = ",".join(stages.FUNDING_COLUMNS) + "\n"
    market = tmp_path / "market_2021.csv.gz"
    funding = tmp_path / "funding_2021.csv.gz"
    prior_market_with_poison_numeric = "2020-12-31 23:55:00,POISON,POISON,POISON,POISON,POISON,POISON,POISON,POISON,POISON\n"
    prior_funding_with_poison_numeric = "POISON,2020-12-31 16:00:00,BTCUSDT,POISON,POISON,POISON,2020-12-31 16:00:00,POISON,mark_price_kline_open\n"
    _write_gzip(
        market,
        [
            market_header,
            prior_market_with_poison_numeric,
            *[_market_line(ts) for ts in pd.date_range(spec.start, periods=spec.market_rows, freq="5min", tz="UTC")],
        ],
    )
    _write_gzip(
        funding,
        [
            funding_header,
            prior_funding_with_poison_numeric,
            *[_funding_line(ts) for ts in pd.date_range(spec.start, periods=spec.funding_rows, freq="8h", tz="UTC")],
        ],
    )

    manifest = stages.prepare_stage_source(
        "2021",
        market_parent=market,
        funding_parent=funding,
        output_root=tmp_path / "out",
        required_schedule_manifest=_schedule(tmp_path, "2021"),
        allow_synthetic_parents=True,
    )

    assert manifest["target_schedule_binding"]["path"].endswith("schedule_2021.json")
    assert manifest["market"]["prior_rows_timestamp_only"] == 1
    assert manifest["funding"]["prior_rows_timestamp_only"] == 1
    assert manifest["source_bindings"]["parent_payload_bytes_hashed"] is False


def test_post_2020_schedule_semantics_are_checked_before_parent_open(
    tmp_path: Path,
) -> None:
    schedule = _schedule(tmp_path, "2021")
    payload = json.loads(schedule.read_text())
    payload["family_ids"] = list(reversed(payload["family_ids"]))
    core = dict(payload)
    core.pop("manifest_hash")
    payload["manifest_hash"] = _canonical_hash(core)
    schedule.write_text(json.dumps(payload, sort_keys=True))
    with pytest.raises(ValueError, match="family"):
        stages.prepare_stage_source(
            "2021",
            market_parent=tmp_path / "missing_market.csv.gz",
            funding_parent=tmp_path / "missing_funding.csv.gz",
            output_root=tmp_path / "out",
            required_schedule_manifest=schedule,
        )


def test_custom_parent_requires_explicit_synthetic_override(
    tmp_path: Path,
) -> None:
    market, funding = _parents(tmp_path)
    with pytest.raises(ValueError, match="frozen sources"):
        stages.prepare_stage_source(
            "2020",
            market_parent=market,
            funding_parent=funding,
            output_root=tmp_path / "out",
        )


def test_schedule_gate_rejects_payload_access_flags(tmp_path: Path) -> None:
    schedule = _schedule(tmp_path, "2021")
    payload = json.loads(schedule.read_text())
    payload["market_or_funding_payload_opened"] = True
    core = dict(payload)
    core.pop("manifest_hash")
    payload["manifest_hash"] = _canonical_hash(core)
    schedule.write_text(json.dumps(payload, sort_keys=True))
    with pytest.raises(ValueError, match="opened payloads"):
        stages.prepare_stage_source(
            "2021",
            market_parent=tmp_path / "missing_market.csv.gz",
            funding_parent=tmp_path / "missing_funding.csv.gz",
            output_root=tmp_path / "out",
            required_schedule_manifest=schedule,
        )
