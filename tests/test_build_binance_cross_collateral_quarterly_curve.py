from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import urllib.parse
from pathlib import Path

import pandas as pd
import pytest

from training.build_binance_cross_collateral_quarterly_curve_2021_2023 import (
    Config,
    DEFAULT_CM_SNAPSHOT,
    DEFAULT_UM_SNAPSHOT,
    _canonical_hash,
    _load_staged_rows,
    _write_deterministic_gzip,
    build,
    combine_pairs,
    fetch_pair,
    rows_to_frame,
)


def _row(
    timestamp: str,
    *,
    open_: float = 100.0,
    high: float = 101.0,
    low: float = 99.0,
    close: float = 100.5,
) -> list[object]:
    start = int(pd.Timestamp(timestamp, tz="UTC").timestamp() * 1_000)
    return [
        start,
        str(open_),
        str(high),
        str(low),
        str(close),
        "10",
        start + 299_999,
        "1000",
        20,
        "5",
        "500",
        "0",
    ]


def test_rows_to_frame_preserves_completed_clock_and_marks_bad_ohlc() -> None:
    rows = [
        _row("2023-01-01 00:00"),
        _row("2023-01-01 00:05", open_=102.0, high=101.0, close=100.0),
    ]
    frame = rows_to_frame(rows, "BTCUSDT")
    assert frame["date"].tolist() == [
        pd.Timestamp("2023-01-01 00:00", tz="UTC"),
        pd.Timestamp("2023-01-01 00:05", tz="UTC"),
    ]
    assert frame["ohlc_valid"].tolist() == [True, False]


def test_rows_to_frame_rejects_grid_gap_and_wrong_close_clock() -> None:
    with pytest.raises(ValueError, match="grid has gaps"):
        rows_to_frame(
            [_row("2023-01-01 00:00"), _row("2023-01-01 00:10")],
            "BTCUSD",
        )
    bad = _row("2023-01-01 00:00")
    bad[6] = int(bad[6]) + 1
    with pytest.raises(ValueError, match="close times"):
        rows_to_frame([bad], "BTCUSD")


def test_fetch_pair_paginates_and_never_retains_end_boundary() -> None:
    rows = [
        _row("2023-01-01 00:00"),
        _row("2023-01-01 00:05"),
        _row("2023-01-01 00:10"),
        _row("2023-01-01 00:15"),
    ]

    def fetcher(url: str, **_: object) -> list[list[object]]:
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
        cursor = int(query["startTime"][0])
        return [row for row in rows if int(row[0]) >= cursor][:2]

    cfg = Config(
        start="2023-01-01 00:00",
        end="2023-01-01 00:15",
        request_pause_seconds=0.0,
    )
    output, requests = fetch_pair("BTCUSDT", cfg, fetcher=fetcher)
    assert [row[0] for row in output] == [row[0] for row in rows[:3]]
    assert requests == 2


def test_combine_pairs_uses_only_joint_rows_and_fails_closed() -> None:
    um = rows_to_frame(
        [_row("2023-01-01 00:00"), _row("2023-01-01 00:05")],
        "BTCUSDT",
    )
    cm = rows_to_frame(
        [
            _row("2023-01-01 00:00"),
            _row("2023-01-01 00:05", open_=102.0, high=101.0),
        ],
        "BTCUSD",
    )
    panel = combine_pairs(um, cm)
    assert len(panel) == 2
    assert panel["source_complete"].tolist() == [True, False]
    assert panel["open_time"].tolist() == [
        pd.Timestamp("2023-01-01 00:00", tz="UTC"),
        pd.Timestamp("2023-01-01 00:05", tz="UTC"),
    ]
    assert panel["available_time"].tolist() == [
        pd.Timestamp("2023-01-01 00:05", tz="UTC"),
        pd.Timestamp("2023-01-01 00:10", tz="UTC"),
    ]
    assert panel["contract_segment"].unique().tolist() == ["20230331"]
    assert not panel["is_roll_boundary"].any()


def test_deterministic_gzip_has_stable_bytes(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {"date": [pd.Timestamp("2023-01-01", tz="UTC")], "value": [1.5]}
    )
    first = tmp_path / "first.csv.gz"
    second = tmp_path / "second.csv.gz"
    assert _write_deterministic_gzip(frame, first) == _write_deterministic_gzip(
        frame, second
    )
    assert first.read_bytes() == second.read_bytes()
    assert "value" in gzip.decompress(first.read_bytes()).decode()


def test_build_rejects_any_post_2023_request() -> None:
    with pytest.raises(ValueError, match="physically bounded"):
        build(Config(end="2024-01-02"))


def test_load_staged_rows_accepts_plain_and_deterministic_gzip(tmp_path: Path) -> None:
    rows = [_row("2023-01-01 00:00")]
    raw = json.dumps(rows, separators=(",", ":")).encode()
    plain = tmp_path / "rows.json"
    compressed = tmp_path / "rows.json.gz"
    plain.write_bytes(raw)
    with compressed.open("wb") as target:
        with gzip.GzipFile(filename="", mode="wb", fileobj=target, mtime=0) as out:
            out.write(raw)
    for path in (plain, compressed):
        loaded, record = _load_staged_rows(str(path), "BTCUSDT")
        assert loaded == rows
        assert record["staged_input_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert record["staged_uncompressed_json_sha256"] == hashlib.sha256(raw).hexdigest()


def test_clean_checkout_default_replay_rebuilds_frozen_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for source_name in (DEFAULT_UM_SNAPSHOT, DEFAULT_CM_SNAPSHOT):
        source = Path(source_name).resolve()
        target = tmp_path / source_name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    frozen = json.loads(
        Path(
            "results/binance_cross_collateral_quarterly_curve_2021_2023_manifest.json"
        ).read_text()
    )
    monkeypatch.chdir(tmp_path)
    rebuilt = build(Config())
    assert rebuilt["manifest_hash"] == frozen["manifest_hash"]
    assert rebuilt["file"]["sha256"] == frozen["file"]["sha256"]
    assert rebuilt["pairs"]["um"]["raw_snapshot"]["sha256"] == (
        frozen["pairs"]["um"]["raw_snapshot"]["sha256"]
    )
    assert rebuilt["pairs"]["cm"]["raw_snapshot"]["sha256"] == (
        frozen["pairs"]["cm"]["raw_snapshot"]["sha256"]
    )


def test_frozen_manifest_hash_ignores_only_created_at() -> None:
    path = Path(
        "results/binance_cross_collateral_quarterly_curve_2021_2023_manifest.json"
    )
    if not path.exists():
        pytest.skip("physical source manifest is created by the network build")
    payload = json.loads(path.read_text())
    body = {
        key: value
        for key, value in payload.items()
        if key not in {"manifest_hash", "created_at"}
    }
    assert _canonical_hash(body) == payload["manifest_hash"]
    assert payload["protocol"]["outcomes_opened"] is False
    assert payload["protocol"]["post_2023_rows_requested"] is False
    assert payload["combined"]["incomplete_rows"] == 1
