from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import zipfile
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from training import build_binance_coinm_eth_liquidation_snapshot as builder


FEATURE_COLUMNS = (
    "event_count",
    "short_liquidation_event_count",
    "long_liquidation_event_count",
    "short_liquidation_contracts",
    "long_liquidation_contracts",
    "total_liquidation_contracts",
    "signed_liquidation_contracts",
    "liquidation_imbalance",
)
EXPECTED_OUTPUT_COLUMNS = (
    "date",
    "feature_available_time",
    "source_valid",
    *FEATURE_COLUMNS,
)


def _row(
    timestamp: int = 1_687_656_471_926,
    *,
    side: str = "BUY",
    quantity: int | float = 3,
    price: float = 1_900.0,
    average_price: float = 1_899.0,
) -> str:
    return ",".join(
        map(
            str,
            (
                timestamp,
                side,
                "LIMIT",
                "IOC",
                quantity,
                price,
                average_price,
                "FILLED",
                quantity,
                quantity,
            ),
        )
    )


def _archive(rows: list[str], *, header: str | None = None) -> bytes:
    columns = header or ",".join(builder.RAW_COLUMNS)
    payload = (columns + "\n" + "\n".join(rows) + "\n").encode()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("liquidation.csv", payload)
    return buffer.getvalue()


def test_config_and_urls_are_frozen_to_official_eth_coinm_archive() -> None:
    day = date(2023, 6, 25)
    cfg = builder.Config()

    assert cfg.start == "2023-06-25"
    assert cfg.end == "2024-10-15"
    assert builder.archive_url(day) == (
        "https://data.binance.vision/data/futures/cm/daily/liquidationSnapshot/"
        "ETHUSD_PERP/ETHUSD_PERP-liquidationSnapshot-2023-06-25.zip"
    )
    assert builder.checksum_url(day) == builder.archive_url(day) + ".CHECKSUM"


def test_read_archive_removes_only_exact_duplicate_rows() -> None:
    row = _row()
    distinct_same_millisecond = _row(quantity=2)

    frame, removed = builder.read_archive(_archive([row, row, distinct_same_millisecond]))

    assert len(frame) == 2
    assert removed == 1
    assert frame["time"].nunique() == 1
    assert frame["accumulated_fill_quantity"].tolist() == [3.0, 2.0]


def test_read_archive_rejects_bad_columns_sides_and_contract_quantities() -> None:
    with pytest.raises(ValueError, match="unexpected liquidation columns"):
        builder.read_archive(_archive([_row()], header="bad,column"))
    with pytest.raises(ValueError, match="unknown side"):
        builder.read_archive(_archive([_row(side="HOLD")]))
    with pytest.raises(ValueError, match="positive"):
        builder.read_archive(_archive([_row(quantity=0)]))
    with pytest.raises(ValueError, match="contract quantities must be integers"):
        builder.read_archive(_archive([_row(quantity=3.5)]))


def test_aggregate_outputs_price_free_contract_features_with_side_mapping() -> None:
    buy = _row(timestamp=1_687_656_471_926, side="BUY", quantity=3)
    sell = _row(
        timestamp=1_687_656_472_926,
        side="SELL",
        quantity=2,
        price=1_901.0,
        average_price=1_902.0,
    )
    events, _ = builder.read_archive(_archive([buy, sell]))

    output = builder.aggregate_day(events, date(2023, 6, 25))

    assert tuple(output.columns) == EXPECTED_OUTPUT_COLUMNS
    assert not any(
        forbidden in column
        for column in output.columns
        for forbidden in ("price", "notional", "usd")
    )
    row = output.loc[output["event_count"].eq(2)].iloc[0]
    assert row["short_liquidation_event_count"] == 1
    assert row["long_liquidation_event_count"] == 1
    assert row["short_liquidation_contracts"] == 3
    assert row["long_liquidation_contracts"] == 2
    assert row["total_liquidation_contracts"] == 5
    assert row["signed_liquidation_contracts"] == 1
    assert row["liquidation_imbalance"] == pytest.approx(0.2)
    assert row["feature_available_time"] == row["date"] + pd.Timedelta(
        minutes=5, seconds=1
    )
    assert output.loc[output["event_count"].eq(0), "source_valid"].all()


def test_process_day_verifies_checksum_and_fails_missing_archive_closed() -> None:
    payload = _archive([_row()])
    digest = hashlib.sha256(payload).hexdigest()

    def fetcher(url: str, **_: object) -> bytes:
        return (digest + "  archive.zip\n").encode() if url.endswith("CHECKSUM") else payload

    result = builder.process_day(date(2023, 6, 25), builder.Config(), fetcher=fetcher)
    assert result["available"] is True
    assert result["archive_sha256"] == digest
    assert result["snapshot_rows"] == 1

    def missing(_: str, **__: object) -> bytes:
        raise FileNotFoundError

    absent = builder.process_day(date(2023, 6, 25), builder.Config(), fetcher=missing)
    assert absent["available"] is False
    assert not absent["frame"]["source_valid"].any()
    assert absent["frame"].loc[:, FEATURE_COLUMNS].isna().all().all()


def test_process_day_rejects_wrong_checksum_and_wrong_archive_date() -> None:
    wrong_day_payload = _archive([_row(timestamp=1_687_742_871_926)])
    wrong_day_digest = hashlib.sha256(wrong_day_payload).hexdigest()

    def wrong_day(url: str, **_: object) -> bytes:
        if url.endswith("CHECKSUM"):
            return (wrong_day_digest + "  archive.zip\n").encode()
        return wrong_day_payload

    with pytest.raises(ValueError, match="another UTC date"):
        builder.process_day(date(2023, 6, 25), builder.Config(), fetcher=wrong_day)

    payload = _archive([_row()])

    def wrong_checksum(url: str, **_: object) -> bytes:
        return b"0" * 64 + b"  archive.zip\n" if url.endswith("CHECKSUM") else payload

    with pytest.raises(ValueError, match="checksum"):
        builder.process_day(date(2023, 6, 25), builder.Config(), fetcher=wrong_checksum)


def test_build_is_deterministic_and_records_missing_days(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_process_day(day: date, _: builder.Config) -> dict[str, object]:
        if day == date(2023, 6, 26):
            return builder._missing_day(day)
        frame = builder._day_grid(day, source_valid=True)
        frame.loc[0, "event_count"] = 1
        frame.loc[0, "short_liquidation_event_count"] = 1
        frame.loc[0, "short_liquidation_contracts"] = 4
        frame.loc[0, "total_liquidation_contracts"] = 4
        frame.loc[0, "signed_liquidation_contracts"] = 4
        frame.loc[0, "liquidation_imbalance"] = 1.0
        return {
            "date": day.isoformat(),
            "available": True,
            "archive_url": "https://example.invalid/a.zip",
            "checksum_url": "https://example.invalid/a.zip.CHECKSUM",
            "archive_sha256": "a" * 64,
            "expected_archive_sha256": "a" * 64,
            "checksum_payload_sha256": "b" * 64,
            "raw_rows": 2,
            "snapshot_rows": 1,
            "duplicate_rows_removed": 1,
            "event_bars": 1,
            "first_time_ms": 1,
            "last_time_ms": 1,
            "frame": frame,
        }

    monkeypatch.setattr(builder, "process_day", fake_process_day)
    cfg = replace(
        builder.Config(),
        start="2023-06-25",
        end="2023-06-27",
        workers=2,
        output_dir=str(tmp_path / "data"),
        manifest=str(tmp_path / "manifest.json"),
    )

    first = builder.build(cfg)
    first_manifest_bytes = Path(cfg.manifest).read_bytes()
    first_data_bytes = Path(first["file"]["path"]).read_bytes()
    second = builder.build(cfg)

    assert first == second
    assert Path(cfg.manifest).read_bytes() == first_manifest_bytes
    assert Path(second["file"]["path"]).read_bytes() == first_data_bytes
    assert first["missing_archive_dates"] == ["2023-06-26"]
    assert first["file"]["rows"] == 576
    assert first["file"]["source_valid_rows"] == 288
    assert hashlib.sha256(Path(first["file"]["path"]).read_bytes()).hexdigest() == first[
        "file"
    ]["sha256"]
    loaded = json.loads(Path(cfg.manifest).read_text())
    assert loaded["protocol"]["outcomes_opened"] is False
    assert loaded["protocol"]["source_only"] is True


def test_build_enforces_physical_archive_bounds() -> None:
    with pytest.raises(ValueError, match="physically bounded"):
        builder.build(replace(builder.Config(), start="2023-06-24"))
    with pytest.raises(ValueError, match="physically bounded"):
        builder.build(replace(builder.Config(), end="2024-10-16"))


def test_module_is_directly_executable_without_import_failure() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "training/build_binance_coinm_eth_liquidation_snapshot.py",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--output-dir" in result.stdout
