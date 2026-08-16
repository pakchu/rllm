from __future__ import annotations

import gzip
import json
from datetime import datetime, timedelta, timezone

import pytest

from training import collect_deribit_btc_option_surface_snapshot as collector


def fixture_payloads():
    instruments = {
        "result": [
            {
                "instrument_name": "BTC-25DEC26-50000-C",
                "kind": "option",
                "is_active": True,
                "option_type": "call",
                "strike": 50000,
                "expiration_timestamp": 1798185600000,
                "creation_timestamp": 1766650000000,
            },
            {
                "instrument_name": "BTC-25DEC26-50000-P",
                "kind": "option",
                "is_active": True,
                "option_type": "put",
                "strike": 50000,
                "expiration_timestamp": 1798185600000,
                "creation_timestamp": 1766650000000,
            },
        ]
    }
    summaries = {
        "result": [
            {
                "instrument_name": suffix,
                "creation_timestamp": 1766650000000,
                "mark_iv": iv,
                "mark_price": 0.1,
                "open_interest": 12.0,
                "underlying_price": 60000.0,
            }
            for suffix, iv in (
                ("BTC-25DEC26-50000-P", 70.0),
                ("BTC-25DEC26-50000-C", 65.0),
            )
        ]
    }
    return instruments, summaries


def test_collect_is_sorted_and_causally_available_after_both_responses():
    instruments, summaries = fixture_payloads()
    payloads = iter((instruments, summaries))
    start = datetime(2026, 8, 16, tzinfo=timezone.utc)
    times = iter((start, start + timedelta(seconds=1), start + timedelta(seconds=2)))
    snapshot = collector.collect(fetch=lambda *_: next(payloads), clock=lambda: next(times))
    assert snapshot["feature_available_time"] == "2026-08-16T00:00:02.000000Z"
    assert [row["instrument"]["instrument_name"] for row in snapshot["rows"]] == [
        "BTC-25DEC26-50000-C",
        "BTC-25DEC26-50000-P",
    ]
    assert snapshot["coverage"]["joined_options"] == 2
    core = {key: value for key, value in snapshot.items() if key != "manifest_hash"}
    assert snapshot["manifest_hash"] == collector.canonical_hash(core)


def test_write_snapshot_is_deterministic_and_refuses_overwrite(tmp_path):
    instruments, summaries = fixture_payloads()
    payloads = iter((instruments, summaries))
    start = datetime(2026, 8, 16, tzinfo=timezone.utc)
    times = iter((start, start, start))
    snapshot = collector.collect(fetch=lambda *_: next(payloads), clock=lambda: next(times))
    path = collector.write_snapshot(snapshot, tmp_path)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        assert json.load(handle) == snapshot
    with pytest.raises(FileExistsError):
        collector.write_snapshot(snapshot, tmp_path)


def test_rejects_incomplete_surface_coverage():
    instruments, summaries = fixture_payloads()
    summaries["result"] = summaries["result"][:1]
    with pytest.raises(RuntimeError, match="coverage below 95%"):
        collector.validate_and_join(instruments["result"], summaries["result"])


def test_rejects_deribit_error():
    with pytest.raises(RuntimeError, match="Deribit get_instruments error"):
        collector.result_rows({"error": {"code": 1}}, "get_instruments")
