from __future__ import annotations

import csv
from decimal import Decimal
import gzip
import json
import urllib.parse
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from training import download_coinmetrics_btc_address_reservoir_daily as source


def _completion(value: str) -> float:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp()


def _row(
    day: str,
    *,
    balance: object = "1000000",
    active: object = "250000",
    available: str | None = None,
) -> dict[str, object]:
    available = available or f"{day}T00:00:00"
    available_dt = datetime.fromisoformat(available) + source.timedelta(days=1)
    return {
        "asset": "btc",
        "time": f"{day}T00:00:00.000000000Z",
        "AdrBalCnt": balance,
        "AdrActCnt": active,
        "AssetEODCompletionTime": available_dt.replace(
            tzinfo=timezone.utc
        ).timestamp(),
    }


def _payload(
    rows: list[dict[str, object]], next_url: str | None = None
) -> dict[str, object]:
    payload: dict[str, object] = {"data": rows}
    if next_url is not None:
        token = urllib.parse.parse_qs(
            urllib.parse.urlparse(next_url).query
        ).get("next_page_token", ["test-token"])[0]
        payload.update(
            next_page_token=token,
            next_page_url=next_url,
        )
    return payload


def _cfg(tmp_path: Path, **changes: object) -> source.Config:
    cfg = source.Config(
        output_csv=str(tmp_path / "source.csv.gz"),
        manifest_output=str(tmp_path / "manifest.json"),
        start="2023-12-28",
        end_exclusive="2024-01-01",
        page_size=2,
        request_pause_sec=0.0,
    )
    return replace(cfg, **changes)


def _next_url(cfg: source.Config, token: str) -> str:
    return source.source_url(cfg) + "&" + urllib.parse.urlencode(
        {"next_page_token": token}
    )


def test_source_url_preserves_end_exclusive_contract(tmp_path: Path) -> None:
    parsed = urllib.parse.urlparse(source.source_url(_cfg(tmp_path)))
    query = urllib.parse.parse_qs(parsed.query)
    assert query == {
        "assets": ["btc"],
        "metrics": ["AdrBalCnt,AdrActCnt,AssetEODCompletionTime"],
        "frequency": ["1d"],
        "start_time": ["2023-12-28"],
        "end_time": ["2023-12-31"],
        "page_size": ["2"],
    }


def test_run_downloads_complete_paginated_source_and_writes_audit(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    pages = {
        None: _payload(
            [_row("2023-12-28"), _row("2023-12-29")],
            _next_url(cfg, "next"),
        ),
        "next": _payload([_row("2023-12-30"), _row("2023-12-31")]),
    }
    calls: list[str] = []

    def fetch(url: str) -> dict[str, object]:
        calls.append(url)
        token = urllib.parse.parse_qs(urllib.parse.urlparse(url).query).get(
            "next_page_token", [None]
        )[0]
        return pages[token]

    manifest = source.run(cfg, fetch=fetch, sleep=lambda _: None)
    assert len(calls) == 2
    audit = manifest["source_audit"]
    assert audit["response_pages"] == 2
    assert audit["response_page_lengths"] == [2, 2]
    assert len(audit["response_page_sha256"]) == 2
    assert audit["expected_rows"] == audit["observed_rows"] == 4
    assert audit["first_observation"] == "2023-12-28T00:00:00Z"
    assert audit["last_observation"] == "2023-12-31T00:00:00Z"
    assert manifest["outcome_boundary"] == {
        "btc_market_rows_loaded": 0,
        "funding_rows_loaded": 0,
        "return_or_pnl_fields": 0,
        "post_2023_source_rows_loaded": 0,
        "raw_api_pages_persisted": False,
    }
    with gzip.open(cfg.output_csv, "rt", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["observation_date"] for row in rows] == [
        "2023-12-28T00:00:00Z",
        "2023-12-29T00:00:00Z",
        "2023-12-30T00:00:00Z",
        "2023-12-31T00:00:00Z",
    ]
    assert rows[0]["AdrBalCnt"] == "1000000"
    assert rows[0]["AdrActCnt"] == "250000"
    assert json.loads(Path(cfg.manifest_output).read_text()) == manifest


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda row: row.update(asset="eth"), "asset must be exactly"),
        (
            lambda row: row.update(time="2023-12-28T01:00:00Z"),
            "UTC midnight",
        ),
        (
            lambda row: row.update(time="2023-12-28T01:00:00+01:00"),
            "exact ISO-8601 UTC",
        ),
        (
            lambda row: row.update(time="2023-12-28T00:00:00.000000001Z"),
            "sub-microseconds",
        ),
        (
            lambda row: row.update(
                AssetEODCompletionTime=_completion("2023-12-28T23:59:59")
            ),
            r"observation \+ 1 day",
        ),
        (lambda row: row.update(AdrBalCnt="0"), "positive integer"),
        (lambda row: row.update(AdrActCnt="1.5"), "positive integer"),
        (lambda row: row.update(AdrBalCnt=True), "positive integer"),
        (
            lambda row: row.update(AssetEODCompletionTime=float("nan")),
            "positive and finite",
        ),
        (
            lambda row: row.update(
                AssetEODCompletionTime=(
                    str(int(_completion("2023-12-28T23:59:59")))
                    + ".999999999"
                )
            ),
            "sub-microsecond",
        ),
    ],
)
def test_row_semantics_fail_closed(
    tmp_path: Path, mutator, message: str
) -> None:
    row = _row("2023-12-28")
    mutator(row)
    with pytest.raises(ValueError, match=message):
        source.download_rows(
            _cfg(
                tmp_path,
                start="2023-12-28",
                end_exclusive="2023-12-29",
                page_size=1,
            ),
            fetch=lambda _: _payload([row]),
            sleep=lambda _: None,
        )


def test_row_schema_rejects_missing_and_unexpected_status_fields(
    tmp_path: Path,
) -> None:
    row = _row("2023-12-28")
    row["status"] = "reviewed"
    with pytest.raises(ValueError, match="schema drift.*unexpected"):
        source.download_rows(
            _cfg(
                tmp_path,
                start="2023-12-28",
                end_exclusive="2023-12-29",
                page_size=1,
            ),
            fetch=lambda _: _payload([row]),
            sleep=lambda _: None,
        )
    row = _row("2023-12-28")
    del row["AdrActCnt"]
    with pytest.raises(ValueError, match="schema drift.*missing"):
        source.download_rows(
            _cfg(
                tmp_path,
                start="2023-12-28",
                end_exclusive="2023-12-29",
                page_size=1,
            ),
            fetch=lambda _: _payload([row]),
            sleep=lambda _: None,
        )


def test_http_json_decode_preserves_epoch_fraction_before_causal_check(
    tmp_path: Path,
) -> None:
    encoded = (
        b'{"data":[{"asset":"btc",'
        b'"time":"2023-12-28T00:00:00.000000000Z",'
        b'"AdrBalCnt":"1000000","AdrActCnt":"250000",'
        b'"AssetEODCompletionTime":1703807999.999999999}],'
        b'"next_page_url":null}'
    )
    payload = source._decode_payload(encoded)
    completion = payload["data"][0]["AssetEODCompletionTime"]
    assert completion == Decimal("1703807999.999999999")
    with pytest.raises(ValueError, match="sub-microsecond"):
        source.download_rows(
            _cfg(
                tmp_path,
                start="2023-12-28",
                end_exclusive="2023-12-29",
                page_size=1,
            ),
            fetch=lambda _: payload,
            sleep=lambda _: None,
        )


def test_duplicate_missing_boundary_and_pagination_loop_are_rejected(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path, page_size=4)
    duplicate = [
        _row("2023-12-28"),
        _row("2023-12-28"),
        _row("2023-12-30"),
        _row("2023-12-31"),
    ]
    with pytest.raises(RuntimeError, match="duplicate day"):
        source.download_rows(
            cfg, fetch=lambda _: _payload(duplicate), sleep=lambda _: None
        )

    missing = [_row("2023-12-28"), _row("2023-12-30"), _row("2023-12-31")]
    with pytest.raises(RuntimeError, match="incomplete.*missing_count=1"):
        source.download_rows(
            cfg, fetch=lambda _: _payload(missing), sleep=lambda _: None
        )

    outside = [_row("2024-01-01")]
    with pytest.raises(ValueError, match="outside the frozen"):
        source.download_rows(
            cfg, fetch=lambda _: _payload(outside), sleep=lambda _: None
        )

    loop_url = _next_url(cfg, "loop")

    def loop_fetch(url: str) -> dict[str, object]:
        day = "2023-12-29" if url == loop_url else "2023-12-28"
        return _payload([_row(day)], loop_url)

    with pytest.raises(RuntimeError, match="pagination loop"):
        source.download_rows(cfg, fetch=loop_fetch, sleep=lambda _: None)


def test_pagination_rejects_scope_drift_empty_pages_and_no_progress(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    first = [_row("2023-12-28")]
    bad_urls = [
        "https://example.com/v4/timeseries/asset-metrics?next_page_token=x",
        source.source_url(cfg).replace("assets=btc", "assets=eth")
        + "&next_page_token=x",
        source.source_url(cfg) + "&metric_status=reviewed&next_page_token=x",
    ]
    for bad_url in bad_urls:
        with pytest.raises(ValueError, match="next_page_url"):
            source.download_rows(
                cfg,
                fetch=lambda _, bad_url=bad_url: _payload(first, bad_url),
                sleep=lambda _: None,
            )

    with pytest.raises(RuntimeError, match="empty non-terminal"):
        source.download_rows(
            cfg,
            fetch=lambda _: _payload([], _next_url(cfg, "next")),
            sleep=lambda _: None,
        )

    long_cfg = _cfg(
        tmp_path / "long",
        start="2023-12-01",
        end_exclusive="2023-12-11",
        page_size=10,
    )
    calls = 0

    def endless_unique(_: str) -> dict[str, object]:
        nonlocal calls
        day = f"2023-12-{calls + 1:02d}"
        calls += 1
        return _payload([_row(day)], _next_url(long_cfg, str(calls)))

    with pytest.raises(RuntimeError, match="maximum page count"):
        source.download_rows(
            long_cfg, fetch=endless_unique, sleep=lambda _: None
        )


def test_payload_and_configuration_contracts_fail_closed(tmp_path: Path) -> None:
    cfg = _cfg(
        tmp_path,
        start="2023-12-28",
        end_exclusive="2023-12-29",
        page_size=1,
    )
    with pytest.raises(RuntimeError, match="API error"):
        source.download_rows(
            cfg,
            fetch=lambda _: {"error": {"message": "bad"}},
            sleep=lambda _: None,
        )
    with pytest.raises(ValueError, match="must be a list"):
        source.download_rows(
            cfg,
            fetch=lambda _: {"data": {}},
            sleep=lambda _: None,
        )
    with pytest.raises(ValueError, match="next_page_url"):
        source.download_rows(
            cfg,
            fetch=lambda _: {"data": [_row("2023-12-28")], "next_page_url": ""},
            sleep=lambda _: None,
        )
    with pytest.raises(ValueError, match="response schema drift"):
        source.download_rows(
            cfg,
            fetch=lambda _: {
                "data": [_row("2023-12-28")],
                "future_return_series": [1.0],
            },
            sleep=lambda _: None,
        )
    with pytest.raises(RuntimeError, match="exceeds frozen"):
        source.download_rows(
            cfg,
            fetch=lambda _: _payload(
                [_row("2023-12-28"), _row("2023-12-28")]
            ),
            sleep=lambda _: None,
        )
    for bad in (
        replace(cfg, start="2023-12-28T00:00:00"),
        replace(cfg, end_exclusive="2023-12-28"),
        replace(cfg, page_size=0),
        replace(cfg, page_size=1.5),
        replace(cfg, page_size=10_001),
        replace(cfg, timeout_sec=0.0),
        replace(cfg, request_pause_sec=-1.0),
        replace(cfg, maximum_retries=-1),
        replace(cfg, maximum_retries=0.5),
    ):
        with pytest.raises(ValueError):
            source.source_url(bad)


def test_output_and_manifest_are_byte_deterministic(tmp_path: Path) -> None:
    rows = [
        _row("2023-12-28"),
        _row("2023-12-29"),
        _row("2023-12-30"),
        _row("2023-12-31"),
    ]
    cfg = _cfg(tmp_path, page_size=4)
    first = source.run(
        cfg, fetch=lambda _: _payload(rows), sleep=lambda _: None
    )
    output_bytes = Path(cfg.output_csv).read_bytes()
    manifest_bytes = Path(cfg.manifest_output).read_bytes()
    second = source.run(
        cfg, fetch=lambda _: _payload(rows), sleep=lambda _: None
    )
    assert Path(cfg.output_csv).read_bytes() == output_bytes
    assert Path(cfg.manifest_output).read_bytes() == manifest_bytes
    assert first["manifest_hash"] == second["manifest_hash"]
