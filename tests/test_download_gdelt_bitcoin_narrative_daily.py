from __future__ import annotations

import csv
import gzip
import json
import urllib.parse
from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from training import download_gdelt_bitcoin_narrative_daily as gdelt


def _cfg(tmp_path: Path) -> gdelt.Config:
    return gdelt.Config(
        cache_dir=str(tmp_path / "cache"),
        daily_output=str(tmp_path / "daily.csv.gz"),
        raw_bundle_output=str(tmp_path / "raw.jsonl.gz"),
        manifest_output=str(tmp_path / "manifest.json"),
        request_pause_seconds=0.0,
        maximum_retries=0,
    )


def _response(url: str) -> bytes:
    params = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
    query = params["query"][0]
    query_index = dict(
        (value, index) for index, (_, value) in enumerate(gdelt.QUERIES)
    )[query]
    start = datetime.strptime(params["startdatetime"][0], "%Y%m%d%H%M%S").date()
    inclusive_end = datetime.strptime(params["enddatetime"][0], "%Y%m%d%H%M%S").date()
    data = []
    cursor = start
    while cursor <= inclusive_end:
        data.append(
            {
                "date": cursor.strftime("%Y%m%dT000000Z"),
                "value": query_index * 1_000 + (cursor - start).days,
                "norm": 100_000 + (cursor - date(2020, 1, 1)).days,
            }
        )
        cursor += timedelta(days=1)
    return json.dumps(
        {
            "query_details": {"title": query, "date_resolution": "day"},
            "timeline": [{"series": "Article Count", "data": data}],
        }
    ).encode("utf-8")


def test_request_window_is_one_complete_half_open_source_interval() -> None:
    windows = gdelt.request_windows(date(2020, 1, 1), date(2024, 1, 1))
    assert windows == [(date(2020, 1, 1), date(2024, 1, 1))]


def test_request_url_translates_half_open_end_to_last_included_second() -> None:
    url = gdelt.request_url(gdelt.QUERIES[0][1], date(2023, 10, 1), date(2024, 1, 1))
    params = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
    assert params["startdatetime"] == ["20231001000000"]
    assert params["enddatetime"] == ["20231231235959"]


def test_downloader_freezes_daily_counts_and_is_resumable(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    calls: list[str] = []

    def fetch(url: str) -> bytes:
        calls.append(url)
        return _response(url)

    manifest = gdelt.run(cfg, fetch=fetch, sleep=lambda _: None)
    assert len(calls) == len(gdelt.QUERIES) == 4
    assert manifest["source_audit"]["daily_rows"] == 1_461
    assert manifest["requests"]["count"] == 4
    assert manifest["outcome_boundary"]["btc_market_rows_read"] == 0
    assert manifest["builder"]["sha256"] == gdelt.sha256_file(gdelt.BUILDER)
    with gzip.open(cfg.daily_output, "rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1_461
    assert rows[0] == {
        "date": "2020-01-01",
        "available_at": "2020-01-03T00:15:00Z",
        "global_article_count": "100000",
        "broad_article_count": "0",
        "failure_article_count": "1000",
        "constraint_article_count": "2000",
        "adoption_article_count": "3000",
    }
    with gzip.open(cfg.raw_bundle_output, "rt", encoding="utf-8") as handle:
        raw_records = [json.loads(line) for line in handle]
    assert len(raw_records) == 4
    assert all("payload" in row and "rows" not in row for row in raw_records)

    calls.clear()
    repeated = gdelt.run(cfg, fetch=fetch, sleep=lambda _: None)
    assert calls == []
    assert repeated["manifest_hash"] == manifest["manifest_hash"]


def test_response_parser_rejects_missing_dates_and_coarse_resolution() -> None:
    start, end = date(2020, 1, 1), date(2020, 1, 3)
    payload = json.loads(_response(gdelt.request_url(gdelt.QUERIES[0][1], start, end)))
    payload["timeline"][0]["data"].pop()
    with pytest.raises(ValueError, match="incomplete"):
        gdelt.parse_timeline_response(
            json.dumps(payload).encode(),
            start=start,
            end=end,
            expected_query=gdelt.QUERIES[0][1],
        )
    payload = json.loads(_response(gdelt.request_url(gdelt.QUERIES[0][1], start, end)))
    payload["query_details"]["date_resolution"] = "week"
    with pytest.raises(ValueError, match="daily resolution"):
        gdelt.parse_timeline_response(
            json.dumps(payload).encode(),
            start=start,
            end=end,
            expected_query=gdelt.QUERIES[0][1],
        )

    payload = json.loads(_response(gdelt.request_url(gdelt.QUERIES[0][1], start, end)))
    with pytest.raises(ValueError, match="query identity"):
        gdelt.parse_timeline_response(
            json.dumps(payload).encode(),
            start=start,
            end=end,
            expected_query=gdelt.QUERIES[1][1],
        )


def test_resume_cache_rejects_contract_change(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    gdelt.collect_responses(cfg, fetch=_response, sleep=lambda _: None)
    changed = replace(cfg, availability_lag_hours=72)
    with pytest.raises(ValueError, match="contract are frozen"):
        gdelt.collect_responses(changed, fetch=_response, sleep=lambda _: None)


def test_finalized_manifest_self_hash_is_enforced(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    gdelt.run(cfg, fetch=_response, sleep=lambda _: None)
    manifest_path = Path(cfg.manifest_output)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["outcome_boundary"]["economic_metrics_computed"] = True
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="manifest hash changed"):
        gdelt.run(cfg, fetch=_response, sleep=lambda _: None)


def test_default_contract_never_requests_post_2023_news() -> None:
    cfg = gdelt.Config()
    assert cfg.end_date_exclusive == "2024-01-01"
    assert gdelt.source_contract(cfg)["required_date_resolution"] == "day"
    assert all(
        end <= date(2024, 1, 1)
        for _, end in gdelt.request_windows(
            gdelt.parse_date(cfg.start_date), gdelt.parse_date(cfg.end_date_exclusive)
        )
    )
    with pytest.raises(ValueError, match="contract are frozen"):
        gdelt.validate_config(replace(cfg, end_date_exclusive="2024-04-01"))
