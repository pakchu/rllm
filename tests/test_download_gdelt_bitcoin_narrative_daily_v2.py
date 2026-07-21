from __future__ import annotations

import csv
import gzip
import json
import urllib.parse
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from training import download_gdelt_bitcoin_narrative_daily_v2 as gdelt


EXTRA_FAILURE_ZERO_DATE = date(2021, 1, 15)


def _cfg(tmp_path: Path) -> gdelt.Config:
    return gdelt.Config(
        cache_dir=str(tmp_path / "cache"),
        daily_output=str(tmp_path / "daily.csv.gz"),
        raw_bundle_output=str(tmp_path / "raw.jsonl.gz"),
        manifest_output=str(tmp_path / "manifest.json"),
        request_pause_seconds=0.0,
        maximum_retries=0,
    )


def _response(url: str, *, norm_offset: int = 0) -> bytes:
    params = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
    query = params["query"][0]
    query_id = dict((value, query_id) for query_id, value in gdelt.QUERIES)[query]
    start = datetime.strptime(params["startdatetime"][0], "%Y%m%d%H%M%S").date()
    inclusive_end = datetime.strptime(params["enddatetime"][0], "%Y%m%d%H%M%S").date()
    values = {"broad": 100, "failure": 10, "constraint": 20, "adoption": 30}
    outage_dates = {
        date.fromisoformat(value) for value in gdelt.KNOWN_GLOBAL_OUTAGE_DATES
    }
    data = []
    cursor = start
    while cursor <= inclusive_end:
        omitted = cursor in outage_dates or (
            query_id == "failure" and cursor == EXTRA_FAILURE_ZERO_DATE
        )
        if not omitted:
            data.append(
                {
                    "date": cursor.strftime("%Y%m%dT000000Z"),
                    "value": values[query_id],
                    "norm": 100_000 + (cursor - date(2020, 1, 1)).days + norm_offset,
                }
            )
        cursor += timedelta(days=1)
    return json.dumps(
        {
            "query_details": {"title": query, "date_resolution": "day"},
            "timeline": [{"series": "Article Count", "data": data}],
        }
    ).encode()


def test_v2_bootstrap_binds_the_v1_dependency() -> None:
    assert gdelt.sha256_file(gdelt.V1_DEPENDENCY) == gdelt.V1_DEPENDENCY_SHA256


def test_v2_fills_sparse_bins_and_freezes_global_outages(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    calls: list[str] = []

    def fetch(url: str) -> bytes:
        calls.append(url)
        return _response(url)

    manifest = gdelt.run(cfg, fetch=fetch, sleep=lambda _: None)
    audit = manifest["source_audit"]
    assert audit["daily_rows"] == 1461
    assert audit["global_outage_dates"] == list(gdelt.KNOWN_GLOBAL_OUTAGE_DATES)
    assert audit["missing_bins_by_query"] == {
        "broad": 2,
        "failure": 3,
        "constraint": 2,
        "adoption": 2,
    }
    assert manifest["requests"]["count"] == 4
    with gzip.open(cfg.daily_output, "rt", encoding="utf-8", newline="") as handle:
        rows = {row["date"]: row for row in csv.DictReader(handle)}
    assert rows["2020-10-20"] == {
        "date": "2020-10-20",
        "available_at": "2020-10-22T00:15:00Z",
        "global_article_count": "0",
        "broad_article_count": "0",
        "failure_article_count": "0",
        "constraint_article_count": "0",
        "adoption_article_count": "0",
    }
    assert rows[EXTRA_FAILURE_ZERO_DATE.isoformat()]["broad_article_count"] == "100"
    assert rows[EXTRA_FAILURE_ZERO_DATE.isoformat()]["failure_article_count"] == "0"
    assert len(calls) == 4
    calls.clear()
    repeated = gdelt.run(cfg, fetch=fetch, sleep=lambda _: None)
    assert calls == []
    assert repeated["manifest_hash"] == manifest["manifest_hash"]


def test_sparse_parser_still_rejects_wrong_resolution() -> None:
    start, end = date(2020, 1, 1), date(2020, 1, 12)
    url = gdelt.request_url(gdelt.QUERIES[0][1], start, end)
    payload = json.loads(_response(url))
    payload["timeline"][0]["data"].pop()
    rows, _ = gdelt.parse_sparse_timeline_response(
        json.dumps(payload).encode(),
        start=start,
        end=end,
        expected_query=gdelt.QUERIES[0][1],
    )
    assert len(rows) == 10
    payload["query_details"]["date_resolution"] = "week"
    with pytest.raises(ValueError, match="daily resolution"):
        gdelt.parse_sparse_timeline_response(
            json.dumps(payload).encode(),
            start=start,
            end=end,
            expected_query=gdelt.QUERIES[0][1],
        )


def test_assembly_rejects_norm_disagreement(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    records = gdelt.collect_responses(cfg, fetch=_response, sleep=lambda _: None)
    first_constraint = next(
        record for record in records if record["query_id"] == "constraint"
    )
    first_date = next(iter(first_constraint["rows"]))
    value, norm = first_constraint["rows"][first_date]
    first_constraint["rows"][first_date] = (value, norm + 1)
    with pytest.raises(ValueError, match="global norm differs"):
        gdelt.assemble_daily_rows(cfg, records)


def test_assembly_rejects_an_unfrozen_global_outage(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    records = gdelt.collect_responses(cfg, fetch=_response, sleep=lambda _: None)
    extra = date(2022, 2, 2)
    for record in records:
        record["rows"].pop(extra)
    with pytest.raises(ValueError, match="outage date set changed"):
        gdelt.assemble_daily_rows(cfg, records)


def test_assembly_rejects_category_count_above_broad(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    records = gdelt.collect_responses(cfg, fetch=_response, sleep=lambda _: None)
    failure = next(record for record in records if record["query_id"] == "failure")
    target = date(2022, 2, 2)
    _, norm = failure["rows"][target]
    failure["rows"][target] = (101, norm)
    with pytest.raises(ValueError, match="category count exceeds broad"):
        gdelt.assemble_daily_rows(cfg, records)


def test_finalized_v2_manifest_self_hash_is_enforced(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    gdelt.run(cfg, fetch=_response, sleep=lambda _: None)
    manifest_path = Path(cfg.manifest_output)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["outcome_boundary"]["economic_metrics_computed"] = True
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="manifest hash changed"):
        gdelt.run(cfg, fetch=_response, sleep=lambda _: None)
