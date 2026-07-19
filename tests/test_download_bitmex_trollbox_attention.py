from __future__ import annotations

import gzip
import json
import urllib.error
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from training import download_bitmex_trollbox_attention as trollbox


def _row(identifier: int, date: str, user: str, message: str) -> dict[str, object]:
    return {
        "id": identifier,
        "date": date,
        "user": user,
        "message": message,
        "html": message,
        "channelID": 1,
    }


def _source() -> list[dict[str, object]]:
    return [
        _row(10, "2020-01-01T00:00:01.000Z", "alice", "alice says hi"),
        _row(11, "2020-01-01T00:01:00.000Z", "alice", "bb"),
        _row(12, "2020-01-01T00:00:30.000Z", "bob", "ccc"),
        _row(20, "2020-01-01T00:07:00.000Z", "carol", "dddd"),
        _row(30, "2020-01-01T00:15:00.000Z", "dave", "stop"),
    ]


def _cfg(tmp_path: Path, **changes: object) -> trollbox.Config:
    cfg = trollbox.Config(
        page_dir=str(tmp_path / "pages"),
        aggregate_output=str(tmp_path / "aggregate.csv.gz"),
        state_output=str(tmp_path / "state.json"),
        manifest_output=str(tmp_path / "manifest.json"),
        end_exclusive="2020-01-01 00:15:00+00:00",
        page_size=2,
        request_pause_sec=0.0,
    )
    return replace(cfg, **changes)


def _fetcher(source: list[dict[str, object]], calls: list[int]):
    def fetch(params: dict[str, object]) -> list[dict[str, object]]:
        cursor = int(params["start"])
        calls.append(cursor)
        exact = next(
            (index for index, row in enumerate(source) if int(row["id"]) == cursor),
            None,
        )
        start = exact if exact is not None else cursor
        return source[start : start + int(params["count"])]

    return fetch


def test_downloader_is_resumable_private_and_aggregates_zero_bars(
    tmp_path: Path,
) -> None:
    calls: list[int] = []
    cfg = _cfg(tmp_path)
    manifest = trollbox.run(
        cfg,
        fetch=_fetcher(_source(), calls),
        sleep=lambda _: None,
    )
    assert calls == [0, 11, 12, 20]
    assert manifest["source_audit"]["messages"] == 4
    assert manifest["source_audit"]["pages"] == 3
    assert manifest["privacy"]["sender_username_field_persisted"] is False
    assert manifest["privacy"]["message_text_committed"] is False
    assert manifest["source_audit"][
        "selected_maximum_raw_timestamp_regression_seconds"
    ] == 30.0
    with gzip.open(cfg.aggregate_output, "rt", encoding="utf-8") as handle:
        public_artifacts = json.dumps(manifest) + handle.read()
    assert "alice" not in public_artifacts
    assert "alice says hi" not in public_artifacts

    frame = pd.read_csv(cfg.aggregate_output)
    assert frame["message_count"].tolist() == [3, 1, 0]
    assert frame["unique_participant_count"].tolist() == [2, 1, 0]
    assert frame["maximum_participant_share"].tolist() == [2 / 3, 1.0, 0.0]
    assert frame["character_count"].tolist() == [18, 4, 0]

    private = ""
    private_rows: list[dict[str, object]] = []
    for path in sorted(Path(cfg.page_dir).glob("*.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                private += line
                private_rows.append(json.loads(line))
    assert '"user":"alice"' not in private
    assert "user_hash" in private
    assert '"message":"alice says hi"' in private
    row_12 = next(row for row in private_rows if row["id"] == 12)
    assert row_12["date"] == "2020-01-01T00:00:30.000Z"
    assert row_12["available_date"] == "2020-01-01T00:01:00.000Z"

    manifest_file_hash = trollbox.sha256_file(cfg.manifest_output)
    aggregate_file_hash = trollbox.sha256_file(cfg.aggregate_output)
    calls.clear()
    repeated = trollbox.run(
        cfg,
        fetch=_fetcher(_source(), calls),
        sleep=lambda _: None,
    )
    assert calls == []
    assert repeated["source_audit"]["raw_stream_sha256"] == manifest[
        "source_audit"
    ]["raw_stream_sha256"]
    assert trollbox.sha256_file(cfg.manifest_output) == manifest_file_hash
    assert trollbox.sha256_file(cfg.aggregate_output) == aggregate_file_hash


def test_atomic_resume_starts_after_last_committed_id(tmp_path: Path) -> None:
    source = _source()
    cfg = _cfg(tmp_path)
    calls: list[int] = []

    class Interrupted(RuntimeError):
        pass

    def first_fetch(params: dict[str, object]) -> list[dict[str, object]]:
        cursor = int(params["start"])
        calls.append(cursor)
        if len(calls) == 2:
            raise Interrupted
        return [row for row in source if int(row["id"]) >= cursor][:2]

    with pytest.raises(Interrupted):
        trollbox.download_pages(cfg, fetch=first_fetch, sleep=lambda _: None)
    state = json.loads(Path(cfg.state_output).read_text())
    assert state["last_fetched_id"] == 11
    assert state["messages"] == 2

    calls.clear()
    resumed = trollbox.download_pages(
        cfg,
        fetch=_fetcher(source, calls),
        sleep=lambda _: None,
    )
    assert calls == [11, 12, 20]
    assert resumed["complete"] is True
    assert resumed["messages"] == 4


def test_downloader_rejects_channel_and_id_defects(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path, page_size=10)
    other_channel = _source()
    other_channel[0] = {**other_channel[0], "channelID": 2}
    with pytest.raises(ValueError, match="another channel"):
        trollbox.download_pages(
            cfg,
            fetch=lambda _: other_channel,
            sleep=lambda _: None,
        )

    duplicate = _source()
    duplicate[1] = {**duplicate[1], "id": duplicate[0]["id"]}
    with pytest.raises(RuntimeError, match="IDs are not strictly increasing"):
        trollbox.download_pages(
            _cfg(tmp_path / "duplicate", page_size=10),
            fetch=lambda _: duplicate,
            sleep=lambda _: None,
        )


def test_downloader_rejects_invalid_page_size(tmp_path: Path) -> None:
    for page_size in (1, 501):
        with pytest.raises(ValueError, match="page_size"):
            trollbox.download_pages(
                _cfg(tmp_path / str(page_size), page_size=page_size),
                fetch=lambda _: [],
                sleep=lambda _: None,
            )


def test_downloader_repeats_existing_id_instead_of_assuming_contiguous_ids(
    tmp_path: Path,
) -> None:
    source = [
        _row(10, "2020-01-01T00:00:01.000Z", "alice", "one"),
        _row(20, "2020-01-01T00:07:00.000Z", "bob", "two"),
        _row(30, "2020-01-01T00:15:00.000Z", "carol", "cutoff"),
    ]
    calls: list[int] = []
    cfg = _cfg(tmp_path)
    manifest = trollbox.run(
        cfg,
        fetch=_fetcher(source, calls),
        sleep=lambda _: None,
    )
    assert calls == [0, 20]
    assert manifest["source_audit"]["messages"] == 2
    assert manifest["source_audit"]["last_id"] == 20


def test_short_page_failure_does_not_leave_private_page(tmp_path: Path) -> None:
    cfg = _cfg(
        tmp_path,
        end_exclusive="2020-01-02T00:00:00Z",
        page_size=3,
    )
    with pytest.raises(RuntimeError, match="short page"):
        trollbox.download_pages(
            cfg,
            fetch=lambda _: _source()[:2],
            sleep=lambda _: None,
        )
    assert not Path(cfg.state_output).exists()
    assert list(Path(cfg.page_dir).glob("page_*.jsonl.gz")) == []


def test_aggregate_write_is_atomic_on_stream_failure(tmp_path: Path) -> None:
    output = tmp_path / "aggregate.csv.gz"
    output.write_bytes(b"previous-complete-output")

    def broken_rows():
        yield {
            "date": "2020-01-01 00:00:00",
            "message_count": 0,
            "unique_participant_count": 0,
            "maximum_participant_share": 0.0,
            "character_count": 0,
        }
        raise RuntimeError("stream failed")

    with pytest.raises(RuntimeError, match="stream failed"):
        trollbox._write_aggregate(output, broken_rows())
    assert output.read_bytes() == b"previous-complete-output"
    assert not output.with_suffix(output.suffix + ".tmp").exists()


class _HTTPResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self) -> "_HTTPResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def test_http_page_retries_rate_limit_server_and_transport_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path, request_pause_sec=0.1, maximum_retries=4)
    responses: list[object] = [
        urllib.error.HTTPError(
            trollbox.ENDPOINT,
            429,
            "rate limited",
            {"Retry-After": "0.25"},
            None,
        ),
        urllib.error.HTTPError(
            trollbox.ENDPOINT,
            503,
            "temporarily unavailable",
            {"Retry-After": "not-a-number"},
            None,
        ),
        urllib.error.URLError("temporary network failure"),
        _HTTPResponse(b"[]"),
    ]
    sleeps: list[float] = []

    def urlopen(*_: object, **__: object) -> object:
        response = responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    monkeypatch.setattr(trollbox.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(trollbox.time, "sleep", sleeps.append)

    assert trollbox._http_page(cfg, {"start": 0}) == []
    assert responses == []
    assert sleeps == [0.25, 2.0, 4.0]


def test_http_page_does_not_retry_nonretryable_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path, maximum_retries=4)
    calls = 0

    def urlopen(*_: object, **__: object) -> object:
        nonlocal calls
        calls += 1
        raise urllib.error.HTTPError(
            trollbox.ENDPOINT,
            400,
            "bad request",
            {},
            None,
        )

    monkeypatch.setattr(trollbox.urllib.request, "urlopen", urlopen)
    with pytest.raises(urllib.error.HTTPError):
        trollbox._http_page(cfg, {"start": 0})
    assert calls == 1
