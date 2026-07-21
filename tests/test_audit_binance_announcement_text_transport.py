from __future__ import annotations

import json
from typing import Any

import pytest

from training import audit_binance_announcement_text_transport as audit


def _list_payload(page: int) -> bytes:
    rows = [
        {
            "id": page * 100 + offset,
            "code": f"{page * 100 + offset:032x}",
            "title": f"Announcement {page}-{offset}",
            "releaseDate": 1_700_000_000_000 - page * 100_000 - offset,
        }
        for offset in range(20)
    ]
    return json.dumps(
        {
            "code": "000000",
            "data": {
                "catalogs": [
                    {
                        "catalogId": audit.CATALOG_ID,
                        "total": 4_000,
                        "articles": rows,
                    }
                ]
            },
        }
    ).encode()


def _detail_payload(code: str, *, updated: bool) -> bytes:
    suffix = " This announcement was updated on 2024-09-24." if updated else ""
    return json.dumps(
        {
            "code": "000000",
            "data": {
                "id": 7,
                "code": code,
                "title": "Frozen title",
                "body": "Current body." + suffix,
                "publishDate": 1_700_000_000_000,
                "version": "1",
                "lastUpdateTime": 0,
            },
        }
    ).encode()


def _telegram_page(before: int) -> bytes:
    first = before - 21
    ids = [value for value in range(first, before) if value != before - 10]
    blocks = []
    for index, message_id in enumerate(ids):
        edited = "edited &nbsp;" if index == 2 else ""
        blocks.append(
            f'''<div class="tgme_widget_message_wrap"><div
            class="tgme_widget_message js-widget_message"
            data-post="binance_announcements/{message_id}">
            <a href="https://www.binance.com/en/support/announcement/detail/{message_id:032x}">link</a>
            <span class="tgme_widget_message_meta">{edited}<a>
            <time datetime="2022-01-01T00:{index:02d}:00+00:00">time</time>
            </a></span></div></div>'''
        )
    return "".join(blocks).encode()


def _fetch(url: str) -> bytes:
    if url == audit.DEVELOPER_DOCS_INDEX:
        return "\n".join(f"- [doc]({path})" for path in audit.DOC_PATHS).encode()
    if url == audit.DEVELOPER_INTRODUCTION_DOC:
        return (
            b"Undocumented behaviors or unofficial interfaces should not be relied upon "
            b"in production systems"
        )
    if url == audit.GENERAL_INFO_DOC:
        return (
            b'# WebSocket API Basic Information wss://api.binance.com/sapi/wss '
            b'X-MBX-APIKEY "command": "SUBSCRIBE" Each connection to the base URL '
            b'is valid for up to 24 hours'
        )
    if url == audit.ANNOUNCEMENT_DOC:
        return (
            b'# Announcements When Binance publishes announcements in English '
            b'`com_announcement_en` "publishDate": "title": "body":'
        )
    if url == audit.CHANGELOG_DOC:
        return b"## 2025-07-21 Added announcement subscription feature"
    if url.startswith(audit.BAPI_LIST):
        page = int(url.split("pageNo=", 1)[1].split("&", 1)[0])
        return _list_payload(page)
    if url.startswith(audit.BAPI_DETAIL):
        code = url.split("articleCode=", 1)[1]
        return _detail_payload(code, updated=code == audit.UPDATED_ARTICLE_CODE)
    if url.startswith(audit.TELEGRAM_ARCHIVE):
        before = int(url.split("before=", 1)[1])
        return _telegram_page(before)
    raise AssertionError(f"unexpected URL: {url}")


def test_source_audit_retires_current_snapshots_before_candidate() -> None:
    report = audit.build_report(_fetch)

    assert report["current_snapshot_probe"]["rows"] == 120
    assert report["telegram_archive_probe"]["rows"] == 140
    assert report["telegram_archive_probe"]["edited_rows"] == 7
    assert report["checks"]["official_realtime_websocket_documented"] is True
    assert report["checks"]["official_realtime_launch_is_changelog_bound"] is True
    assert (
        report["checks"]["official_policy_warns_against_undocumented_interfaces"]
        is True
    )
    assert report["checks"]["official_history_or_revision_replay_documented"] is False
    assert report["checks"]["known_updated_article_is_exposed_as_current_snapshot_only"] is True
    assert report["historical_original_text_vintage_replayable"] is False
    assert report["decision"]["status"] == "retired_before_candidate_preregistration"
    assert report["decision"]["historical_backtest_authorized"] is False
    assert report["decision"]["forward_shadow_collection_authorized"] is True
    assert report["outcome_boundary"]["economic_outcomes_opened"] is False


def test_telegram_parser_marks_edits_and_unreplayed_ids() -> None:
    rows = audit.parse_telegram_page(_telegram_page(1_000))
    assert len(rows) == 20
    assert sum(row["edited"] for row in rows) == 1
    assert audit._missing_ids(rows) == [990]
    assert all(len(row["announcement_codes"]) == 1 for row in rows)


def test_list_page_rejects_noncausal_sort_order() -> None:
    payload = json.loads(_list_payload(1))
    articles: list[dict[str, Any]] = payload["data"]["catalogs"][0]["articles"]
    articles[1]["releaseDate"] = articles[0]["releaseDate"] + 1
    with pytest.raises(ValueError, match="ordering drift"):
        audit.parse_list_page(json.dumps(payload).encode(), expected_page=1)


def test_detail_requires_current_article_identity() -> None:
    raw = _detail_payload("0" * 32, updated=False)
    with pytest.raises(ValueError, match="code mismatch"):
        audit.parse_detail(raw, expected_code="1" * 32)
