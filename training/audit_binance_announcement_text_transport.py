"""Audit Binance announcement text transport before opening market outcomes.

The audit is deliberately source-only.  It checks whether an historical
announcement corpus can be reconstructed as it was observable at publication
time.  A current-page snapshot is not treated as an historical text vintage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


PROTOCOL_VERSION = "binance_announcement_text_transport_audit_v1"
AS_OF_DATE = "2026-07-21"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AUDITOR_SOURCE = Path("training/audit_binance_announcement_text_transport.py")
DEFAULT_REPORT = Path(
    "results/binance_announcement_text_transport_rejection_2026-07-21.json"
)

DEVELOPER_DOCS_INDEX = "https://developers.binance.com/en/docs/llms.txt"
DEVELOPER_INTRODUCTION_DOC = (
    "https://developers.binance.com/en/docs/introduction.md"
)
GENERAL_INFO_DOC = (
    "https://developers.binance.com/en/docs/products/announcements/general-info.md"
)
ANNOUNCEMENT_DOC = (
    "https://developers.binance.com/en/docs/products/announcements/announcement.md"
)
CHANGELOG_DOC = (
    "https://developers.binance.com/en/docs/products/announcements/cms-log.md"
)
BAPI_LIST = "https://www.binance.com/bapi/composite/v1/public/cms/article/list/query"
BAPI_DETAIL = (
    "https://www.binance.com/bapi/composite/v1/public/cms/article/detail/query"
)
TELEGRAM_ARCHIVE = "https://t.me/s/binance_announcements"
USER_AGENT = "rllm-source-audit/1.0"

CATALOG_ID = 49
LIST_PAGES = (1, 50, 100, 150, 200, 218)
TELEGRAM_BEFORE_IDS = (1_000, 4_000, 6_000, 7_000, 7_474, 8_000, 8_755)
UPDATED_ARTICLE_CODE = "e140f5fa16fd483e9c5d69aca2c84968"
REALTIME_LAUNCH_ARTICLE_CODE = "a72645c63e4a4062b77db52b86fef1bb"
DOC_PATHS = (
    "/en/docs/products/announcements/general-info.md",
    "/en/docs/products/announcements/cms-log.md",
    "/en/docs/products/announcements/announcement.md",
)

Fetch = Callable[[str], bytes]
_MESSAGE_SPLIT_RE = re.compile(
    r'(?=<div class="tgme_widget_message_wrap\b)', re.IGNORECASE
)
_MESSAGE_ID_RE = re.compile(
    r'data-post="binance_announcements/(\d+)"', re.IGNORECASE
)
_DATETIME_RE = re.compile(r'<time[^>]+datetime="([^"]+)"', re.IGNORECASE)
_EDITED_RE = re.compile(
    r'tgme_widget_message_meta">\s*edited(?:\s|&nbsp;)', re.IGNORECASE
)
_ANNOUNCEMENT_CODE_RE = re.compile(
    r'https://www\.binance\.com/[^"\s]*/support/announcement/detail/'
    r'([0-9a-f]{32}|\d{12})',
    re.IGNORECASE,
)


def _path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_url(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
        return response.read()


def _decode_json(raw: bytes, source: str) -> Mapping[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{source} did not return canonical JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{source} root must be an object")
    return value


def parse_list_page(raw: bytes, *, expected_page: int) -> list[dict[str, Any]]:
    payload = _decode_json(raw, "Binance announcement list")
    if payload.get("code") != "000000":
        raise ValueError("Binance announcement list did not succeed")
    data = payload.get("data")
    catalogs = data.get("catalogs") if isinstance(data, dict) else None
    if not isinstance(catalogs, list) or len(catalogs) != 1:
        raise ValueError("Binance announcement list catalog shape drift")
    catalog = catalogs[0]
    if not isinstance(catalog, dict) or catalog.get("catalogId") != CATALOG_ID:
        raise ValueError("Binance announcement list catalog mismatch")
    articles = catalog.get("articles")
    if not isinstance(articles, list) or not articles:
        raise ValueError(f"Binance announcement list page {expected_page} is empty")

    projected: list[dict[str, Any]] = []
    prior_release: int | None = None
    for article in articles:
        if not isinstance(article, dict):
            raise ValueError("Binance announcement list article shape drift")
        row = {
            "id": article.get("id"),
            "code": article.get("code"),
            "title": article.get("title"),
            "releaseDate": article.get("releaseDate"),
        }
        if (
            isinstance(row["id"], bool)
            or not isinstance(row["id"], int)
            or not isinstance(row["code"], str)
            or re.fullmatch(r"(?:[0-9a-f]{32}|\d{12})", row["code"]) is None
            or not isinstance(row["title"], str)
            or not row["title"].strip()
            or isinstance(row["releaseDate"], bool)
            or not isinstance(row["releaseDate"], int)
        ):
            raise ValueError("Binance announcement list article schema drift")
        release = int(row["releaseDate"])
        if prior_release is not None and release > prior_release:
            raise ValueError("Binance announcement list ordering drift")
        prior_release = release
        projected.append(row)
    return projected


def parse_detail(raw: bytes, *, expected_code: str) -> dict[str, Any]:
    payload = _decode_json(raw, "Binance announcement detail")
    if payload.get("code") != "000000":
        raise ValueError("Binance announcement detail did not succeed")
    data = payload.get("data")
    if not isinstance(data, dict) or data.get("code") != expected_code:
        raise ValueError("Binance announcement detail code mismatch")
    title = data.get("title")
    body = data.get("body")
    publish_date = data.get("publishDate")
    if (
        not isinstance(title, str)
        or not title.strip()
        or not isinstance(body, str)
        or not body.strip()
        or isinstance(publish_date, bool)
        or not isinstance(publish_date, int)
    ):
        raise ValueError("Binance announcement detail schema drift")
    version = data.get("version")
    last_update_time = data.get("lastUpdateTime")
    return {
        "code": expected_code,
        "id": data.get("id"),
        "publish_date_ms": publish_date,
        "title_sha256": hashlib.sha256(title.encode()).hexdigest(),
        "body_sha256": hashlib.sha256(body.encode()).hexdigest(),
        "body_bytes": len(body.encode()),
        "version": version,
        "last_update_time": last_update_time,
        "body_declares_update": bool(
            re.search(r"\b(?:updated|update)\s+on\b", body, re.IGNORECASE)
        ),
        "revision_history_field_present": any(
            key in data
            for key in ("revisions", "revisionHistory", "versions", "originalBody")
        ),
    }


def parse_telegram_page(raw: bytes) -> list[dict[str, Any]]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Telegram archive did not return UTF-8") from exc
    messages: list[dict[str, Any]] = []
    for block in _MESSAGE_SPLIT_RE.split(text):
        message_id_match = _MESSAGE_ID_RE.search(block)
        if message_id_match is None:
            continue
        datetime_match = _DATETIME_RE.search(block)
        if datetime_match is None:
            raise ValueError("Telegram archive message lacks a timestamp")
        timestamp = datetime.fromisoformat(datetime_match.group(1))
        if timestamp.tzinfo is None:
            raise ValueError("Telegram archive timestamp lacks a timezone")
        messages.append(
            {
                "message_id": int(message_id_match.group(1)),
                "published_at": timestamp.isoformat(),
                "edited": _EDITED_RE.search(block) is not None,
                "announcement_codes": sorted(
                    set(code.lower() for code in _ANNOUNCEMENT_CODE_RE.findall(block))
                ),
            }
        )
    if not messages:
        raise ValueError("Telegram archive page contains no messages")
    ids = [row["message_id"] for row in messages]
    if ids != sorted(set(ids)):
        raise ValueError("Telegram archive message IDs are not unique and increasing")
    return messages


def _list_url(page: int) -> str:
    return (
        f"{BAPI_LIST}?type=1&catalogId={CATALOG_ID}&pageNo={page}&pageSize=20"
    )


def _detail_url(code: str) -> str:
    return f"{BAPI_DETAIL}?articleCode={code}"


def _telegram_url(before: int) -> str:
    return f"{TELEGRAM_ARCHIVE}?before={before}"


def _missing_ids(rows: Sequence[Mapping[str, Any]]) -> list[int]:
    ids = [int(row["message_id"]) for row in rows]
    return [value for value in range(min(ids), max(ids) + 1) if value not in ids]


def _hash_rows(rows: Iterable[Mapping[str, Any]]) -> str:
    return canonical_hash(list(rows))


def build_report(fetch: Fetch = fetch_url) -> dict[str, Any]:
    docs_raw = {
        "index": fetch(DEVELOPER_DOCS_INDEX),
        "introduction": fetch(DEVELOPER_INTRODUCTION_DOC),
        "general_info": fetch(GENERAL_INFO_DOC),
        "announcement": fetch(ANNOUNCEMENT_DOC),
        "changelog": fetch(CHANGELOG_DOC),
    }
    docs = {key: value.decode("utf-8") for key, value in docs_raw.items()}
    documented_paths = tuple(
        line.split("](", 1)[1].split(")", 1)[0]
        for line in docs["index"].splitlines()
        if "/products/announcements/" in line and "](" in line
    )
    if documented_paths != DOC_PATHS:
        raise RuntimeError("Binance announcement documentation catalog drift")

    list_rows: list[dict[str, Any]] = []
    page_counts: dict[str, int] = {}
    for page in LIST_PAGES:
        rows = parse_list_page(fetch(_list_url(page)), expected_page=page)
        page_counts[str(page)] = len(rows)
        list_rows.extend(rows)
    list_dates = [int(row["releaseDate"]) for row in list_rows]

    details = {
        UPDATED_ARTICLE_CODE: parse_detail(
            fetch(_detail_url(UPDATED_ARTICLE_CODE)),
            expected_code=UPDATED_ARTICLE_CODE,
        ),
        REALTIME_LAUNCH_ARTICLE_CODE: parse_detail(
            fetch(_detail_url(REALTIME_LAUNCH_ARTICLE_CODE)),
            expected_code=REALTIME_LAUNCH_ARTICLE_CODE,
        ),
    }

    telegram_rows: list[dict[str, Any]] = []
    telegram_pages: list[dict[str, Any]] = []
    for before in TELEGRAM_BEFORE_IDS:
        rows = parse_telegram_page(fetch(_telegram_url(before)))
        gaps = _missing_ids(rows)
        telegram_rows.extend(rows)
        telegram_pages.append(
            {
                "before": before,
                "rows": len(rows),
                "first_message_id": rows[0]["message_id"],
                "last_message_id": rows[-1]["message_id"],
                "edited_rows": sum(bool(row["edited"]) for row in rows),
                "missing_ids": gaps,
                "rows_with_announcement_link": sum(
                    bool(row["announcement_codes"]) for row in rows
                ),
                "projected_rows_sha256": _hash_rows(rows),
            }
        )

    general_info = docs["general_info"]
    announcement_doc = docs["announcement"]
    docs_combined = "\n".join(docs.values()).lower()
    updated_detail = details[UPDATED_ARTICLE_CODE]
    edited_rows = sum(bool(row["edited"]) for row in telegram_rows)
    missing_ids = sorted(
        {value for page in telegram_pages for value in page["missing_ids"]}
    )

    checks = {
        "official_realtime_websocket_documented": all(
            token in general_info
            for token in (
                "# WebSocket API Basic Information",
                "wss://api.binance.com/sapi/wss",
                "X-MBX-APIKEY",
                '"command": "SUBSCRIBE"',
                "Each connection to the base URL is valid for up to 24 hours",
            )
        ),
        "official_realtime_payload_contains_publish_time_title_and_body": all(
            token in announcement_doc
            for token in (
                "# Announcements",
                "When Binance publishes announcements in English",
                "`com_announcement_en`",
                '"publishDate":',
                '"title":',
                '"body":',
            )
        ),
        "official_realtime_launch_is_changelog_bound": all(
            token in docs["changelog"]
            for token in ("## 2025-07-21", "Added announcement subscription feature")
        ),
        "official_policy_warns_against_undocumented_interfaces": (
            "Undocumented behaviors or unofficial interfaces should not be relied upon "
            "in production systems" in docs["introduction"]
        ),
        "official_history_or_revision_replay_documented": any(
            token in docs_combined
            for token in ("historical replay", "revision history", "original version")
        ),
        "undocumented_bapi_historical_enumeration_works": len(list_rows) >= 100,
        "bapi_detail_exposes_revision_history": any(
            bool(detail["revision_history_field_present"])
            for detail in details.values()
        ),
        "known_updated_article_is_exposed_as_current_snapshot_only": bool(
            updated_detail["body_declares_update"]
            and not updated_detail["revision_history_field_present"]
            and updated_detail["last_update_time"] in (None, 0)
        ),
        "telegram_archive_is_historically_enumerable": len(telegram_rows) >= 100,
        "telegram_archive_contains_edited_messages": edited_rows > 0,
        "telegram_archive_contains_unreplayed_message_id_gaps": bool(missing_ids),
    }
    historical_vintage_replayable = bool(
        checks["official_history_or_revision_replay_documented"]
        and checks["bapi_detail_exposes_revision_history"]
        and not checks["telegram_archive_contains_edited_messages"]
        and not checks["telegram_archive_contains_unreplayed_message_id_gaps"]
    )
    source_contract_passed = bool(
        checks["official_realtime_websocket_documented"]
        and checks[
            "official_realtime_payload_contains_publish_time_title_and_body"
        ]
        and historical_vintage_replayable
    )

    report: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "as_of_date": AS_OF_DATE,
        "scope": {
            "question": (
                "can Binance announcement text be reconstructed exactly as it was "
                "observable at publication time for leak-free historical evaluation"
            ),
            "source_only": True,
            "candidate_or_model_frozen": False,
        },
        "official_documentation": {
            "urls": {
                "index": DEVELOPER_DOCS_INDEX,
                "introduction": DEVELOPER_INTRODUCTION_DOC,
                "general_info": GENERAL_INFO_DOC,
                "announcement": ANNOUNCEMENT_DOC,
                "changelog": CHANGELOG_DOC,
            },
            "documented_announcement_paths": list(documented_paths),
            "sha256": {
                key: hashlib.sha256(value).hexdigest()
                for key, value in docs_raw.items()
            },
            "documented_launch_date": "2025-07-21",
            "historical_rest_endpoint_documented": False,
        },
        "current_snapshot_probe": {
            "endpoint_documented_by_developer_portal": False,
            "catalog_id": CATALOG_ID,
            "pages": list(LIST_PAGES),
            "page_counts": page_counts,
            "rows": len(list_rows),
            "earliest_release_date_ms": min(list_dates),
            "latest_release_date_ms": max(list_dates),
            "projected_rows_sha256": _hash_rows(list_rows),
            "details": details,
        },
        "telegram_archive_probe": {
            "channel": TELEGRAM_ARCHIVE,
            "pages": telegram_pages,
            "rows": len(telegram_rows),
            "edited_rows": edited_rows,
            "unreplayed_message_id_gaps": missing_ids,
            "projected_rows_sha256": _hash_rows(telegram_rows),
            "revision_or_deletion_history_endpoint_documented": False,
        },
        "checks": checks,
        "historical_original_text_vintage_replayable": historical_vintage_replayable,
        "decision": {
            "status": "retired_before_candidate_preregistration",
            "source_contract_passed": source_contract_passed,
            "historical_backtest_authorized": False,
            "candidate_preregistration_authorized": False,
            "economic_evaluation_authorized": False,
            "forward_shadow_collection_authorized": bool(
                checks["official_realtime_websocket_documented"]
                and checks[
                    "official_realtime_payload_contains_publish_time_title_and_body"
                ]
                and checks["official_realtime_launch_is_changelog_bound"]
            ),
            "reason": (
                "the official WebSocket supports causal forward capture, but no "
                "official historical original-version replay is documented; current "
                "BAPI snapshots and Telegram survivors contain edits or message-ID gaps"
            ),
        },
        "outcome_boundary": {
            "artifact_network_calls": 5 + len(LIST_PAGES) + 2 + len(TELEGRAM_BEFORE_IDS),
            "prior_interactive_source_calls_at_least": 40,
            "btc_market_rows_read": 0,
            "funding_rows_read": 0,
            "future_return_rows_read": 0,
            "return_or_pnl_fields_read": 0,
            "candidate_signal_rows_created": 0,
            "economic_outcomes_opened": False,
            "clean_room_claimed": False,
        },
        "auditor": {
            "path": str(AUDITOR_SOURCE),
            "sha256": sha256_file(AUDITOR_SOURCE),
        },
    }
    report["manifest_hash"] = canonical_hash(report)
    return report


def write_report(report: Mapping[str, Any], output: str | Path) -> None:
    target = _path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_REPORT))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report()
    write_report(report, args.output)
    print(json.dumps(report["decision"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
