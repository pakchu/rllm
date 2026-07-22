"""Seal NFET issue inventory and candidate-search responses without membership."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse

from training import preregister_non_sec_federal_exact_term_source as nfet


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "non_sec_federal_exact_term_candidate_access_v1"
PREREGISTRATION = Path(
    "results/non_sec_federal_exact_term_source_protocol_2026-07-20.json"
)
PREREGISTRATION_SHA256 = (
    "dabf6e0875ccb63ca0cfb2e489f9b0cf144de5d7dad5187923f1bd3f34bf534b"
)
PREREGISTRATION_COMMIT = "6d3c6ed9612364fa6b48fd79d37f1e75a606c4bc"
PREREGISTRATION_IMPLEMENTATION_SHA256 = (
    "48754ca00f1786b9cb0d2e00e85e34c9e9fd72f7f8d874ff18c2f00cd5c15df5"
)
BUILDER = Path("training/seal_non_sec_federal_exact_term_source_access.py")
SEARCH_ENDPOINT = "https://www.federalregister.gov/api/v1/documents.json"
ALLOWED_HOSTS = frozenset({"www.govinfo.gov", "www.federalregister.gov"})
YEARS = (2020, 2021, 2022, 2023)
NEXT_AUTHORIZED_STAGE = (
    "evaluate exact membership from retained official GovInfo MODS/HTML; "
    "still no market or model input"
)
Sleep = Callable[[float], None]
Now = Callable[[], datetime]


@dataclass(frozen=True)
class FetchResult:
    body: bytes
    status: int
    final_url: str
    etag: str | None
    last_modified: str | None


Fetch = Callable[[str], FetchResult]


@dataclass(frozen=True)
class Config:
    archive_root: Path = Path("data/non_sec_federal_exact_term_access_2020_2023")
    candidate_index: Path = Path(
        "data/non_sec_federal_exact_term_access_2020_2023/candidate_index.jsonl.gz"
    )
    resume_state: Path = Path(
        "data/non_sec_federal_exact_term_access_2020_2023/resume_state.json"
    )
    access_seal: Path = Path(
        "results/non_sec_federal_exact_term_source_access_seal_2026-07-20.json"
    )
    timeout_seconds: float = 60.0
    maximum_retries: int = 6
    retry_base_seconds: float = 2.0
    retry_max_seconds: float = 60.0
    request_pause_seconds: float = 0.10
    disk_used_abort_gib: int = 300


def _path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def _recorded_path(path: str | Path) -> str:
    candidate = _path(path).resolve()
    try:
        return candidate.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return candidate.as_posix()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_hash(payload: Any) -> str:
    return sha256_bytes(canonical_json_bytes(payload))


def deterministic_gzip(payload: bytes) -> bytes:
    return gzip.compress(payload, compresslevel=9, mtime=0)


def ensure_disk_budget(path: str | Path, *, abort_gib: int) -> None:
    target = _path(path)
    target.mkdir(parents=True, exist_ok=True)
    used = shutil.disk_usage(target).used
    if used >= abort_gib * 1024**3:
        raise RuntimeError(f"NFET disk usage reached the frozen {abort_gib} GiB limit")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_frozen(path: str | Path, payload: bytes) -> None:
    output = _path(path)
    if output.exists():
        if output.read_bytes() != payload:
            raise RuntimeError(f"refusing to overwrite frozen NFET artifact: {output}")
        return
    _atomic_write(output, payload)


def _load_preregistration() -> dict[str, Any]:
    if sha256_file(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise RuntimeError("NFET preregistration artifact changed before source access")
    if nfet.sha256_file(nfet.SCRIPT_PATH) != PREREGISTRATION_IMPLEMENTATION_SHA256:
        raise RuntimeError(
            "NFET preregistration implementation changed before source access"
        )
    payload = json.loads(_path(PREREGISTRATION).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("NFET preregistration artifact is not an object")
    nfet.validate_manifest(payload)
    if payload["final_exact_member_incidence_opened"] is not False:
        raise RuntimeError("NFET exact membership was already opened")
    return payload


def candidate_query_url(query: str, page: int) -> str:
    if query not in nfet.TERM_QUERIES or page < 1:
        raise ValueError("NFET candidate query escaped the frozen envelope")
    params = [
        ("conditions[term]", query),
        ("conditions[publication_date][gte]", nfet.SOURCE_START),
        ("conditions[publication_date][lte]", "2023-12-31"),
        ("order", "oldest"),
        ("per_page", "1000"),
        ("page", str(page)),
    ]
    return f"{SEARCH_ENDPOINT}?{urlencode(params)}"


def issue_inventory_url(year: int) -> str:
    if year not in YEARS:
        raise ValueError("NFET issue inventory year escaped the frozen envelope")
    return f"https://www.govinfo.gov/sitemap/FR_{year}_sitemap.xml"


def _http_fetch(cfg: Config, url: str, sleep: Sleep = time.sleep) -> FetchResult:
    host = urlparse(url).hostname
    if host not in ALLOWED_HOSTS:
        raise ValueError("NFET fetch host escaped the official allowlist")
    prior_error: Exception | None = None
    for attempt in range(cfg.maximum_retries + 1):
        if attempt:
            delay = min(
                cfg.retry_max_seconds,
                cfg.retry_base_seconds * (2 ** (attempt - 1)),
            )
            sleep(delay)
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "rllm-nfet-source-audit/1.0",
                "Accept": "application/json, application/xml;q=0.9, */*;q=0.1",
                "Accept-Encoding": "identity",
            },
        )
        try:
            with urllib.request.urlopen(  # noqa: S310
                request, timeout=cfg.timeout_seconds
            ) as response:
                status = response.status
                final_url = response.geturl()
                if final_url != url:
                    raise RuntimeError("NFET source redirected outside the frozen URL")
                if status != 200:
                    raise RuntimeError("NFET official source returned a non-200 status")
                etag = response.headers.get("ETag")
                last_modified = response.headers.get("Last-Modified")
                payload = response.read()
            if not payload:
                raise RuntimeError("NFET official source returned empty bytes")
            if cfg.request_pause_seconds:
                sleep(cfg.request_pause_seconds)
            return FetchResult(
                body=payload,
                status=status,
                final_url=final_url,
                etag=etag,
                last_modified=last_modified,
            )
        except urllib.error.HTTPError as exc:
            prior_error = exc
            if exc.code not in {408, 425, 429, 500, 502, 503, 504}:
                raise
        except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            prior_error = exc
    raise RuntimeError(
        f"NFET source request failed after retries: {url}"
    ) from prior_error


def _object_path(archive_root: Path, raw_sha256: str, kind: str) -> Path:
    if kind not in {"issue_xml", "search_json"}:
        raise ValueError("NFET content-addressed object kind is unknown")
    return archive_root / "objects" / raw_sha256[:2] / f"{raw_sha256}.{kind}.gz"


def store_content_addressed(
    archive_root: str | Path, payload: bytes, *, kind: str
) -> dict[str, Any]:
    root = _path(archive_root)
    raw_sha = sha256_bytes(payload)
    object_path = _object_path(root, raw_sha, kind)
    encoded = deterministic_gzip(payload)
    if object_path.exists():
        existing = object_path.read_bytes()
        if existing != encoded or gzip.decompress(existing) != payload:
            raise RuntimeError("NFET content-addressed object bytes changed")
    else:
        _atomic_write(object_path, encoded)
    return {
        "kind": kind,
        "raw_sha256": raw_sha,
        "raw_bytes": len(payload),
        "gzip_sha256": sha256_bytes(encoded),
        "gzip_bytes": len(encoded),
        "object_path": _recorded_path(object_path),
    }


def load_content_addressed(receipt: Mapping[str, Any]) -> bytes:
    path = _path(str(receipt.get("object_path", "")))
    encoded = path.read_bytes()
    if sha256_bytes(encoded) != receipt.get("gzip_sha256"):
        raise RuntimeError("NFET retained object gzip hash mismatch")
    payload = gzip.decompress(encoded)
    if len(payload) != receipt.get("raw_bytes"):
        raise RuntimeError("NFET retained object byte count mismatch")
    if sha256_bytes(payload) != receipt.get("raw_sha256"):
        raise RuntimeError("NFET retained object raw hash mismatch")
    return payload


def _validate_http_receipt(receipt: Mapping[str, Any], *, expected_url: str) -> None:
    http = receipt.get("http")
    if not isinstance(http, dict) or set(http) != {
        "etag",
        "final_url",
        "last_modified",
        "status",
    }:
        raise RuntimeError("NFET retained response HTTP metadata is malformed")
    if http.get("status") != 200 or http.get("final_url") != expected_url:
        raise RuntimeError("NFET retained response escaped the frozen URL")
    validators = (http.get("etag"), http.get("last_modified"))
    if any(
        value is not None
        and (
            not isinstance(value, str)
            or not value.strip()
            or "\r" in value
            or "\n" in value
        )
        for value in validators
    ):
        raise RuntimeError("NFET retained HTTP validator is malformed")


def _state_core(cfg: Config) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "builder_sha256": sha256_file(BUILDER),
        "archive_root": _recorded_path(cfg.archive_root),
        "responses": {},
    }


def _write_state(cfg: Config, state: Mapping[str, Any]) -> None:
    core = {key: value for key, value in state.items() if key != "state_hash"}
    payload = {**core, "state_hash": canonical_hash(core)}
    _atomic_write(
        _path(cfg.resume_state),
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False).encode()
        + b"\n",
    )


def _load_state(cfg: Config) -> dict[str, Any]:
    expected = _state_core(cfg)
    path = _path(cfg.resume_state)
    if not path.exists():
        state = expected
        _write_state(cfg, state)
        return state
    state = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(state, dict):
        raise RuntimeError("NFET resume state is not an object")
    core = {key: value for key, value in state.items() if key != "state_hash"}
    if canonical_hash(core) != state.get("state_hash"):
        raise RuntimeError("NFET resume state hash mismatch")
    for key, value in expected.items():
        if key != "responses" and core.get(key) != value:
            raise RuntimeError("NFET resume state contract changed")
    if not isinstance(core.get("responses"), dict):
        raise RuntimeError("NFET resume response map is malformed")
    return core


def _load_or_fetch(
    cfg: Config,
    state: dict[str, Any],
    *,
    url: str,
    kind: str,
    fetch: Fetch,
    now: Now,
) -> tuple[bytes, dict[str, Any]]:
    responses = state["responses"]
    if url in responses:
        receipt = responses[url]
        if (
            not isinstance(receipt, dict)
            or receipt.get("url") != url
            or receipt.get("kind") != kind
        ):
            raise RuntimeError("NFET resume receipt identity mismatch")
        _validate_http_receipt(receipt, expected_url=url)
        return load_content_addressed(receipt), receipt
    ensure_disk_budget(cfg.archive_root, abort_gib=cfg.disk_used_abort_gib)
    response = fetch(url)
    if not isinstance(response, FetchResult):
        raise TypeError("NFET fetcher returned an unsealed response type")
    if response.status != 200 or response.final_url != url:
        raise RuntimeError("NFET fetched response escaped the frozen URL")
    payload = response.body
    if not payload:
        raise RuntimeError("NFET official source returned empty bytes")
    fetched_at = now()
    if fetched_at.tzinfo is None or fetched_at.utcoffset() is None:
        raise RuntimeError("NFET retrieval timestamp must be timezone-aware")
    receipt = {
        "url": url,
        "fetched_at": fetched_at.astimezone(timezone.utc).isoformat(),
        "http": {
            "status": response.status,
            "final_url": response.final_url,
            "etag": response.etag,
            "last_modified": response.last_modified,
        },
        **store_content_addressed(cfg.archive_root, payload, kind=kind),
    }
    _validate_http_receipt(receipt, expected_url=url)
    responses[url] = receipt
    _write_state(cfg, state)
    return payload, receipt


def parse_search_page(raw: bytes) -> tuple[int, int, list[Mapping[str, Any]]]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("NFET candidate search returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("NFET candidate search root is not an object")
    count = payload.get("count")
    total_pages = payload.get("total_pages")
    results = payload.get("results")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("NFET candidate query must return a positive count")
    if (
        isinstance(total_pages, bool)
        or not isinstance(total_pages, int)
        or total_pages < 1
        or total_pages != (count + 999) // 1000
    ):
        raise ValueError("NFET candidate query total_pages does not reconcile")
    if not isinstance(results, list) or any(
        not isinstance(row, dict) for row in results
    ):
        raise ValueError("NFET candidate query results are malformed")
    if len(results) > 1000:
        raise ValueError("NFET candidate query exceeded frozen page size")
    return count, total_pages, results


def reconcile_query_pages(
    pages: Sequence[tuple[int, int, Sequence[Mapping[str, Any]]]],
) -> tuple[int, list[Mapping[str, Any]]]:
    if not pages:
        raise ValueError("NFET candidate query has no sealed pages")
    count, total_pages, _ = pages[0]
    if len(pages) != total_pages:
        raise ValueError("NFET candidate query page count is incomplete")
    rows: list[Mapping[str, Any]] = []
    seen_document_numbers: set[str] = set()
    for index, (page_count, page_total, page_rows) in enumerate(pages, start=1):
        if page_count != count or page_total != total_pages:
            raise ValueError("NFET candidate pagination metadata changed")
        expected_size = 1000 if index < total_pages else count - 1000 * (index - 1)
        if len(page_rows) != expected_size:
            raise ValueError("NFET candidate page row count does not reconcile")
        for row in page_rows:
            document_number = row.get("document_number")
            if not isinstance(document_number, str):
                raise ValueError("NFET candidate query document number is malformed")
            if document_number in seen_document_numbers:
                raise ValueError(
                    "NFET candidate query repeats a document number across its pages"
                )
            seen_document_numbers.add(document_number)
            rows.append(row)
    if len(rows) != count:
        raise ValueError("NFET candidate query result count is incomplete")
    return count, rows


def normalize_candidate(
    row: Mapping[str, Any], *, query: str, issue_dates: frozenset[str]
) -> dict[str, Any]:
    document_number = row.get("document_number")
    publication_date = row.get("publication_date")
    document_type = row.get("type")
    title = row.get("title")
    pdf_url = row.get("pdf_url")
    if (
        not isinstance(document_number, str)
        or nfet.DOCUMENT_NUMBER_PATTERN.fullmatch(document_number) is None
        or not isinstance(publication_date, str)
        or publication_date not in issue_dates
        or not isinstance(document_type, str)
        or not document_type.strip()
        or not isinstance(title, str)
        or not title.strip()
        or not isinstance(pdf_url, str)
    ):
        raise ValueError("NFET candidate record identity or metadata is malformed")
    if pdf_url != nfet.govinfo_document_urls(publication_date, document_number)["pdf"]:
        raise ValueError("NFET candidate PDF URL differs from predictable GovInfo")
    return {
        "document_number": document_number,
        "publication_date": publication_date,
        "type": document_type,
        "title": title,
        "pdf_url": pdf_url,
        "queries": [query],
    }


def merge_candidate(
    candidates: dict[str, dict[str, Any]], candidate: Mapping[str, Any]
) -> None:
    identity = str(candidate["document_number"])
    existing = candidates.get(identity)
    if existing is None:
        candidates[identity] = dict(candidate)
        return
    for field in ("publication_date", "type", "title", "pdf_url"):
        if existing[field] != candidate[field]:
            raise ValueError(f"NFET candidate duplicate conflicts on {field}")
    queries = set(existing["queries"])
    queries.update(candidate["queries"])
    existing["queries"] = [query for query in nfet.TERM_QUERIES if query in queries]


def _candidate_index_bytes(rows: Sequence[Mapping[str, Any]]) -> tuple[bytes, bytes]:
    raw = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    return raw, deterministic_gzip(raw)


def _validate_candidate_index(
    path: str | Path, expected_rows: int
) -> tuple[bytes, bytes, list[dict[str, Any]]]:
    encoded = _path(path).read_bytes()
    raw = gzip.decompress(encoded)
    lines = raw.splitlines()
    if len(lines) != expected_rows or any(not line for line in lines):
        raise RuntimeError("NFET candidate index row count mismatch")
    rows: list[dict[str, Any]] = []
    for line in lines:
        value = json.loads(line)
        if not isinstance(value, dict) or set(value) != {
            "document_number",
            "pdf_url",
            "publication_date",
            "queries",
            "title",
            "type",
        }:
            raise RuntimeError("NFET candidate index schema mismatch")
        if canonical_json_bytes(value) != line:
            raise RuntimeError("NFET candidate index row is not canonical JSON")
        rows.append(value)
    if raw != b"".join(canonical_json_bytes(row) + b"\n" for row in rows):
        raise RuntimeError("NFET candidate index framing changed")
    ordered = sorted(
        rows, key=lambda row: (row["publication_date"], row["document_number"])
    )
    if rows != ordered or len({row["document_number"] for row in rows}) != len(rows):
        raise RuntimeError("NFET candidate index order or identity changed")
    return raw, encoded, rows


def _source_contract(cfg: Config) -> dict[str, Any]:
    return {
        "years": list(YEARS),
        "queries": list(nfet.TERM_QUERIES),
        "page_size": 1000,
        "integer_pages_only": True,
        "zero_result_query_effect": "REJECT_NO_REPAIR",
        "candidate_dates_must_exist_in_issue_inventory": True,
        "raw_responses_retained": True,
        "deterministic_gzip_mtime": 0,
        "http_status": 200,
        "http_redirect_policy": "REJECT_ANY_FINAL_URL_CHANGE",
        "http_validators": ["ETag", "Last-Modified"],
        "http_validator_absence": "EXPLICIT_NULL_RETAINED",
        "disk_used_abort_gib": cfg.disk_used_abort_gib,
    }


def _seal_core(
    cfg: Config,
    *,
    state: Mapping[str, Any],
    issue_summary: Sequence[Mapping[str, Any]],
    query_summary: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    index_raw: bytes,
    index_gzip: bytes,
) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "outcomes_opened": False,
        "market_clocks_opened": False,
        "exact_official_membership_evaluated": False,
        "candidate_envelope_opened": True,
        "preregistration": {
            "path": str(PREREGISTRATION),
            "sha256": PREREGISTRATION_SHA256,
            "commit": PREREGISTRATION_COMMIT,
        },
        "implementation": {
            "path": str(BUILDER),
            "sha256": sha256_file(BUILDER),
        },
        "parser": {
            "path": str(nfet.SCRIPT_PATH),
            "sha256": PREREGISTRATION_IMPLEMENTATION_SHA256,
        },
        "source_contract": _source_contract(cfg),
        "issue_inventories": list(issue_summary),
        "candidate_queries": list(query_summary),
        "candidate_count": len(candidate_rows),
        "candidate_index": {
            "path": _recorded_path(cfg.candidate_index),
            "rows": len(candidate_rows),
            "raw_sha256": sha256_bytes(index_raw),
            "gzip_sha256": sha256_bytes(index_gzip),
            "gzip_bytes": len(index_gzip),
        },
        "responses": [state["responses"][url] for url in sorted(state["responses"])],
        "archive_root": _recorded_path(cfg.archive_root),
        "forbidden_sources_opened": [],
        "next_authorized_stage": NEXT_AUTHORIZED_STAGE,
    }


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_url: str,
    expected_kind: str,
    archive_root: str | Path,
) -> bytes:
    if set(receipt) != {
        "fetched_at",
        "gzip_bytes",
        "gzip_sha256",
        "http",
        "kind",
        "object_path",
        "raw_bytes",
        "raw_sha256",
        "url",
    }:
        raise RuntimeError("NFET candidate access receipt schema changed")
    if receipt.get("url") != expected_url or receipt.get("kind") != expected_kind:
        raise RuntimeError("NFET candidate access receipt identity changed")
    fetched_at = receipt.get("fetched_at")
    if not isinstance(fetched_at, str):
        raise RuntimeError("NFET candidate access retrieval time is malformed")
    try:
        timestamp = datetime.fromisoformat(fetched_at)
    except ValueError as exc:
        raise RuntimeError("NFET candidate access retrieval time is malformed") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() != timezone.utc.utcoffset(
        None
    ):
        raise RuntimeError("NFET candidate access retrieval time is not UTC")
    _validate_http_receipt(receipt, expected_url=expected_url)
    if (
        not _is_sha256(receipt.get("raw_sha256"))
        or not _is_sha256(receipt.get("gzip_sha256"))
        or isinstance(receipt.get("raw_bytes"), bool)
        or not isinstance(receipt.get("raw_bytes"), int)
        or receipt["raw_bytes"] <= 0
        or isinstance(receipt.get("gzip_bytes"), bool)
        or not isinstance(receipt.get("gzip_bytes"), int)
        or receipt["gzip_bytes"] <= 0
    ):
        raise RuntimeError("NFET candidate access receipt hashes or sizes changed")
    expected_object = _object_path(
        _path(archive_root), str(receipt["raw_sha256"]), expected_kind
    )
    if receipt.get("object_path") != _recorded_path(expected_object):
        raise RuntimeError("NFET candidate access object escaped its archive")
    if expected_object.stat().st_size != receipt["gzip_bytes"]:
        raise RuntimeError("NFET candidate access retained gzip size changed")
    return load_content_addressed(receipt)


def validate_access_seal(
    payload: Mapping[str, Any], *, cfg: Config | None = None
) -> None:
    if set(payload) != {
        "archive_root",
        "candidate_count",
        "candidate_envelope_opened",
        "candidate_index",
        "candidate_queries",
        "exact_official_membership_evaluated",
        "forbidden_sources_opened",
        "implementation",
        "issue_inventories",
        "manifest_hash",
        "market_clocks_opened",
        "next_authorized_stage",
        "outcomes_opened",
        "parser",
        "preregistration",
        "protocol_version",
        "responses",
        "source_contract",
    }:
        raise RuntimeError("NFET candidate access seal schema changed")
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if canonical_hash(core) != payload.get("manifest_hash"):
        raise RuntimeError("NFET candidate access seal hash mismatch")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("NFET candidate access protocol changed")
    for key in (
        "outcomes_opened",
        "market_clocks_opened",
        "exact_official_membership_evaluated",
    ):
        if payload.get(key) is not False:
            raise RuntimeError(f"NFET access seal must keep {key}=false")
    if payload.get("candidate_envelope_opened") is not True:
        raise RuntimeError("NFET candidate access seal must disclose its envelope")
    if payload.get("preregistration") != {
        "path": str(PREREGISTRATION),
        "sha256": PREREGISTRATION_SHA256,
        "commit": PREREGISTRATION_COMMIT,
    }:
        raise RuntimeError("NFET candidate access preregistration binding changed")
    implementation = payload.get("implementation")
    if not isinstance(implementation, dict) or implementation != {
        "path": str(BUILDER),
        "sha256": sha256_file(BUILDER),
    }:
        raise RuntimeError("NFET candidate access implementation binding changed")
    if payload.get("parser") != {
        "path": str(nfet.SCRIPT_PATH),
        "sha256": PREREGISTRATION_IMPLEMENTATION_SHA256,
    }:
        raise RuntimeError("NFET candidate access parser binding changed")
    expected_cfg = cfg or Config()
    if payload.get("source_contract") != _source_contract(expected_cfg):
        raise RuntimeError("NFET candidate access source contract changed")
    if payload.get("forbidden_sources_opened") != []:
        raise RuntimeError("NFET candidate access crossed the source boundary")
    if payload.get("next_authorized_stage") != NEXT_AUTHORIZED_STAGE:
        raise RuntimeError("NFET candidate access next-stage boundary changed")
    archive_root = payload.get("archive_root")
    if not isinstance(archive_root, str) or not archive_root:
        raise RuntimeError("NFET candidate access archive root is malformed")
    if cfg is not None and archive_root != _recorded_path(cfg.archive_root):
        raise RuntimeError("NFET candidate access archive root changed")
    candidate_index = payload.get("candidate_index")
    if not isinstance(candidate_index, dict):
        raise RuntimeError("NFET candidate access index metadata is malformed")
    if set(candidate_index) != {
        "gzip_bytes",
        "gzip_sha256",
        "path",
        "raw_sha256",
        "rows",
    }:
        raise RuntimeError("NFET candidate access index metadata changed")
    if cfg is not None and candidate_index.get("path") != _recorded_path(
        cfg.candidate_index
    ):
        raise RuntimeError("NFET candidate access index path changed")
    try:
        _path(str(candidate_index.get("path", ""))).resolve().relative_to(
            _path(archive_root).resolve()
        )
    except ValueError as exc:
        raise RuntimeError("NFET candidate access index escaped its archive") from exc
    rows_value = candidate_index.get("rows")
    if (
        isinstance(rows_value, bool)
        or not isinstance(rows_value, int)
        or rows_value <= 0
    ):
        raise RuntimeError("NFET candidate access index row count is malformed")
    index_raw, index_gzip, index_rows = _validate_candidate_index(
        str(candidate_index.get("path", "")), rows_value
    )
    if sha256_bytes(index_raw) != candidate_index.get("raw_sha256") or sha256_bytes(
        index_gzip
    ) != candidate_index.get("gzip_sha256"):
        raise RuntimeError("NFET candidate access index hash mismatch")
    if len(index_gzip) != candidate_index.get("gzip_bytes"):
        raise RuntimeError("NFET candidate access index byte count mismatch")
    if payload.get("candidate_count") != rows_value:
        raise RuntimeError("NFET candidate access count differs from the index")

    issue_summary = payload.get("issue_inventories")
    if not isinstance(issue_summary, list) or len(issue_summary) != len(YEARS):
        raise RuntimeError("NFET candidate access issue inventory count changed")
    for year, summary in zip(YEARS, issue_summary, strict=True):
        if not isinstance(summary, dict) or set(summary) != {
            "first_date",
            "last_date",
            "packages",
            "raw_sha256",
            "url",
            "year",
        }:
            raise RuntimeError("NFET candidate access issue summary changed")
        if (
            summary.get("year") != year
            or summary.get("url") != issue_inventory_url(year)
            or isinstance(summary.get("packages"), bool)
            or not isinstance(summary.get("packages"), int)
            or summary["packages"] <= 0
            or not _is_sha256(summary.get("raw_sha256"))
        ):
            raise RuntimeError("NFET candidate access issue summary is malformed")

    query_summary = payload.get("candidate_queries")
    if not isinstance(query_summary, list) or len(query_summary) != len(
        nfet.TERM_QUERIES
    ):
        raise RuntimeError("NFET candidate access query count changed")
    expected_urls = {issue_inventory_url(year) for year in YEARS}
    for query, summary in zip(nfet.TERM_QUERIES, query_summary, strict=True):
        if not isinstance(summary, dict) or set(summary) != {
            "count",
            "page_raw_sha256",
            "query",
            "total_pages",
        }:
            raise RuntimeError("NFET candidate access query summary changed")
        count = summary.get("count")
        total_pages = summary.get("total_pages")
        summary_page_hashes = summary.get("page_raw_sha256")
        if (
            summary.get("query") != query
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            or isinstance(total_pages, bool)
            or not isinstance(total_pages, int)
            or total_pages != (count + 999) // 1000
            or not isinstance(summary_page_hashes, list)
            or len(summary_page_hashes) != total_pages
            or any(not _is_sha256(value) for value in summary_page_hashes)
        ):
            raise RuntimeError("NFET candidate access query summary is malformed")
        expected_urls.update(
            candidate_query_url(query, page) for page in range(1, total_pages + 1)
        )

    responses = payload.get("responses")
    if not isinstance(responses, list) or not responses:
        raise RuntimeError("NFET candidate access response receipts are missing")
    if any(not isinstance(receipt, dict) for receipt in responses):
        raise RuntimeError("NFET candidate access receipt is malformed")
    receipt_urls = [receipt.get("url") for receipt in responses]
    if receipt_urls != sorted(expected_urls):
        raise RuntimeError("NFET candidate access response envelope changed")
    receipt_by_url = {str(receipt["url"]): receipt for receipt in responses}
    if set(receipt_by_url) != expected_urls:
        raise RuntimeError("NFET candidate access response envelope changed")

    issue_dates: set[str] = set()
    for year, summary in zip(YEARS, issue_summary, strict=True):
        url = issue_inventory_url(year)
        receipt = receipt_by_url[url]
        raw = _validate_receipt(
            receipt,
            expected_url=url,
            expected_kind="issue_xml",
            archive_root=archive_root,
        )
        records = nfet.parse_issue_inventory(raw, year)
        dates = [record["publication_date"] for record in records]
        if issue_dates.intersection(dates):
            raise RuntimeError("NFET issue inventory dates overlap across years")
        issue_dates.update(dates)
        if summary != {
            "year": year,
            "url": url,
            "packages": len(records),
            "first_date": dates[0],
            "last_date": dates[-1],
            "raw_sha256": receipt["raw_sha256"],
        }:
            raise RuntimeError("NFET candidate access issue replay changed")

    candidates: dict[str, dict[str, Any]] = {}
    frozen_issue_dates = frozenset(issue_dates)
    for query, summary in zip(nfet.TERM_QUERIES, query_summary, strict=True):
        pages: list[tuple[int, int, list[Mapping[str, Any]]]] = []
        replayed_page_hashes: list[str] = []
        for page in range(1, int(summary["total_pages"]) + 1):
            url = candidate_query_url(query, page)
            receipt = receipt_by_url[url]
            raw = _validate_receipt(
                receipt,
                expected_url=url,
                expected_kind="search_json",
                archive_root=archive_root,
            )
            pages.append(parse_search_page(raw))
            replayed_page_hashes.append(str(receipt["raw_sha256"]))
        count, rows = reconcile_query_pages(pages)
        if (
            count != summary["count"]
            or replayed_page_hashes != summary["page_raw_sha256"]
        ):
            raise RuntimeError("NFET candidate access query replay changed")
        for row in rows:
            merge_candidate(
                candidates,
                normalize_candidate(row, query=query, issue_dates=frozen_issue_dates),
            )
    replay_rows = sorted(
        candidates.values(),
        key=lambda row: (row["publication_date"], row["document_number"]),
    )
    replay_raw, replay_gzip = _candidate_index_bytes(replay_rows)
    if (
        replay_rows != index_rows
        or replay_raw != index_raw
        or replay_gzip != index_gzip
        or len(replay_rows) != payload.get("candidate_count")
    ):
        raise RuntimeError("NFET candidate access deterministic replay changed")


def build(
    cfg: Config,
    *,
    fetch: Fetch | None = None,
    now: Now = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    _load_preregistration()
    if cfg.disk_used_abort_gib != 300:
        raise ValueError("NFET disk limit differs from the frozen protocol")
    if cfg.maximum_retries < 0 or cfg.timeout_seconds <= 0:
        raise ValueError("NFET request configuration is invalid")
    access_path = _path(cfg.access_seal)
    if access_path.exists():
        existing = json.loads(access_path.read_text(encoding="utf-8"))
        if not isinstance(existing, dict):
            raise RuntimeError("NFET existing access seal is not an object")
        validate_access_seal(existing, cfg=cfg)
        _path(cfg.resume_state).unlink(missing_ok=True)
        return existing

    effective_fetch = fetch or (lambda url: _http_fetch(cfg, url))
    ensure_disk_budget(cfg.archive_root, abort_gib=cfg.disk_used_abort_gib)
    state = _load_state(cfg)
    issue_dates: set[str] = set()
    issue_summary: list[dict[str, Any]] = []
    for year in YEARS:
        url = issue_inventory_url(year)
        raw, receipt = _load_or_fetch(
            cfg, state, url=url, kind="issue_xml", fetch=effective_fetch, now=now
        )
        records = nfet.parse_issue_inventory(raw, year)
        dates = [record["publication_date"] for record in records]
        overlap = issue_dates.intersection(dates)
        if overlap:
            raise RuntimeError("NFET issue inventory dates overlap across years")
        issue_dates.update(dates)
        issue_summary.append(
            {
                "year": year,
                "url": url,
                "packages": len(records),
                "first_date": dates[0],
                "last_date": dates[-1],
                "raw_sha256": receipt["raw_sha256"],
            }
        )

    candidates: dict[str, dict[str, Any]] = {}
    query_summary: list[dict[str, Any]] = []
    frozen_issue_dates = frozenset(issue_dates)
    for query in nfet.TERM_QUERIES:
        page_receipts: list[dict[str, Any]] = []
        first_url = candidate_query_url(query, 1)
        first_raw, first_receipt = _load_or_fetch(
            cfg,
            state,
            url=first_url,
            kind="search_json",
            fetch=effective_fetch,
            now=now,
        )
        first = parse_search_page(first_raw)
        pages = [first]
        page_receipts.append(first_receipt)
        for page in range(2, first[1] + 1):
            url = candidate_query_url(query, page)
            raw, receipt = _load_or_fetch(
                cfg,
                state,
                url=url,
                kind="search_json",
                fetch=effective_fetch,
                now=now,
            )
            pages.append(parse_search_page(raw))
            page_receipts.append(receipt)
        count, rows = reconcile_query_pages(pages)
        for row in rows:
            merge_candidate(
                candidates,
                normalize_candidate(row, query=query, issue_dates=frozen_issue_dates),
            )
        query_summary.append(
            {
                "query": query,
                "count": count,
                "total_pages": len(pages),
                "page_raw_sha256": [receipt["raw_sha256"] for receipt in page_receipts],
            }
        )

    candidate_rows = sorted(
        candidates.values(),
        key=lambda row: (row["publication_date"], row["document_number"]),
    )
    if not candidate_rows:
        raise RuntimeError("NFET candidate envelope is empty")
    index_raw, index_gzip = _candidate_index_bytes(candidate_rows)
    _write_frozen(cfg.candidate_index, index_gzip)
    core = _seal_core(
        cfg,
        state=state,
        issue_summary=issue_summary,
        query_summary=query_summary,
        candidate_rows=candidate_rows,
        index_raw=index_raw,
        index_gzip=index_gzip,
    )
    payload = {**core, "manifest_hash": canonical_hash(core)}
    validate_access_seal(payload, cfg=cfg)
    _write_frozen(
        cfg.access_seal,
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False).encode()
        + b"\n",
    )
    _path(cfg.resume_state).unlink(missing_ok=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=Config.archive_root)
    parser.add_argument("--candidate-index", type=Path, default=Config.candidate_index)
    parser.add_argument("--resume-state", type=Path, default=Config.resume_state)
    parser.add_argument("--access-seal", type=Path, default=Config.access_seal)
    args = parser.parse_args()
    cfg = Config(
        archive_root=args.archive_root,
        candidate_index=args.candidate_index,
        resume_state=args.resume_state,
        access_seal=args.access_seal,
    )
    existed_before = _path(cfg.access_seal).exists()
    payload = build(cfg)
    print(
        json.dumps(
            {
                "status": "verified_existing" if existed_before else "created",
                "candidate_count": payload["candidate_count"],
                "manifest_hash": payload["manifest_hash"],
                "exact_official_membership_evaluated": False,
                "outcomes_opened": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
