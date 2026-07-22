"""Build the outcome-blind NFET official-membership source and support gate."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time as datetime_time, timedelta, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

from training import preregister_non_sec_federal_exact_term_source as nfet
from training import seal_non_sec_federal_exact_term_source_access as access


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "non_sec_federal_exact_term_official_membership_v1"
SUPPORT_VERSION = "non_sec_federal_exact_term_source_support_v1"
SOURCE_PROTOCOL = Path(
    "results/non_sec_federal_exact_term_source_protocol_2026-07-20.json"
)
SOURCE_PROTOCOL_SHA256 = (
    "dabf6e0875ccb63ca0cfb2e489f9b0cf144de5d7dad5187923f1bd3f34bf534b"
)
SOURCE_PARSER_SHA256 = (
    "48754ca00f1786b9cb0d2e00e85e34c9e9fd72f7f8d874ff18c2f00cd5c15df5"
)
ACCESS_BUILDER = Path("training/seal_non_sec_federal_exact_term_source_access.py")
ACCESS_BUILDER_SHA256 = (
    "cb40ed10730c473b93b0369757ed390415b40beea936948e46f01502bf16fb8b"
)
ACCESS_SEAL = Path(
    "results/non_sec_federal_exact_term_source_access_seal_2026-07-20.json"
)
ACCESS_SEAL_SHA256 = "de3148e5e13e93b33fe4fd1c34da5ae1513f51202f008b81d9ae64da142b4a3f"
ACCESS_MANIFEST_HASH = (
    "32137e45d4decb08d8629db05ca8ed27af43dca2535d9f6aeeb1cb4daf62a231"
)
CANDIDATE_INDEX_SHA256 = (
    "298103ebfbfde091b10787f3a79f03833bbf67621ac4cdc9f19d6c86e651d5ae"
)
BUILDER = Path("training/build_non_sec_federal_exact_term_source.py")
DETAIL_URL_TEMPLATE = (
    "https://www.federalregister.gov/api/v1/documents/{document_number}.json"
)
CORRECTION_RELATIONSHIP_URL_TEMPLATE = (
    "https://www.federalregister.gov/api/v1/documents/{document_number}"
)
CORRECTION_RELATIONSHIP_URL_PATTERN = re.compile(
    r"https://www\.federalregister\.gov/api/v1/documents/"
    r"([A-Z0-9]+(?:-[A-Z0-9]+)+)",
    re.ASCII,
)
NETWORK_KINDS = frozenset({"mods_xml", "html_raw", "detail_json", "pdf"})
DERIVED_KINDS = frozenset({"canonical_text", "match_json"})
YEARS = (2020, 2021, 2022, 2023)
QUARTERS = tuple(f"{year}-Q{quarter}" for year in YEARS for quarter in range(1, 5))
Fetch = Callable[[str], access.FetchResult]
Now = Callable[[], datetime]


@dataclass(frozen=True)
class Config:
    archive_root: Path = Path("data/non_sec_federal_exact_term_membership_2020_2023")
    decisions: Path = Path(
        "data/non_sec_federal_exact_term_membership_2020_2023/"
        "candidate_decisions.jsonl.gz"
    )
    sec_events: Path = Path(
        "data/non_sec_federal_exact_term_membership_2020_2023/"
        "sec_comparator_events.jsonl.gz"
    )
    selected_source: Path = Path(
        "data/non_sec_federal_exact_term_source_2020_2023.jsonl.gz"
    )
    resume_state: Path = Path(
        "data/non_sec_federal_exact_term_membership_2020_2023/resume_state.json"
    )
    source_manifest: Path = Path(
        "results/non_sec_federal_exact_term_source_manifest_2026-07-20.json"
    )
    support_result: Path = Path(
        "results/non_sec_federal_exact_term_source_support_2026-07-20.json"
    )
    timeout_seconds: float = 60.0
    maximum_retries: int = 6
    retry_base_seconds: float = 2.0
    retry_max_seconds: float = 60.0
    request_pause_seconds: float = 0.10
    disk_used_abort_gib: int = 300
    resume_existing: bool = False


def _path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def _recorded_path(path: str | Path) -> str:
    candidate = _path(path).resolve()
    try:
        return candidate.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return candidate.as_posix()


def sha256_file(path: str | Path) -> str:
    return access.sha256_file(path)


def canonical_json_bytes(payload: Any) -> bytes:
    return access.canonical_json_bytes(payload)


def canonical_hash(payload: Any) -> str:
    return access.canonical_hash(payload)


def deterministic_gzip(payload: bytes) -> bytes:
    return access.deterministic_gzip(payload)


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


def _object_path(archive_root: Path, raw_sha256: str, kind: str) -> Path:
    if kind not in NETWORK_KINDS | DERIVED_KINDS:
        raise ValueError("NFET membership object kind is unknown")
    return archive_root / "objects" / raw_sha256[:2] / f"{raw_sha256}.{kind}.gz"


def store_object(
    archive_root: str | Path, payload: bytes, *, kind: str
) -> dict[str, Any]:
    root = _path(archive_root)
    raw_sha = access.sha256_bytes(payload)
    object_path = _object_path(root, raw_sha, kind)
    encoded = deterministic_gzip(payload)
    if object_path.exists():
        existing = object_path.read_bytes()
        if existing != encoded or gzip.decompress(existing) != payload:
            raise RuntimeError("NFET membership content-addressed object changed")
    else:
        _atomic_write(object_path, encoded)
    return {
        "kind": kind,
        "raw_sha256": raw_sha,
        "raw_bytes": len(payload),
        "gzip_sha256": access.sha256_bytes(encoded),
        "gzip_bytes": len(encoded),
        "object_path": _recorded_path(object_path),
    }


def load_object(receipt: Mapping[str, Any], *, archive_root: str | Path) -> bytes:
    kind = receipt.get("kind")
    raw_sha = receipt.get("raw_sha256")
    if kind not in NETWORK_KINDS | DERIVED_KINDS or not isinstance(raw_sha, str):
        raise RuntimeError("NFET membership receipt identity is malformed")
    expected = _object_path(_path(archive_root), raw_sha, str(kind))
    if receipt.get("object_path") != _recorded_path(expected):
        raise RuntimeError("NFET membership object escaped its archive")
    encoded = expected.read_bytes()
    if access.sha256_bytes(encoded) != receipt.get("gzip_sha256"):
        raise RuntimeError("NFET membership retained gzip hash mismatch")
    if len(encoded) != receipt.get("gzip_bytes"):
        raise RuntimeError("NFET membership retained gzip size mismatch")
    payload = gzip.decompress(encoded)
    if len(payload) != receipt.get("raw_bytes"):
        raise RuntimeError("NFET membership retained raw size mismatch")
    if access.sha256_bytes(payload) != raw_sha:
        raise RuntimeError("NFET membership retained raw hash mismatch")
    return payload


def detail_url(document_number: str) -> str:
    if nfet.DOCUMENT_NUMBER_PATTERN.fullmatch(document_number) is None:
        raise ValueError("NFET detail document number is malformed")
    return DETAIL_URL_TEMPLATE.format(document_number=document_number)


def correction_relationship_url(document_number: str) -> str:
    if nfet.DOCUMENT_NUMBER_PATTERN.fullmatch(document_number) is None:
        raise ValueError("NFET correction relationship document number is malformed")
    return CORRECTION_RELATIONSHIP_URL_TEMPLATE.format(document_number=document_number)


def normalize_correction_metadata(
    detail: Mapping[str, Any], *, current_document_number: str
) -> tuple[str | None, list[str]]:
    correction_of = detail.get("correction_of")
    if correction_of is not None:
        if not isinstance(correction_of, str):
            raise ValueError("NFET correction_of is not null or a detail URL")
        match = CORRECTION_RELATIONSHIP_URL_PATTERN.fullmatch(correction_of)
        if (
            match is None
            or nfet.DOCUMENT_NUMBER_PATTERN.fullmatch(match.group(1)) is None
            or correction_of != correction_relationship_url(match.group(1))
            or match.group(1) == current_document_number
        ):
            raise ValueError("NFET correction_of relationship URL is malformed")
    corrections = detail.get("corrections")
    if not isinstance(corrections, list):
        raise ValueError("NFET corrections is not a list of detail URLs")
    normalized: list[str] = []
    for value in corrections:
        if not isinstance(value, str):
            raise ValueError("NFET corrections contains a non-string URL")
        match = CORRECTION_RELATIONSHIP_URL_PATTERN.fullmatch(value)
        if (
            match is None
            or nfet.DOCUMENT_NUMBER_PATTERN.fullmatch(match.group(1)) is None
            or value != correction_relationship_url(match.group(1))
            or match.group(1) == current_document_number
        ):
            raise ValueError("NFET corrections contains a malformed relationship URL")
        if value in normalized:
            raise ValueError("NFET corrections repeats a detail URL")
        normalized.append(value)
    return correction_of, sorted(normalized)


def _network_config(cfg: Config) -> access.Config:
    return access.Config(
        archive_root=cfg.archive_root,
        timeout_seconds=cfg.timeout_seconds,
        maximum_retries=cfg.maximum_retries,
        retry_base_seconds=cfg.retry_base_seconds,
        retry_max_seconds=cfg.retry_max_seconds,
        request_pause_seconds=cfg.request_pause_seconds,
        disk_used_abort_gib=cfg.disk_used_abort_gib,
    )


def _state_core(cfg: Config) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "source_protocol_sha256": SOURCE_PROTOCOL_SHA256,
        "access_seal_sha256": ACCESS_SEAL_SHA256,
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


def _load_state(cfg: Config, *, allow_existing: bool = False) -> dict[str, Any]:
    expected = _state_core(cfg)
    path = _path(cfg.resume_state)
    if not path.exists():
        _write_state(cfg, expected)
        return expected
    if not allow_existing:
        raise RuntimeError(
            "NFET membership resume state exists; explicit resume authorization required"
        )
    state = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(state, dict):
        raise RuntimeError("NFET membership resume state is not an object")
    core = {key: value for key, value in state.items() if key != "state_hash"}
    if canonical_hash(core) != state.get("state_hash"):
        raise RuntimeError("NFET membership resume state hash mismatch")
    for key, value in expected.items():
        if key != "responses" and core.get(key) != value:
            raise RuntimeError("NFET membership resume contract changed")
    if not isinstance(core.get("responses"), dict):
        raise RuntimeError("NFET membership resume response map is malformed")
    return core


def _validate_http(receipt: Mapping[str, Any], *, expected_url: str) -> None:
    http = receipt.get("http")
    if not isinstance(http, dict) or set(http) != {
        "etag",
        "final_url",
        "last_modified",
        "status",
    }:
        raise RuntimeError("NFET membership HTTP receipt is malformed")
    if http.get("status") != 200 or http.get("final_url") != expected_url:
        raise RuntimeError("NFET membership response escaped the frozen URL")
    for value in (http.get("etag"), http.get("last_modified")):
        if value is not None and (
            not isinstance(value, str)
            or not value.strip()
            or "\r" in value
            or "\n" in value
        ):
            raise RuntimeError("NFET membership HTTP validator is malformed")


def _validate_network_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_url: str,
    expected_kind: str,
    archive_root: str | Path,
    not_after: datetime | None = None,
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
        raise RuntimeError("NFET membership network receipt schema changed")
    if receipt.get("url") != expected_url or receipt.get("kind") != expected_kind:
        raise RuntimeError("NFET membership network receipt identity changed")
    fetched_at = receipt.get("fetched_at")
    if not isinstance(fetched_at, str):
        raise RuntimeError("NFET membership retrieval time is malformed")
    try:
        timestamp = datetime.fromisoformat(fetched_at)
    except ValueError as exc:
        raise RuntimeError("NFET membership retrieval time is malformed") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() != timedelta(0):
        raise RuntimeError("NFET membership retrieval time is not UTC")
    if not_after is not None:
        if not_after.tzinfo is None or not_after.utcoffset() is None:
            raise RuntimeError("NFET membership resume clock is not timezone-aware")
        if timestamp > not_after.astimezone(timezone.utc):
            raise RuntimeError(
                "NFET membership receipt is dated after the resume clock"
            )
    _validate_http(receipt, expected_url=expected_url)
    return load_object(receipt, archive_root=archive_root)


def _load_or_fetch(
    cfg: Config,
    state: dict[str, Any],
    *,
    url: str,
    kind: str,
    fetch: Fetch,
    now: Now,
) -> tuple[bytes, dict[str, Any]]:
    if kind not in NETWORK_KINDS:
        raise ValueError("NFET membership fetch kind is not a network object")
    responses = state["responses"]
    if url in responses:
        receipt = responses[url]
        if not isinstance(receipt, dict):
            raise RuntimeError("NFET membership resume receipt is malformed")
        checked_at = now()
        return (
            _validate_network_receipt(
                receipt,
                expected_url=url,
                expected_kind=kind,
                archive_root=cfg.archive_root,
                not_after=checked_at,
            ),
            receipt,
        )
    access.ensure_disk_budget(cfg.archive_root, abort_gib=cfg.disk_used_abort_gib)
    response = fetch(url)
    if not isinstance(response, access.FetchResult):
        raise TypeError("NFET membership fetcher returned an unsealed response")
    if response.status != 200 or response.final_url != url or not response.body:
        raise RuntimeError("NFET membership official response is invalid")
    fetched_at = now()
    if fetched_at.tzinfo is None or fetched_at.utcoffset() is None:
        raise RuntimeError("NFET membership timestamp must be timezone-aware")
    receipt = {
        "url": url,
        "fetched_at": fetched_at.astimezone(timezone.utc).isoformat(),
        "http": {
            "status": response.status,
            "final_url": response.final_url,
            "etag": response.etag,
            "last_modified": response.last_modified,
        },
        **store_object(cfg.archive_root, response.body, kind=kind),
    }
    _validate_network_receipt(
        receipt,
        expected_url=url,
        expected_kind=kind,
        archive_root=cfg.archive_root,
    )
    responses[url] = receipt
    _write_state(cfg, state)
    return response.body, receipt


def _derived_receipt(
    cfg: Config,
    payload: bytes,
    *,
    kind: str,
    derived_from_raw_sha256: str,
) -> dict[str, Any]:
    if kind not in DERIVED_KINDS:
        raise ValueError("NFET membership derived object kind is unknown")
    receipt = {
        "derived_from_raw_sha256": derived_from_raw_sha256,
        "parser_sha256": SOURCE_PARSER_SHA256,
        **store_object(cfg.archive_root, payload, kind=kind),
    }
    return receipt


def _validate_derived_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_kind: str,
    expected_source_sha256: str,
    archive_root: str | Path,
) -> bytes:
    if set(receipt) != {
        "derived_from_raw_sha256",
        "gzip_bytes",
        "gzip_sha256",
        "kind",
        "object_path",
        "parser_sha256",
        "raw_bytes",
        "raw_sha256",
    }:
        raise RuntimeError("NFET membership derived receipt schema changed")
    if (
        receipt.get("kind") != expected_kind
        or receipt.get("derived_from_raw_sha256") != expected_source_sha256
        or receipt.get("parser_sha256") != SOURCE_PARSER_SHA256
    ):
        raise RuntimeError("NFET membership derived receipt binding changed")
    return load_object(receipt, archive_root=archive_root)


def _load_json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"NFET {label} is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"NFET {label} is not a JSON object")
    return payload


def _load_inputs() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if sha256_file(SOURCE_PROTOCOL) != SOURCE_PROTOCOL_SHA256:
        raise RuntimeError("NFET source protocol changed before membership access")
    if sha256_file(nfet.SCRIPT_PATH) != SOURCE_PARSER_SHA256:
        raise RuntimeError("NFET source parser changed before membership access")
    if sha256_file(ACCESS_BUILDER) != ACCESS_BUILDER_SHA256:
        raise RuntimeError("NFET candidate-access builder changed")
    if sha256_file(ACCESS_SEAL) != ACCESS_SEAL_SHA256:
        raise RuntimeError("NFET candidate-access seal changed")
    protocol = _load_json_object(_path(SOURCE_PROTOCOL).read_bytes(), label="protocol")
    nfet.validate_manifest(protocol)
    seal = _load_json_object(_path(ACCESS_SEAL).read_bytes(), label="access seal")
    access.validate_access_seal(seal)
    if seal.get("manifest_hash") != ACCESS_MANIFEST_HASH:
        raise RuntimeError("NFET candidate-access manifest binding changed")
    index_meta = seal.get("candidate_index")
    if not isinstance(index_meta, dict):
        raise RuntimeError("NFET candidate index metadata is malformed")
    index_path = _path(str(index_meta.get("path", "")))
    if sha256_file(index_path) != CANDIDATE_INDEX_SHA256:
        raise RuntimeError("NFET candidate index changed before membership access")
    raw = gzip.decompress(index_path.read_bytes())
    candidates = [json.loads(line) for line in raw.splitlines()]
    if (
        len(candidates) != 486
        or any(not isinstance(candidate, dict) for candidate in candidates)
        or len({candidate["document_number"] for candidate in candidates}) != 486
    ):
        raise RuntimeError("NFET candidate index identity changed")
    return protocol, candidates


def historical_available_at(publication_date: str) -> str:
    available_date = date.fromisoformat(publication_date) + timedelta(days=1)
    available = datetime.combine(
        available_date, datetime_time(hour=12), tzinfo=timezone.utc
    )
    return available.isoformat()


def validate_candidate_govinfo_identity(
    document_number: str,
    publication_date: str,
    mods: Mapping[str, Any],
) -> tuple[str, ...]:
    urls = nfet.govinfo_document_urls(publication_date, document_number)
    expected = {
        "collection_code": "FR",
        "access_id": document_number,
        "fr_doc_number": document_number,
        "identifier_fr_doc_number": document_number,
        "host_date_issued": publication_date,
        "host_uri": f"https://www.govinfo.gov/app/details/FR-{publication_date}",
        "other_format_urls": tuple(sorted((urls["html"], urls["pdf"]))),
    }
    for field, value in expected.items():
        if mods.get(field) != value:
            raise ValueError(f"NFET GovInfo MODS {field} identity mismatch")
    granule_class = mods.get("granule_class")
    if not isinstance(granule_class, str) or not granule_class.strip():
        raise ValueError("NFET GovInfo MODS granule_class is empty")
    agency_names = mods.get("agency_names")
    if not isinstance(agency_names, (list, tuple)) or not all(
        isinstance(name, str) and name for name in agency_names
    ):
        raise ValueError("NFET GovInfo MODS agency_names are malformed")
    return tuple(agency_names)


def evaluate_candidate_sources(
    candidate: Mapping[str, Any],
    *,
    mods_raw: bytes,
    html_raw: bytes,
    detail_raw: bytes,
    pdf_raw: bytes | None,
) -> dict[str, Any]:
    document_number = candidate.get("document_number")
    publication_date = candidate.get("publication_date")
    if not isinstance(document_number, str) or not isinstance(publication_date, str):
        raise ValueError("NFET candidate identity is malformed")
    detail = _load_json_object(detail_raw, label="FederalRegister detail")
    canonical_text = nfet.official_html_membership_view(html_raw)
    if not canonical_text:
        raise ValueError("NFET canonical GovInfo HTML text is empty")
    matches = nfet.exact_term_matches(canonical_text)
    if matches and (pdf_raw is None or not pdf_raw.startswith(b"%PDF-")):
        raise ValueError("NFET positive GovInfo PDF is missing or malformed")
    identity: Mapping[str, Any] | None = None
    if matches:
        mods = nfet.parse_govinfo_mods(mods_raw)
        validate_candidate_govinfo_identity(document_number, publication_date, mods)
        identity = nfet.reconcile_positive_identity(
            document_number, publication_date, mods, detail
        )
        normalize_correction_metadata(detail, current_document_number=document_number)
    return {
        "member": bool(matches),
        "identity_stratum": identity["stratum"] if identity is not None else None,
        "member_stratum": identity["stratum"] if identity is not None else None,
        "govinfo_agency_names": (
            list(identity["govinfo_agency_names"]) if identity is not None else None
        ),
        "detail_agency_slugs": (
            list(identity["detail_agency_slugs"]) if identity is not None else None
        ),
        "canonical_text": canonical_text,
        "exact_matches": matches,
        "historical_available_at": historical_available_at(publication_date),
    }


def _decision_from_sources(
    cfg: Config,
    candidate: Mapping[str, Any],
    *,
    mods_raw: bytes,
    mods_receipt: Mapping[str, Any],
    html_raw: bytes,
    html_receipt: Mapping[str, Any],
    detail_raw: bytes,
    detail_receipt: Mapping[str, Any],
    pdf_raw: bytes | None,
    pdf_receipt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    evaluated = evaluate_candidate_sources(
        candidate,
        mods_raw=mods_raw,
        html_raw=html_raw,
        detail_raw=detail_raw,
        pdf_raw=pdf_raw,
    )
    canonical_text = str(evaluated.pop("canonical_text"))
    matches = evaluated["exact_matches"]
    canonical_receipt = _derived_receipt(
        cfg,
        canonical_text.encode("utf-8"),
        kind="canonical_text",
        derived_from_raw_sha256=str(html_receipt["raw_sha256"]),
    )
    match_receipt = None
    if matches:
        match_receipt = _derived_receipt(
            cfg,
            canonical_json_bytes(matches),
            kind="match_json",
            derived_from_raw_sha256=str(canonical_receipt["raw_sha256"]),
        )
    return {
        "candidate": dict(candidate),
        **evaluated,
        "receipts": {
            "mods": dict(mods_receipt),
            "html": dict(html_receipt),
            "detail": dict(detail_receipt),
            "canonical_text": canonical_receipt,
            "pdf": dict(pdf_receipt) if pdf_receipt is not None else None,
            "matches": match_receipt,
        },
    }


def _validate_decision(
    cfg: Config,
    decision: Mapping[str, Any],
    *,
    expected_candidate: Mapping[str, Any],
) -> dict[str, Any]:
    if set(decision) != {
        "candidate",
        "detail_agency_slugs",
        "exact_matches",
        "govinfo_agency_names",
        "historical_available_at",
        "identity_stratum",
        "member",
        "member_stratum",
        "receipts",
    } or decision.get("candidate") != dict(expected_candidate):
        raise RuntimeError("NFET membership decision schema or candidate changed")
    receipts = decision.get("receipts")
    if not isinstance(receipts, dict) or set(receipts) != {
        "canonical_text",
        "detail",
        "html",
        "matches",
        "mods",
        "pdf",
    }:
        raise RuntimeError("NFET membership decision receipts changed")
    document_number = str(expected_candidate["document_number"])
    publication_date = str(expected_candidate["publication_date"])
    urls = nfet.govinfo_document_urls(publication_date, document_number)
    network_specs = (
        ("mods", urls["mods"], "mods_xml"),
        ("html", urls["html"], "html_raw"),
        ("detail", detail_url(document_number), "detail_json"),
    )
    loaded: dict[str, bytes] = {}
    for name, url, kind in network_specs:
        receipt = receipts.get(name)
        if not isinstance(receipt, dict):
            raise RuntimeError("NFET membership required receipt is missing")
        loaded[name] = _validate_network_receipt(
            receipt,
            expected_url=url,
            expected_kind=kind,
            archive_root=cfg.archive_root,
        )
    canonical_receipt = receipts.get("canonical_text")
    if not isinstance(canonical_receipt, dict):
        raise RuntimeError("NFET canonical text receipt is missing")
    canonical_raw = _validate_derived_receipt(
        canonical_receipt,
        expected_kind="canonical_text",
        expected_source_sha256=str(receipts["html"]["raw_sha256"]),
        archive_root=cfg.archive_root,
    )
    expected_canonical = nfet.official_html_membership_view(loaded["html"]).encode(
        "utf-8"
    )
    if canonical_raw != expected_canonical:
        raise RuntimeError("NFET canonical membership text changed")
    expected_matches = nfet.exact_term_matches(canonical_raw.decode("utf-8"))

    pdf_receipt = receipts.get("pdf")
    match_receipt = receipts.get("matches")
    pdf_raw: bytes | None = None
    if expected_matches:
        if not isinstance(pdf_receipt, dict) or not isinstance(match_receipt, dict):
            raise RuntimeError("NFET positive PDF or match receipt is missing")
        pdf_raw = _validate_network_receipt(
            pdf_receipt,
            expected_url=urls["pdf"],
            expected_kind="pdf",
            archive_root=cfg.archive_root,
        )
        matches_raw = _validate_derived_receipt(
            match_receipt,
            expected_kind="match_json",
            expected_source_sha256=str(canonical_receipt["raw_sha256"]),
            archive_root=cfg.archive_root,
        )
        if matches_raw != canonical_json_bytes(expected_matches):
            raise RuntimeError("NFET exact match record changed")
    elif pdf_receipt is not None or match_receipt is not None:
        raise RuntimeError(
            "NFET negative candidate retained forbidden positive objects"
        )

    evaluated = evaluate_candidate_sources(
        expected_candidate,
        mods_raw=loaded["mods"],
        html_raw=loaded["html"],
        detail_raw=loaded["detail"],
        pdf_raw=pdf_raw,
    )
    evaluated.pop("canonical_text")
    expected = {
        "candidate": dict(expected_candidate),
        **evaluated,
        "receipts": dict(receipts),
    }
    if decision != expected:
        raise RuntimeError("NFET membership decision deterministic replay changed")
    return expected


def _process_candidate(
    cfg: Config,
    state: dict[str, Any],
    candidate: Mapping[str, Any],
    *,
    fetch: Fetch,
    now: Now,
) -> dict[str, Any]:
    identity = str(candidate["document_number"])
    publication_date = str(candidate["publication_date"])
    urls = nfet.govinfo_document_urls(publication_date, identity)
    mods_raw, mods_receipt = _load_or_fetch(
        cfg,
        state,
        url=urls["mods"],
        kind="mods_xml",
        fetch=fetch,
        now=now,
    )
    html_raw, html_receipt = _load_or_fetch(
        cfg,
        state,
        url=urls["html"],
        kind="html_raw",
        fetch=fetch,
        now=now,
    )
    detail_raw, detail_receipt = _load_or_fetch(
        cfg,
        state,
        url=detail_url(identity),
        kind="detail_json",
        fetch=fetch,
        now=now,
    )
    matches = nfet.exact_term_matches(nfet.official_html_membership_view(html_raw))
    pdf_raw: bytes | None = None
    pdf_receipt: dict[str, Any] | None = None
    if matches:
        pdf_raw, pdf_receipt = _load_or_fetch(
            cfg,
            state,
            url=urls["pdf"],
            kind="pdf",
            fetch=fetch,
            now=now,
        )
    decision = _decision_from_sources(
        cfg,
        candidate,
        mods_raw=mods_raw,
        mods_receipt=mods_receipt,
        html_raw=html_raw,
        html_receipt=html_receipt,
        detail_raw=detail_raw,
        detail_receipt=detail_receipt,
        pdf_raw=pdf_raw,
        pdf_receipt=pdf_receipt,
    )
    _validate_decision(cfg, decision, expected_candidate=candidate)
    return decision


def _event_document(decision: Mapping[str, Any]) -> dict[str, Any]:
    candidate = decision["candidate"]
    receipts = decision["receipts"]
    pdf = receipts["pdf"]
    return {
        "document_number": candidate["document_number"],
        "type": candidate["type"],
        "title": candidate["title"],
        "candidate_queries": candidate["queries"],
        "govinfo_agency_names": decision["govinfo_agency_names"],
        "detail_agency_slugs": decision["detail_agency_slugs"],
        "exact_matches": decision["exact_matches"],
        "canonical_text": {
            "path": receipts["canonical_text"]["object_path"],
            "sha256": receipts["canonical_text"]["raw_sha256"],
        },
        "official_html_raw_sha256": receipts["html"]["raw_sha256"],
        "govinfo_mods_raw_sha256": receipts["mods"]["raw_sha256"],
        "federalregister_detail_raw_sha256": receipts["detail"]["raw_sha256"],
        "govinfo_pdf_raw_sha256": pdf["raw_sha256"],
        "govinfo_pdf_bytes": pdf["raw_bytes"],
    }


def build_events(
    decisions: Sequence[Mapping[str, Any]], *, stratum: str
) -> list[dict[str, Any]]:
    if stratum not in {"primary_non_sec", "sec_comparator"}:
        raise ValueError("NFET event stratum is unknown")
    by_date: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for decision in decisions:
        if decision.get("member") is True and decision.get("member_stratum") == stratum:
            candidate = decision.get("candidate")
            if not isinstance(candidate, dict):
                raise ValueError("NFET event candidate is malformed")
            by_date[str(candidate["publication_date"])].append(decision)
    events: list[dict[str, Any]] = []
    for publication_date in sorted(by_date):
        documents = sorted(
            (_event_document(decision) for decision in by_date[publication_date]),
            key=lambda row: row["document_number"],
        )
        events.append(
            {
                "source_id": "NFET" if stratum == "primary_non_sec" else "NFET_SEC",
                "event_id": f"{stratum}:{publication_date}",
                "stratum": stratum,
                "publication_date": publication_date,
                "historical_available_at": historical_available_at(publication_date),
                "document_count": len(documents),
                "documents": documents,
            }
        )
    return events


def _fraction_record(
    value: Fraction | None, *, label: str | None = None
) -> dict[str, Any]:
    if value is None:
        return {"value": None, "fraction": None, "label": label}
    return {
        "value": float(value),
        "fraction": f"{value.numerator}/{value.denominator}",
        "label": label,
    }


def evaluate_source_quality(
    decisions: Sequence[Mapping[str, Any]], quality_contract: Mapping[str, Any]
) -> dict[str, Any]:
    primary = [
        decision
        for decision in decisions
        if decision.get("member") is True
        and decision.get("member_stratum") == "primary_non_sec"
    ]
    sec = [
        decision
        for decision in decisions
        if decision.get("member") is True
        and decision.get("member_stratum") == "sec_comparator"
    ]

    def publication_date(decision: Mapping[str, Any]) -> str:
        candidate = decision.get("candidate")
        if not isinstance(candidate, dict):
            raise ValueError("NFET quality candidate is malformed")
        value = candidate.get("publication_date")
        if not isinstance(value, str):
            raise ValueError("NFET quality publication date is malformed")
        return value

    def year_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        counts = Counter(publication_date(row)[:4] for row in rows)
        return {str(year): counts[str(year)] for year in YEARS}

    def unique_days_by_year(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        days: dict[str, set[str]] = {str(year): set() for year in YEARS}
        for row in rows:
            value = publication_date(row)
            days[value[:4]].add(value)
        return {year: len(values) for year, values in days.items()}

    primary_dates = [publication_date(row) for row in primary]
    sec_dates = [publication_date(row) for row in sec]
    primary_years = year_counts(primary)
    sec_years = year_counts(sec)
    primary_days_by_year = unique_days_by_year(primary)

    quarter_counts = Counter()
    for value in primary_dates:
        parsed = date.fromisoformat(value)
        quarter_counts[f"{parsed.year}-Q{(parsed.month - 1) // 3 + 1}"] += 1
    by_quarter = {quarter: quarter_counts[quarter] for quarter in QUARTERS}

    month_counts = Counter(value[:7] for value in primary_dates)
    if month_counts:
        max_month_count = max(month_counts.values())
        max_month = min(
            month for month, count in month_counts.items() if count == max_month_count
        )
        max_month_share = Fraction(max_month_count, len(primary))
    else:
        max_month = None
        max_month_count = 0
        max_month_share = None

    agency_weights: dict[str, Fraction] = defaultdict(Fraction)
    for decision in primary:
        agencies = decision.get("govinfo_agency_names")
        if not isinstance(agencies, list) or not agencies:
            raise ValueError("NFET quality agency set is malformed")
        contribution = Fraction(1, len(agencies))
        for agency in agencies:
            if not isinstance(agency, str) or not agency:
                raise ValueError("NFET quality agency name is malformed")
            agency_weights[agency] += contribution
    if agency_weights:
        max_agency_weight = max(agency_weights.values())
        max_agency = min(
            agency
            for agency, weight in agency_weights.items()
            if weight == max_agency_weight
        )
        max_agency_share = max_agency_weight / len(primary)
    else:
        max_agency = None
        max_agency_share = None

    primary_contract = quality_contract.get("primary_non_sec")
    sec_contract = quality_contract.get("sec_comparator")
    integrity_contract = quality_contract.get("integrity")
    if (
        not isinstance(primary_contract, dict)
        or not isinstance(sec_contract, dict)
        or not isinstance(integrity_contract, dict)
    ):
        raise ValueError("NFET quality contract is malformed")

    gates = {
        "primary.minimum_documents": len(primary)
        >= int(primary_contract["minimum_documents"]),
        "primary.minimum_documents_each_year": all(
            count >= int(primary_contract["minimum_documents_each_year"])
            for count in primary_years.values()
        ),
        "primary.minimum_unique_publication_days": len(set(primary_dates))
        >= int(primary_contract["minimum_unique_publication_days"]),
        "primary.minimum_unique_publication_days_each_year": all(
            count >= int(primary_contract["minimum_unique_publication_days_each_year"])
            for count in primary_days_by_year.values()
        ),
        "primary.minimum_documents_each_quarter": all(
            count >= int(primary_contract["minimum_documents_each_quarter"])
            for count in by_quarter.values()
        ),
        "primary.maximum_month_share": max_month_share is not None
        and max_month_share <= Fraction(str(primary_contract["maximum_month_share"])),
        "primary.maximum_fractional_agency_share": max_agency_share is not None
        and max_agency_share
        <= Fraction(str(primary_contract["maximum_fractional_agency_share"])),
        "sec.minimum_documents": len(sec) >= int(sec_contract["minimum_documents"]),
        "sec.minimum_documents_each_year": all(
            count >= int(sec_contract["minimum_documents_each_year"])
            for count in sec_years.values()
        ),
        "sec.minimum_unique_publication_days": len(set(sec_dates))
        >= int(sec_contract["minimum_unique_publication_days"]),
        "integrity.candidate_page_reconciliation_fraction": integrity_contract[
            "candidate_page_reconciliation_fraction"
        ]
        == 1.0,
        "integrity.candidate_issue_inventory_reconciliation_fraction": integrity_contract[
            "candidate_issue_inventory_reconciliation_fraction"
        ]
        == 1.0,
        "integrity.positive_mods_html_pdf_detail_reconciliation_fraction": integrity_contract[
            "positive_mods_html_pdf_detail_reconciliation_fraction"
        ]
        == 1.0,
        "integrity.no_quarantine_or_imputation": integrity_contract[
            "quarantine_or_imputation_allowed"
        ]
        is False,
        "integrity.deterministic_rebuild": integrity_contract[
            "deterministic_rebuild_required"
        ]
        is True,
    }
    failed = [name for name, passed in gates.items() if not passed]
    return {
        "status": "PASS" if not failed else "REJECT",
        "failure_effect": quality_contract.get("failure_effect"),
        "failed_gates": failed,
        "gates": gates,
        "metrics": {
            "candidate_documents": len(decisions),
            "exact_members": len(primary) + len(sec),
            "nonmembers": len(decisions) - len(primary) - len(sec),
            "primary_non_sec": {
                "documents": len(primary),
                "documents_by_year": primary_years,
                "unique_publication_days": len(set(primary_dates)),
                "unique_publication_days_by_year": primary_days_by_year,
                "documents_by_quarter": by_quarter,
                "maximum_month": max_month,
                "maximum_month_documents": max_month_count,
                "maximum_month_share": _fraction_record(
                    max_month_share, label=max_month
                ),
                "maximum_fractional_agency_share": _fraction_record(
                    max_agency_share, label=max_agency
                ),
            },
            "sec_comparator": {
                "documents": len(sec),
                "documents_by_year": sec_years,
                "unique_publication_days": len(set(sec_dates)),
            },
        },
    }


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> tuple[bytes, bytes]:
    raw = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    return raw, deterministic_gzip(raw)


def _output_record(
    path: str | Path, raw: bytes, encoded: bytes, rows: int
) -> dict[str, Any]:
    return {
        "path": _recorded_path(path),
        "rows": rows,
        "raw_sha256": access.sha256_bytes(raw),
        "gzip_sha256": access.sha256_bytes(encoded),
        "gzip_bytes": len(encoded),
    }


def _manifest_core(
    cfg: Config,
    *,
    decisions_record: Mapping[str, Any],
    selected_record: Mapping[str, Any],
    sec_record: Mapping[str, Any],
    decisions: Sequence[Mapping[str, Any]],
    responses: Mapping[str, Any],
    quality: Mapping[str, Any],
) -> dict[str, Any]:
    response_kinds = Counter(
        receipt["kind"] for receipt in responses.values() if isinstance(receipt, dict)
    )
    members = [decision for decision in decisions if decision["member"] is True]
    primary = [
        decision
        for decision in members
        if decision["member_stratum"] == "primary_non_sec"
    ]
    sec = [
        decision
        for decision in members
        if decision["member_stratum"] == "sec_comparator"
    ]
    return {
        "protocol_version": PROTOCOL_VERSION,
        "source_id": "NFET",
        "source_protocol": {
            "path": str(SOURCE_PROTOCOL),
            "sha256": SOURCE_PROTOCOL_SHA256,
            "parser_path": str(nfet.SCRIPT_PATH),
            "parser_sha256": SOURCE_PARSER_SHA256,
        },
        "candidate_access": {
            "path": str(ACCESS_SEAL),
            "sha256": ACCESS_SEAL_SHA256,
            "manifest_hash": ACCESS_MANIFEST_HASH,
            "builder_path": str(ACCESS_BUILDER),
            "builder_sha256": ACCESS_BUILDER_SHA256,
            "candidate_index_sha256": CANDIDATE_INDEX_SHA256,
        },
        "implementation": {
            "path": str(BUILDER),
            "sha256": sha256_file(BUILDER),
        },
        "boundary": {
            "final_exact_member_incidence_opened": True,
            "semantic_model_opened": False,
            "market_clocks_opened": False,
            "outcomes_opened": False,
            "forbidden_sources_opened": [],
        },
        "retention_contract": {
            "all_candidates": [
                "GovInfo MODS",
                "raw GovInfo HTML",
                "canonical visible text",
                "FederalRegister detail JSON",
            ],
            "positive_only": ["GovInfo PDF", "exact match records"],
            "negative_pdf_requests": 0,
            "govinfo_identity_and_agency_reconciled_for_positive_members": True,
            "detail_identity_and_agency_reconciled_for_positive_members": True,
            "raw_html_role": "transport audit only",
            "canonical_text_role": "membership identity",
            "detail_corrections_role": (
                "positive relationship fields are schema-validated; all raw detail "
                "objects remain hash-bound audit evidence, and relationships never "
                "enter structured candidate decisions, features, or events"
            ),
            "storage": "content-addressed deterministic gzip mtime=0",
            "disk_used_abort_gib": cfg.disk_used_abort_gib,
        },
        "candidate_count": len(decisions),
        "incidence": {
            "exact_members": len(members),
            "primary_non_sec_documents": len(primary),
            "sec_comparator_documents": len(sec),
            "nonmembers": len(decisions) - len(members),
        },
        "network_responses": {
            "count": len(responses),
            "by_kind": dict(sorted(response_kinds.items())),
        },
        "outputs": {
            "candidate_decisions": dict(decisions_record),
            "selected_primary_events": dict(selected_record),
            "sec_comparator_events": dict(sec_record),
        },
        "source_quality_status": quality["status"],
        "source_quality_failed_gates": quality["failed_gates"],
        "support_result_path": _recorded_path(cfg.support_result),
        "next_authorized_stage": (
            "outcome-blind novelty access seal"
            if quality["status"] == "PASS"
            else "retire NFET without repair"
        ),
    }


def _support_core(
    cfg: Config,
    *,
    source_manifest: Mapping[str, Any],
    quality_contract: Mapping[str, Any],
    quality: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "protocol_version": SUPPORT_VERSION,
        "source_id": "NFET",
        "source_manifest": {
            "path": _recorded_path(cfg.source_manifest),
            "sha256": sha256_file(cfg.source_manifest),
            "manifest_hash": source_manifest["manifest_hash"],
        },
        "outcomes_opened": False,
        "market_clocks_opened": False,
        "semantic_model_opened": False,
        "quality_contract": dict(quality_contract),
        "status": quality["status"],
        "failure_effect": quality["failure_effect"],
        "failed_gates": quality["failed_gates"],
        "gates": quality["gates"],
        "metrics": quality["metrics"],
        "next_authorized_stage": (
            "outcome-blind novelty access seal"
            if quality["status"] == "PASS"
            else "retire NFET without repair"
        ),
    }


def _read_output(
    record: Mapping[str, Any], *, expected_path: str | Path
) -> tuple[bytes, bytes, list[dict[str, Any]]]:
    if set(record) != {
        "gzip_bytes",
        "gzip_sha256",
        "path",
        "raw_sha256",
        "rows",
    } or record.get("path") != _recorded_path(expected_path):
        raise RuntimeError("NFET membership output metadata changed")
    path = _path(str(record["path"]))
    encoded = path.read_bytes()
    if len(encoded) != record.get("gzip_bytes") or access.sha256_bytes(
        encoded
    ) != record.get("gzip_sha256"):
        raise RuntimeError("NFET membership output gzip changed")
    raw = gzip.decompress(encoded)
    if access.sha256_bytes(raw) != record.get("raw_sha256"):
        raise RuntimeError("NFET membership output raw hash changed")
    lines = raw.splitlines()
    if len(lines) != record.get("rows") or any(not line for line in lines):
        raise RuntimeError("NFET membership output row count changed")
    rows: list[dict[str, Any]] = []
    for line in lines:
        row = json.loads(line)
        if not isinstance(row, dict) or canonical_json_bytes(row) != line:
            raise RuntimeError("NFET membership output row is not canonical")
        rows.append(row)
    return raw, encoded, rows


def validate_source_manifest(
    cfg: Config,
    payload: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any] | None = None,
    candidates: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if canonical_hash(core) != payload.get("manifest_hash"):
        raise RuntimeError("NFET source manifest hash changed")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("NFET source manifest protocol changed")
    if payload.get("source_protocol") != {
        "path": str(SOURCE_PROTOCOL),
        "sha256": SOURCE_PROTOCOL_SHA256,
        "parser_path": str(nfet.SCRIPT_PATH),
        "parser_sha256": SOURCE_PARSER_SHA256,
    }:
        raise RuntimeError("NFET source manifest protocol binding changed")
    if payload.get("candidate_access") != {
        "path": str(ACCESS_SEAL),
        "sha256": ACCESS_SEAL_SHA256,
        "manifest_hash": ACCESS_MANIFEST_HASH,
        "builder_path": str(ACCESS_BUILDER),
        "builder_sha256": ACCESS_BUILDER_SHA256,
        "candidate_index_sha256": CANDIDATE_INDEX_SHA256,
    }:
        raise RuntimeError("NFET source manifest access binding changed")
    if payload.get("implementation") != {
        "path": str(BUILDER),
        "sha256": sha256_file(BUILDER),
    }:
        raise RuntimeError("NFET source manifest implementation changed")
    if payload.get("boundary") != {
        "final_exact_member_incidence_opened": True,
        "semantic_model_opened": False,
        "market_clocks_opened": False,
        "outcomes_opened": False,
        "forbidden_sources_opened": [],
    }:
        raise RuntimeError("NFET source manifest crossed its boundary")
    loaded_protocol, loaded_candidates = (
        (dict(protocol), list(candidates))
        if protocol is not None and candidates is not None
        else _load_inputs()
    )
    outputs = payload.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != {
        "candidate_decisions",
        "sec_comparator_events",
        "selected_primary_events",
    }:
        raise RuntimeError("NFET source manifest outputs changed")
    _, _, decisions = _read_output(
        outputs["candidate_decisions"], expected_path=cfg.decisions
    )
    if len(decisions) != len(loaded_candidates) or payload.get(
        "candidate_count"
    ) != len(decisions):
        raise RuntimeError("NFET source manifest candidate count changed")
    for decision, candidate in zip(decisions, loaded_candidates, strict=True):
        _validate_decision(cfg, decision, expected_candidate=candidate)
    primary_events = build_events(decisions, stratum="primary_non_sec")
    sec_events = build_events(decisions, stratum="sec_comparator")
    primary_raw, primary_gzip = _jsonl_bytes(primary_events)
    sec_raw, sec_gzip = _jsonl_bytes(sec_events)
    actual_primary_raw, actual_primary_gzip, actual_primary = _read_output(
        outputs["selected_primary_events"], expected_path=cfg.selected_source
    )
    actual_sec_raw, actual_sec_gzip, actual_sec = _read_output(
        outputs["sec_comparator_events"], expected_path=cfg.sec_events
    )
    if (
        actual_primary != primary_events
        or actual_sec != sec_events
        or actual_primary_raw != primary_raw
        or actual_primary_gzip != primary_gzip
        or actual_sec_raw != sec_raw
        or actual_sec_gzip != sec_gzip
    ):
        raise RuntimeError("NFET source event deterministic replay changed")
    quality_contract = loaded_protocol["source_quality_gates"]
    quality = evaluate_source_quality(decisions, quality_contract)
    expected_incidence = {
        "exact_members": quality["metrics"]["exact_members"],
        "primary_non_sec_documents": quality["metrics"]["primary_non_sec"]["documents"],
        "sec_comparator_documents": quality["metrics"]["sec_comparator"]["documents"],
        "nonmembers": quality["metrics"]["nonmembers"],
    }
    if (
        payload.get("incidence") != expected_incidence
        or payload.get("source_quality_status") != quality["status"]
        or payload.get("source_quality_failed_gates") != quality["failed_gates"]
    ):
        raise RuntimeError("NFET source manifest incidence or quality changed")
    expected_urls = set()
    replayed_responses: dict[str, Mapping[str, Any]] = {}
    referenced_paths = set()
    kind_counts = Counter()
    for decision in decisions:
        candidate = decision["candidate"]
        urls = nfet.govinfo_document_urls(
            candidate["publication_date"], candidate["document_number"]
        )
        expected_urls.update(
            (urls["mods"], urls["html"], detail_url(candidate["document_number"]))
        )
        receipts = decision["receipts"]
        for name in ("mods", "html", "detail", "canonical_text"):
            referenced_paths.add(receipts[name]["object_path"])
        for name in ("mods", "html", "detail"):
            receipt = receipts[name]
            replayed_responses[receipt["url"]] = receipt
        kind_counts.update(("mods_xml", "html_raw", "detail_json"))
        if decision["member"]:
            expected_urls.add(urls["pdf"])
            referenced_paths.add(receipts["pdf"]["object_path"])
            referenced_paths.add(receipts["matches"]["object_path"])
            replayed_responses[receipts["pdf"]["url"]] = receipts["pdf"]
            kind_counts["pdf"] += 1
    actual_objects = {
        _recorded_path(path)
        for path in _path(cfg.archive_root).glob("objects/*/*.gz")
        if path.is_file()
    }
    if any(_path(cfg.archive_root).rglob("*.tmp")):
        raise RuntimeError("NFET membership archive has incomplete temporary files")
    if actual_objects != referenced_paths:
        raise RuntimeError(
            "NFET membership archive has missing or unreferenced objects"
        )
    network = payload.get("network_responses")
    expected_network = {
        "count": len(expected_urls),
        "by_kind": dict(sorted(kind_counts.items())),
    }
    if network != expected_network:
        raise RuntimeError("NFET source manifest network envelope changed")
    if set(replayed_responses) != expected_urls:
        raise RuntimeError("NFET source manifest response URLs changed")
    expected_core = _manifest_core(
        cfg,
        decisions_record=outputs["candidate_decisions"],
        selected_record=outputs["selected_primary_events"],
        sec_record=outputs["sec_comparator_events"],
        decisions=decisions,
        responses=replayed_responses,
        quality=quality,
    )
    if core != expected_core:
        raise RuntimeError("NFET source manifest deterministic replay changed")
    return decisions, quality


def validate_support_result(
    cfg: Config,
    payload: Mapping[str, Any],
    *,
    source_manifest: Mapping[str, Any],
    quality_contract: Mapping[str, Any],
    quality: Mapping[str, Any],
) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if canonical_hash(core) != payload.get("manifest_hash"):
        raise RuntimeError("NFET support result hash changed")
    expected_core = _support_core(
        cfg,
        source_manifest=source_manifest,
        quality_contract=quality_contract,
        quality=quality,
    )
    if core != expected_core:
        raise RuntimeError("NFET support result deterministic replay changed")


def build(
    cfg: Config,
    *,
    fetch: Fetch | None = None,
    now: Now = lambda: datetime.now(timezone.utc),
    progress: Callable[[int, int, Mapping[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol, candidates = _load_inputs()
    if cfg.disk_used_abort_gib != 300:
        raise ValueError("NFET membership disk limit differs from the freeze")
    if cfg.maximum_retries < 0 or cfg.timeout_seconds <= 0:
        raise ValueError("NFET membership request configuration is invalid")
    manifest_path = _path(cfg.source_manifest)
    support_path = _path(cfg.support_result)
    if manifest_path.exists():
        source_manifest = _load_json_object(
            manifest_path.read_bytes(), label="source manifest"
        )
        decisions, quality = validate_source_manifest(
            cfg,
            source_manifest,
            protocol=protocol,
            candidates=candidates,
        )
        del decisions
        if not support_path.exists():
            support_core = _support_core(
                cfg,
                source_manifest=source_manifest,
                quality_contract=protocol["source_quality_gates"],
                quality=quality,
            )
            support = {
                **support_core,
                "manifest_hash": canonical_hash(support_core),
            }
            _write_frozen(
                cfg.support_result,
                json.dumps(
                    support, indent=2, ensure_ascii=False, allow_nan=False
                ).encode()
                + b"\n",
            )
        else:
            support = _load_json_object(
                support_path.read_bytes(), label="support result"
            )
        validate_support_result(
            cfg,
            support,
            source_manifest=source_manifest,
            quality_contract=protocol["source_quality_gates"],
            quality=quality,
        )
        _path(cfg.resume_state).unlink(missing_ok=True)
        return source_manifest, support
    if support_path.exists():
        raise RuntimeError("NFET support result exists without source manifest")

    access.ensure_disk_budget(cfg.archive_root, abort_gib=cfg.disk_used_abort_gib)
    state = _load_state(cfg, allow_existing=cfg.resume_existing)
    network_cfg = _network_config(cfg)
    effective_fetch = fetch or (lambda url: access._http_fetch(network_cfg, url))
    decisions: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        decision = _process_candidate(
            cfg,
            state,
            candidate,
            fetch=effective_fetch,
            now=now,
        )
        decisions.append(decision)
        if progress is not None:
            progress(index, len(candidates), decision)

    primary_events = build_events(decisions, stratum="primary_non_sec")
    sec_events = build_events(decisions, stratum="sec_comparator")
    quality_contract = protocol["source_quality_gates"]
    quality = evaluate_source_quality(decisions, quality_contract)

    decisions_raw, decisions_gzip = _jsonl_bytes(decisions)
    selected_raw, selected_gzip = _jsonl_bytes(primary_events)
    sec_raw, sec_gzip = _jsonl_bytes(sec_events)
    _write_frozen(cfg.decisions, decisions_gzip)
    _write_frozen(cfg.selected_source, selected_gzip)
    _write_frozen(cfg.sec_events, sec_gzip)
    decisions_record = _output_record(
        cfg.decisions, decisions_raw, decisions_gzip, len(decisions)
    )
    selected_record = _output_record(
        cfg.selected_source, selected_raw, selected_gzip, len(primary_events)
    )
    sec_record = _output_record(cfg.sec_events, sec_raw, sec_gzip, len(sec_events))

    manifest_core = _manifest_core(
        cfg,
        decisions_record=decisions_record,
        selected_record=selected_record,
        sec_record=sec_record,
        decisions=decisions,
        responses=state["responses"],
        quality=quality,
    )
    source_manifest = {
        **manifest_core,
        "manifest_hash": canonical_hash(manifest_core),
    }
    validate_source_manifest(
        cfg,
        source_manifest,
        protocol=protocol,
        candidates=candidates,
    )
    _write_frozen(
        cfg.source_manifest,
        json.dumps(
            source_manifest, indent=2, ensure_ascii=False, allow_nan=False
        ).encode()
        + b"\n",
    )
    support_core = _support_core(
        cfg,
        source_manifest=source_manifest,
        quality_contract=quality_contract,
        quality=quality,
    )
    support = {**support_core, "manifest_hash": canonical_hash(support_core)}
    validate_support_result(
        cfg,
        support,
        source_manifest=source_manifest,
        quality_contract=quality_contract,
        quality=quality,
    )
    _write_frozen(
        cfg.support_result,
        json.dumps(support, indent=2, ensure_ascii=False, allow_nan=False).encode()
        + b"\n",
    )
    _path(cfg.resume_state).unlink(missing_ok=True)
    return source_manifest, support


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=Config.archive_root)
    parser.add_argument("--decisions", type=Path, default=Config.decisions)
    parser.add_argument("--sec-events", type=Path, default=Config.sec_events)
    parser.add_argument("--selected-source", type=Path, default=Config.selected_source)
    parser.add_argument("--resume-state", type=Path, default=Config.resume_state)
    parser.add_argument("--source-manifest", type=Path, default=Config.source_manifest)
    parser.add_argument("--support-result", type=Path, default=Config.support_result)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="explicitly authorize reuse of the hash-verified resume state",
    )
    args = parser.parse_args()
    cfg = Config(
        archive_root=args.archive_root,
        decisions=args.decisions,
        sec_events=args.sec_events,
        selected_source=args.selected_source,
        resume_state=args.resume_state,
        source_manifest=args.source_manifest,
        support_result=args.support_result,
        resume_existing=args.resume,
    )
    existed_before = _path(cfg.source_manifest).exists()

    def report(index: int, total: int, decision: Mapping[str, Any]) -> None:
        if index == 1 or index % 10 == 0 or index == total:
            print(
                json.dumps(
                    {
                        "progress": f"{index}/{total}",
                        "document_number": decision["candidate"]["document_number"],
                        "member": decision["member"],
                        "member_stratum": decision["member_stratum"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    source_manifest, support = build(cfg, progress=report)
    print(
        json.dumps(
            {
                "status": "verified_existing" if existed_before else "created",
                "candidate_count": source_manifest["candidate_count"],
                "incidence": source_manifest["incidence"],
                "source_quality_status": support["status"],
                "failed_gates": support["failed_gates"],
                "manifest_hash": source_manifest["manifest_hash"],
                "outcomes_opened": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
