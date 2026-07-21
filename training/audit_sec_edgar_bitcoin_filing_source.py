"""Build and audit an outcome-blind SEC EDGAR Bitcoin filing source clock."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import time
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Sequence
from zoneinfo import ZoneInfo


PROTOCOL_VERSION = "sec_edgar_bitcoin_8k_6k_source_audit_v1"
AS_OF_DATE = "2026-07-21"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AUDITOR_SOURCE = Path("training/audit_sec_edgar_bitcoin_filing_source.py")
DEFAULT_SOURCE = Path("data/sec_edgar_bitcoin_8k_6k_source_2018_2023.jsonl.gz")
DEFAULT_REPORT = Path("results/sec_edgar_bitcoin_8k_6k_source_audit_2026-07-21.json")

EFTS_ENDPOINT = "https://efts.sec.gov/LATEST/search-index"
SUBMISSIONS_BASE = "https://data.sec.gov/submissions"
ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"
OFFICIAL_DOCS = {
    "api": "https://www.sec.gov/edgar/sec-api-documentation",
    "access": (
        "https://www.sec.gov/search-filings/edgar-search-assistance/"
        "accessing-edgar-data"
    ),
    "submit": (
        "https://www.sec.gov/submit-filings/filer-support-resources/"
        "how-do-i-guides/attach-submit-filing-through-edgar-filing-website"
    ),
    "adjust": (
        "https://www.sec.gov/submit-filings/filer-support-resources/"
        "how-do-i-guides/request-filing-date-adjustment"
    ),
    "full_text": "https://www.sec.gov/edgar/search/index.html",
}
QUERY = {
    "q": "bitcoin",
    "forms": "8-K,6-K",
    "startdt": "2018-01-01",
    "enddt": "2023-12-31",
    "sort": "asc",
}
PAGE_SIZE = 100
ALLOWED_FORMS = frozenset({"8-K", "8-K/A", "6-K", "6-K/A"})
EMITTABLE_FORMS = frozenset({"8-K", "6-K"})
YEARS = tuple(str(year) for year in range(2018, 2024))
PERIODS = {
    "source_train": frozenset({"2018", "2019", "2020"}),
    "source_test": frozenset({"2021", "2022"}),
    "selection": frozenset({"2023"}),
}
DEFAULT_THRESHOLDS: Mapping[str, float | int] = {
    "source_train_min_accessions": 350,
    "source_train_min_days": 250,
    "source_test_min_accessions": 1_000,
    "source_test_min_days": 400,
    "selection_min_accessions": 600,
    "selection_min_days": 200,
    "all_min_documents": 3_000,
    "all_min_ciks": 250,
    "period_max_top1_share": 0.10,
    "period_max_top5_share": 0.36,
    "period_max_hhi": 0.04,
}
ACCESSION_RE = re.compile(r"^\d{10}-\d{2}-\d{6}$")
DOCUMENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")
CIK_RE = re.compile(r"^\d{10}$")
HEADER_ACCEPTANCE_RE = re.compile(br"<ACCEPTANCE-DATETIME>(\d{14})")
Fetch = Callable[[str], bytes]


@dataclass(frozen=True)
class Config:
    user_agent: str
    source_output: Path = DEFAULT_SOURCE
    report_output: Path = DEFAULT_REPORT
    min_interval_seconds: float = 0.11


class RateLimitedFetcher:
    def __init__(self, user_agent: str, min_interval_seconds: float) -> None:
        if not user_agent.strip() or "@" not in user_agent:
            raise ValueError("SEC user agent must declare a contact address")
        if min_interval_seconds < 0.1:
            raise ValueError("SEC request interval must respect the 10 requests/second cap")
        self.user_agent = user_agent
        self.min_interval_seconds = min_interval_seconds
        self.last_request = 0.0

    def __call__(self, url: str) -> bytes:
        delay = self.min_interval_seconds - (time.monotonic() - self.last_request)
        if delay > 0:
            time.sleep(delay)
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": self.user_agent,
                "Accept-Encoding": "identity",
            },
        )
        with urllib.request.urlopen(request, timeout=45) as response:  # noqa: S310
            payload = response.read()
        self.last_request = time.monotonic()
        return payload


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


def _json(raw: bytes, source: str) -> Mapping[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{source} returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{source} root must be an object")
    return value


def _query_url(offset: int) -> str:
    from urllib.parse import urlencode

    return f"{EFTS_ENDPOINT}?{urlencode({**QUERY, 'from': offset, 'size': PAGE_SIZE})}"


def _parse_total(value: Any) -> int:
    if isinstance(value, dict):
        if value.get("relation") != "eq":
            raise ValueError("EFTS total is not exact")
        value = value.get("value")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("EFTS total must be a positive exact integer")
    return value


def parse_search_page(raw: bytes) -> tuple[int, list[dict[str, Any]]]:
    payload = _json(raw, "SEC EFTS")
    query = payload.get("query")
    expected_sort = [
        {"file_date": {"order": "asc"}},
        {"_id": {"order": "asc"}},
    ]
    if not isinstance(query, dict) or query.get("sort") != expected_sort:
        raise ValueError("SEC EFTS did not honor the deterministic sort contract")
    hits_root = payload.get("hits")
    if not isinstance(hits_root, dict):
        raise ValueError("SEC EFTS hits shape drift")
    total = _parse_total(hits_root.get("total"))
    hits = hits_root.get("hits")
    if not isinstance(hits, list):
        raise ValueError("SEC EFTS page hits must be a list")
    rows: list[dict[str, Any]] = []
    prior_key: tuple[str, str] | None = None
    for hit in hits:
        if not isinstance(hit, dict) or not isinstance(hit.get("_source"), dict):
            raise ValueError("SEC EFTS hit shape drift")
        identity = hit.get("_id")
        if not isinstance(identity, str) or identity.count(":") != 1:
            raise ValueError("SEC EFTS hit identity drift")
        accession, document = identity.split(":", 1)
        source = hit["_source"]
        ciks = source.get("ciks")
        form = source.get("form")
        file_date = source.get("file_date")
        if (
            ACCESSION_RE.fullmatch(accession) is None
            or DOCUMENT_RE.fullmatch(document) is None
            or not isinstance(ciks, list)
            or not ciks
            or any(not isinstance(cik, str) or CIK_RE.fullmatch(cik) is None for cik in ciks)
            or form not in ALLOWED_FORMS
            or not isinstance(file_date, str)
            or file_date[:4] not in YEARS
        ):
            raise ValueError("SEC EFTS source row schema drift")
        key = (file_date, identity)
        if prior_key is not None and key <= prior_key:
            raise ValueError("SEC EFTS page ordering drift")
        prior_key = key
        rows.append(
            {
                "accession": accession,
                "document": document,
                "file_date": file_date,
                "search_form": form,
                "ciks": sorted(set(ciks)),
                "sequence": source.get("sequence"),
                "file_description": source.get("file_description"),
            }
        )
    return total, rows


def download_search_rows(fetch: Fetch) -> tuple[list[dict[str, Any]], int, int]:
    rows: list[dict[str, Any]] = []
    total: int | None = None
    offset = 0
    calls = 0
    prior_key: tuple[str, str, str] | None = None
    while total is None or offset < total:
        page_total, page_rows = parse_search_page(fetch(_query_url(offset)))
        calls += 1
        if total is None:
            total = page_total
            if total > 10_000:
                raise RuntimeError("SEC EFTS source exceeds the bounded query contract")
        elif page_total != total:
            raise RuntimeError("SEC EFTS total changed during pagination")
        if not page_rows:
            raise RuntimeError("SEC EFTS pagination ended before the exact total")
        for row in page_rows:
            key = (
                str(row["file_date"]),
                str(row["accession"]),
                str(row["document"]),
            )
            if prior_key is not None and key <= prior_key:
                raise RuntimeError("SEC EFTS cross-page ordering drift")
            prior_key = key
        rows.extend(page_rows)
        offset += len(page_rows)
    if total is None or len(rows) != total:
        raise RuntimeError("SEC EFTS returned a non-exact row count")
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["accession"]), str(row["document"]))
        prior = unique.get(key)
        if prior is not None and prior != row:
            raise RuntimeError("SEC EFTS duplicate document metadata disagrees")
        unique[key] = row
    return (
        sorted(
            unique.values(), key=lambda row: (row["accession"], row["document"])
        ),
        calls,
        total,
    )


def _columns(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    filings = payload.get("filings")
    if isinstance(filings, dict) and isinstance(filings.get("recent"), dict):
        return filings["recent"]
    return payload


def _ingest_acceptance_columns(
    payload: Mapping[str, Any],
    targets: set[str],
    resolved: MutableMapping[str, tuple[str, str]],
) -> None:
    columns = _columns(payload)
    accessions = columns.get("accessionNumber")
    timestamps = columns.get("acceptanceDateTime")
    forms = columns.get("form")
    if (
        not isinstance(accessions, list)
        or not isinstance(timestamps, list)
        or not isinstance(forms, list)
    ):
        raise ValueError("SEC submissions column shape drift")
    if len(accessions) != len(timestamps) or len(accessions) != len(forms):
        raise ValueError("SEC submissions column lengths disagree")
    for accession, timestamp, form in zip(accessions, timestamps, forms):
        if not isinstance(accession, str) or not isinstance(form, str):
            raise ValueError("SEC submissions accession or form type drift")
        if accession not in targets:
            continue
        if not isinstance(timestamp, str) or not timestamp.endswith("Z"):
            raise ValueError("SEC acceptance timestamp is not UTC")
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if form not in ALLOWED_FORMS:
            raise ValueError("SEC submissions form drift")
        value = (timestamp, form)
        prior = resolved.get(accession)
        if prior is not None and prior != value:
            raise RuntimeError("SEC submissions metadata disagrees across CIKs")
        resolved[accession] = value


def resolve_acceptance(
    rows: Sequence[Mapping[str, Any]], fetch: Fetch
) -> tuple[dict[str, tuple[str, str]], int, int]:
    targets_by_cik: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        for cik in row["ciks"]:
            targets_by_cik[str(cik)].add(str(row["accession"]))
    resolved: dict[str, tuple[str, str]] = {}
    calls = 0
    supplemental_calls = 0
    for cik, targets in sorted(targets_by_cik.items()):
        main = _json(
            fetch(f"{SUBMISSIONS_BASE}/CIK{int(cik):010d}.json"),
            "SEC submissions",
        )
        calls += 1
        _ingest_acceptance_columns(main, targets, resolved)
        missing = targets.difference(resolved)
        filings = main.get("filings")
        files = filings.get("files") if isinstance(filings, dict) else None
        if missing and not isinstance(files, list):
            raise ValueError("SEC submissions supplemental file list drift")
        for file_row in files or []:
            if not isinstance(file_row, dict):
                raise ValueError("SEC submissions supplemental metadata drift")
            if file_row.get("filingTo", "") < QUERY["startdt"]:
                continue
            if file_row.get("filingFrom", "") > QUERY["enddt"]:
                continue
            name = file_row.get("name")
            if not isinstance(name, str) or DOCUMENT_RE.fullmatch(name) is None:
                raise ValueError("SEC submissions supplemental filename drift")
            supplement = _json(
                fetch(f"{SUBMISSIONS_BASE}/{name}"), "SEC submissions supplement"
            )
            calls += 1
            supplemental_calls += 1
            _ingest_acceptance_columns(supplement, targets, resolved)
            missing = targets.difference(resolved)
            if not missing:
                break
        if missing:
            raise RuntimeError(
                f"SEC acceptance metadata unresolved for {len(missing)} accessions"
            )
    expected = {str(row["accession"]) for row in rows}
    if set(resolved) != expected:
        raise RuntimeError("SEC acceptance universe does not match EFTS accessions")
    return resolved, calls, supplemental_calls


def attach_acceptance(
    rows: Sequence[Mapping[str, Any]], resolved: Mapping[str, tuple[str, str]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        timestamp, form = resolved[str(row["accession"])]
        if form != row["search_form"]:
            raise RuntimeError("EFTS and submissions form metadata disagree")
        row["acceptance_datetime"] = timestamp
        row["form"] = form
        row["amendment"] = form.endswith("/A")
        output.append(row)
    return sorted(
        output,
        key=lambda row: (
            row["acceptance_datetime"],
            row["accession"],
            row["document"],
        ),
    )


def archive_urls(row: Mapping[str, Any]) -> tuple[str, str]:
    cik = str(int(row["ciks"][0]))
    accession = str(row["accession"])
    directory = accession.replace("-", "")
    base = f"{ARCHIVES_BASE}/{cik}/{directory}"
    return (
        f"{base}/{row['document']}",
        f"{base}/{accession}-index-headers.html",
    )


def _header_acceptance_candidates(raw: bytes) -> dict[str, str]:
    match = HEADER_ACCEPTANCE_RE.search(raw)
    if match is None:
        raise ValueError("SEC archive header lacks acceptance datetime")
    raw_value = match.group(1).decode()
    naive = datetime.strptime(raw_value, "%Y%m%d%H%M%S")
    direct_utc = naive.replace(tzinfo=timezone.utc).isoformat(
        timespec="milliseconds"
    ).replace("+00:00", "Z")
    eastern_utc = naive.replace(
        tzinfo=ZoneInfo("America/New_York")
    ).astimezone(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )
    return {
        "raw": raw_value,
        "direct_utc": direct_utc,
        "eastern_to_utc": eastern_utc,
    }


def probe_archive_samples(
    rows: Sequence[Mapping[str, Any]], fetch: Fetch
) -> tuple[list[dict[str, Any]], int]:
    samples: list[dict[str, Any]] = []
    calls = 0
    for year in YEARS:
        eligible = [
            row
            for row in rows
            if str(row["file_date"]).startswith(year)
            and str(row["document"]).lower().endswith((".htm", ".html", ".txt"))
        ]
        if not eligible:
            raise RuntimeError(f"SEC archive has no probe document for {year}")
        row = eligible[0]
        document_url, header_url = archive_urls(row)
        first = fetch(document_url)
        second = fetch(document_url)
        header = fetch(header_url)
        calls += 3
        first_hash = hashlib.sha256(first).hexdigest()
        second_hash = hashlib.sha256(second).hexdigest()
        header_candidates = _header_acceptance_candidates(header)
        submissions_acceptance = str(row["acceptance_datetime"])
        if submissions_acceptance == header_candidates["direct_utc"]:
            match_mode = "direct_utc"
        elif submissions_acceptance == header_candidates["eastern_to_utc"]:
            match_mode = "eastern_to_utc"
        else:
            match_mode = None
        samples.append(
            {
                "year": year,
                "accession": row["accession"],
                "document": row["document"],
                "document_url": document_url,
                "header_url": header_url,
                "bytes": len(first),
                "sha256": first_hash,
                "double_fetch_hash_stable": first_hash == second_hash,
                "contains_bitcoin": re.search(br"bitcoin", first, re.IGNORECASE)
                is not None,
                "header_acceptance_raw": header_candidates["raw"],
                "header_acceptance_candidates": {
                    "direct_utc": header_candidates["direct_utc"],
                    "eastern_to_utc": header_candidates["eastern_to_utc"],
                },
                "submissions_acceptance_datetime": submissions_acceptance,
                "header_match_mode": match_mode,
                "header_matches_submissions": match_mode is not None,
            }
        )
    return samples, calls


def write_source(rows: Sequence[Mapping[str, Any]], output: str | Path) -> None:
    target = _path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                    ).encode()
                    + b"\n"
                )


def _unique_accession_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    unique: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        accession = str(row["accession"])
        prior = unique.get(accession)
        if prior is not None and (
            prior["acceptance_datetime"] != row["acceptance_datetime"]
            or prior["form"] != row["form"]
            or prior["ciks"] != row["ciks"]
        ):
            raise RuntimeError("SEC accession-level metadata disagrees across documents")
        unique[accession] = row
    return unique


def source_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    unique = _unique_accession_rows(rows)
    emittable = {
        accession: row
        for accession, row in unique.items()
        if str(row["form"]) in EMITTABLE_FORMS
    }
    periods: dict[str, Any] = {}
    for name, years in PERIODS.items():
        selected = [
            row
            for row in emittable.values()
            if str(row["acceptance_datetime"])[:4] in years
        ]
        cik_counts: defaultdict[str, float] = defaultdict(float)
        for row in selected:
            ciks = list(row["ciks"])
            for cik in ciks:
                cik_counts[str(cik)] += 1 / len(ciks)
        shares = sorted(
            (count / len(selected) for count in cik_counts.values()), reverse=True
        )
        periods[name] = {
            "years": sorted(years),
            "accessions": len(selected),
            "event_days": len(
                {str(row["acceptance_datetime"])[:10] for row in selected}
            ),
            "ciks": len(cik_counts),
            "top1_share": shares[0],
            "top5_share": sum(shares[:5]),
            "hhi": sum(share * share for share in shares),
        }
    year_counts = Counter(
        str(row["acceptance_datetime"])[:4] for row in emittable.values()
    )
    return {
        "documents": len(rows),
        "accessions": len(unique),
        "emittable_documents": sum(not bool(row["amendment"]) for row in rows),
        "emittable_accessions": len(emittable),
        "ciks": len(
            {str(cik) for row in emittable.values() for cik in row["ciks"]}
        ),
        "event_days": len(
            {str(row["acceptance_datetime"])[:10] for row in emittable.values()}
        ),
        "amendment_accessions": sum(
            bool(row["amendment"]) for row in unique.values()
        ),
        "filing_date_differs_from_acceptance_utc_date": sum(
            str(row["file_date"]) != str(row["acceptance_datetime"])[:10]
            for row in unique.values()
        ),
        "year_accessions": dict(sorted(year_counts.items())),
        "first_acceptance": min(
            str(row["acceptance_datetime"]) for row in unique.values()
        ),
        "last_acceptance": max(
            str(row["acceptance_datetime"]) for row in unique.values()
        ),
        "periods": periods,
    }


def evaluate_support_gates(
    metrics: Mapping[str, Any],
    thresholds: Mapping[str, float | int] = DEFAULT_THRESHOLDS,
) -> dict[str, bool]:
    periods = metrics["periods"]
    gates = {
        "source_train_accessions": periods["source_train"]["accessions"]
        >= thresholds["source_train_min_accessions"],
        "source_train_event_days": periods["source_train"]["event_days"]
        >= thresholds["source_train_min_days"],
        "source_test_accessions": periods["source_test"]["accessions"]
        >= thresholds["source_test_min_accessions"],
        "source_test_event_days": periods["source_test"]["event_days"]
        >= thresholds["source_test_min_days"],
        "selection_accessions": periods["selection"]["accessions"]
        >= thresholds["selection_min_accessions"],
        "selection_event_days": periods["selection"]["event_days"]
        >= thresholds["selection_min_days"],
        "document_breadth": metrics["documents"] >= thresholds["all_min_documents"],
        "issuer_breadth": metrics["ciks"] >= thresholds["all_min_ciks"],
    }
    for period, values in periods.items():
        gates[f"{period}_top1_concentration"] = (
            values["top1_share"] <= thresholds["period_max_top1_share"]
        )
        gates[f"{period}_top5_concentration"] = (
            values["top5_share"] <= thresholds["period_max_top5_share"]
        )
        gates[f"{period}_hhi"] = values["hhi"] <= thresholds["period_max_hhi"]
    return gates


def build_report(
    fetch: Fetch,
    source_output: str | Path = DEFAULT_SOURCE,
    thresholds: Mapping[str, float | int] = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    network_calls = 0

    def get(url: str) -> bytes:
        nonlocal network_calls
        network_calls += 1
        return fetch(url)

    docs_raw = {name: get(url) for name, url in OFFICIAL_DOCS.items()}
    docs_text = {name: value.decode("utf-8", "replace").lower() for name, value in docs_raw.items()}
    documentation_checks = {
        "submissions_history_api_documented": "submissions history by filer"
        in docs_text["api"],
        "fair_access_cap_documented": "10 requests/second" in docs_text["access"],
        "accepted_filing_cannot_be_rescinded": "cannot rescind"
        in docs_text["submit"],
        "acceptance_time_is_not_adjustable": "acceptance time" in docs_text["adjust"]
        and "denied" in docs_text["adjust"],
        "full_text_since_2001_documented": "since 2001" in docs_text["full_text"],
    }
    search_rows, search_calls, raw_search_hits = download_search_rows(get)
    resolved, submissions_calls, supplemental_calls = resolve_acceptance(
        search_rows, get
    )
    rows = attach_acceptance(search_rows, resolved)
    write_source(rows, source_output)
    samples, sample_calls = probe_archive_samples(rows, get)
    metrics = source_metrics(rows)
    support_gates = evaluate_support_gates(metrics, thresholds)
    transport_checks = {
        "all_search_hits_are_unique_documents": len(rows) == raw_search_hits,
        "all_accessions_have_acceptance_datetime": metrics["accessions"]
        == len(resolved),
        "all_sample_documents_are_hash_stable": all(
            sample["double_fetch_hash_stable"] for sample in samples
        ),
        "all_sample_documents_contain_bitcoin": all(
            sample["contains_bitcoin"] for sample in samples
        ),
        "all_sample_headers_match_submissions": all(
            sample["header_matches_submissions"] for sample in samples
        ),
        "source_stops_before_2024": metrics["last_acceptance"]
        < "2024-01-01T00:00:00.000Z",
    }
    source_contract_passed = all(
        [
            *documentation_checks.values(),
            *transport_checks.values(),
            *support_gates.values(),
        ]
    )
    report: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "as_of_date": AS_OF_DATE,
        "scope": {
            "source_only": True,
            "candidate_frozen": False,
            "model_or_prompt_opened": False,
            "query": dict(QUERY),
            "emittable_forms": sorted(EMITTABLE_FORMS),
            "amendments_may_not_emit": True,
        },
        "official_documentation": {
            "urls": dict(OFFICIAL_DOCS),
            "sha256": {
                name: hashlib.sha256(value).hexdigest()
                for name, value in docs_raw.items()
            },
            "checks": documentation_checks,
        },
        "transport": {
            "efts_endpoint": EFTS_ENDPOINT,
            "efts_is_discovery_transport_not_runtime_dependency": True,
            "raw_search_hits": raw_search_hits,
            "deduplicated_documents": len(rows),
            "search_calls": search_calls,
            "submissions_calls": submissions_calls,
            "supplemental_submissions_calls": supplemental_calls,
            "archive_sample_calls": sample_calls,
            "checks": transport_checks,
            "samples": samples,
        },
        "source_artifact": {
            "path": str(source_output),
            "sha256": sha256_file(source_output),
            "rows": len(rows),
            "canonical_rows_sha256": canonical_hash(rows),
        },
        "metrics": metrics,
        "thresholds": dict(thresholds),
        "support_gates": support_gates,
        "decision": {
            "status": (
                "passed_for_candidate_preregistration"
                if source_contract_passed
                else "retired_before_candidate_preregistration"
            ),
            "source_contract_passed": source_contract_passed,
            "candidate_preregistration_authorized": source_contract_passed,
            "semantic_model_execution_authorized": False,
            "economic_evaluation_authorized": False,
            "2024_or_later_source_authorized": False,
        },
        "outcome_boundary": {
            "artifact_network_calls": network_calls,
            "prior_interactive_source_calls_at_least": 390,
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
    parser.add_argument("--user-agent", default=os.environ.get("SEC_USER_AGENT", ""))
    parser.add_argument("--source-output", default=str(DEFAULT_SOURCE))
    parser.add_argument("--report-output", default=str(DEFAULT_REPORT))
    parser.add_argument("--min-interval-seconds", type=float, default=0.11)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = Config(
        user_agent=args.user_agent,
        source_output=Path(args.source_output),
        report_output=Path(args.report_output),
        min_interval_seconds=args.min_interval_seconds,
    )
    fetch = RateLimitedFetcher(config.user_agent, config.min_interval_seconds)
    report = build_report(fetch, config.source_output)
    write_report(report, config.report_output)
    print(json.dumps(report["decision"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
