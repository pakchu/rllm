"""Evaluate only the frozen source-support gates for the GNRC family.

This stage may read the committed 2020-2023 GDELT source artifacts. It must not
open BTC market, funding, return, label, PnL, or post-2023 news data.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import importlib
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

PROTOCOL_VERSION = "gdelt_narrative_rotation_clearing_source_support_v1"
AS_OF_DATE = "2026-07-20"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = Path(
    "results/gdelt_narrative_rotation_clearing_preregistration_2026-07-20.json"
)
PREREGISTRATION_SHA256 = (
    "ae175a242db1fa850164789e4a3e6f3f39b4ac8eae0fb877ce79e915ae3d67f3"
)
PREREGISTRATION_SOURCE = Path(
    "training/preregister_gdelt_narrative_rotation_clearing.py"
)
PREREGISTRATION_SOURCE_SHA256 = (
    "68c8402c4b04f9d301a76bf4ed202d2488154de03365467a13abf435e5ffe587"
)
PREREGISTRATION_DOCUMENT = Path(
    "docs/gdelt-narrative-rotation-clearing-preregistration-2026-07-20.md"
)
PREREGISTRATION_DOCUMENT_SHA256 = (
    "50b1e2550b8ec3e36b4db39873ed404b734d689f3094ff5da1d9d0bb10e2a388"
)
TRANSPORT_AMENDMENT = Path(
    "results/gdelt_source_transport_v2_amendment_2026-07-22.json"
)
TRANSPORT_AMENDMENT_SHA256 = (
    "9244fc5ab203abe1866a1960c9b652ec725a8e37a1196ea5e784c742d1bc9f18"
)
V1_SOURCE = Path("training/download_gdelt_bitcoin_narrative_daily.py")
V1_SOURCE_SHA256 = "d756990d979e901033891ad6a8c565783dc58e8a4a9e286d6e866929dd74889e"
V2_SOURCE = Path("training/download_gdelt_bitcoin_narrative_daily_v2.py")
V2_SOURCE_SHA256 = "10f22a4a7e45080369dd989add6765caac3ad3a91c72f5e4bb26986904671569"
SOURCE_MANIFEST = Path(
    "results/gdelt_bitcoin_narrative_source_manifest_2026-07-20.json"
)
DAILY_SOURCE = Path("data/gdelt_bitcoin_narrative_daily_2020_2023.csv.gz")
RAW_SOURCE = Path("data/gdelt_bitcoin_narrative_timeline_raw_2020_2023.jsonl.gz")
PROTOCOL_DOCUMENT = Path(
    "docs/gdelt-narrative-rotation-clearing-source-support-protocol-2026-07-20.md"
)
EVALUATOR_SOURCE = Path("training/evaluate_gdelt_narrative_source_support.py")
SOURCE_ACCESS_SEAL = Path("results/gdelt_gnrc_source_access_seal_2026-07-22.json")
DEFAULT_OUTPUT = Path(
    "results/gdelt_narrative_rotation_clearing_source_support_2026-07-20.json"
)
EXPECTED_DAILY_ROWS = 1461
EXPECTED_FIRST_DATE = "2020-01-01"
EXPECTED_LAST_DATE = "2023-12-31"
SOURCE_SEAL_FIELDS = (
    "protocol_version",
    "preregistration_path",
    "preregistration_sha256",
    "transport_amendment_path",
    "transport_amendment_sha256",
    "source_manifest_path",
    "source_manifest_sha256",
    "daily_source_path",
    "daily_source_sha256",
    "raw_source_path",
    "raw_source_sha256",
    "evaluator_source_path",
    "evaluator_source_sha256",
    "protocol_document_path",
    "protocol_document_sha256",
    "feature_values_inspected_before_seal",
    "market_outcomes_opened_before_seal",
    "sealed_at",
)


def repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with repository_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    with repository_path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"GNRC JSON input must be an object: {path}")
    return payload


def _bootstrap_frozen_modules() -> tuple[Any, Any]:
    expected_hashes = {
        PREREGISTRATION: PREREGISTRATION_SHA256,
        PREREGISTRATION_SOURCE: PREREGISTRATION_SOURCE_SHA256,
        PREREGISTRATION_DOCUMENT: PREREGISTRATION_DOCUMENT_SHA256,
        TRANSPORT_AMENDMENT: TRANSPORT_AMENDMENT_SHA256,
        V1_SOURCE: V1_SOURCE_SHA256,
        V2_SOURCE: V2_SOURCE_SHA256,
    }
    for path, expected_hash in expected_hashes.items():
        if sha256_file(path) != expected_hash:
            raise RuntimeError(
                f"GNRC frozen executable input changed before import: {path}"
            )
    preregistration = _load_json(PREREGISTRATION)
    prereg_unhashed = dict(preregistration)
    prereg_manifest_hash = prereg_unhashed.pop("manifest_hash", None)
    if prereg_manifest_hash != canonical_hash(prereg_unhashed):
        raise RuntimeError("GNRC preregistration internal hash changed before import")
    if (
        preregistration.get("preregistration_source_sha256")
        != PREREGISTRATION_SOURCE_SHA256
        or preregistration.get("preregistration_document_sha256")
        != PREREGISTRATION_DOCUMENT_SHA256
    ):
        raise RuntimeError("GNRC preregistration executable binding changed")
    amendment = _load_json(TRANSPORT_AMENDMENT)
    amendment_unhashed = dict(amendment)
    amendment_manifest_hash = amendment_unhashed.pop("manifest_hash", None)
    if amendment_manifest_hash != canonical_hash(amendment_unhashed):
        raise RuntimeError(
            "GNRC transport amendment internal hash changed before import"
        )
    if (
        amendment.get("original_preregistration", {}).get("sha256")
        != PREREGISTRATION_SHA256
        or amendment.get("transport", {}).get("v1_builder_sha256") != V1_SOURCE_SHA256
        or amendment.get("transport", {}).get("v2_builder_sha256") != V2_SOURCE_SHA256
    ):
        raise RuntimeError("GNRC transport amendment executable binding changed")
    prereg_module = importlib.import_module(
        "training.preregister_gdelt_narrative_rotation_clearing"
    )
    source_module = importlib.import_module(
        "training.download_gdelt_bitcoin_narrative_daily_v2"
    )
    return prereg_module, source_module


prereg, source = _bootstrap_frozen_modules()


def validate_preregistration() -> dict[str, Any]:
    if sha256_file(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise ValueError("GNRC preregistration artifact hash changed")
    payload = _load_json(PREREGISTRATION)
    manifest_hash = payload.get("manifest_hash")
    unhashed = dict(payload)
    unhashed.pop("manifest_hash", None)
    if manifest_hash != canonical_hash(unhashed):
        raise ValueError("GNRC preregistration manifest hash changed")
    if payload.get("protocol_version") != prereg.PROTOCOL_VERSION:
        raise ValueError("GNRC preregistration protocol changed")
    if tuple(row["variant_id"] for row in payload.get("variants", [])) != (
        prereg.FAMILY_VARIANT_IDS
    ):
        raise ValueError("GNRC preregistration family changed")
    if payload.get("outcome_boundary", {}).get("outcomes_opened") is not False:
        raise ValueError("GNRC preregistration opened an outcome")
    return payload


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_source_access_seal(payload: Mapping[str, Any]) -> None:
    if set(payload) != set(SOURCE_SEAL_FIELDS):
        raise ValueError("GNRC source-access seal fields changed")
    if payload["protocol_version"] != "gdelt_gnrc_source_access_seal_v1":
        raise ValueError("GNRC source-access seal protocol changed")
    expected_paths = {
        "preregistration_path": PREREGISTRATION,
        "transport_amendment_path": TRANSPORT_AMENDMENT,
        "source_manifest_path": SOURCE_MANIFEST,
        "daily_source_path": DAILY_SOURCE,
        "raw_source_path": RAW_SOURCE,
        "evaluator_source_path": EVALUATOR_SOURCE,
        "protocol_document_path": PROTOCOL_DOCUMENT,
    }
    for field, expected in expected_paths.items():
        if payload[field] != str(expected):
            raise ValueError(f"GNRC source-access seal path changed: {field}")
    hash_bindings = {
        "preregistration_sha256": PREREGISTRATION,
        "transport_amendment_sha256": TRANSPORT_AMENDMENT,
        "source_manifest_sha256": SOURCE_MANIFEST,
        "daily_source_sha256": DAILY_SOURCE,
        "raw_source_sha256": RAW_SOURCE,
        "evaluator_source_sha256": EVALUATOR_SOURCE,
        "protocol_document_sha256": PROTOCOL_DOCUMENT,
    }
    for field, path in hash_bindings.items():
        if not _is_sha256(payload[field]) or sha256_file(path) != payload[field]:
            raise ValueError(f"GNRC source-access seal hash changed: {field}")
    if (
        payload["preregistration_sha256"] != PREREGISTRATION_SHA256
        or payload["transport_amendment_sha256"] != TRANSPORT_AMENDMENT_SHA256
    ):
        raise ValueError("GNRC source-access seal ancestry changed")
    if payload["feature_values_inspected_before_seal"] is not False:
        raise ValueError("GNRC source feature values were inspected before sealing")
    if payload["market_outcomes_opened_before_seal"] is not False:
        raise ValueError("GNRC market outcomes were opened before source sealing")
    prereg.parse_utc(str(payload["sealed_at"]))


def validate_source_manifest(source_seal: Mapping[str, Any]) -> dict[str, Any]:
    validate_source_access_seal(source_seal)
    payload = _load_json(SOURCE_MANIFEST)
    manifest_hash = payload.get("manifest_hash")
    unhashed = dict(payload)
    unhashed.pop("manifest_hash", None)
    if manifest_hash != source.canonical_hash(unhashed):
        raise ValueError("GNRC source manifest hash changed")
    if source_seal["source_manifest_sha256"] != sha256_file(SOURCE_MANIFEST):
        raise ValueError("GNRC source manifest differs from its outer seal")
    if payload.get("protocol_version") != source.PROTOCOL_VERSION:
        raise ValueError("GNRC source protocol changed")
    expected_contract = source.source_contract(source.Config())
    if payload.get("contract") != expected_contract:
        raise ValueError("GNRC source contract differs from its amendment")
    if payload.get("contract_hash") != source.canonical_hash(expected_contract):
        raise ValueError("GNRC source contract hash changed")
    if payload.get("builder") != {
        "path": str(source.BUILDER),
        "sha256": V2_SOURCE_SHA256,
        "v1_dependency_path": str(source.V1_DEPENDENCY),
        "v1_dependency_sha256": V1_SOURCE_SHA256,
    }:
        raise ValueError("GNRC source builder identity changed")
    requests = payload.get("requests", {})
    if requests.get("count") != 4 or len(requests.get("response_hashes", [])) != 4:
        raise ValueError("GNRC source request family is incomplete")
    expected_query_ids = {query_id for query_id, _ in source.QUERIES}
    response_hashes = requests["response_hashes"]
    if {row.get("query_id") for row in response_hashes} != expected_query_ids:
        raise ValueError("GNRC source request identities changed")
    if any(
        row.get("start") != source.FROZEN_START_DATE
        or row.get("end_exclusive") != source.FROZEN_END_DATE_EXCLUSIVE
        or not isinstance(row.get("response_sha256"), str)
        or len(row["response_sha256"]) != 64
        or any(
            character not in "0123456789abcdef" for character in row["response_sha256"]
        )
        for row in response_hashes
    ):
        raise ValueError("GNRC source request interval or hash changed")
    audit = payload.get("source_audit", {})
    if (
        audit.get("daily_rows") != EXPECTED_DAILY_ROWS
        or audit.get("first_date") != EXPECTED_FIRST_DATE
        or audit.get("last_date") != EXPECTED_LAST_DATE
        or audit.get("first_available_at") != "2020-01-03T00:15:00Z"
        or audit.get("last_available_at") != "2024-01-02T00:15:00Z"
        or audit.get("date_resolution") != "day"
        or audit.get("global_norm_consistent_across_available_queries") is not True
        or tuple(audit.get("global_outage_dates", []))
        != source.KNOWN_GLOBAL_OUTAGE_DATES
        or audit.get("global_outage_days") != 2
        or audit.get("known_global_outage_dates_match") is not True
        or set(audit.get("missing_bins_by_query", {}))
        != {query_id for query_id, _ in source.QUERIES}
        or audit.get("missing_bins_by_query", {}).get("broad") != 2
        or any(
            not isinstance(value, int) or value < 2
            for value in audit.get("missing_bins_by_query", {}).values()
        )
    ):
        raise ValueError("GNRC source audit is incomplete")
    outputs = payload.get("outputs", {})
    if (
        outputs.get("daily_path") != str(DAILY_SOURCE)
        or outputs.get("raw_bundle_path") != str(RAW_SOURCE)
        or tuple(outputs.get("daily_columns", [])) != source.DAILY_COLUMNS
        or outputs.get("daily_sha256") != source_seal["daily_source_sha256"]
        or outputs.get("raw_bundle_sha256") != source_seal["raw_source_sha256"]
    ):
        raise ValueError("GNRC source output binding changed")
    boundary = payload.get("outcome_boundary", {})
    if boundary != {
        "btc_market_rows_read": 0,
        "funding_rows_read": 0,
        "future_return_rows_read": 0,
        "return_or_pnl_fields_read": 0,
        "post_2023_news_rows_requested": 0,
        "economic_metrics_computed": False,
    }:
        raise ValueError("GNRC source manifest crossed the outcome boundary")
    return payload


def load_daily_rows(path: str | Path = DAILY_SOURCE) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with gzip.open(
            repository_path(path), "rt", encoding="utf-8", newline=""
        ) as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != source.DAILY_COLUMNS:
                raise ValueError("GNRC daily source columns changed")
            for raw in reader:
                if None in raw or any(
                    raw[column] is None for column in source.DAILY_COLUMNS
                ):
                    raise ValueError("GNRC daily source row shape changed")
                row: dict[str, Any] = {
                    "date": raw["date"],
                    "available_at": raw["available_at"],
                }
                for column in source.DAILY_COLUMNS[2:]:
                    value = raw[column]
                    if not value.isdigit() or (value != "0" and value.startswith("0")):
                        raise ValueError("GNRC daily source count is not canonical")
                    row[column] = int(value)
                prereg.validate_count_row_clock(row)
                global_count = row["global_article_count"]
                broad_count = row["broad_article_count"]
                category_counts = [
                    row["failure_article_count"],
                    row["constraint_article_count"],
                    row["adoption_article_count"],
                ]
                if (
                    global_count < 0
                    or broad_count > global_count
                    or any(count > broad_count for count in category_counts)
                    or (
                        global_count == 0
                        and any(count != 0 for count in (broad_count, *category_counts))
                    )
                ):
                    raise ValueError("GNRC daily source subset counts are inconsistent")
                rows.append(row)
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise ValueError("GNRC daily source artifact is unreadable") from error
    expected_dates = [
        date(2020, 1, 1) + timedelta(days=index) for index in range(EXPECTED_DAILY_ROWS)
    ]
    observed_dates = [date.fromisoformat(str(row["date"])) for row in rows]
    if observed_dates != expected_dates:
        raise ValueError("GNRC daily source must equal the complete 2020-2023 grid")
    zero_global_dates = tuple(
        str(row["date"]) for row in rows if row["global_article_count"] == 0
    )
    if zero_global_dates != source.KNOWN_GLOBAL_OUTAGE_DATES:
        raise ValueError("GNRC daily source global outage set changed")
    return rows


def _split_bounds() -> dict[str, tuple[datetime, datetime]]:
    return {
        "train": (
            prereg.parse_utc(prereg.FROZEN_CONFIG.train_start),
            prereg.parse_utc(prereg.FROZEN_CONFIG.train_end_exclusive),
        ),
        "selection": (
            prereg.parse_utc(prereg.FROZEN_CONFIG.selection_start),
            prereg.parse_utc(prereg.FROZEN_CONFIG.selection_end_exclusive),
        ),
    }


def evaluate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != EXPECTED_DAILY_ROWS:
        raise ValueError("GNRC source-support input row count changed")
    row_by_date = {
        date.fromisoformat(str(row["date"])): index for index, row in enumerate(rows)
    }
    if len(row_by_date) != len(rows):
        raise ValueError("GNRC source-support dates are duplicated")
    score_cache: dict[tuple[int, int, date], dict[str, Any]] = {}

    def decision_for(
        source_date: date, score: str, fast_days: int, slow_days: int
    ) -> Any:
        cache_key = (fast_days, slow_days, source_date)
        if cache_key not in score_cache:
            index = row_by_date.get(source_date)
            if index is None or index + 1 < slow_days:
                raise ValueError("GNRC source-support feature history is incomplete")
            score_cache[cache_key] = prereg.compute_score_state(
                rows[index - slow_days + 1 : index + 1], fast_days, slow_days
            )
        state = score_cache[cache_key]
        score_state = state[score]
        return prereg.ScoreDecision(
            source_date=source_date,
            available_at=state["available_at"],
            long_score=score_state["long_score"],
            short_score=score_state["short_score"],
            evidence_ok=state["evidence_ok"],
        )

    support_by_variant: dict[str, dict[str, Any]] = {}
    support_checks: dict[str, dict[str, bool]] = {}
    for variant in prereg.variants():
        split_support: dict[str, dict[str, Any]] = {}
        for split_name, (split_start, split_end) in _split_bounds().items():
            expected_dates = prereg.expected_split_source_dates(
                split_start=split_start,
                split_end_exclusive=split_end,
                hold_days=variant["hold_days"],
            )
            decisions = [
                decision_for(
                    source_date,
                    variant["score"],
                    variant["fast_days"],
                    variant["slow_days"],
                )
                for source_date in expected_dates
            ]
            scheduled = prereg.schedule_events(
                decisions,
                threshold=variant["threshold"],
                hold_days=variant["hold_days"],
                split_start=split_start,
                split_end_exclusive=split_end,
            )
            split_support[split_name] = prereg.support_rates(scheduled)
        checks = prereg.evaluate_variant_support(
            split_support["train"], split_support["selection"]
        )
        variant_id = str(variant["variant_id"])
        support_by_variant[variant_id] = {
            "policy": variant,
            "train": split_support["train"],
            "selection": split_support["selection"],
            "checks": checks,
            "passes": all(checks.values()),
        }
        support_checks[variant_id] = checks
    family = prereg.evaluate_family_support(support_checks)
    return {
        "variant_support": support_by_variant,
        "family_support": family,
        "decision": family["decision"],
    }


def build_report() -> dict[str, Any]:
    source_seal = _load_json(SOURCE_ACCESS_SEAL)
    preregistration = validate_preregistration()
    source_manifest = validate_source_manifest(source_seal)
    rows = load_daily_rows()
    evaluation = evaluate_rows(rows)
    report: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "as_of_date": AS_OF_DATE,
        "family": prereg.FAMILY,
        "preregistration": {
            "path": str(PREREGISTRATION),
            "sha256": PREREGISTRATION_SHA256,
            "manifest_hash": preregistration["manifest_hash"],
        },
        "source_manifest": {
            "path": str(SOURCE_MANIFEST),
            "sha256": sha256_file(SOURCE_MANIFEST),
            "manifest_hash": source_manifest["manifest_hash"],
            "outer_seal_path": str(SOURCE_ACCESS_SEAL),
            "outer_seal_sha256": sha256_file(SOURCE_ACCESS_SEAL),
        },
        "source_artifacts": {
            "daily_path": str(DAILY_SOURCE),
            "daily_sha256": sha256_file(DAILY_SOURCE),
            "daily_rows_read": len(rows),
            "raw_path": str(RAW_SOURCE),
            "raw_sha256": sha256_file(RAW_SOURCE),
            "raw_responses_read": 0,
        },
        "evaluator": {
            "path": str(EVALUATOR_SOURCE),
            "sha256": sha256_file(EVALUATOR_SOURCE),
            "protocol_document": str(PROTOCOL_DOCUMENT),
            "protocol_document_sha256": sha256_file(PROTOCOL_DOCUMENT),
        },
        **evaluation,
        "outcome_boundary": {
            "btc_market_rows_read": 0,
            "funding_rows_read": 0,
            "future_return_rows_read": 0,
            "return_or_pnl_fields_read": 0,
            "post_2023_news_rows_read": 0,
            "economic_metrics_computed": False,
            "outcomes_opened": False,
        },
        "failure_action": "retire_without_threshold_sign_window_or_hold_repair",
    }
    report["manifest_hash"] = canonical_hash(report)
    return report


def write_once(path: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    destination = repository_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(
            f"GNRC source-support report is write-once: {destination}"
        )
    report = build_report()
    try:
        with destination.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except FileExistsError as error:
        raise FileExistsError(
            f"GNRC source-support report is write-once: {destination}"
        ) from error
    return report
