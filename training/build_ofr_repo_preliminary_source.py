"""Build an outcome-blind 2019-2023 OFR preliminary repo source panel."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import tempfile
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


PROTOCOL_VERSION = "ofr_repo_preliminary_source_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path("training/build_ofr_repo_preliminary_source.py")
SOURCE_DECISION = Path(
    "docs/ofr-repo-segmentation-source-axis-decision-2026-07-23.md"
)
SOURCE_DECISION_SHA256 = (
    "6c383ee224c794e0fea7414e5f7c20f523001b869f0e7bf999f800247faf5445"
)
API_HOST = "data.financialresearch.gov"
MNEMONICS_URL = (
    "https://data.financialresearch.gov/v1/metadata/mnemonics?dataset=repo"
)
DATASET_URL = (
    "https://data.financialresearch.gov/v1/series/dataset?"
    "dataset=repo&vintage=p&start_date=2019-01-01&end_date=2023-12-31"
)
START_DATE = date(2019, 1, 1)
END_DATE = date(2023, 12, 31)
UTC = timezone.utc
PRELIMINARY_FEED_FLOOR_UTC = datetime(2020, 9, 10, tzinfo=UTC)
ALLOWED_SEGMENTS = frozenset({"DVP", "GCF", "TRI", "TRIV1"})
ALLOWED_MEASURES = frozenset({"AR", "OV", "TV"})
EXPECTED_METADATA_SERIES = 164
EXPECTED_PRELIMINARY_SERIES = 82
EXPECTED_FINAL_SERIES = 82
EXPECTED_SERIES_BY_SEGMENT = {
    "DVP": 18,
    "GCF": 24,
    "TRI": 20,
    "TRIV1": 20,
}
EXPECTED_SERIES_BY_MEASURE = {
    "AR": 34,
    "OV": 14,
    "TV": 34,
}
TOP_LEVEL_FIELDS = frozenset({"short_name", "long_name", "timeseries"})
SERIES_FIELDS = frozenset({"timeseries", "metadata"})
SUBSERIES_FIELDS = frozenset({"aggregation", "disclosure_edits"})
OBSERVATION_COLUMNS = (
    "mnemonic",
    "observation_date",
    "available_at_utc",
    "value",
    "disclosure_edit",
    "segment",
    "measure",
    "subset",
    "series_name",
)
GZIP_MAGIC = b"\x1f\x8b"
MAX_DECODED_JSON_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class Config:
    output_dir: str = "data/ofr_repo_preliminary_2019_2023"
    fetch: bool = False
    request_timeout_seconds: int = 180


@dataclass(frozen=True)
class FetchResponse:
    body: bytes
    final_url: str
    status: int
    content_type: str
    redirect_chain: tuple[str, ...] = ()


@dataclass(frozen=True)
class SeriesDefinition:
    mnemonic: str
    series_name: str
    segment: str
    measure: str
    subset: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class Observation:
    mnemonic: str
    observation_date: date
    available_at_utc: datetime
    value: Decimal | None
    disclosure_edit: bool
    segment: str
    measure: str
    subset: str
    series_name: str


Fetch = Callable[[str, int], FetchResponse]
Clock = Callable[[], datetime]


def _repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        raise RuntimeError("path must be repository-relative")
    resolved = (REPOSITORY_ROOT / candidate).resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError("path must remain repository-relative") from exc
    return resolved


def _assert_repository_member(path: Path) -> None:
    try:
        path.resolve().relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError("artifact path must remain inside repository") from exc


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _repository_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256_bytes(raw)


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _gzip_bytes(payload: bytes) -> bytes:
    return gzip.compress(payload, compresslevel=9, mtime=0)


def _atomic_write(path: Path, payload: bytes) -> None:
    _assert_repository_member(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("OFR source redirect rejected")


def _default_fetch(url: str, timeout: int) -> FetchResponse:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "rllm-ofr-source-audit/1.0",
        },
    )
    opener = urllib.request.build_opener(_RejectRedirects())
    with opener.open(request, timeout=timeout) as response:
        return FetchResponse(
            body=response.read(),
            final_url=response.geturl(),
            status=int(response.status),
            content_type=str(response.headers.get("Content-Type", "")),
            redirect_chain=(),
        )


def _validate_response(request_url: str, response: FetchResponse) -> None:
    if response.redirect_chain:
        raise RuntimeError("OFR source redirect chain must be empty")
    if response.status != 200:
        raise RuntimeError(f"OFR source HTTP status {response.status}")
    expected = urllib.parse.urlsplit(request_url)
    observed = urllib.parse.urlsplit(response.final_url)
    if observed.scheme != "https" or observed.netloc != API_HOST:
        raise RuntimeError("OFR source redirected outside official host")
    if observed != expected:
        raise RuntimeError("OFR source response URL changed")
    if "json" not in response.content_type.lower():
        raise RuntimeError("OFR source response is not JSON")
    if not response.body:
        raise RuntimeError("OFR source response is empty")


def _request_specs() -> tuple[tuple[str, str], ...]:
    return (("mnemonics", MNEMONICS_URL), ("preliminary", DATASET_URL))


def _raw_paths(root: Path) -> dict[str, Path]:
    paths = {
        "mnemonics": root / "raw/repo_mnemonics.json.gz",
        "preliminary": root / "raw/repo_preliminary_2019_2023.json.gz",
    }
    for path in paths.values():
        _assert_repository_member(path)
    return paths


def _ledger_entry(
    *, name: str, request_url: str, response: FetchResponse, retrieved_at: datetime
) -> dict[str, Any]:
    if retrieved_at.tzinfo is None:
        raise RuntimeError("retrieval clock must be timezone-aware")
    return {
        "name": name,
        "request_url": request_url,
        "final_url": response.final_url,
        "retrieved_at_utc": retrieved_at.astimezone(UTC).isoformat(),
        "http_status": response.status,
        "content_type": response.content_type,
        "redirect_chain": list(response.redirect_chain),
        "bytes": len(response.body),
        "sha256": sha256_bytes(response.body),
    }


def _validate_cached_ledger(
    ledger: Any, payloads: Mapping[str, bytes]
) -> list[dict[str, Any]]:
    specs = _request_specs()
    if not isinstance(ledger, list) or len(ledger) != len(specs):
        raise RuntimeError("cached OFR ledger shape changed")
    required = {
        "name",
        "request_url",
        "final_url",
        "retrieved_at_utc",
        "http_status",
        "content_type",
        "redirect_chain",
        "bytes",
        "sha256",
    }
    validated: list[dict[str, Any]] = []
    for raw, (name, url) in zip(ledger, specs):
        if not isinstance(raw, dict) or set(raw) != required:
            raise RuntimeError("cached OFR ledger schema changed")
        string_fields = {
            "name",
            "request_url",
            "final_url",
            "retrieved_at_utc",
            "content_type",
            "sha256",
        }
        if any(type(raw[field]) is not str for field in string_fields):
            raise RuntimeError("cached OFR ledger field types changed")
        if type(raw["http_status"]) is not int or type(raw["bytes"]) is not int:
            raise RuntimeError("cached OFR ledger field types changed")
        if raw["bytes"] < 0:
            raise RuntimeError("cached OFR ledger byte count changed")
        if raw["name"] != name or raw["request_url"] != url:
            raise RuntimeError("cached OFR ledger identity changed")
        try:
            retrieved = datetime.fromisoformat(str(raw["retrieved_at_utc"]))
        except ValueError as exc:
            raise RuntimeError("cached OFR retrieval timestamp changed") from exc
        if retrieved.tzinfo is None:
            raise RuntimeError("cached OFR retrieval timestamp lost timezone")
        redirect_chain = raw["redirect_chain"]
        if not isinstance(redirect_chain, list) or any(
            not isinstance(item, str) for item in redirect_chain
        ):
            raise RuntimeError("cached OFR redirect chain schema changed")
        payload = payloads[name]
        response = FetchResponse(
            body=payload,
            final_url=raw["final_url"],
            status=raw["http_status"],
            content_type=raw["content_type"],
            redirect_chain=tuple(redirect_chain),
        )
        _validate_response(url, response)
        if raw["bytes"] != len(payload) or raw["sha256"] != sha256_bytes(payload):
            raise RuntimeError("cached OFR source metadata mismatch")
        validated.append(raw)
    return validated


def acquire_sources(
    cfg: Config,
    *,
    fetcher: Fetch = _default_fetch,
    clock: Clock = lambda: datetime.now(UTC),
) -> tuple[dict[str, bytes], list[dict[str, Any]]]:
    root = _repository_path(cfg.output_dir)
    raw_paths = _raw_paths(root)
    ledger_path = root / "raw/fetch_ledger.json"
    _assert_repository_member(ledger_path)
    if all(path.exists() for path in raw_paths.values()) and ledger_path.exists():
        if cfg.fetch:
            raise RuntimeError("frozen OFR source cache exists; refusing refresh")
        payloads = {
            name: gzip.decompress(path.read_bytes())
            for name, path in raw_paths.items()
        }
        ledger = _validate_cached_ledger(
            json.loads(ledger_path.read_text(encoding="utf-8")), payloads
        )
        return payloads, ledger
    if any(path.exists() for path in raw_paths.values()) or ledger_path.exists():
        raise RuntimeError("partial OFR source cache exists")
    if not cfg.fetch:
        raise RuntimeError("source cache absent; rerun with --fetch")
    payloads: dict[str, bytes] = {}
    ledger: list[dict[str, Any]] = []
    for name, url in _request_specs():
        response = fetcher(url, cfg.request_timeout_seconds)
        _validate_response(url, response)
        payloads[name] = response.body
        ledger.append(
            _ledger_entry(
                name=name,
                request_url=url,
                response=response,
                retrieved_at=clock(),
            )
        )
    for name, path in raw_paths.items():
        _atomic_write(path, _gzip_bytes(payloads[name]))
    _atomic_write(ledger_path, _canonical_json(ledger))
    return payloads, ledger


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be an object")
    return value


def _text(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise RuntimeError(f"{label} must be text")
    return value.strip()


def _date(value: Any, label: str) -> date:
    try:
        return date.fromisoformat(_text(value, label))
    except ValueError as exc:
        raise RuntimeError(f"{label} is not an ISO date") from exc


def _decimal(value: Any, label: str, *, optional: bool = False) -> Decimal | None:
    if optional and value is None:
        return None
    if isinstance(value, bool):
        raise RuntimeError(f"{label} must be numeric")
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise RuntimeError(f"{label} must be decimal") from exc
    if not result.is_finite():
        raise RuntimeError(f"{label} must be finite")
    return result


def _decode_json_document(payload: bytes, label: str) -> Any:
    decoded = payload
    if payload.startswith(GZIP_MAGIC):
        try:
            decoded = gzip.decompress(payload)
        except (EOFError, OSError) as exc:
            raise RuntimeError(f"{label} transport gzip is invalid") from exc
        if decoded.startswith(GZIP_MAGIC):
            raise RuntimeError(f"{label} transport gzip is nested")
    if len(decoded) > MAX_DECODED_JSON_BYTES:
        raise RuntimeError(f"{label} decoded JSON exceeds frozen size limit")
    try:
        return json.loads(decoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is invalid JSON") from exc


def _parse_mnemonic(value: str) -> tuple[str, str, str]:
    if not value.startswith("REPO-") or not value.endswith("-P"):
        raise RuntimeError(f"non-preliminary repo mnemonic: {value}")
    middle = value[len("REPO-") : -len("-P")]
    pieces = middle.split("_")
    if len(pieces) < 2:
        raise RuntimeError(f"malformed repo mnemonic: {value}")
    segment, measure = pieces[:2]
    subset = "_".join(pieces[2:]) or "TOTAL"
    if segment not in ALLOWED_SEGMENTS:
        raise RuntimeError(f"unknown repo segment: {segment}")
    if measure not in ALLOWED_MEASURES:
        raise RuntimeError(f"unknown repo measure: {measure}")
    return segment, measure, subset


def _parse_mnemonic_catalog(payload: bytes) -> dict[str, str]:
    document = _decode_json_document(payload, "OFR mnemonic response")
    if not isinstance(document, list) or not document:
        raise RuntimeError("OFR mnemonic response must be a nonempty array")
    output: dict[str, str] = {}
    for value in document:
        row = _mapping(value, "mnemonic row")
        if set(row) != {"mnemonic", "series_name"}:
            raise RuntimeError("OFR mnemonic schema changed")
        mnemonic = _text(row["mnemonic"], "mnemonic")
        series_name = _text(row["series_name"], "series_name")
        if mnemonic in output:
            raise RuntimeError("duplicate OFR mnemonic")
        output[mnemonic] = series_name
    for mnemonic in output:
        if not mnemonic.startswith("REPO-") or not mnemonic.endswith(("-P", "-F")):
            raise RuntimeError(f"foreign mnemonic in repo catalog: {mnemonic}")
    return output


def parse_mnemonics(payload: bytes) -> dict[str, str]:
    catalog = _parse_mnemonic_catalog(payload)
    preliminary = {
        mnemonic: name
        for mnemonic, name in catalog.items()
        if mnemonic.endswith("-P")
    }
    if not preliminary:
        raise RuntimeError("OFR mnemonic response contains no preliminary repo series")
    for mnemonic in preliminary:
        _parse_mnemonic(mnemonic)
    return preliminary


def _vintage_base_name(series_name: str, vintage: str) -> str:
    suffix = f" ({vintage})"
    if not series_name.endswith(suffix):
        raise RuntimeError(f"OFR {vintage.lower()} series name label changed")
    return series_name[: -len(suffix)]


def _validate_preliminary_final_correspondence(
    catalog: Mapping[str, str],
) -> None:
    preliminary = {key: value for key, value in catalog.items() if key.endswith("-P")}
    final = {key: value for key, value in catalog.items() if key.endswith("-F")}
    expected_final = {f"{mnemonic[:-1]}F" for mnemonic in preliminary}
    if set(final) != expected_final:
        raise RuntimeError("OFR preliminary/final mnemonic correspondence changed")
    for mnemonic, preliminary_name in preliminary.items():
        final_mnemonic = f"{mnemonic[:-1]}F"
        if _vintage_base_name(
            preliminary_name, "Preliminary"
        ) != _vintage_base_name(final[final_mnemonic], "Final"):
            raise RuntimeError("OFR preliminary/final series names disagree")


def validate_expected_source_shape(
    mnemonic_payload: bytes, definitions: Sequence[SeriesDefinition]
) -> None:
    """Bind the production build to the source-only metadata shape already audited."""

    catalog = _parse_mnemonic_catalog(mnemonic_payload)
    _validate_preliminary_final_correspondence(catalog)
    preliminary = {mnemonic for mnemonic in catalog if mnemonic.endswith("-P")}
    final = {mnemonic for mnemonic in catalog if mnemonic.endswith("-F")}
    if len(catalog) != EXPECTED_METADATA_SERIES:
        raise RuntimeError(
            "OFR repo metadata series count changed: "
            f"expected={EXPECTED_METADATA_SERIES} observed={len(catalog)}"
        )
    if len(preliminary) != EXPECTED_PRELIMINARY_SERIES:
        raise RuntimeError(
            "OFR preliminary series count changed: "
            f"expected={EXPECTED_PRELIMINARY_SERIES} observed={len(preliminary)}"
        )
    if len(final) != EXPECTED_FINAL_SERIES:
        raise RuntimeError(
            "OFR final series count changed: "
            f"expected={EXPECTED_FINAL_SERIES} observed={len(final)}"
        )
    definition_mnemonics = {row.mnemonic for row in definitions}
    if len(definition_mnemonics) != len(definitions):
        raise RuntimeError("duplicate OFR normalized series definition")
    if definition_mnemonics != preliminary:
        raise RuntimeError("OFR dataset and metadata preliminary series differ")
    segment_counts = Counter(row.segment for row in definitions)
    measure_counts = Counter(row.measure for row in definitions)
    if dict(segment_counts) != EXPECTED_SERIES_BY_SEGMENT:
        raise RuntimeError(
            "OFR preliminary segment counts changed: "
            f"expected={EXPECTED_SERIES_BY_SEGMENT} observed={dict(segment_counts)}"
        )
    if dict(measure_counts) != EXPECTED_SERIES_BY_MEASURE:
        raise RuntimeError(
            "OFR preliminary measure counts changed: "
            f"expected={EXPECTED_SERIES_BY_MEASURE} observed={dict(measure_counts)}"
        )


def _metadata_text(
    metadata: Mapping[str, Any], group: str, field: str, *, allow_empty: bool = False
) -> str:
    section = _mapping(metadata.get(group), f"metadata.{group}")
    return _text(
        section.get(field), f"metadata.{group}.{field}", allow_empty=allow_empty
    )


def _validate_metadata(
    metadata: Mapping[str, Any], *, mnemonic: str, series_name: str
) -> None:
    if _text(metadata.get("mnemonic"), "metadata.mnemonic") != mnemonic:
        raise RuntimeError("OFR metadata mnemonic mismatch")
    if _metadata_text(metadata, "description", "vintage") != "Preliminary":
        raise RuntimeError("OFR series is not preliminary vintage")
    if (
        _metadata_text(metadata, "description", "vintage_approach")
        != "Preliminary"
    ):
        raise RuntimeError("OFR preliminary vintage approach changed")
    if _metadata_text(metadata, "description", "name") != series_name:
        raise RuntimeError("OFR mnemonic and metadata names disagree")
    if _metadata_text(metadata, "schedule", "observation_frequency") != "Daily":
        raise RuntimeError("OFR repo observation frequency changed")
    if _metadata_text(metadata, "schedule", "observation_period") != "Single Day":
        raise RuntimeError("OFR repo observation period changed")
    if _metadata_text(metadata, "release", "frequency") != "Daily":
        raise RuntimeError("OFR repo release frequency changed")
    if (
        _metadata_text(metadata, "release", "long_name")
        != "OFR U.S. Repo Markets Data Release"
    ):
        raise RuntimeError("OFR repo release identity changed")
    if _metadata_text(metadata, "release", "short_name") != "U.S. Repo Markets":
        raise RuntimeError("OFR repo release short name changed")
    if (
        _metadata_text(metadata, "release", "href")
        != "/short-term-funding-monitor/datasets/repo/"
    ):
        raise RuntimeError("OFR repo release href changed")
    unit = _mapping(metadata.get("unit"), "metadata.unit")
    unit_type = _text(unit.get("type"), "metadata.unit.type")
    if unit_type not in {"Rate", "Volume"}:
        raise RuntimeError("OFR repo unit type changed")


def _pairs(value: Any, label: str) -> list[tuple[date, Any]]:
    if not isinstance(value, list):
        raise RuntimeError(f"{label} must be an array")
    rows: list[tuple[date, Any]] = []
    seen: set[date] = set()
    for raw in value:
        if not isinstance(raw, list) or len(raw) != 2:
            raise RuntimeError(f"{label} row must be [date, value]")
        observation_day = _date(raw[0], f"{label} date")
        if observation_day in seen:
            raise RuntimeError(f"duplicate date in {label}")
        seen.add(observation_day)
        rows.append((observation_day, raw[1]))
    if any(current[0] <= previous[0] for previous, current in zip(rows, rows[1:])):
        raise RuntimeError(f"{label} dates must be strictly increasing")
    return rows


def _availability(observation_day: date) -> datetime:
    delayed = datetime.combine(observation_day + timedelta(days=8), time(), UTC)
    return max(delayed, PRELIMINARY_FEED_FLOOR_UTC)


def parse_dataset(
    payload: bytes, mnemonic_names: Mapping[str, str]
) -> tuple[list[SeriesDefinition], list[Observation]]:
    document = _decode_json_document(payload, "OFR dataset response")
    root = _mapping(document, "OFR dataset")
    if set(root) != TOP_LEVEL_FIELDS:
        raise RuntimeError("OFR dataset top-level schema changed")
    if _text(root["short_name"], "short_name") != "U.S. Repo Markets":
        raise RuntimeError("OFR dataset short name changed")
    if _text(root["long_name"], "long_name") != "OFR U.S. Repo Markets Data Release":
        raise RuntimeError("OFR dataset long name changed")
    raw_series = _mapping(root["timeseries"], "timeseries")
    if not raw_series:
        raise RuntimeError("OFR dataset contains no series")
    if set(raw_series) != set(mnemonic_names):
        missing = sorted(set(mnemonic_names) - set(raw_series))
        extra = sorted(set(raw_series) - set(mnemonic_names))
        raise RuntimeError(
            f"OFR preliminary series set changed: missing={missing[:3]} extra={extra[:3]}"
        )
    definitions: list[SeriesDefinition] = []
    observations: list[Observation] = []
    for mnemonic in sorted(raw_series):
        series_name = mnemonic_names[mnemonic]
        segment, measure, subset = _parse_mnemonic(mnemonic)
        raw = _mapping(raw_series[mnemonic], f"series {mnemonic}")
        if set(raw) != SERIES_FIELDS:
            raise RuntimeError("OFR series schema changed")
        metadata = _mapping(raw["metadata"], f"metadata {mnemonic}")
        _validate_metadata(metadata, mnemonic=mnemonic, series_name=series_name)
        timeseries = _mapping(raw["timeseries"], f"timeseries {mnemonic}")
        if "aggregation" not in timeseries or not set(timeseries).issubset(
            SUBSERIES_FIELDS
        ):
            raise RuntimeError("OFR subseries schema changed")
        aggregation = _pairs(timeseries["aggregation"], f"{mnemonic}.aggregation")
        disclosure = _pairs(
            timeseries.get("disclosure_edits", []),
            f"{mnemonic}.disclosure_edits",
        )
        disclosure_dates = {observation_day for observation_day, _ in disclosure}
        if any(value is not None for _, value in disclosure):
            raise RuntimeError("OFR disclosure-edit marker stopped being null")
        definitions.append(
            SeriesDefinition(
                mnemonic=mnemonic,
                series_name=series_name,
                segment=segment,
                measure=measure,
                subset=subset,
                metadata=dict(metadata),
            )
        )
        for observation_day, raw_value in aggregation:
            if observation_day < START_DATE or observation_day > END_DATE:
                raise RuntimeError("OFR observation escaped frozen source window")
            value = _decimal(raw_value, f"{mnemonic} value", optional=True)
            if value is not None and measure in {"OV", "TV", "UV"} and value < 0:
                raise RuntimeError("OFR volume must be nonnegative")
            observations.append(
                Observation(
                    mnemonic=mnemonic,
                    observation_date=observation_day,
                    available_at_utc=_availability(observation_day),
                    value=value,
                    disclosure_edit=observation_day in disclosure_dates,
                    segment=segment,
                    measure=measure,
                    subset=subset,
                    series_name=series_name,
                )
            )
        aggregation_dates = {observation_day for observation_day, _ in aggregation}
        if not disclosure_dates.issubset(aggregation_dates):
            raise RuntimeError("OFR disclosure edit has no aggregation row")
    identities = {(row.mnemonic, row.observation_date) for row in observations}
    if len(identities) != len(observations):
        raise RuntimeError("duplicate OFR normalized observation")
    observations.sort(key=lambda row: (row.observation_date, row.mnemonic))
    return definitions, observations


def build_panel(
    payloads: Mapping[str, bytes]
) -> tuple[list[SeriesDefinition], list[Observation]]:
    if set(payloads) != {"mnemonics", "preliminary"}:
        raise RuntimeError("OFR source payload identities changed")
    names = parse_mnemonics(payloads["mnemonics"])
    definitions, observations = parse_dataset(payloads["preliminary"], names)
    if not definitions or not observations:
        raise RuntimeError("OFR normalized source is empty")
    return definitions, observations


def _observation_row(row: Observation) -> dict[str, str]:
    return {
        "mnemonic": row.mnemonic,
        "observation_date": row.observation_date.isoformat(),
        "available_at_utc": row.available_at_utc.isoformat(),
        "value": "" if row.value is None else format(row.value, "f"),
        "disclosure_edit": "1" if row.disclosure_edit else "0",
        "segment": row.segment,
        "measure": row.measure,
        "subset": row.subset,
        "series_name": row.series_name,
    }


def _csv_bytes(columns: Sequence[str], rows: Iterable[Mapping[str, str]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _definition_row(row: SeriesDefinition) -> dict[str, Any]:
    return {
        "mnemonic": row.mnemonic,
        "series_name": row.series_name,
        "segment": row.segment,
        "measure": row.measure,
        "subset": row.subset,
        "metadata": row.metadata,
    }


def write_outputs(
    cfg: Config,
    definitions: Sequence[SeriesDefinition],
    observations: Sequence[Observation],
    ledger: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    root = _repository_path(cfg.output_dir)
    metadata_path = root / "ofr_repo_preliminary_metadata_2019_2023.json.gz"
    observation_path = root / "ofr_repo_preliminary_observations_2019_2023.csv.gz"
    metadata_payload = _gzip_bytes(
        _canonical_json([_definition_row(row) for row in definitions])
    )
    observation_payload = _gzip_bytes(
        _csv_bytes(
            OBSERVATION_COLUMNS,
            (_observation_row(row) for row in observations),
        )
    )
    _atomic_write(metadata_path, metadata_payload)
    _atomic_write(observation_path, observation_payload)
    series_by_segment = Counter(row.segment for row in definitions)
    series_by_measure = Counter(row.measure for row in definitions)
    rows_by_year = Counter(row.observation_date.year for row in observations)
    retrieval_times = sorted(str(row["retrieved_at_utc"]) for row in ledger)
    core: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "source_decision": {
            "path": str(SOURCE_DECISION),
            "sha256": SOURCE_DECISION_SHA256,
        },
        "builder": {
            "path": str(SCRIPT_PATH),
            "sha256": sha256_file(SCRIPT_PATH),
        },
        # Fetch mode and timeout are transport controls, not source semantics.
        # Excluding them makes a fetched build byte-identical to offline replay.
        "config": {"output_dir": cfg.output_dir},
        "source_window": [START_DATE.isoformat(), END_DATE.isoformat()],
        "availability_policy": {
            "observation_lag_elapsed_days": 8,
            "preliminary_feed_floor_utc": PRELIMINARY_FEED_FLOOR_UTC.isoformat(),
        },
        "official_urls": {
            "mnemonics": MNEMONICS_URL,
            "preliminary_dataset": DATASET_URL,
        },
        "fetch_ledger": list(ledger),
        "generated_from_latest_retrieval_utc": retrieval_times[-1],
        "metadata": {
            "path": str(metadata_path.relative_to(REPOSITORY_ROOT)),
            "sha256": sha256_bytes(metadata_payload),
            "series": len(definitions),
            "series_by_segment": dict(sorted(series_by_segment.items())),
            "series_by_measure": dict(sorted(series_by_measure.items())),
        },
        "observations": {
            "path": str(observation_path.relative_to(REPOSITORY_ROOT)),
            "sha256": sha256_bytes(observation_payload),
            "rows": len(observations),
            "columns": list(OBSERVATION_COLUMNS),
            "first_observation_date": observations[0].observation_date.isoformat(),
            "last_observation_date": observations[-1].observation_date.isoformat(),
            "rows_by_year": {
                str(year): rows_by_year[year]
                for year in range(START_DATE.year, END_DATE.year + 1)
            },
            "null_rows": sum(row.value is None for row in observations),
            "disclosure_edit_rows": sum(
                row.disclosure_edit for row in observations
            ),
            "unique_series_dates": len(
                {(row.mnemonic, row.observation_date) for row in observations}
            ),
        },
        "source_checks": {
            "source_decision_hash_matches": True,
            "all_series_preliminary": all(
                row.mnemonic.endswith("-P")
                and _metadata_text(row.metadata, "description", "vintage")
                == "Preliminary"
                for row in definitions
            ),
            "all_series_daily": all(
                _metadata_text(row.metadata, "schedule", "observation_frequency")
                == "Daily"
                for row in definitions
            ),
            "segments_recognized": all(
                row.segment in ALLOWED_SEGMENTS for row in definitions
            ),
            "measures_recognized": all(
                row.measure in ALLOWED_MEASURES for row in definitions
            ),
            "dates_inside_frozen_window": all(
                START_DATE <= row.observation_date <= END_DATE
                for row in observations
            ),
            "availability_matches_frozen_clock": all(
                row.available_at_utc == _availability(row.observation_date)
                for row in observations
            ),
            "prepublication_rows_not_backdated": all(
                row.available_at_utc >= PRELIMINARY_FEED_FLOOR_UTC
                for row in observations
                if row.observation_date < PRELIMINARY_FEED_FLOOR_UTC.date()
            ),
            "preliminary_final_definitions_correspond": True,
            "series_dates_unique": len(
                {(row.mnemonic, row.observation_date) for row in observations}
            )
            == len(observations),
            "unknown_envelope_fields_rejected": True,
            "final_or_asof_rows_read_zero": True,
        },
        "research_boundary": {
            "metadata_rows_read": len(definitions),
            "preliminary_source_rows_read": len(observations),
            "final_source_rows_read": 0,
            "candidate_features_computed": [],
            "candidate_incidence_opened": False,
            "btc_market_rows_read": 0,
            "funding_rows_read": 0,
            "return_rows_read": 0,
            "pnl_cagr_mdd_opened": False,
        },
        "next_action": (
            "audit source metadata and missingness, then preregister exactly one "
            "candidate before incidence"
        ),
    }
    if not all(core["source_checks"].values()):
        raise RuntimeError("OFR source output invariant failed")
    core["manifest_hash"] = canonical_hash(core)
    _atomic_write(root / "build_manifest.json", _canonical_json(core))
    return core


def build(
    cfg: Config,
    *,
    fetcher: Fetch = _default_fetch,
    clock: Clock = lambda: datetime.now(UTC),
) -> dict[str, Any]:
    if sha256_file(SOURCE_DECISION) != SOURCE_DECISION_SHA256:
        raise RuntimeError("OFR source decision hash mismatch")
    payloads, ledger = acquire_sources(cfg, fetcher=fetcher, clock=clock)
    definitions, observations = build_panel(payloads)
    validate_expected_source_shape(payloads["mnemonics"], definitions)
    return write_outputs(cfg, definitions, observations, ledger)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", default="data/ofr_repo_preliminary_2019_2023"
    )
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--request-timeout-seconds", type=int, default=180)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build(
        Config(
            output_dir=args.output_dir,
            fetch=args.fetch,
            request_timeout_seconds=args.request_timeout_seconds,
        )
    )
    print(
        json.dumps(
            {
                "status": "built",
                "series": report["metadata"]["series"],
                "observations": report["observations"]["rows"],
                "manifest_hash": report["manifest_hash"],
                "candidate_incidence_opened": report["research_boundary"]
                ["candidate_incidence_opened"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
