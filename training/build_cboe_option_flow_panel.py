"""Freeze the official Cboe daily option-flow panel used by CIHM-1.

The source is Cboe's date-addressable daily market-statistics page.  This
module parses only option ratios and volumes.  It never reads crypto prices,
funding, returns, portfolio state, or labels.

Cboe does not expose this table as a static bulk CSV.  The frozen panel keeps
the SHA-256 of every HTML response beside its normalized values so a later
vintage can be audited without pretending that the current web page is a
point-in-time archive.  Raw HTML is intentionally not retained.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence


BASE_URL = "https://www.cboe.com/us/options/market_statistics/daily/"
HISTORICAL_URL = "https://www.cboe.com/us/options/market_statistics/historical_data/"
SOURCE_SNAPSHOT_DATE = "2026-07-18"
START_DATE = date(2020, 1, 1)
END_DATE_EXCLUSIVE = date(2024, 1, 1)
SCHEMA_VERSION = 1
FROZEN_COVERAGE = (1_006, "2020-01-02", "2023-12-29")
FROZEN_PANEL_SHA256 = "35ef106ef01e3abadbcb4a6227187dd1d7cf2722191bd146bac06d08d1684a78"

RATIO_NAMES = {
    "total_pcr": "TOTAL PUT/CALL RATIO",
    "index_pcr": "INDEX PUT/CALL RATIO",
    "equity_pcr": "EQUITY PUT/CALL RATIO",
    "vix_pcr": "CBOE VOLATILITY INDEX (VIX) PUT/CALL RATIO",
    "spx_pcr": "SPX + SPXW PUT/CALL RATIO",
}
VOLUME_NAMES = {
    "total": "SUM OF ALL PRODUCTS",
    "index": "INDEX OPTIONS",
    "equity": "EQUITY OPTIONS",
    "vix": "CBOE VOLATILITY INDEX (VIX)",
    "spx": "SPX + SPXW",
}
PANEL_COLUMNS = (
    "observation_date",
    *RATIO_NAMES,
    "total_call_volume",
    "total_put_volume",
    "total_volume",
    "index_call_volume",
    "index_put_volume",
    "index_volume",
    "equity_call_volume",
    "equity_put_volume",
    "equity_volume",
    "vix_call_volume",
    "vix_put_volume",
    "vix_volume",
    "spx_call_volume",
    "spx_put_volume",
    "spx_volume",
    "response_sha256",
)
_FLIGHT_CHUNK = re.compile(
    r'self\.__next_f\.push\(\[1,("(?:\\.|[^"\\])*")\]\)</script>'
)


@dataclass(frozen=True)
class BuildConfig:
    output_dir: str = "data/cboe_option_flow_2020_2023"
    import_jsonl: str | None = None
    from_snapshot: bool = False


def source_url(observation: str) -> str:
    parsed = date.fromisoformat(observation)
    if parsed < START_DATE or parsed >= END_DATE_EXCLUSIVE:
        raise ValueError("Cboe option-flow date escaped the frozen horizon")
    return f"{BASE_URL}?dt={observation}"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _balanced_object(text: str, key: str = '"optionsData":') -> Mapping[str, Any] | None:
    start_key = text.index(key) + len(key)
    suffix = text[start_key:].lstrip()
    if suffix.startswith("null"):
        return None
    start = text.index("{", start_key)
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                value = json.loads(text[start : index + 1])
                if not isinstance(value, dict):
                    raise ValueError("Cboe optionsData is not an object")
                return value
    raise ValueError("Cboe optionsData object is unbalanced")


def parse_html_response(payload: bytes) -> Mapping[str, Any] | None:
    """Extract Next.js ``optionsData``; ``None`` means an official no-data date."""
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Cboe daily response is not UTF-8 HTML") from exc
    chunks = [json.loads(match.group(1)) for match in _FLIGHT_CHUNK.finditer(text)]
    candidates = [chunk for chunk in chunks if '"optionsData"' in chunk]
    if len(candidates) != 1:
        raise ValueError("Cboe daily response must contain one optionsData payload")
    return _balanced_object(candidates[0])


def _finite_nonnegative(value: Any, *, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Cboe {field} must be numeric") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"Cboe {field} must be finite and nonnegative")
    return parsed


def _volume(value: Any, *, field: str) -> int:
    parsed = _finite_nonnegative(value, field=field)
    if not parsed.is_integer():
        raise ValueError(f"Cboe {field} must be an integer")
    return int(parsed)


def normalize_options_data(
    options_data: Mapping[str, Any],
    *,
    observation: str,
    response_sha256: str,
) -> dict[str, str]:
    parsed_date = date.fromisoformat(observation)
    if parsed_date < START_DATE or parsed_date >= END_DATE_EXCLUSIVE:
        raise ValueError("Cboe option-flow row escaped the frozen horizon")
    if not re.fullmatch(r"[0-9a-f]{64}", response_sha256):
        raise ValueError("Cboe response SHA-256 is invalid")

    ratio_items = options_data.get("ratios")
    if not isinstance(ratio_items, list):
        raise ValueError("Cboe ratio schema changed")
    ratios: dict[str, float] = {}
    for item in ratio_items:
        if not isinstance(item, dict) or "name" not in item or "value" not in item:
            raise ValueError("Cboe ratio item schema changed")
        ratios[str(item["name"])] = _finite_nonnegative(
            item["value"], field=str(item["name"])
        )

    row: dict[str, str] = {"observation_date": observation}
    for field, source_name in RATIO_NAMES.items():
        if source_name not in ratios:
            raise ValueError(f"Cboe ratio disappeared: {source_name}")
        row[field] = format(ratios[source_name], ".6f")

    parsed_volumes: dict[str, tuple[int, int, int]] = {}
    for prefix, source_name in VOLUME_NAMES.items():
        items = options_data.get(source_name)
        if not isinstance(items, list):
            raise ValueError(f"Cboe volume group disappeared: {source_name}")
        matches = [item for item in items if isinstance(item, dict) and item.get("name") == "VOLUME"]
        if len(matches) != 1:
            raise ValueError(f"Cboe VOLUME row changed: {source_name}")
        item = matches[0]
        call = _volume(item.get("call"), field=f"{prefix}_call_volume")
        put = _volume(item.get("put"), field=f"{prefix}_put_volume")
        total = _volume(item.get("total"), field=f"{prefix}_volume")
        if call + put != total:
            raise ValueError(f"Cboe call+put invariant failed: {source_name}")
        parsed_volumes[prefix] = (call, put, total)
        row[f"{prefix}_call_volume"] = str(call)
        row[f"{prefix}_put_volume"] = str(put)
        row[f"{prefix}_volume"] = str(total)

    total_volume = parsed_volumes["total"][2]
    if any(parsed_volumes[prefix][2] > total_volume for prefix in ("index", "equity")):
        raise ValueError("Cboe component volume exceeds all-products volume")
    index_volume = parsed_volumes["index"][2]
    if any(parsed_volumes[prefix][2] > index_volume for prefix in ("vix", "spx")):
        raise ValueError("Cboe index product volume exceeds index total")

    for prefix in ("index", "equity", "vix", "spx"):
        call, put, _ = parsed_volumes[prefix]
        if call == 0:
            raise ValueError(f"Cboe {prefix} call volume is zero")
        exact = put / call
        stated = float(row[f"{prefix}_pcr"])
        if abs(stated - exact) > 0.011:
            raise ValueError(f"Cboe rounded put/call ratio mismatch: {prefix}")

    row["response_sha256"] = response_sha256
    if tuple(row) != PANEL_COLUMNS:
        raise ValueError("normalized Cboe panel column order changed")
    return row


def normalize_preparsed_record(record: Mapping[str, Any]) -> dict[str, str]:
    """Validate a compact extraction made from the official HTML response."""
    observation = str(record.get("date", ""))
    response_sha256 = str(record.get("response_sha256", ""))
    ratios = [
        {"name": source_name, "value": record[field]}
        for field, source_name in RATIO_NAMES.items()
    ]
    options_data: dict[str, Any] = {"ratios": ratios}
    for prefix, source_name in VOLUME_NAMES.items():
        options_data[source_name] = [
            {
                "name": "VOLUME",
                "call": record[f"{prefix}_call_volume"],
                "put": record[f"{prefix}_put_volume"],
                "total": record[f"{prefix}_volume"],
            }
        ]
    return normalize_options_data(
        options_data,
        observation=observation,
        response_sha256=response_sha256,
    )


def panel_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    rows = [normalize_preparsed_record(record) for record in records]
    rows.sort(key=lambda row: row["observation_date"])
    dates = [row["observation_date"] for row in rows]
    if len(dates) != len(set(dates)):
        raise ValueError("duplicate Cboe option-flow date")
    expected_rows, expected_first, expected_last = FROZEN_COVERAGE
    if len(rows) != expected_rows or dates[0] != expected_first or dates[-1] != expected_last:
        raise ValueError("Cboe option-flow coverage changed")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=PANEL_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode()


def write_gzip(path: str | Path, payload: bytes) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as handle:
            handle.write(payload)


def read_gzip(path: str | Path) -> bytes:
    with gzip.open(path, "rb") as handle:
        return handle.read()


def artifact_paths(output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    return {
        "panel": root / "cboe_option_flow_2020-01-01_2023-12-31.csv.gz",
        "manifest": root / "build_manifest.json",
    }


def _read_jsonl(path: str | Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row {number} is not an object")
        rows.append(value)
    return rows


def validate_snapshot(path: str | Path) -> tuple[bytes, list[dict[str, str]]]:
    payload = read_gzip(path)
    reader = csv.DictReader(io.StringIO(payload.decode()))
    if tuple(reader.fieldnames or ()) != PANEL_COLUMNS:
        raise ValueError("frozen Cboe option-flow panel schema changed")
    rows = list(reader)
    # Reuse every value invariant by translating the normalized row back through
    # the compact-record validator.
    compact = [{"date": row["observation_date"], **{k: v for k, v in row.items() if k != "observation_date"}} for row in rows]
    replay = panel_bytes(compact)
    if replay != payload:
        raise ValueError("frozen Cboe option-flow panel is not deterministic")
    return payload, rows


def build(config: BuildConfig = BuildConfig()) -> dict[str, Any]:
    if config.import_jsonl is not None and config.from_snapshot:
        raise ValueError("choose either import_jsonl or from_snapshot")
    paths = artifact_paths(config.output_dir)
    if config.import_jsonl is not None:
        payload = panel_bytes(_read_jsonl(config.import_jsonl))
        write_gzip(paths["panel"], payload)
    elif config.from_snapshot:
        payload, _ = validate_snapshot(paths["panel"])
    else:
        raise ValueError("initial freeze requires --import-jsonl; later replay uses --from-snapshot")

    panel_sha = sha256_file(paths["panel"])
    if FROZEN_PANEL_SHA256 and panel_sha != FROZEN_PANEL_SHA256:
        raise RuntimeError("frozen Cboe option-flow panel hash changed")
    _, rows = validate_snapshot(paths["panel"])
    response_ledger = [
        {"date": row["observation_date"], "response_sha256": row["response_sha256"]}
        for row in rows
    ]
    core: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_snapshot_date": SOURCE_SNAPSHOT_DATE,
        "builder": "training/build_cboe_option_flow_panel.py",
        "config": asdict(config),
        "official_sources": {
            "daily": BASE_URL,
            "historical_information": HISTORICAL_URL,
            "date_query_template": f"{BASE_URL}?dt=YYYY-MM-DD",
        },
        "source_contract": {
            "provider": "Cboe Global Markets",
            "table": "U.S. options daily market statistics",
            "research_horizon": [START_DATE.isoformat(), END_DATE_EXCLUSIVE.isoformat()],
            "raw_html_retained": False,
            "per_response_sha256_retained": True,
            "current_page_not_claimed_as_point_in_time_archive": True,
            "market_or_label_rows_read": 0,
        },
        "panel": {
            "path": str(paths["panel"]),
            "sha256": panel_sha,
            "rows": len(rows),
            "first": rows[0]["observation_date"],
            "last": rows[-1]["observation_date"],
            "columns": list(PANEL_COLUMNS),
            "response_ledger_hash": canonical_hash(response_ledger),
        },
    }
    manifest = {**core, "manifest_hash": canonical_hash(core)}
    paths["manifest"].write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=BuildConfig.output_dir)
    parser.add_argument("--import-jsonl")
    parser.add_argument("--from-snapshot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build(
        BuildConfig(
            output_dir=args.output_dir,
            import_jsonl=args.import_jsonl,
            from_snapshot=args.from_snapshot,
        )
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
