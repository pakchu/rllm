"""Seal GNRC pre-2024 market inputs without parsing any outcome value."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "gdelt_gnrc_premarket_access_seal_v1"
SOURCE_SUPPORT_REPORT = Path(
    "results/gdelt_narrative_rotation_clearing_source_support_2026-07-20.json"
)
SOURCE_SUPPORT_REPORT_SHA256 = (
    "1b35c6fef694f1b352129cd3b40ae85832834561f61b731bccaf4d8b24c2a5e4"
)
EVALUATOR_SOURCE = Path(
    "training/evaluate_gdelt_narrative_economic_selection.py"
)
EVALUATOR_SOURCE_SHA256 = (
    "7437fef90dd159d63e56a6226d0a14c8c17133442dabce0bd6a3338c3f5769b6"
)
PROTOCOL_DOCUMENT = Path(
    "docs/gdelt-narrative-rotation-clearing-economic-selection-protocol-"
    "2026-07-20.md"
)
PROTOCOL_DOCUMENT_SHA256 = (
    "bb570db9e18dbf77540af5e1e4ccc2bdeff439295cda031e557b3558dca8af2c"
)
TEST_SOURCE = Path(
    "tests/test_evaluate_gdelt_narrative_economic_selection.py"
)
TEST_SOURCE_SHA256 = (
    "acdc5950248a1d9fbff0950c6091d701c8448cda9a3eb529dc58347f2cecb0b3"
)
MARKET_DATA = Path(
    "data/binance_um_kline_reference_btc_2020_2023/"
    "BTCUSDT_5m_2020-01-01_2023-12-31.csv.gz"
)
MARKET_DATA_SHA256 = (
    "e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d"
)
MARKET_MANIFEST = Path(
    "data/binance_um_kline_reference_btc_2020_2023/build_manifest.json"
)
MARKET_MANIFEST_SHA256 = (
    "c04fbbd299cc748a6745c0ef030787da4d560833c744c81c98dd8840efc7913e"
)
FUNDING_DATA = Path("data/binance_um_btcusdt_funding_marks_2020_2023.csv.gz")
FUNDING_DATA_SHA256 = (
    "3284bbb6bb67946acb673c6b67459543e217f752589e1d47b6c7c3b659f733e6"
)
FUNDING_MANIFEST = Path(
    "results/binance_um_btcusdt_funding_marks_2020_2023_manifest_2026-07-17.json"
)
FUNDING_MANIFEST_SHA256 = (
    "a0b2d27e1aa8cf2d9ab8cb659b598ee0a6d7bd25401c9e10ae92d1a74415845b"
)
DEFAULT_OUTPUT = Path(
    "results/gdelt_gnrc_premarket_access_seal_2026-07-22.json"
)
SEALED_AT = "2026-07-22T00:05:00Z"
EXPECTED_MARKET_COLUMNS = (
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base",
    "taker_buy_quote",
)
EXPECTED_FUNDING_COLUMNS = (
    "funding_time_ms",
    "funding_time_utc",
    "symbol",
    "funding_rate",
    "settlement_mark_price",
    "mark_open_time_ms",
    "mark_open_time_utc",
    "funding_time_offset_ms",
    "mark_source",
)


def repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with repository_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    with repository_path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"GNRC premarket metadata is not an object: {path}")
    return payload


def validate_manifest_metadata(
    market: Mapping[str, Any], funding: Mapping[str, Any]
) -> None:
    config = market.get("config", {})
    protocol = market.get("protocol", {})
    if (
        config.get("symbol") != "BTCUSDT"
        or config.get("interval") != "5m"
        or config.get("start") != "2020-01-01"
        or config.get("end") != "2024-01-01"
        or market.get("combined_output") != str(MARKET_DATA)
        or market.get("combined_sha256") != MARKET_DATA_SHA256
        or market.get("rows") != 1_461 * 24 * 12
        or market.get("first_date") != "2020-01-01 00:00:00"
        or market.get("last_date") != "2023-12-31 23:55:00"
        or tuple(market.get("columns", ())) != EXPECTED_MARKET_COLUMNS
        or protocol.get("source") != "official Binance USD-M daily kline archives"
        or protocol.get("archive_checksums_verified") is not True
        or protocol.get("end_is_exclusive") is not True
        or protocol.get("outcomes_opened") is not False
    ):
        raise ValueError("GNRC premarket market-manifest metadata changed")

    funding_core = {
        key: value
        for key, value in funding.items()
        if key not in {"manifest_hash", "created_at"}
    }
    data = funding.get("data", {})
    mapping = funding.get("mapping", {})
    if (
        funding.get("manifest_hash") != canonical_hash(funding_core)
        or funding.get("protocol_version")
        != "btc_um_funding_settlement_marks_2020_2023_v1"
        or funding.get("outcomes_opened") is not False
        or funding.get("strategy_outcomes_calculated") != []
        or data.get("path") != str(FUNDING_DATA)
        or data.get("sha256") != FUNDING_DATA_SHA256
        or data.get("rows") != 1_461 * 3
        or tuple(data.get("columns", ())) != EXPECTED_FUNDING_COLUMNS
        or mapping.get("funding_time") != "exact returned fundingTime retained"
        or mapping.get("mark")
        != "open of floor(fundingTime, 8h) official mark-price kline"
        or mapping.get("maximum_allowed_timestamp_offset_ms") != 60_000
    ):
        raise ValueError("GNRC premarket funding-manifest metadata changed")


def build_seal() -> dict[str, Any]:
    frozen = {
        SOURCE_SUPPORT_REPORT: SOURCE_SUPPORT_REPORT_SHA256,
        EVALUATOR_SOURCE: EVALUATOR_SOURCE_SHA256,
        PROTOCOL_DOCUMENT: PROTOCOL_DOCUMENT_SHA256,
        TEST_SOURCE: TEST_SOURCE_SHA256,
        MARKET_DATA: MARKET_DATA_SHA256,
        MARKET_MANIFEST: MARKET_MANIFEST_SHA256,
        FUNDING_DATA: FUNDING_DATA_SHA256,
        FUNDING_MANIFEST: FUNDING_MANIFEST_SHA256,
    }
    for path, expected_hash in frozen.items():
        if sha256_file(path) != expected_hash:
            raise ValueError(f"GNRC premarket frozen input changed: {path}")
    validate_manifest_metadata(
        _load_json(MARKET_MANIFEST), _load_json(FUNDING_MANIFEST)
    )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "source_support_report_path": str(SOURCE_SUPPORT_REPORT),
        "source_support_report_sha256": SOURCE_SUPPORT_REPORT_SHA256,
        "evaluator_source_path": str(EVALUATOR_SOURCE),
        "evaluator_source_sha256": EVALUATOR_SOURCE_SHA256,
        "protocol_document_path": str(PROTOCOL_DOCUMENT),
        "protocol_document_sha256": PROTOCOL_DOCUMENT_SHA256,
        "test_source_path": str(TEST_SOURCE),
        "test_source_sha256": TEST_SOURCE_SHA256,
        "market_data_path": str(MARKET_DATA),
        "market_data_sha256": MARKET_DATA_SHA256,
        "market_manifest_path": str(MARKET_MANIFEST),
        "market_manifest_sha256": MARKET_MANIFEST_SHA256,
        "funding_data_path": str(FUNDING_DATA),
        "funding_data_sha256": FUNDING_DATA_SHA256,
        "funding_manifest_path": str(FUNDING_MANIFEST),
        "funding_manifest_sha256": FUNDING_MANIFEST_SHA256,
        "market_values_inspected_before_seal": False,
        "funding_values_inspected_before_seal": False,
        "post_2023_outcomes_inspected_before_seal": False,
        "sealed_at": SEALED_AT,
    }


def write_once(path: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    destination = repository_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"GNRC premarket seal is write-once: {destination}")
    payload = build_seal()
    try:
        with destination.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except FileExistsError as error:
        raise FileExistsError(
            f"GNRC premarket seal is write-once: {destination}"
        ) from error
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(write_once(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
