"""Record the terminal HVKSRR-24 historical-event transport rejection."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import preregister_high_volatility_kalshi_sp500_range_repricing_relay as prereg


PREREG = prereg.DEFAULT_OUTPUT
DEFAULT_OUTPUT = Path("results/high_volatility_kalshi_sp500_range_repricing_relay_support_2026-08-12.json")
FAILED_EVENT = "INX-23AUG01"
FAILED_URL = f"{prereg.API_BASE}/events/{FAILED_EVENT}?with_nested_markets=true"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def build() -> dict[str, Any]:
    frozen = json.loads(PREREG.read_text())
    prereg.validate(frozen)
    core = {
        "protocol_version": "high_volatility_kalshi_sp500_range_repricing_relay_source_rejection_v1",
        "policy_id": "HVKSRR-24",
        "as_of_date": "2026-08-12",
        "stage": "source_contract",
        "preregistration": str(PREREG),
        "preregistration_sha256": sha256_file(PREREG),
        "preregistration_manifest_hash": frozen["manifest_hash"],
        "outcomes_opened": False,
        "historical_candidate_market_prices_opened": False,
        "funding_opened": False,
        "gross9_rows_opened": False,
        "source_incidence_boundary": {
            "series_metadata_opened": True,
            "query_series": "KXINX",
            "listed_event_count": 1094,
            "listed_newest_event": "KXINX-26AUG14H1600",
            "listed_oldest_event": "INX-22APR28",
            "historical_candidate_repricings_computed": False,
            "candidate_clocks_written": False,
        },
        "first_failure": {
            "gate": "historical_event_identity_resolves_on_preregistered_official_endpoint",
            "event_ticker": FAILED_EVENT,
            "event_was_returned_by_series_query": True,
            "url": FAILED_URL,
            "http_status": 404,
            "exception": "urllib.error.HTTPError: HTTP Error 404: Not Found",
            "failed_before_nested_market_or_candlestick_payload": True,
        },
        "gates": {
            "official_series_pagination": True,
            "historical_event_replay": False,
            "source_support": False,
            "gross9_novelty": None,
            "economics": None,
            "rv20_q90": None,
        },
        "decision": "REJECT_NO_REPAIR",
        "advance": False,
        "repair_prohibition": "Do not substitute historical/markets, infer nested markets from ticker grammar, change accepted lineage, shorten the source interval, or select another Kalshi series, endpoint, anchor, side, hold, threshold, subset, or control for HVKSRR-24.",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core) or value != build():
        raise RuntimeError("HVKSRR source rejection drift")
    if value["advance"] is not False or value["decision"] != "REJECT_NO_REPAIR":
        raise RuntimeError("HVKSRR rejection boundary drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    value = build()
    validate(value)
    args.output.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
