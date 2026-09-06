"""Record the terminal DDAELR-24 DOJ API date-identity failure."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

from training import preregister_doj_digital_asset_enforcement_lifecycle_relay as prereg


PAGE = Path("data/doj_digital_asset_enforcement_lifecycle_relay_sources_2023_2026/raw_page_cache/page-00447.json")
PAGE_SHA = "75029514546e85ef113585b4c71f9fd180a98502171a9123cd6af3fa1d10ec86"
RAW_FAILURE = Path("data/doj_digital_asset_enforcement_lifecycle_relay_sources_2023_2026/terminal_page_00447.json.gz")
OUTPUT = Path("results/doj_digital_asset_enforcement_lifecycle_relay_source_failure_2026-08-09.json")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def run() -> dict[str, Any]:
    raw = PAGE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PAGE_SHA:
        raise RuntimeError("DDAELR terminal DOJ page drift")
    payload = json.loads(raw)
    row = payload["results"][8]
    observed = {key: row.get(key) for key in ("uuid", "date", "title", "url")}
    if observed != {"uuid": "", "date": "", "title": "", "url": ""}:
        raise RuntimeError("DDAELR first invalid row drift")
    RAW_FAILURE.parent.mkdir(parents=True, exist_ok=True)
    with RAW_FAILURE.open("wb") as target:
        with gzip.GzipFile(filename="", mode="wb", fileobj=target, mtime=0, compresslevel=9) as stream:
            stream.write(raw)
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    core = {
        "protocol_version": "ddaelr_24_terminal_source_failure_v1",
        "policy_id": "DDAELR-24",
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": sha(prereg.DEFAULT_OUTPUT), "manifest_hash": registration["manifest_hash"]},
        "first_failed_gate": "official_source_date_identity",
        "failure": {
            "api_page": 447,
            "record_index": 8,
            "page_sha256": PAGE_SHA,
            "observed": observed,
            "reason": "blank DOJ API date cannot define the frozen date-only 22:00 UTC causal availability clock",
        },
        "raw_failure_page": {"path": str(RAW_FAILURE), "sha256": sha(RAW_FAILURE)},
        "doj_source_rows_opened_before_failure": 447 * 50 + 9,
        "candidate_incidence_computed": False,
        "btc_source_rows_opened": 0,
        "gross9_rows_opened": 0,
        "postentry_return_or_pnl_opened": False,
        "advance_to_gross9_novelty": False,
        "advance_to_economic_outcomes": False,
        "decision": "terminal_source_reject_no_repair",
        "forbidden_repairs": ["drop blank row", "infer date", "substitute created or updated", "title-only universe", "change availability clock"],
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
