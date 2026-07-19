"""Freeze the legacy DVOL-rich BTC-price-follow clock used by OPDR novelty checks."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import preregister_cross_venue_vol_disagreement_alpha as legacy  # noqa: E402
from training.build_binance_aggtrade_microstructure import _write_gzip_csv  # noqa: E402


LEGACY_SOURCE_SHA256 = (
    "b313360c8c1f7acc5f744a96efdfa9d6aeecb9f5d9340abc500748410a92f9a3"
)
LEGACY_IMPLEMENTATION_SHA256 = (
    "ecd0e5748d60a2d5f358209e909f94803628c88264b06bfa63f833a9426861b3"
)
LEGACY_SUPPORT_SHA256 = (
    "cc7d3b8c123ebaccf3d048f67d38609533fc7dd67f5c6b30a63e7fd3f0bea0fc"
)
EXPECTED_CANDIDATE = legacy.Candidate(
    family="dvol_rich_move_follow",
    vol_tail_quantile=0.80,
    price_tail_quantile=0.80,
    hold_hours=48,
)
EXPECTED_CANONICAL_CLOCK_HASH = (
    "1c42483b2a7f4512dbda4690072d105efc383518358def86afbe1f120db536e6"
)
EXPECTED_ROWS = 29


@dataclass(frozen=True)
class Config:
    source: str = legacy.Config.input_csv
    legacy_implementation: str = (
        "training/preregister_cross_venue_vol_disagreement_alpha.py"
    )
    legacy_support: str = (
        "results/cross_venue_vol_disagreement_support_2026-07-19.json"
    )
    output_clock: str = (
        "data/opdr_old_dvol_price_follow_comparator_2023h2.csv.gz"
    )
    output_result: str = (
        "results/opdr_old_dvol_price_follow_comparator_freeze_2026-07-19.json"
    )


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_legacy_support(cfg: Config) -> dict[str, Any]:
    if _sha256(cfg.source) != LEGACY_SOURCE_SHA256:
        raise ValueError("legacy DVOL price-follow source bytes changed")
    if _sha256(cfg.legacy_implementation) != LEGACY_IMPLEMENTATION_SHA256:
        raise ValueError("legacy DVOL price-follow implementation bytes changed")
    if _sha256(cfg.legacy_support) != LEGACY_SUPPORT_SHA256:
        raise ValueError("legacy DVOL price-follow support bytes changed")
    support = json.loads(Path(cfg.legacy_support).read_text(encoding="utf-8"))
    if support.get("protocol", {}).get("outcomes_opened") is not False:
        raise ValueError("legacy support artifact unexpectedly opened outcomes")
    matches = [
        row
        for row in support.get("candidates", [])
        if row.get("name") == EXPECTED_CANDIDATE.name
    ]
    if len(matches) != 1:
        raise ValueError("legacy comparator candidate is missing or duplicated")
    selected = matches[0]
    if selected.get("candidate") != asdict(EXPECTED_CANDIDATE):
        raise ValueError("legacy comparator candidate parameters changed")
    if selected.get("clock_hash") != EXPECTED_CANONICAL_CLOCK_HASH:
        raise ValueError("legacy comparator canonical clock hash changed")
    if selected.get("support", {}).get("total") != EXPECTED_ROWS:
        raise ValueError("legacy comparator support count changed")
    return selected


def build_clock(cfg: Config = Config()) -> pd.DataFrame:
    """Recreate the frozen comparator without opening any OPDR outcome source."""

    _load_legacy_support(cfg)
    frame = legacy.load_source(cfg.source)
    thresholds = legacy.build_thresholds(frame)
    onset, side = legacy.candidate_clock(frame, thresholds, EXPECTED_CANDIDATE)
    clock = legacy.nonoverlapping_schedule(
        frame,
        onset,
        side,
        hold_hours=EXPECTED_CANDIDATE.hold_hours,
    )
    canonical = legacy.canonical_hash(clock.to_dict(orient="records"))
    if canonical != EXPECTED_CANONICAL_CLOCK_HASH:
        raise ValueError("legacy comparator clock cannot be reproduced")
    if len(clock) != EXPECTED_ROWS:
        raise ValueError("legacy comparator row count cannot be reproduced")
    return clock


def build_report(cfg: Config, clock: pd.DataFrame) -> dict[str, Any]:
    selected = _load_legacy_support(cfg)
    core: dict[str, Any] = {
        "protocol_version": "opdr_old_dvol_price_follow_comparator_freeze_v1",
        "as_of_date": "2026-07-19",
        "opdr_outcomes_opened": False,
        "opdr_outcome_sources_opened": [],
        "legacy_comparator_feature_source_opened": True,
        "btc_execution_rows_loaded_for_opdr": 0,
        "funding_rows_loaded_for_opdr": 0,
        "source": {
            "path": cfg.source,
            "sha256": LEGACY_SOURCE_SHA256,
            "physically_truncated_before": legacy.SELECTION_END,
        },
        "legacy_implementation": {
            "path": cfg.legacy_implementation,
            "sha256": LEGACY_IMPLEMENTATION_SHA256,
        },
        "legacy_support": {
            "path": cfg.legacy_support,
            "sha256": LEGACY_SUPPORT_SHA256,
        },
        "candidate": asdict(EXPECTED_CANDIDATE),
        "canonical_clock_hash": EXPECTED_CANONICAL_CLOCK_HASH,
        "clock": {
            "path": cfg.output_clock,
            "sha256": _sha256(cfg.output_clock),
            "rows": int(len(clock)),
            "first_entry": str(clock["entry_time"].min()),
            "last_exit": str(clock["exit_time"].max()),
        },
        "support": selected["support"],
        "implementation_sha256": _sha256(__file__),
    }
    return {**core, "manifest_hash": _canonical_hash(core)}


def run(cfg: Config = Config()) -> dict[str, Any]:
    clock = build_clock(cfg)
    output_clock = Path(cfg.output_clock)
    output_clock.parent.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(clock, output_clock)
    report = build_report(cfg, clock)
    output_result = Path(cfg.output_result)
    output_result.parent.mkdir(parents=True, exist_ok=True)
    output_result.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=Config.source)
    parser.add_argument(
        "--legacy-implementation", default=Config.legacy_implementation
    )
    parser.add_argument("--legacy-support", default=Config.legacy_support)
    parser.add_argument("--output-clock", default=Config.output_clock)
    parser.add_argument("--output-result", default=Config.output_result)
    report = run(Config(**vars(parser.parse_args())))
    print(
        json.dumps(
            {
                "rows": report["clock"]["rows"],
                "clock_sha256": report["clock"]["sha256"],
                "manifest_hash": report["manifest_hash"],
                "opdr_outcomes_opened": report["opdr_outcomes_opened"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
