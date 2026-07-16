"""Build PFCR-2's outcome-blind 2023-2024 episode-onset clock."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.build_post_funding_cross_sectional_crowding_release_support import (
    CLOCK_COLUMNS,
    assert_clock_contract as assert_pfcr1_clock_contract,
    build_clock as build_pfcr1_clock,
    causal_betas,
    load_feature_sources,
    support_stats,
)
from training.export_leave_one_out_residual_exhaustion_sources import (
    deterministic_csv_gz,
    sha256_file,
)
from training.preregister_post_funding_crowding_release_episode_v2 import (
    EPISODE_COOLDOWN_HOURS,
    canonical_hash,
    protocol,
)


PREREGISTRATION = Path(
    "results/post_funding_crowding_release_episode_v2_preregistration_2026-07-17.json"
)
EXPECTED_PREREGISTRATION_SHA256 = (
    "14af65aa684033d85210a6d28d98571b00cf0b07d2dcdbe6206a5de7a864f59b"
)
EXPECTED_PROTOCOL_HASH = "7dc7d51af83f4d4ce9822439799277d39ddd87a181b8728b468beedc5080d3d1"
DEFAULT_CLOCK = Path(
    "data/post_funding_crowding_release_episode_v2_clock_2023_2024.csv.gz"
)
DEFAULT_MANIFEST = Path(
    "results/post_funding_crowding_release_episode_v2_support_2026-07-17.json"
)
DEFAULT_DOCS = Path(
    "docs/post-funding-crowding-release-episode-v2-support-2026-07-17.md"
)
POLICY_ID = "PFCR02"
PARENT_POLICY_ID = "PFCR01"
EPISODE_COOLDOWN = pd.Timedelta(hours=EPISODE_COOLDOWN_HOURS)


def _verify_preregistration() -> dict[str, Any]:
    if sha256_file(PREREGISTRATION) != EXPECTED_PREREGISTRATION_SHA256:
        raise RuntimeError("PFCR-2 preregistration file changed")
    payload = json.loads(PREREGISTRATION.read_text())
    if payload.get("protocol_hash") != EXPECTED_PROTOCOL_HASH:
        raise RuntimeError("PFCR-2 preregistration identity changed")
    if canonical_hash(payload["protocol"]) != EXPECTED_PROTOCOL_HASH:
        raise RuntimeError("PFCR-2 preregistration body changed")
    if canonical_hash(protocol()) != EXPECTED_PROTOCOL_HASH:
        raise RuntimeError("PFCR-2 implementation protocol drifted")
    boundary = payload["protocol"]["evidence_boundary"]
    if boundary["exact_pfcr2_clock_opened"]:
        raise RuntimeError("PFCR-2 preregistration already opened its exact clock")
    if boundary["pfcr1_or_pfcr2_post_entry_returns_opened"]:
        raise RuntimeError("PFCR returns were opened before PFCR-2 support")
    return payload


def episode_onset_clock(parent_clock: pd.DataFrame) -> pd.DataFrame:
    """Keep the first eligible event at least 36h after the prior accepted event."""

    if parent_clock.empty:
        return pd.DataFrame(columns=CLOCK_COLUMNS)
    ordered = parent_clock.copy()
    ordered["settlement_time"] = pd.to_datetime(ordered["settlement_time"], errors="raise")
    ordered = ordered.sort_values("settlement_time", kind="stable").reset_index(drop=True)
    accepted: list[int] = []
    previous_settlement: pd.Timestamp | None = None
    for index, event in ordered["settlement_time"].items():
        settlement = pd.Timestamp(event)
        if previous_settlement is not None and settlement < previous_settlement + EPISODE_COOLDOWN:
            continue
        accepted.append(index)
        previous_settlement = settlement
    clock = ordered.loc[accepted, list(CLOCK_COLUMNS)].reset_index(drop=True)
    clock["policy_id"] = POLICY_ID
    return clock


def build_clock(funding: pd.DataFrame, beta: pd.DataFrame) -> pd.DataFrame:
    return episode_onset_clock(build_pfcr1_clock(funding, beta))


def assert_clock_contract(clock: pd.DataFrame) -> None:
    if clock.empty:
        return
    checked = clock.copy()
    for column in ("settlement_time", "feature_available_time", "entry_time", "exit_time"):
        checked[column] = pd.to_datetime(checked[column], errors="raise")
    if not checked["policy_id"].eq(POLICY_ID).all():
        raise RuntimeError("PFCR-2 policy identity changed")
    gaps = checked["settlement_time"].diff().dropna()
    if (gaps < EPISODE_COOLDOWN).any():
        raise RuntimeError("PFCR-2 episode cooldown changed")
    parent_compatible = checked.copy()
    parent_compatible["policy_id"] = PARENT_POLICY_ID
    assert_pfcr1_clock_contract(parent_compatible)


def _markdown(result: dict[str, Any]) -> str:
    stats = result["support"]
    return f"""# PFCR-2 episode-onset outcome-blind support — 2026-07-17

- Post-entry returns/PnL calculated: **no**
- Parent eligible events before cooldown: `{result['parent_eligible_events']}`
- Accepted 36-hour episode onsets: `{stats['events']}`; years `{stats['year_counts']}`
- Unique ordered pairs: `{stats['unique_ordered_pairs']}`
- Maximum pair share: `{stats['maximum_ordered_pair_share']:.2%}`
- Maximum month share: `{stats['maximum_month_share']:.2%}`
- Long symbols: `{stats['long_symbols']}`
- Short symbols: `{stats['short_symbols']}`
- Support gate: **{'PASS' if stats['passes_support'] else 'REJECT'}**
- Clock SHA-256: `{result['clock_sha256']}`

The event sequence is frozen by scanning PFCR-1-eligible settlements in time
order and accepting only the first event at least 36 hours after the previous
accepted settlement. No post-entry price, return, PnL, 2025, or 2026 source
was read while building or qualifying this clock.
"""


def run(
    clock_path: str | Path = DEFAULT_CLOCK,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    docs_path: str | Path = DEFAULT_DOCS,
) -> dict[str, Any]:
    preregistration = _verify_preregistration()
    close, funding, records = load_feature_sources()
    beta = causal_betas(close)
    parent_clock = build_pfcr1_clock(funding, beta)
    clock = episode_onset_clock(parent_clock)
    assert_clock_contract(clock)
    stats = support_stats(clock)
    clock_output = Path(clock_path)
    deterministic_csv_gz(clock, clock_output)
    reread = pd.read_csv(clock_output)
    assert_clock_contract(reread)
    result: dict[str, Any] = {
        "protocol_version": "pfcr_v2_episode_support_2026-07-17",
        "preregistration_protocol_hash": preregistration["protocol_hash"],
        "post_entry_returns_calculated": False,
        "2023_fit_opened": False,
        "2024_test_opened": False,
        "2025_eval_opened": False,
        "2026_holdout_opened": False,
        "common_settlements": int(len(funding)),
        "parent_eligible_events": int(len(parent_clock)),
        "episode_cooldown_hours": EPISODE_COOLDOWN_HOURS,
        "clock_path": str(clock_output),
        "clock_sha256": sha256_file(clock_output),
        "source_records": records,
        "support": stats,
    }
    result["manifest_hash"] = canonical_hash(result)
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    manifest_output = Path(manifest_path)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    docs_output = Path(docs_path)
    docs_output.parent.mkdir(parents=True, exist_ok=True)
    docs_output.write_text(_markdown(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clock", default=str(DEFAULT_CLOCK))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--docs", default=str(DEFAULT_DOCS))
    args = parser.parse_args()
    print(json.dumps(run(args.clock, args.manifest, args.docs), indent=2))


if __name__ == "__main__":
    main()
