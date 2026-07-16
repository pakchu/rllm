"""Preregister PFCR-2 before its exact event clock or returns are opened."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.preregister_post_funding_cross_sectional_crowding_release import (
    canonical_hash,
    protocol as pfcr1_protocol,
)


DEFAULT_OUTPUT = Path(
    "results/post_funding_crowding_release_episode_v2_preregistration_2026-07-17.json"
)
DEFAULT_DOCS = Path(
    "docs/post-funding-crowding-release-episode-v2-preregistration-2026-07-17.md"
)
EPISODE_COOLDOWN_HOURS = 36


def protocol() -> dict[str, Any]:
    """Return the immutable PFCR-2 protocol derived without opening returns."""

    frozen = copy.deepcopy(pfcr1_protocol())
    frozen.update(
        {
            "protocol_version": "pfcr_v2_episode_onset_2026-07-17",
            "name": "PFCR-2 — Post-Funding Crowding Release Episode Onset",
            "claim": (
                "The first extreme cross-sectional funding-spread settlement after a "
                "36-hour quiet period identifies a new crowding episode; over the next "
                "four hours, the highest-funding alt underperforms the lowest-funding alt."
            ),
            "protocol_derivation": {
                "parent_protocol": "PFCR-1",
                "parent_support_gate_result": "rejected for monthly concentration",
                "parent_post_entry_returns_calculated": False,
                "information_used": "event timestamps, symbols, and support concentration only",
                "outcome_or_post_entry_price_used": False,
                "support_only_candidates_inspected": [
                    "first eligible event per UTC week",
                    "maximum-spread eligible event per UTC week",
                    "accepted-settlement cooldown 36h",
                    "accepted-settlement cooldown 48h",
                    "accepted-settlement cooldown 60h",
                    "accepted-settlement cooldown 66h",
                    "accepted-settlement cooldown 72h",
                    "accepted-settlement cooldown 84h",
                    "accepted-settlement cooldown 96h",
                ],
                "selection_rule": (
                    "choose the shortest inspected deterministic cooldown satisfying all "
                    "unchanged PFCR-1 support gates"
                ),
                "selected_cooldown_hours": EPISODE_COOLDOWN_HOURS,
                "new_protocol_not_parent_repair": True,
            },
        }
    )
    frozen["evidence_boundary"] = {
        "underlying_alt_market_and_funding_rows_seen_elsewhere": True,
        "pfcr1_outcome_blind_support_clock_opened": True,
        "pfcr1_or_pfcr2_post_entry_returns_opened": False,
        "exact_pfcr2_clock_opened": False,
        "2023_fit_opened": False,
        "2024_test_opened": False,
        "2025_eval_opened": False,
        "2026_holdout_opened": False,
        "historical_results_can_promote_live": False,
        "forward_shadow_required": True,
    }
    frozen["novelty_boundary"]["versus_pfcr1"] = (
        "episode-onset clock accepts only the first eligible settlement at least 36 hours "
        "after the previous accepted settlement; PFCR-1 accepted every non-overlapping event"
    )
    frozen["clock"].update(
        {
            "episode_onset": (
                "scan eligible settlements chronologically and accept only when no prior "
                "accepted settlement exists or settlement_time >= prior accepted "
                "settlement_time + 36 hours"
            ),
            "episode_cooldown_hours": EPISODE_COOLDOWN_HOURS,
            "cooldown_anchor": "previous accepted settlement timestamp",
        }
    )
    frozen["selection_2023_2024"]["singleton_no_parameter_ranking"] = True
    frozen["sequential_oos"]["no_threshold_sign_hold_pair_or_beta_repair"] = True
    frozen["stop_rule"] = (
        "Reject before outcomes if the frozen 36-hour episode clock fails support. Reject "
        "before 2025 if the singleton fails 2023-2024. Reject before 2026 if it fails "
        "2025. Never alter cooldown, sign, threshold, hold, pair, or beta after an outcome "
        "window opens."
    )
    return frozen


def markdown(payload: dict[str, Any]) -> str:
    return f"""# PFCR-2 episode-onset preregistration — 2026-07-17

## Mechanism

PFCR-2 retains PFCR-1's causal post-settlement, six-alt, beta-neutral pair.
It accepts only the first eligible settlement at least **36 hours** after the
previous accepted settlement. This treats repeated extreme settlements as one
crowding episode rather than independent signals.

## Outcome-blind derivation

PFCR-1 was rejected before any post-entry return was calculated because its
maximum monthly event share was 24.29%, above the frozen 20% support limit.
Only timestamps, selected symbols, and support concentration were inspected.
Weekly first/maximum rules and 36/48/60/66/72/84/96-hour cooldowns were checked;
36 hours was the shortest inspected deterministic cooldown satisfying every
unchanged support gate. PFCR-2 is therefore a separately frozen protocol, not
an outcome-based repair of PFCR-1.

## Qualification

The support and return gates remain unchanged: at least 60 events and 25 per
year, broad pairs/symbols, maximum pair share 25%, maximum month share 20%,
then positive 2023 and 2024 returns, each-year CAGR/strict-MDD >=1.5,
combined ratio >=3, strict MDD <=15%, at least 60 trades, and all frozen
robustness controls. Only a pass can open 2025, then 2026.

Protocol hash: `{payload['protocol_hash']}`
"""


def run(
    output: str | Path = DEFAULT_OUTPUT,
    docs_output: str | Path = DEFAULT_DOCS,
) -> dict[str, Any]:
    frozen = protocol()
    payload = {
        "protocol": frozen,
        "protocol_hash": canonical_hash(frozen),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    docs_path = Path(docs_output)
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(markdown(payload))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--docs-output", default=str(DEFAULT_DOCS))
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.docs_output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
