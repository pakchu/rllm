#!/usr/bin/env python3
"""Execute the source-only support gate for preregistered PSIM-D8-CDP1."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from training import preregister_psim_d8_cross_protocol_disagreement_persistence as prereg
from training import preregister_psim_d8_rllm1 as d8


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "psim_d8_cross_protocol_disagreement_source_support_v1"
PREREGISTRATION = prereg.DEFAULT_OUTPUT
PREREGISTRATION_SHA256 = (
    "111063c83f06b4527d0c6427f7124997629a290b52da89481867fa21b34f04f1"
)
DEFAULT_OUTPUT = Path(
    "results/psim_d8_cross_protocol_disagreement_persistence_"
    "source_support_2026-07-29.json"
)

EXACT_FIELDS = (
    "event_type",
    "window_revision_count_bucket",
    "window_age_bucket",
    "update_gap_bucket",
    "dependency_delta_state",
    "dependency_edge_delta_count_bucket",
    "line_change_count_bucket",
    "changed_section_count_bucket",
)


@dataclass
class EwmaState:
    fast: float | None = None
    slow: float | None = None
    nonmissing_cards: int = 0


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes((REPO_ROOT / path).read_bytes())


def _jaccard_distance(left: Sequence[Any], right: Sequence[Any]) -> float:
    left_set = {str(value) for value in left}
    right_set = {str(value) for value in right}
    union = left_set | right_set
    if not union:
        return 0.0
    return 1.0 - len(left_set & right_set) / len(union)


def unit_disagreement(unit: Mapping[str, Any]) -> float | None:
    if (
        unit.get("memorization_excluded") is not False
        or unit.get("counterpart_state")
        not in {"SAME_DAY_CARTESIAN", "TRAILING_90D"}
    ):
        return None
    ethereum = unit.get("ethereum")
    bitcoin = unit.get("bitcoin")
    if not isinstance(ethereum, Mapping) or not isinstance(bitcoin, Mapping):
        return None
    components = [
        0.0 if ethereum.get(field) == bitcoin.get(field) else 1.0
        for field in EXACT_FIELDS
    ]
    components.append(
        _jaccard_distance(
            ethereum.get("changed_sections", []),
            bitcoin.get("changed_sections", []),
        )
    )
    score = sum(components) / len(components)
    if not 0.0 <= score <= 1.0:
        raise RuntimeError("unit disagreement escaped [0, 1]")
    return score


def daily_disagreement(card: Mapping[str, Any]) -> tuple[float | None, int]:
    d8._validate_card(card)
    if card.get("schedule") != "ARCHIVE_D90":
        raise RuntimeError("unexpected schedule in CDP1 card stream")
    units = card["local_payload"]["relation_units"]
    scores = [
        score
        for unit in units
        if (score := unit_disagreement(unit)) is not None
    ]
    if not scores:
        return None, 0
    return sum(scores) / len(scores), len(scores)


def update_ewmas(state: EwmaState, score: float | None) -> None:
    if score is None:
        return
    if not 0.0 <= score <= 1.0:
        raise ValueError("daily disagreement must be in [0, 1]")
    if state.fast is None or state.slow is None:
        state.fast = score
        state.slow = score
    else:
        fast_alpha = 1.0 - math.exp(math.log(0.5) / 3.0)
        slow_alpha = 1.0 - math.exp(math.log(0.5) / 30.0)
        state.fast = fast_alpha * score + (1.0 - fast_alpha) * state.fast
        state.slow = slow_alpha * score + (1.0 - slow_alpha) * state.slow
    state.nonmissing_cards += 1


def signal_for(
    state: EwmaState,
    *,
    slow_floor: float,
    gap: float,
) -> str:
    if (
        state.nonmissing_cards < 30
        or state.fast is None
        or state.slow is None
        or state.slow < slow_floor
    ):
        return "flat"
    delta = state.fast - state.slow
    if delta >= gap:
        return "short"
    if -delta >= gap:
        return "long"
    return "flat"


def _load_cards() -> list[dict[str, Any]]:
    raw = (REPO_ROOT / prereg.D8_CARDS).read_bytes()
    if sha256_bytes(raw) != prereg.D8_CARDS_SHA256:
        raise RuntimeError("PSIM-D8 card payload drift")
    decompressed = gzip.decompress(raw)
    if sha256_bytes(decompressed) != prereg.D8_CARDS_ROWS_SHA256:
        raise RuntimeError("PSIM-D8 canonical card rows drift")
    rows = [
        json.loads(line)
        for line in decompressed.decode("utf-8").splitlines()
        if line
    ]
    selected = [
        row
        for row in rows
        if row.get("schedule") == "ARCHIVE_D90"
        and "2022-01-01" <= str(row.get("decision_at", ""))[:10] <= "2023-12-31"
    ]
    for row in selected:
        d8._validate_card(row)
    return selected


def _candidate_parameters(candidate_id: str) -> tuple[float, float]:
    _, slow_token, gap_token = candidate_id.split("_")
    return int(slow_token[1:]) / 100.0, int(gap_token[1:]) / 100.0


def _candidate_stats(
    observations: Sequence[Mapping[str, Any]],
    candidate_id: str,
) -> dict[str, Any]:
    slow_floor, gap = _candidate_parameters(candidate_id)
    state = EwmaState()
    accepted: list[dict[str, str]] = []
    skip_next_card = False
    raw = Counter()
    for row in observations:
        update_ewmas(state, row["score"])
        signal = signal_for(state, slow_floor=slow_floor, gap=gap)
        if signal == "flat":
            skip_next_card = False if skip_next_card else skip_next_card
            continue
        raw[signal] += 1
        if skip_next_card:
            skip_next_card = False
            continue
        decision_at = str(row["decision_at"])
        accepted.append(
            {
                "decision_at": decision_at,
                "direction": signal,
            }
        )
        skip_next_card = True
    months = Counter(row["decision_at"][:7] for row in accepted)
    quarters = {
        f"{row['decision_at'][:4]}Q{(int(row['decision_at'][5:7]) - 1) // 3 + 1}"
        for row in accepted
    }
    directions = Counter(row["direction"] for row in accepted)
    top_month_share = max(months.values(), default=0) / max(len(accepted), 1)
    return {
        "candidate_id": candidate_id,
        "slow_floor": slow_floor,
        "fast_slow_gap": gap,
        "raw_signal_days": dict(sorted(raw.items())),
        "accepted_signal_days": len(accepted),
        "accepted_long_days": directions["long"],
        "accepted_short_days": directions["short"],
        "active_quarters": sorted(quarters),
        "active_quarter_count": len(quarters),
        "top_calendar_month_share": top_month_share,
        "accepted_signal_manifest_hash": prereg.canonical_hash(accepted),
    }


def build_source_support() -> dict[str, Any]:
    if sha256_file(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise RuntimeError("CDP1 preregistration artifact drift")
    cards = _load_cards()
    observations = []
    eligible_units = Counter()
    nonmissing = Counter()
    for card in cards:
        score, unit_count = daily_disagreement(card)
        year = str(card["decision_at"])[:4]
        eligible_units[year] += unit_count
        if score is not None:
            nonmissing[year] += 1
        observations.append(
            {
                "decision_at": card["decision_at"],
                "score": score,
            }
        )
    by_year = {
        year: [
            row
            for row in observations
            if str(row["decision_at"]).startswith(year)
        ]
        for year in ("2022", "2023")
    }
    family: dict[str, dict[str, Any]] = {}
    eligible_candidates: list[str] = []
    for candidate_id in prereg._candidate_ids():
        yearly = {
            year: _candidate_stats(rows, candidate_id)
            for year, rows in by_year.items()
        }
        family[candidate_id] = yearly
        if all(
            stats["accepted_signal_days"] >= 24
            and stats["active_quarter_count"] >= 3
            and stats["top_calendar_month_share"] <= 0.50
            for stats in yearly.values()
        ):
            eligible_candidates.append(candidate_id)
    has_long = any(
        family[candidate_id][year]["accepted_long_days"] > 0
        for candidate_id in eligible_candidates
        for year in ("2022", "2023")
    )
    has_short = any(
        family[candidate_id][year]["accepted_short_days"] > 0
        for candidate_id in eligible_candidates
        for year in ("2022", "2023")
    )
    passed = bool(eligible_candidates) and has_long and has_short
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "candidate": prereg.POLICY_ID,
        "preregistration": {
            "path": PREREGISTRATION.as_posix(),
            "sha256": PREREGISTRATION_SHA256,
        },
        "source_access": {
            "cards_path": prereg.D8_CARDS.as_posix(),
            "cards_sha256": prereg.D8_CARDS_SHA256,
            "card_rows_parsed": len(cards),
            "events_payload_opened": False,
            "market_payload_opened": False,
            "funding_payload_opened": False,
            "economic_metrics_computed": 0,
        },
        "source_summary": {
            "nonmissing_daily_scores": dict(sorted(nonmissing.items())),
            "eligible_relation_units": dict(sorted(eligible_units.items())),
            "daily_score_values_disclosed": False,
        },
        "family_source_incidence": family,
        "eligible_candidate_ids": eligible_candidates,
        "eligible_candidate_count": len(eligible_candidates),
        "family_has_long_incidence": has_long,
        "family_has_short_incidence": has_short,
        "decision": "pass" if passed else "reject",
        "terminal_action": (
            "AUTHORIZE_2022_SELECTION_PREREGISTERED_FAMILY_ONLY"
            if passed
            else "TERMINAL_REJECT_CDP1_SOURCE_SUPPORT_NO_THRESHOLD_REPAIR"
        ),
        "2022_market_or_funding_authorized": passed,
        "2023_market_or_funding_authorized": False,
        "threshold_repair_rank2_or_family_extension_allowed": False,
    }
    return {**core, "result_hash": prereg.canonical_hash(core)}


def write_source_support(path: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    result = build_source_support()
    output = path if path.is_absolute() else REPO_ROOT / path
    rendered = json.dumps(
        result,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    if output.exists():
        if output.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"source-support result drift: {output}")
        return result
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = write_source_support(args.output)
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "eligible_candidate_count": result[
                    "eligible_candidate_count"
                ],
                "result_hash": result["result_hash"],
                "terminal_action": result["terminal_action"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
