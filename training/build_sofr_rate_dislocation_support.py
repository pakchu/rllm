"""Replay SFRD-1's exact source-only clock without opening market outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from training import preregister_sofr_rate_dislocation as prereg
from training import sofr_rate_dislocation_clock as clock


PREREGISTRATION = prereg.DEFAULT_OUTPUT
PREREGISTRATION_SHA256 = (
    "cbb80c25e4b4c627b95d1992ce4ad00043acfff586e0fe3fcd086af5b4e80b06"
)
PREREGISTRATION_COMMIT = "3a5e4659db98dc02422671410c7d5adce9931b3a"
DEFAULT_OUTPUT = "results/sofr_rate_dislocation_support_2026-07-17.json"
WINDOWS = {
    "train": ("2021-01-01", "2023-01-01"),
    "2021": ("2021-01-01", "2022-01-01"),
    "2022": ("2022-01-01", "2023-01-01"),
    "2023": ("2023-01-01", "2024-01-01"),
    "2023_h1": ("2023-01-01", "2023-07-01"),
    "2023_h2": ("2023-07-01", "2024-01-01"),
}


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def load_preregistration(path: str | Path = PREREGISTRATION) -> dict[str, Any]:
    if _sha256(path) != PREREGISTRATION_SHA256:
        raise ValueError("SFRD-1 preregistration file changed")
    payload = _load_json(path)
    prereg.validate_manifest(payload, verify_sources=False)
    if payload["policy"]["policy_id"] != "SFRD-1":
        raise ValueError("unexpected SFRD-1 policy identity")
    return payload


def verify_source_only_contract(registration: dict[str, Any]) -> None:
    source = registration["source_contract"]
    if _sha256(source["sofr"]) != source["sofr_sha256"]:
        raise ValueError("SFRD-1 SOFR panel changed")
    if _sha256(source["sofr_manifest"]) != source["sofr_manifest_sha256"]:
        raise ValueError("SFRD-1 SOFR source manifest changed")
    manifest = _load_json(source["sofr_manifest"])
    if manifest.get("output_sha256") != source["sofr_sha256"]:
        raise ValueError("SFRD-1 source manifest does not bind the panel")
    protocol = manifest.get("protocol", {})
    if protocol.get("outcomes_opened") is not False:
        raise ValueError("SFRD-1 source manifest opened outcomes")
    if protocol.get("crypto_market_fields_opened") is not False:
        raise ValueError("SFRD-1 source manifest opened crypto market fields")
    if manifest.get("config", {}).get("end_year") != 2023:
        raise ValueError("SFRD-1 source is not capped at 2023")


def replay_clock(registration: dict[str, Any]) -> list[clock.Event]:
    source = registration["source_contract"]
    support = registration["support_verification"]
    if _sha256(support["clock_ledger"]) != support["clock_ledger_sha256"]:
        raise ValueError("SFRD-1 frozen clock ledger changed")
    rebuilt = clock.build_events(clock.read_source(source["sofr"]))
    ledger = clock.read_event_ledger(support["clock_ledger"])
    if rebuilt != ledger:
        raise ValueError("SFRD-1 exact rebuilt clock differs from frozen ledger")
    if len(rebuilt) != support["clock_ledger_events_full_source"]:
        raise ValueError("SFRD-1 full-source event count changed")
    return rebuilt


def _window_checks(
    registration: dict[str, Any], events: list[clock.Event]
) -> tuple[dict[str, dict[str, Any]], dict[str, bool]]:
    support = registration["support_verification"]
    expected = support["expected_preflight_counts"]
    summaries = {
        name: clock.event_summary(events, start, end)
        for name, (start, end) in WINDOWS.items()
    }
    exact_replay = {
        name: summaries[name]["count"] == expected[name] for name in WINDOWS
    }
    exact_replay.update(
        {
            "train_long": summaries["train"]["long"] == expected["train_long"],
            "train_short": summaries["train"]["short"] == expected["train_short"],
            "2023_long": summaries["2023"]["long"] == expected["2023_long"],
            "2023_short": summaries["2023"]["short"] == expected["2023_short"],
            "train_max_month_count": (
                summaries["train"]["max_single_month_count"]
                == expected["train_max_single_month_count"]
            ),
            "train_max_month_share": (
                summaries["train"]["max_single_month_share"]
                == expected["train_max_single_month_share"]
            ),
            "2023_max_month_count": (
                summaries["2023"]["max_single_month_count"]
                == expected["2023_max_single_month_count"]
            ),
            "2023_max_month_share": (
                summaries["2023"]["max_single_month_share"]
                == expected["2023_max_single_month_share"]
            ),
        }
    )
    floors = {
        "train_count": summaries["train"]["count"]
        >= support["minimum_nonoverlap_train"],
        "2021_count": summaries["2021"]["count"] >= support["minimum_2021"],
        "2022_count": summaries["2022"]["count"] >= support["minimum_2022"],
        "2023_count": summaries["2023"]["count"] >= support["minimum_2023"],
        "2023_h1_count": summaries["2023_h1"]["count"]
        >= support["minimum_2023_h1"],
        "2023_h2_count": summaries["2023_h2"]["count"]
        >= support["minimum_2023_h2"],
        "train_each_side": min(
            summaries["train"]["long"], summaries["train"]["short"]
        )
        >= support["minimum_train_each_side"],
        "2023_each_side": min(
            summaries["2023"]["long"], summaries["2023"]["short"]
        )
        >= support["minimum_2023_each_side"],
        "train_month_concentration": summaries["train"]["max_single_month_share"]
        <= support["maximum_single_month_share_train"],
        "2023_month_concentration": summaries["2023"]["max_single_month_share"]
        <= support["maximum_single_month_share_2023"],
    }
    return summaries, {**exact_replay, **floors}


def _causality_checks(events: list[clock.Event]) -> dict[str, bool]:
    five_minutes = 5 * 60
    five_days = 5 * 24 * 60 * 60
    parsed = [
        (
            clock._parse_utc(event.sofr_available_at_utc),
            clock._parse_utc(event.entry_time),
            clock._parse_utc(event.exit_time),
        )
        for event in events
    ]
    return {
        "entry_exactly_one_bar_after_rate_availability": all(
            int((entry - available).total_seconds()) == five_minutes
            for available, entry, _ in parsed
        ),
        "exit_exactly_five_days_after_entry": all(
            int((exit_time - entry).total_seconds()) == five_days
            for _, entry, exit_time in parsed
        ),
        "globally_nonoverlapping": all(
            parsed[index][1] >= parsed[index - 1][2]
            for index in range(1, len(parsed))
        ),
        "integer_rank_denominator_frozen": all(
            event.rank_twice_denominator == 240 for event in events
        ),
        "inclusive_tail_and_side_mapping": all(
            (
                event.state == 1
                and event.rank_twice_numerator >= 204
                and event.side == -1
            )
            or (
                event.state == -1
                and event.rank_twice_numerator <= 36
                and event.side == 1
            )
            for event in events
        ),
        "pre_2024_only": all(exit_time.year <= 2023 for _, _, exit_time in parsed),
    }


def build(output_path: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    registration = load_preregistration()
    verify_source_only_contract(registration)
    events = replay_clock(registration)
    windows, support_checks = _window_checks(registration, events)
    causality_checks = _causality_checks(events)
    all_checks = {**support_checks, **causality_checks}
    passed = all(all_checks.values())
    result = {
        "protocol_version": "sofr_rate_dislocation_support_v1",
        "as_of_date": "2026-07-17",
        "policy_id": "SFRD-1",
        "candidate_class": "source-only-screened exploratory singleton",
        "preregistration": PREREGISTRATION,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "preregistration_manifest_hash": registration["manifest_hash"],
        "outcomes_opened": False,
        "outcome_sources_opened": [],
        "source_files_opened": [
            registration["source_contract"]["sofr"],
            registration["source_contract"]["sofr_manifest"],
            registration["support_verification"]["clock_ledger"],
        ],
        "clock_ledger_sha256": registration["support_verification"][
            "clock_ledger_sha256"
        ],
        "full_source_events": len(events),
        "windows": windows,
        "checks": all_checks,
        "support_replay_passed": passed,
        "advance_to_stage1_outcomes": passed,
        "interpretation": (
            "PASS means exact implementation and source-density replay only; density "
            "through 2023 was screened in-sample and is not OOS alpha evidence"
        ),
        "sealed": ["2024", "2025", "2026_ytd"],
        "performance_statistics": {
            "absolute_return": None,
            "CAGR": None,
            "strict_MDD": None,
            "CAGR_to_strict_MDD": None,
        },
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build(args.output)
    print(
        json.dumps(
            {
                "output": args.output,
                "support_replay_passed": result["support_replay_passed"],
                "advance_to_stage1_outcomes": result["advance_to_stage1_outcomes"],
                "outcomes_opened": result["outcomes_opened"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
