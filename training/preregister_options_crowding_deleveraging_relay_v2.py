"""Write the availability-corrected, outcome-blind OCDR-12A preregistration."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from training import preregister_options_crowding_deleveraging_relay as v1


DEFAULT_OUTPUT = Path(
    "results/options_crowding_deleveraging_relay_preregistration_v2_2026-08-08.json"
)
VETO = Path("results/options_crowding_deleveraging_relay_source_support_veto_2026-08-08.json")
VETO_SHA = "bce9029503503977f1586e7e6428a741f92ddd7f2707e8c78257b219b4c839f5"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict:
    if sha256(VETO) != VETO_SHA:
        raise RuntimeError("OCDR v1 source-support veto drift")
    prior = v1.build()
    core = {key: value for key, value in prior.items() if key != "manifest_hash"}
    core["protocol_version"] = "options_crowding_deleveraging_relay_v2"
    core["policy"] = {**core["policy"], "policy_id": "OCDR-12A"}
    core["v1_terminal_source_support_veto"] = {
        "path": str(VETO),
        "sha256": VETO_SHA,
        "candidate_incidence_opened": False,
        "economic_outcomes_opened": False,
    }
    core["causal_clock"] = {
        **core["causal_clock"],
        "oi_change": (
            "sum_open_interest from exact archive periods [T-65m,T-60m) and "
            "[T-5m,T); each value becomes available only at period ts+5m, "
            "which must be <=T and strictly before the T+5m entry"
        ),
        "oi_archive_availability": (
            "for source=open_interest_hist, observed_at is deterministically the "
            "documented completed 5m period end ts+5m; NULL database ingestion "
            "metadata is not a value imputation and is never used as availability"
        ),
    }
    core["source_plan"] = {
        **core["source_plan"],
        "oi": (
            "Postgres open_interest_binance BTCUSDT period=5m source=open_interest_hist; "
            "materialize ts, values and source, and derive availability solely as ts+5m"
        ),
    }
    core["research_boundary"] = {
        **core["research_boundary"],
        "v1_source_metadata_known": True,
        "v2_candidate_incidence_opened": False,
        "v2_price_or_return_rows_opened": False,
        "mechanism_threshold_side_hold_changed_from_v1": False,
    }
    return {**core, "manifest_hash": v1.canonical_hash(core)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.write_text(json.dumps(build(), indent=2, ensure_ascii=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
