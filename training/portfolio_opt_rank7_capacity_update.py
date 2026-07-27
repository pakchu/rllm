#!/usr/bin/env python3
"""Re-optimize the Gross8 universe with the frozen Rank7 1.5x capacity result.

The signal universe and selection/evaluation protocol remain identical to the
2026-07-16 Gross8 search.  The only candidate-space change is that the existing
0.50x Rank7 path may receive a multiplier up to 3.0, matching the leverage
selected from 2023-2024 by the preregistered Rank7 battery.  A duplicate Rank7
sleeve is deliberately not introduced.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.portfolio_opt_added_alpha_update import Config, run


DEFAULT_CONFIG = Config(
    output="results/portfolio_rank7_capacity_update_2026-07-28.json",
    docs_output="docs/portfolio-rank7-capacity-update-2026-07-28.md",
    candidate_config=(
        "configs/shadow/portfolio_rank7_capacity_candidate_2026-07-28.json"
    ),
    rank7_family_gross_cap=3.0,
    rank7_capacity_evidence=(
        "results/expanding_extratrees_rank7_leverage_battery_2026-07-27.json"
    ),
    comparison_portfolio=(
        "configs/live/portfolio_added_alpha_mainnet_live_2026-07-18.json"
    ),
    comparison_label="Previous Gross8 live",
    report_title="Rank7-capacity portfolio allocation update (2026-07-28)",
    candidate_name="portfolio_rank7_capacity_shadow_candidate_2026_07_28",
    candidate_as_of="2026-07-28",
)


def main() -> None:
    report = run(DEFAULT_CONFIG)
    selected = report["frozen_pre2025_top1"]
    print(
        json.dumps(
            {
                "output": DEFAULT_CONFIG.output,
                "protocol_hash": report["protocol_hash"],
                "deployment_disposition": report["deployment_disposition"],
                "weights": selected["weights"],
                "gross": selected["gross"],
                "future_veto_passed": selected["future_veto_passed"],
                "stats": selected["stats"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
