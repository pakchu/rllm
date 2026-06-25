#!/usr/bin/env python3
"""Run the current audited event-candidate policy preset.

This is intentionally a research/paper-test runner.  The current preset is not
live-ready because recent 2026 diagnostics show the strict gate is protective and
relaxed recent trading loses money.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"
if VENV_PYTHON.exists() and Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), *sys.argv])

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.event_candidate_pairwise_walkforward import run
from training.event_candidate_policy_config import DEFAULT_POLICY_PATH, build_walk_forward_cfg, load_policy_preset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=str(DEFAULT_POLICY_PATH))
    parser.add_argument("--output", default="results/current_event_candidate_policy/report.json")
    parser.add_argument("--work-dir", default="results/current_event_candidate_policy")
    parser.add_argument("--input-jsonl", default=None, help="Override preset input_jsonl for controlled reruns")
    parser.add_argument("--market-csv", default=None, help="Override preset market_csv for controlled reruns")
    parser.add_argument("--print-config-only", action="store_true", help="Print resolved walk-forward config without running")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preset = load_policy_preset(args.policy)
    cfg = build_walk_forward_cfg(
        preset,
        output=args.output,
        work_dir=args.work_dir,
        input_jsonl=args.input_jsonl,
        market_csv=args.market_csv,
    )
    if args.print_config_only:
        print(json.dumps({"policy": preset["name"], "live_ready": preset["live_ready"], "config": cfg.__dict__}, indent=2, ensure_ascii=False))
        return
    Path(args.work_dir).mkdir(parents=True, exist_ok=True)
    print(json.dumps(run(cfg), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
