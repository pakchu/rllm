"""Freeze the LVRT-R0 evaluator source before it can open prices."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from training import evaluate_liquidity_vacuum_replenishment as evaluate


DEFAULT_OUTPUT = str(evaluate.EVALUATION_FREEZE)


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_manifest(
    commit: str,
    *,
    policy_clock_sha256: dict[str, str],
    policy_clock_rows: dict[str, int],
) -> dict[str, Any]:
    if len(commit) != 40:
        raise ValueError("evaluator source commit must be a full Git hash")
    core: dict[str, Any] = {
        "protocol": "LVRT-R0 evaluator pre-outcome freeze v1",
        "outcomes_opened": False,
        "evaluation_source": str(evaluate.EVALUATION_SOURCE),
        "evaluation_source_sha256": _sha256(evaluate.EVALUATION_SOURCE),
        "evaluation_source_commit": commit,
        "support_commit": evaluate.SUPPORT_COMMIT,
        "support_source_sha256": evaluate.SUPPORT_SOURCE_SHA256,
        "support_result_sha256": evaluate.SUPPORT_RESULT_SHA256,
        "event_clock_sha256": evaluate.EVENT_CLOCK_SHA256,
        "policy_clock_sha256": policy_clock_sha256,
        "policy_clock_rows": policy_clock_rows,
        "funding_data_sha256": evaluate.FUNDING_DATA_SHA256,
        "funding_manifest_sha256": evaluate.FUNDING_MANIFEST_SHA256,
        "opened_windows": [],
        "sealed_windows": [*evaluate.WINDOWS, "2024", "2025", "2026_ytd"],
        "mutable_parameters": [],
        "returns_or_prices_loaded_during_freeze": False,
        "evaluation_config": evaluate.asdict(evaluate.EvaluationConfig()),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate_manifest(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if canonical_hash(core) != payload.get("manifest_hash"):
        raise RuntimeError("LVRT-R0 evaluator freeze hash mismatch")
    if payload.get("outcomes_opened") is not False:
        raise RuntimeError("LVRT-R0 evaluator freeze cannot open outcomes")
    if payload.get("returns_or_prices_loaded_during_freeze") is not False:
        raise RuntimeError("LVRT-R0 evaluator freeze loaded outcomes")
    if payload.get("mutable_parameters") != []:
        raise RuntimeError("LVRT-R0 evaluator freeze permits mutable parameters")
    expected = set(evaluate.POLICY_NAMES)
    if set(payload.get("policy_clock_sha256", {})) != expected:
        raise RuntimeError("LVRT-R0 evaluator freeze clock hash set changed")
    if set(payload.get("policy_clock_rows", {})) != expected:
        raise RuntimeError("LVRT-R0 evaluator freeze clock row set changed")
    if payload["policy_clock_sha256"]["primary"] != evaluate.EVENT_CLOCK_SHA256:
        raise RuntimeError("LVRT-R0 evaluator freeze primary clock hash changed")


def write_once(path: str | Path, payload: dict[str, Any]) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if output.exists():
        existing = json.loads(output.read_text())
        validate_manifest(existing)
        if existing != payload:
            raise RuntimeError("refusing to overwrite LVRT-R0 evaluator freeze")
        return "verified_existing"
    with output.open("x", encoding="utf-8") as handle:
        handle.write(content)
    return "created"


def current_clean_commit() -> str:
    source = str(evaluate.EVALUATION_SOURCE)
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", source], text=True
    ).strip()
    if status:
        raise RuntimeError("LVRT-R0 evaluator source is not clean at HEAD")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    committed = subprocess.check_output(["git", "show", f"HEAD:{source}"])
    committed_sha256 = hashlib.sha256(committed).hexdigest()
    if committed_sha256 != _sha256(evaluate.EVALUATION_SOURCE):
        raise RuntimeError("LVRT-R0 evaluator source is not reproducible from HEAD")
    return commit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if Path(evaluate.DEFAULT_OUTPUT).exists():
        raise RuntimeError("LVRT-R0 outcome result already exists")
    _, schedules, _ = evaluate.verify_support_and_replay()
    clock_hashes = {
        name: evaluate._clock_sha256(schedule)
        for name, schedule in schedules.items()
    }
    clock_rows = {name: int(len(schedule)) for name, schedule in schedules.items()}
    payload = build_manifest(
        current_clean_commit(),
        policy_clock_sha256=clock_hashes,
        policy_clock_rows=clock_rows,
    )
    status = write_once(args.output, payload)
    print(
        json.dumps(
            {
                "status": status,
                "outcomes_opened": False,
                "evaluation_source_commit": payload["evaluation_source_commit"],
                "evaluation_source_sha256": payload["evaluation_source_sha256"],
                "output": args.output,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
