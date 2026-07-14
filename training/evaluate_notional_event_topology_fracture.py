"""Frozen pre-2024 train/selection evaluator for NETF v1."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from training.evaluate_metaorder_fragmentation_impact_curvature import (
    EvaluationConfig,
    _sha256,
    simulate_schedule,
)
from training.preregister_metaorder_fragmentation_impact_curvature import (
    Config as SourceConfig,
    load_causal_frame,
)
from training.preregister_notional_event_topology_fracture import (
    CANDIDATES,
    Config as SignalConfig,
    compute_netf,
    nonoverlapping_netf_schedule,
)


PREREGISTRATION_COMMIT = "26e6b3d"
PREREGISTRATION_SOURCE = Path(
    "training/preregister_notional_event_topology_fracture.py"
)
PREREGISTRATION_SOURCE_SHA256 = (
    "2229c93495d246e949dc768860c4df942189a86b6f700183b09ae54c2873c578"
)
PREREGISTRATION_DOCUMENT = Path(
    "docs/notional-event-topology-fracture-preregistration-2026-07-14.md"
)
PREREGISTRATION_DOCUMENT_SHA256 = (
    "cbd969059da8be16d1e5f6f8c6b3d3c3146d653265a0a44cb9a280972613ebb3"
)
PREREGISTRATION_RESULT = Path(
    "results/notional_event_topology_fracture_support_2026-07-14.json"
)
PREREGISTRATION_RESULT_SHA256 = (
    "4062da454fbd83e10e04dc9b3b01d0884277023d68e209219bb1bcc76d38588e"
)
EXECUTION_SOURCE = Path(
    "training/evaluate_metaorder_fragmentation_impact_curvature.py"
)
EXECUTION_SOURCE_SHA256 = (
    "1589a52605386570485a7e6be3b8f3aa9439a498abb60eaa42272ac62d4cbed3"
)
DEFAULT_OUTPUT = (
    "results/notional_event_topology_fracture_selection_2026-07-14.json"
)

WINDOWS: dict[str, tuple[str, str]] = {
    "train": ("2020-01-01", "2023-01-01"),
    "select2023": ("2023-01-01", "2024-01-01"),
    "select2023_h1": ("2023-01-01", "2023-07-01"),
    "select2023_h2": ("2023-07-01", "2024-01-01"),
}


def _verify_preregistration() -> dict[str, Any]:
    hashes = (
        (PREREGISTRATION_SOURCE, PREREGISTRATION_SOURCE_SHA256),
        (PREREGISTRATION_DOCUMENT, PREREGISTRATION_DOCUMENT_SHA256),
        (PREREGISTRATION_RESULT, PREREGISTRATION_RESULT_SHA256),
        (EXECUTION_SOURCE, EXECUTION_SOURCE_SHA256),
    )
    for path, expected in hashes:
        if _sha256(path) != expected:
            raise ValueError(f"frozen NETF artifact changed: {path}")
    result = json.loads(PREREGISTRATION_RESULT.read_text())
    protocol = result.get("protocol", {})
    if protocol.get("outcomes_opened_for_netf") is not False:
        raise ValueError("NETF support artifact did not preserve unopened outcomes")
    if result.get("all_candidates_pass_support") is not True:
        raise ValueError("NETF candidates did not pass frozen support floors")
    if result.get("config") != asdict(SignalConfig()):
        raise ValueError("NETF config differs from frozen support artifact")
    frozen_candidates = [item["candidate"] for item in result.get("candidates", [])]
    if frozen_candidates != [asdict(candidate) for candidate in CANDIDATES]:
        raise ValueError("NETF candidates differ from frozen support artifact")
    calibration = result.get("support_calibration", {})
    if calibration.get("selected_tension_quantile") != SignalConfig.tension_quantile:
        raise ValueError("NETF support stopping rule is not frozen")
    if calibration.get("further_support_repairs_allowed") is not False:
        raise ValueError("NETF support artifact permits post-freeze repair")
    return result


def qualification(candidate: dict[str, Any]) -> dict[str, Any]:
    windows = candidate["windows"]
    train = windows["train"]
    select = windows["select2023"]
    failures: list[str] = []
    for name, metrics in (("train", train), ("select2023", select)):
        if metrics["absolute_return_pct"] <= 0.0:
            failures.append(f"{name}: non-positive absolute return")
        if metrics["cagr_to_strict_mdd"] < 3.0:
            failures.append(f"{name}: CAGR/strict-MDD below 3")
        if metrics["strict_mdd_pct"] > 15.0:
            failures.append(f"{name}: strict MDD above 15%")
        if metrics["weekly_cluster_sign_flip"]["p_value_one_sided"] >= 0.10:
            failures.append(f"{name}: weekly-cluster p-value not below 0.10")
    for name in ("select2023_h1", "select2023_h2"):
        metrics = windows[name]
        if metrics["absolute_return_pct"] <= 0.0:
            failures.append(f"{name}: non-positive absolute return")
        if metrics["trade_count"] < 20:
            failures.append(f"{name}: fewer than 20 trades")
    if select["trade_count"] < 60:
        failures.append("select2023: fewer than 60 trades")
    return {"qualifies": not failures, "failures": failures}


def select_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    qualified = [item for item in candidates if item["qualification"]["qualifies"]]
    if not qualified:
        return {
            "selected_candidate": None,
            "rejected": True,
            "reason": "no frozen NETF candidate passed every selection gate",
        }
    selected = sorted(
        qualified,
        key=lambda item: (
            -min(
                item["windows"]["train"]["cagr_to_strict_mdd"],
                item["windows"]["select2023"]["cagr_to_strict_mdd"],
            ),
            item["windows"]["select2023"]["strict_mdd_pct"],
            item["candidate"]["name"],
        ),
    )[0]
    return {
        "selected_candidate": selected["candidate"]["name"],
        "rejected": False,
        "reason": "passed frozen NETF train/select gates",
    }


def run_evaluation(output: str = DEFAULT_OUTPUT) -> dict[str, Any]:
    preregistration = _verify_preregistration()
    frame, source = load_causal_frame(SourceConfig())
    signal_cfg = SignalConfig()
    execution_cfg = replace(EvaluationConfig(), output=output)
    candidates: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        signal = compute_netf(frame, candidate, signal_cfg)
        windows: dict[str, Any] = {}
        for name, (start, end) in WINDOWS.items():
            schedule = nonoverlapping_netf_schedule(
                signal,
                frame,
                start=start,
                end=end,
            )
            windows[name] = simulate_schedule(
                frame,
                schedule,
                start=start,
                end=end,
                cfg=execution_cfg,
            )
        item = {"candidate": asdict(candidate), "windows": windows}
        item["qualification"] = qualification(item)
        candidates.append(item)
    return {
        "protocol": {
            "name": "NETF frozen pre-2024 selection evaluation",
            "preregistration_commit": PREREGISTRATION_COMMIT,
            "preregistration_source_sha256": PREREGISTRATION_SOURCE_SHA256,
            "preregistration_document_sha256": PREREGISTRATION_DOCUMENT_SHA256,
            "preregistration_result_sha256": PREREGISTRATION_RESULT_SHA256,
            "execution_source_sha256": EXECUTION_SOURCE_SHA256,
            "outcomes_opened_for_netf": True,
            "opened_windows": list(WINDOWS),
            "sealed_windows": ["test2024", "eval2025", "ytd2026"],
            "signal_parameters_mutable": False,
            "entry": "next 5m open",
            "exit": "scheduled future 5m open",
            "strict_mdd": "complete held path, favorable extreme first then adverse extreme",
            "cagr": "full wall-clock split including idle cash",
        },
        "execution_config": asdict(execution_cfg),
        "signal_config": preregistration["config"],
        "source": source,
        "candidates": candidates,
        "selection": select_candidate(candidates),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run_evaluation(args.output)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "selection": result["selection"],
                "candidates": [
                    {
                        "name": item["candidate"]["name"],
                        "qualification": item["qualification"],
                        "windows": {
                            name: {
                                key: metrics[key]
                                for key in (
                                    "absolute_return_pct",
                                    "cagr_pct",
                                    "strict_mdd_pct",
                                    "cagr_to_strict_mdd",
                                    "trade_count",
                                )
                            }
                            for name, metrics in item["windows"].items()
                        },
                    }
                    for item in result["candidates"]
                ],
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
