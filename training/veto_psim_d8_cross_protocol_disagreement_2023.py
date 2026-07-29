#!/usr/bin/env python3
"""Run the untouched 2023 future veto for frozen CDP_S50_G05."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from training import preregister_psim_d8_cross_protocol_disagreement_persistence as prereg
from training import run_psim_d8_cross_protocol_disagreement_source_support as source
from training import select_psim_d8_cross_protocol_disagreement_2022 as selection


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "psim_d8_cross_protocol_disagreement_2023_veto_v1"
POLICY_ID = "CDP_S50_G05"
SELECTION_RESULT = selection.RESULT_PATH
SELECTION_RESULT_SHA256 = (
    "33cb78065b04a1103b048a0607a4d68f40ada58bd9911b49d1f4da49c73d2f4e"
)
SELECTION_RESULT_HASH = (
    "f4530edbe8f26ad5ab4cd549de24fcc9c4d1ffc0dde8d6d3e3420bd88dd4a76b"
)
SELECTION_RUNNER_SHA256 = (
    "08aa17bdb495724d0df47312e1c285723d8708450698c8877385c4ed11ce5f7c"
)
RUNNER_PATH = Path(
    "training/veto_psim_d8_cross_protocol_disagreement_2023.py"
)
ATTEMPT_PATH = Path(
    "results/psim_d8_cross_protocol_disagreement_2023_veto_"
    "attempt_2026-07-29.json"
)
RESULT_PATH = Path(
    "results/psim_d8_cross_protocol_disagreement_2023_veto_"
    "result_2026-07-29.json"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with (REPO_ROOT / path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads((REPO_ROOT / path).read_text(encoding="utf-8"))


def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
    target = REPO_ROOT / path
    rendered = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    if target.exists():
        if target.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"write-once artifact drift: {path}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(rendered, encoding="utf-8")


def _selection_metric(result: Mapping[str, Any]) -> dict[str, Any]:
    rows = [
        row
        for row in result["candidate_metrics"]
        if row["candidate_id"] == POLICY_ID
    ]
    if len(rows) != 1:
        raise RuntimeError("frozen CDP1 top1 metric is missing")
    return dict(rows[0])


def prepare_attempt() -> dict[str, Any]:
    result = _read_json(SELECTION_RESULT)
    if (
        _sha256_file(SELECTION_RESULT) != SELECTION_RESULT_SHA256
        or result.get("result_hash") != SELECTION_RESULT_HASH
        or result.get("decision") != "pass"
        or result.get("selected_top1") != POLICY_ID
        or result.get("authorize_2023_future_veto") is not True
        or _sha256_file(selection.RUNNER_PATH) != SELECTION_RUNNER_SHA256
    ):
        raise RuntimeError("CDP1 2022 selection authority drift")
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "untouched_2023_future_veto",
        "candidate_id": POLICY_ID,
        "runner": {
            "path": RUNNER_PATH.as_posix(),
            "sha256": _sha256_file(RUNNER_PATH),
        },
        "selection_result": {
            "path": SELECTION_RESULT.as_posix(),
            "sha256": SELECTION_RESULT_SHA256,
            "result_hash": SELECTION_RESULT_HASH,
        },
        "frozen_selection_metric": _selection_metric(result),
        "veto_contract": prereg.build_preregistration()[
            "evaluation_contract"
        ]["future_veto_gate"],
        "access_boundary": {
            "2023_market_rows_parsed": 0,
            "2023_funding_rows_parsed": 0,
            "2023_economic_metrics_computed": 0,
        },
    }
    payload = {**core, "attempt_hash": prereg.canonical_hash(core)}
    _write_once(ATTEMPT_PATH, payload)
    return payload


def _signals_2023() -> list[dict[str, Any]]:
    slow_floor, gap = source._candidate_parameters(POLICY_ID)
    cards = [
        card
        for card in source._load_cards()
        if str(card["decision_at"]).startswith("2023")
    ]
    state = source.EwmaState()
    skip_next = False
    accepted: list[dict[str, Any]] = []
    for card in cards:
        score, _ = source.daily_disagreement(card)
        source.update_ewmas(state, score)
        signal = source.signal_for(state, slow_floor=slow_floor, gap=gap)
        if signal == "flat":
            if skip_next:
                skip_next = False
            continue
        if skip_next:
            skip_next = False
            continue
        decision = pd.Timestamp(card["decision_at"])
        entry = decision + pd.Timedelta(minutes=5)
        accepted.append(
            {
                "decision_at": str(card["decision_at"]),
                "entry_time": entry,
                "exit_time": entry + pd.Timedelta(days=1),
                "direction": signal,
                "side": 1 if signal == "long" else -1,
            }
        )
        skip_next = True
    manifest = prereg.canonical_hash(
        [
            {
                "decision_at": row["decision_at"],
                "direction": row["direction"],
            }
            for row in accepted
        ]
    )
    expected = _read_json(source.DEFAULT_OUTPUT)["family_source_incidence"][
        POLICY_ID
    ]["2023"]["accepted_signal_manifest_hash"]
    if manifest != expected:
        raise RuntimeError("2023 frozen signal reconstruction changed")
    return [
        row
        for row in accepted
        if row["exit_time"] <= pd.Timestamp("2024-01-01", tz="UTC")
    ]


def _load_2023_market() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames = []
    bindings = []
    for month in range(1, 13):
        stem = f"BTCUSDT_5m_2023-{month:02d}"
        payload_path = selection.MONTHLY_MARKET_DIR / f"{stem}.csv.gz"
        manifest_path = selection.MONTHLY_MARKET_DIR / f"{stem}.json"
        manifest = _read_json(manifest_path)
        observed_hash = _sha256_file(payload_path)
        if manifest.get("output_sha256") != observed_hash:
            raise RuntimeError(f"2023 monthly market authority drift: {stem}")
        frame = pd.read_csv(REPO_ROOT / payload_path, compression="gzip")
        if len(frame) != int(manifest["rows"]):
            raise RuntimeError(f"2023 monthly market row drift: {stem}")
        frames.append(frame)
        bindings.append(
            {
                "path": payload_path.as_posix(),
                "sha256": observed_hash,
                "rows": len(frame),
            }
        )
    market = pd.concat(frames, ignore_index=True)
    market["date"] = pd.to_datetime(market["date"], utc=True)
    expected = pd.date_range(
        "2023-01-01",
        "2023-12-31 23:55:00",
        freq="5min",
        tz="UTC",
    )
    if not pd.DatetimeIndex(market["date"]).equals(expected):
        raise RuntimeError("2023 market grid is not exact")
    return market, bindings


def _load_2023_funding() -> list[dict[str, Any]]:
    rows = []
    with gzip.open(REPO_ROOT / prereg.FUNDING, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            timestamp = str(row["funding_time_utc"])
            if not timestamp.startswith("2023"):
                continue
            rows.append(
                {
                    "funding_time": pd.Timestamp(row["mark_open_time_utc"]),
                    "funding_rate": float(row["funding_rate"]),
                    "settlement_mark_price": float(
                        row["settlement_mark_price"]
                    ),
                }
            )
    expected = pd.date_range(
        "2023-01-01", "2023-12-31 16:00:00", freq="8h", tz="UTC"
    )
    if [row["funding_time"] for row in rows] != list(expected):
        raise RuntimeError("2023 exact funding grid changed")
    return rows


def veto_checks(
    metrics: Mapping[str, Any],
    selection_metric: Mapping[str, Any],
) -> dict[str, bool]:
    base = metrics["base"]
    stress = metrics["stress"]
    return {
        "base_net_return_strictly_positive": (
            float(base["absolute_return"]) > 0.0
        ),
        "stress_net_return_not_negative": (
            float(stress["absolute_return"]) >= 0.0
        ),
        "base_return_over_strict_mdd_minimum": (
            float(base["cagr_to_strict_mdd"]) >= 0.50
        ),
        "strict_mdd_not_more_than_selection_multiple": (
            float(base["strict_mdd"])
            <= 2.0 * float(selection_metric["base"]["strict_mdd"])
        ),
        "minimum_closed_trades": int(base["closed_trades"]) >= 15,
    }


def execute_veto() -> dict[str, Any]:
    attempt = _read_json(ATTEMPT_PATH)
    if (
        attempt.get("runner", {}).get("sha256") != _sha256_file(RUNNER_PATH)
        or attempt.get("attempt_hash")
        != prereg.canonical_hash(
            {k: v for k, v in attempt.items() if k != "attempt_hash"}
        )
    ):
        raise RuntimeError("2023 veto attempt drift")
    market, market_bindings = _load_2023_market()
    funding = _load_2023_funding()
    signals = _signals_2023()
    metrics = {
        "base": selection._evaluate(
            market, funding, signals, cost_rate=0.0006
        ),
        "stress": selection._evaluate(
            market, funding, signals, cost_rate=0.0010
        ),
    }
    checks = veto_checks(metrics, attempt["frozen_selection_metric"])
    passed = all(checks.values())
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "candidate_id": POLICY_ID,
        "attempt_hash": attempt["attempt_hash"],
        "selection_result_hash": SELECTION_RESULT_HASH,
        "market_bindings": market_bindings,
        "funding_binding": {
            "path": prereg.FUNDING.as_posix(),
            "frozen_full_payload_sha256": prereg.FUNDING_SHA256,
            "rows_parsed_2023": len(funding),
        },
        "metrics": metrics,
        "veto_checks": checks,
        "decision": "pass" if passed else "reject",
        "terminal_action": (
            "AUTHORIZE_SEPARATE_SAME_GROSS_PORTFOLIO_PREREGISTRATION"
            if passed
            else "TERMINAL_REJECT_CDP1_NO_REPAIR_OR_RANK2"
        ),
        "authorize_same_gross_portfolio_preregistration": passed,
        "authorize_live_promotion": False,
        "post_veto_repair_rank2_or_threshold_change_allowed": False,
    }
    return {**core, "result_hash": prereg.canonical_hash(core)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("prepare-attempt", "execute"), nargs="?", default="execute"
    )
    args = parser.parse_args()
    if args.action == "prepare-attempt":
        payload = prepare_attempt()
        print(json.dumps({"attempt_hash": payload["attempt_hash"]}, sort_keys=True))
        return
    result = execute_veto()
    _write_once(RESULT_PATH, result)
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "result_hash": result["result_hash"],
                "veto_checks": result["veto_checks"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
