"""Freeze the outcome-blind UGCI-288 source and policy contract.

This module hashes preregistration inputs only.  It must not parse stablecoin
event rows, comparator clocks, BTC market data, funding, labels, or PnL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "usdc_gross_clearing_imbalance_preregistration_v1"
CANDIDATE = "UGCI-288"
AS_OF_DATE = "2026-07-22"

SOURCE_CSV = Path(
    "data/ethereum_stablecoin_issuance_redemption_2020_2023/"
    "ethereum_usdt_usdc_issuance_redemption_2020_2023.csv.gz"
)
SOURCE_CSV_SHA256 = "70ba3799ba84dc671051623a8d167b1731f043cf84a686b9878a67fcd52e5901"
SOURCE_MANIFEST = Path(
    "results/ethereum_stablecoin_issuance_redemption_source_manifest_2026-07-21.json"
)
SOURCE_MANIFEST_SHA256 = (
    "8ec9ab08c413bf6f5f8170fb800b05105522d4cf1a7932943c214288701e31fe"
)
PREREGISTRATION_SOURCE = Path("training/preregister_usdc_gross_clearing_imbalance.py")
PREREGISTRATION_DOCUMENT = Path(
    "docs/usdc-gross-clearing-imbalance-preregistration-2026-07-22.md"
)
DEFAULT_OUTPUT = Path(
    "results/usdc_gross_clearing_imbalance_preregistration_2026-07-22.json"
)

COMPARATORS: tuple[dict[str, Any], ...] = (
    {
        "candidate": "AMTR-48",
        "path": "data/authorized_minter_turnaround_relay_clocks_2020_2023.csv.gz",
        "sha256": "30875029daa4d6e2eff9a59f53d45eda57dbced05988df089c38a6c81abfa0f6",
        "controls": ["primary", "cross_minter"],
        "entry_column": "entry_time",
        "comparison_start": "2021-01-01T00:00:00Z",
        "comparison_end_exclusive": "2024-01-01T00:00:00Z",
    },
    {
        "candidate": "SQFD-6",
        "path": "data/stablecoin_quote_flow_diffusion_clocks_2023_2026.csv.gz",
        "sha256": "a81e144eea1e80ae5439fc66db1fad5bbd00cd9ac177e25142b5cfb5a07bcc5b",
        "controls": ["primary", "no_usdt_lag", "no_participation"],
        "entry_column": "entry_time",
        "comparison_start": "2023-09-01T00:00:00Z",
        "comparison_end_exclusive": "2024-01-01T00:00:00Z",
    },
    {
        "candidate": "SDDR-12",
        "path": "data/stablecoin_denominator_dislocation_clocks_2023.csv.gz",
        "sha256": "eaf2d6c187af9855e76474d2951fcdc12267174980a72649b73d068982ca8c69",
        "controls": ["primary"],
        "entry_column": "entry_time",
        "comparison_start": "2023-09-01T00:00:00Z",
        "comparison_end_exclusive": "2024-01-01T00:00:00Z",
    },
    {
        "candidate": "UCBR-12",
        "path": "data/usdt_collateral_breadth_relay_clocks_2023.csv.gz",
        "sha256": "20b3ee9f82696222a3adbde0045dfde53e0e240e85162e463166aa8fe90b1a8f",
        "controls": ["primary"],
        "entry_column": "entry_time",
        "comparison_start": "2023-09-01T00:00:00Z",
        "comparison_end_exclusive": "2024-01-01T00:00:00Z",
    },
)


@dataclass(frozen=True)
class PolicyConfig:
    asset: str = "usdc_eth"
    events: tuple[str, ...] = ("mint", "burn")
    event_clock: str = "available_at"
    packet_hours: int = 6
    include_zero_event_packets: bool = True
    lookback_days: int = 180
    minimum_history_packets: int = 360
    gross_tail_quantile: float = 0.95
    quantile_method: str = "nearest_rank"
    minimum_imbalance_ratio: float = 0.60
    entry_delay_minutes: int = 10
    hold_bars: int = 288
    bar_minutes: int = 5
    leverage: float = 0.5
    global_nonoverlap: bool = True
    warmup_start: str = "2020-01-01T00:00:00Z"
    train_start: str = "2021-01-01T00:00:00Z"
    train_end_exclusive: str = "2023-01-01T00:00:00Z"
    selection_start: str = "2023-01-01T00:00:00Z"
    selection_end_exclusive: str = "2024-01-01T00:00:00Z"


FROZEN_CONFIG = PolicyConfig()

SUPPORT_GATES = {
    "minimum_train_events": 120,
    "minimum_selection_events": 50,
    "minimum_events_each_train_year": 45,
    "minimum_events_each_selection_half": 20,
    "minimum_side_share_train": 0.25,
    "maximum_side_share_train": 0.75,
    "minimum_side_share_selection": 0.25,
    "maximum_side_share_selection": 0.75,
    "maximum_entry_month_share": 0.15,
    "maximum_exact_entry_jaccard": 0.10,
    "novelty_containment_hours": 6,
    "maximum_bidirectional_novelty_containment": 0.35,
    "minimum_common_candidate_events": 10,
    "minimum_common_comparator_events": 5,
    "stop_if_failed": True,
}

SOURCE_ONLY_CONTROLS = (
    "no_gross_tail",
    "no_imbalance_floor",
    "stale_6h",
)


def _path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_frozen_inputs() -> None:
    if sha256_file(SOURCE_CSV) != SOURCE_CSV_SHA256:
        raise ValueError("UGCI-288 source CSV hash mismatch")
    if sha256_file(SOURCE_MANIFEST) != SOURCE_MANIFEST_SHA256:
        raise ValueError("UGCI-288 source manifest hash mismatch")
    manifest = json.loads(_path(SOURCE_MANIFEST).read_text(encoding="utf-8"))
    boundary = manifest.get("outcome_boundary", {})
    if boundary.get("source_only") is not True:
        raise ValueError("UGCI-288 source manifest is not source-only")
    if boundary.get("pnl_cagr_mdd_opened") is not False:
        raise ValueError("UGCI-288 source manifest opened outcomes")
    for key in (
        "btc_market_rows_read",
        "funding_rows_read",
        "future_return_rows_read",
        "post_2023_contract_event_rows_read",
    ):
        if boundary.get(key) != 0:
            raise ValueError(f"UGCI-288 source manifest violates {key}")
    if manifest.get("output", {}).get("sha256") != SOURCE_CSV_SHA256:
        raise ValueError("UGCI-288 source manifest no longer binds the source CSV")
    for comparator in COMPARATORS:
        if sha256_file(comparator["path"]) != comparator["sha256"]:
            raise ValueError(
                f"UGCI-288 comparator hash mismatch: {comparator['candidate']}"
            )


def _validate_config(cfg: PolicyConfig = FROZEN_CONFIG) -> None:
    if cfg != FROZEN_CONFIG:
        raise ValueError("UGCI-288 policy configuration is frozen")
    if cfg.packet_hours != 6 or 24 % cfg.packet_hours:
        raise ValueError("UGCI-288 packet grid must remain six-hour UTC")
    if cfg.hold_bars * cfg.bar_minutes != 24 * 60:
        raise ValueError("UGCI-288 hold must remain exactly 24 hours")
    if cfg.entry_delay_minutes != 2 * cfg.bar_minutes:
        raise ValueError("UGCI-288 must retain one full latency bar plus next open")
    if not 0 < cfg.gross_tail_quantile < 1:
        raise ValueError("UGCI-288 gross quantile is invalid")
    if not 0 < cfg.minimum_imbalance_ratio <= 1:
        raise ValueError("UGCI-288 imbalance floor is invalid")


def preregistration_payload() -> dict[str, Any]:
    """Return the frozen contract without parsing any source or comparator row."""
    _validate_config()
    _validate_frozen_inputs()
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "candidate": CANDIDATE,
        "as_of_date": AS_OF_DATE,
        "decision": "freeze_source_support_before_any_outcome",
        "source": {
            "csv": str(SOURCE_CSV),
            "csv_sha256": SOURCE_CSV_SHA256,
            "manifest": str(SOURCE_MANIFEST),
            "manifest_sha256": SOURCE_MANIFEST_SHA256,
            "rows_parsed_during_preregistration": 0,
            "numeric_fields_parsed_during_preregistration": 0,
            "required_asset": "usdc_eth",
            "required_events": ["mint", "burn"],
            "availability_field": "available_at",
        },
        "policy": {
            "config": asdict(FROZEN_CONFIG),
            "packet_formulas": {
                "mint_raw": "sum(amount_raw where event == mint)",
                "burn_raw": "sum(amount_raw where event == burn)",
                "gross_raw": "mint_raw + burn_raw",
                "net_raw": "mint_raw - burn_raw",
                "imbalance_ratio": "abs(net_raw) / gross_raw",
            },
            "threshold": (
                "nearest-rank q95 of gross_raw on the strictly prior 180-day "
                "six-hour grid, current packet excluded"
            ),
            "direction": "LONG if net_raw > 0; SHORT if net_raw < 0",
            "entry": "packet_end + 10 minutes BTCUSDT USD-M perpetual open",
            "exit": "scheduled open after exactly 288 five-minute bars",
            "source_only_controls": list(SOURCE_ONLY_CONTROLS),
            "later_economic_controls": [
                "direction_flip",
                "deterministic_random_side",
                "double_cost",
                "extra_latency_1h",
            ],
        },
        "support_gate": dict(SUPPORT_GATES),
        "novelty_comparators": [dict(item) for item in COMPARATORS],
        "outcome_boundary": {
            "outcomes_opened": False,
            "outcome_sources_opened": False,
            "post_2023_source_rows_opened": False,
            "btc_market_rows_read": 0,
            "funding_rows_read": 0,
            "future_return_rows_read": 0,
            "return_or_pnl_fields_read": 0,
            "comparator_rows_read": 0,
            "network_calls": 0,
            "subprocess_calls": 0,
        },
        "one_way_sequence": {
            "support_before_evaluator": True,
            "evaluator_committed_before_market": True,
            "train_2021_2022_before_selection_2023": True,
            "post_2023_remains_sealed": True,
            "failure_action": "retire UGCI-288 without repair",
        },
        "later_metrics_required": [
            "absolute_return",
            "full_calendar_cagr",
            "strict_mdd",
            "cagr_to_strict_mdd",
            "trades",
            "long_short_counts",
            "realized_funding",
            "base_and_stress_costs",
        ],
        "rllm_boundary": {
            "allowed_only_after_deterministic_economics_pass": True,
            "may_abstain_or_size_only_under_separate_freeze": True,
            "may_create_retime_reverse_or_repair_events": False,
        },
        "preregistration_source": str(PREREGISTRATION_SOURCE),
        "preregistration_source_sha256": sha256_file(PREREGISTRATION_SOURCE),
        "preregistration_document": str(PREREGISTRATION_DOCUMENT),
        "preregistration_document_sha256": sha256_file(PREREGISTRATION_DOCUMENT),
    }
    payload["manifest_hash"] = canonical_hash(payload)
    return payload


def write_payload(output: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    destination = _path(output)
    if destination.exists():
        raise FileExistsError(f"UGCI-288 artifact is write-once: {destination}")
    payload = preregistration_payload()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = write_payload(args.output)
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
