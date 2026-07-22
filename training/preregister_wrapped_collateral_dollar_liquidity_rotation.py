"""Freeze WCDR-2016 before source incidence or BTC outcomes are opened."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence


POLICY_ID = "WCDR-2016"
PROTOCOL_VERSION = "wrapped_collateral_dollar_liquidity_rotation_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(
    "training/preregister_wrapped_collateral_dollar_liquidity_rotation.py"
)
MECHANISM_DECISION = Path(
    "docs/wrapped-collateral-dollar-liquidity-rotation-"
    "mechanism-decision-2026-07-23.md"
)
MECHANISM_DECISION_SHA256 = (
    "4fe1176bcb221bf67a62c6a196440efdd3f2fcd22ff3b30a078ce868c6dc1e26"
)
DEFAULT_OUTPUT = Path(
    "results/wrapped_collateral_dollar_liquidity_rotation_"
    "preregistration_2026-07-23.json"
)

WBTC_SOURCE = Path(
    "data/wbtc_custody_bridge_flow_2020_2023/"
    "wbtc_mint_burn_2020_2023.csv.gz"
)
WBTC_SOURCE_SHA256 = (
    "bfcc6ebc2ded0cd8a57e5cda83a77daafe4de325adf606b23ba43ecf486b3b4e"
)
WBTC_MANIFEST = Path(
    "results/wbtc_custody_bridge_flow_source_manifest_2026-07-23.json"
)
WBTC_MANIFEST_FILE_SHA256 = (
    "e95267d55f390a35bf609580e014c67d44adabe67526022d5e80d555964274e8"
)
WBTC_MANIFEST_HASH = (
    "4e4344a7f2841803dc8da625ee1320f79e1821d54cb2366a5464728507b4bcab"
)
WBTC_BUILDER = Path("training/build_wbtc_custody_bridge_flow_source.py")
WBTC_BUILDER_SHA256 = (
    "70816fbcc94d5ecd11f99e3b1ebc3087e396c8f6972adcefd7c3a308f7c6fdbf"
)
WBTC_HEADER = (
    "asset",
    "contract_address",
    "event",
    "event_sign",
    "amount_raw",
    "decimals",
    "actor_address",
    "block_number",
    "block_hash",
    "block_timestamp",
    "transaction_hash",
    "transaction_index",
    "semantic_log_index",
    "companion_transfer_log_index",
    "confirmation_block_number",
    "confirmation_block_hash",
    "available_at",
)
WBTC_ALLOWED_COLUMNS = (
    "event",
    "event_sign",
    "amount_raw",
    "actor_address",
    "block_number",
    "transaction_index",
    "semantic_log_index",
    "available_at",
)

STABLECOIN_SOURCE = Path(
    "data/ethereum_stablecoin_issuance_redemption_2020_2023/"
    "ethereum_usdt_usdc_issuance_redemption_2020_2023.csv.gz"
)
STABLECOIN_SOURCE_SHA256 = (
    "70ba3799ba84dc671051623a8d167b1731f043cf84a686b9878a67fcd52e5901"
)
STABLECOIN_MANIFEST = Path(
    "results/ethereum_stablecoin_issuance_redemption_"
    "source_manifest_2026-07-21.json"
)
STABLECOIN_MANIFEST_FILE_SHA256 = (
    "8ec9ab08c413bf6f5f8170fb800b05105522d4cf1a7932943c214288701e31fe"
)
STABLECOIN_MANIFEST_HASH = (
    "a0c7740db64f7779fade68d76985c629cabe81983bf594e8258cef16a5725a1b"
)
STABLECOIN_BUILDER = Path(
    "training/build_ethereum_stablecoin_issuance_redemption.py"
)
STABLECOIN_BUILDER_SHA256 = (
    "af699d91404e44298573ca148c6cf1b90b68d9aac2972da9ad228a26b9e8a9a5"
)
STABLECOIN_HEADER = (
    "asset",
    "contract_address",
    "event",
    "event_sign",
    "amount_raw",
    "decimals",
    "indexed_address_1",
    "indexed_address_2",
    "data_address",
    "block_number",
    "block_hash",
    "block_timestamp",
    "transaction_hash",
    "transaction_index",
    "log_index",
    "confirmation_block_number",
    "confirmation_block_hash",
    "available_at",
)
USDC_ALLOWED_COLUMNS = (
    "asset",
    "event",
    "event_sign",
    "amount_raw",
    "indexed_address_1",
    "block_number",
    "transaction_index",
    "log_index",
    "available_at",
)

EXPECTED_OUTCOME_BOUNDARY = {
    "source_file_bytes_hashed": True,
    "source_csv_headers_read": 2,
    "source_value_rows_read": 0,
    "source_feature_rows_derived": 0,
    "candidate_incidence_rows_derived": 0,
    "comparator_rows_read": 0,
    "btc_market_rows_read": 0,
    "funding_rows_read": 0,
    "future_return_rows_read": 0,
    "pnl_cagr_mdd_opened": False,
    "post_2023_contract_event_rows_read": 0,
    "network_calls": 0,
    "subprocess_calls": 0,
}


@dataclass(frozen=True)
class Config:
    output: str = str(DEFAULT_OUTPUT)


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    if str(path).startswith("~") or candidate.is_absolute() or ".." in candidate.parts:
        raise RuntimeError("WCDR path must be repository-relative")
    root = REPOSITORY_ROOT.resolve(strict=True)
    current = REPOSITORY_ROOT
    for part in candidate.parts:
        current /= part
        if current.is_symlink():
            raise RuntimeError("WCDR repository path contains a symlink")
        if not current.exists():
            break
    target = REPOSITORY_ROOT / candidate
    try:
        target.resolve(strict=False).relative_to(root)
    except ValueError as error:
        raise RuntimeError("WCDR path escapes the repository") from error
    return target


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: str | Path) -> str:
    return _sha256_path(_repository_path(path))


def _validate_regular_file(
    path: str | Path, expected_sha256: str, label: str
) -> str:
    target = _repository_path(path)
    if target.is_symlink() or not target.is_file():
        raise RuntimeError(f"{label} is missing or not a regular file")
    observed = _sha256_path(target)
    if observed != expected_sha256:
        raise RuntimeError(f"{label} SHA-256 drift")
    return observed


def _read_gzip_header(path: str | Path) -> tuple[str, ...]:
    target = _repository_path(path)
    with gzip.open(target, "rt", encoding="utf-8", newline="") as handle:
        try:
            return tuple(next(csv.reader(handle)))
        except StopIteration as error:
            raise RuntimeError("WCDR source CSV is empty") from error


def _source_binding(
    *,
    name: str,
    source: Path,
    source_sha256: str,
    manifest: Path,
    manifest_file_sha256: str,
    manifest_hash: str,
    builder: Path,
    builder_sha256: str,
    expected_header: Sequence[str],
    allowed_columns: Sequence[str],
) -> dict[str, Any]:
    _validate_regular_file(source, source_sha256, f"WCDR {name} source")
    header = _read_gzip_header(source)
    if header != tuple(expected_header):
        raise RuntimeError(f"WCDR {name} source header drift")
    if any(column not in header for column in allowed_columns):
        raise RuntimeError(f"WCDR {name} allowed-column drift")
    _validate_regular_file(
        manifest, manifest_file_sha256, f"WCDR {name} source manifest"
    )
    _validate_regular_file(builder, builder_sha256, f"WCDR {name} source builder")
    return {
        "source": str(source),
        "source_sha256": source_sha256,
        "manifest": str(manifest),
        "manifest_file_sha256": manifest_file_sha256,
        "manifest_hash": manifest_hash,
        "builder": str(builder),
        "builder_sha256": builder_sha256,
        "header": list(header),
        "allowed_columns": list(allowed_columns),
        "value_rows_read_during_preregistration": 0,
    }


def source_bindings() -> dict[str, Any]:
    return {
        "wbtc": _source_binding(
            name="WBTC",
            source=WBTC_SOURCE,
            source_sha256=WBTC_SOURCE_SHA256,
            manifest=WBTC_MANIFEST,
            manifest_file_sha256=WBTC_MANIFEST_FILE_SHA256,
            manifest_hash=WBTC_MANIFEST_HASH,
            builder=WBTC_BUILDER,
            builder_sha256=WBTC_BUILDER_SHA256,
            expected_header=WBTC_HEADER,
            allowed_columns=WBTC_ALLOWED_COLUMNS,
        ),
        "usdc": _source_binding(
            name="USDC",
            source=STABLECOIN_SOURCE,
            source_sha256=STABLECOIN_SOURCE_SHA256,
            manifest=STABLECOIN_MANIFEST,
            manifest_file_sha256=STABLECOIN_MANIFEST_FILE_SHA256,
            manifest_hash=STABLECOIN_MANIFEST_HASH,
            builder=STABLECOIN_BUILDER,
            builder_sha256=STABLECOIN_BUILDER_SHA256,
            expected_header=STABLECOIN_HEADER,
            allowed_columns=USDC_ALLOWED_COLUMNS,
        ),
    }


def policy_payload() -> dict[str, Any]:
    return {
        "candidate": POLICY_ID,
        "singleton": True,
        "multiple_testing_hypotheses": 1,
        "economic_hypothesis": {
            "long": (
                "WBTC net burn with USDC net mint is a "
                "collateral-unwind/dollar-replenishment state"
            ),
            "short": (
                "WBTC net mint with USDC net burn is collateral expansion "
                "against dollar contraction"
            ),
            "no_trade": "same-sign or zero-sign source states are ambiguous",
            "wbtc_mint_long_or_burn_short_claimed": False,
        },
        "source_rows": {
            "wbtc": {
                "asset": "wbtc_eth",
                "events": ["mint", "burn"],
                "decimals": 8,
            },
            "usdc": {
                "asset": "usdc_eth",
                "events": ["mint", "burn"],
                "decimals": 6,
                "usdt_or_black_funds_allowed": False,
            },
        },
        "causal_state": {
            "decision_grid": "00:00:00 UTC each calendar day",
            "source_cutoff": "decision_time - 6 elapsed hours",
            "only_clock_field": "available_at",
            "block_timestamp_forbidden": True,
            "windows": {
                "wbtc_calendar_days": 30,
                "usdc_calendar_days": 7,
                "interval": "cutoff-lookback < available_at <= cutoff",
            },
            "wbtc_validity": {
                "gross_raw_positive": True,
                "minimum_events": 3,
                "minimum_distinct_actors": 2,
                "maximum_top_actor_gross_share": 0.90,
                "actor_must_be_nonzero_ethereum_address": True,
            },
            "usdc_validity": {
                "gross_raw_positive": True,
                "minimum_events": 30,
            },
            "side": {
                "long": "wbtc_net_raw < 0 and usdc_net_raw > 0",
                "short": "wbtc_net_raw > 0 and usdc_net_raw < 0",
                "otherwise": "no candidate",
            },
            "forbidden_transforms": [
                "amount clipping",
                "quantile threshold",
                "forward fill",
                "full-sample normalization",
                "entity or exchange address classification",
            ],
        },
        "execution": {
            "decision_time": "daily UTC anchor",
            "entry_delay_minutes": 5,
            "hold_bars_5m": 2016,
            "hold_elapsed_days": 7,
            "notional_exposure": 0.5,
            "global_nonoverlap": True,
            "accept_when_entry_at_or_after_prior_exit": True,
            "split_crossing_action": "skip",
            "stops_take_profit_or_trailing_exit": False,
        },
        "windows": {
            "warmup": ["2020-01-01T00:00:00Z", "2021-01-01T00:00:00Z"],
            "train": ["2021-01-01T00:00:00Z", "2023-01-01T00:00:00Z"],
            "selection": ["2023-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "sealed_from": "2024-01-01T00:00:00Z",
        },
        "source_support_gates": {
            "train_total_minimum": 50,
            "selection_total_minimum": 20,
            "each_train_year_minimum": 20,
            "each_train_half_year_minimum": 8,
            "each_selection_half_year_minimum": 8,
            "train_each_side_minimum": 12,
            "selection_each_side_minimum": 4,
            "maximum_month_share": 0.20,
            "maximum_consecutive_same_side": 10,
            "train_distinct_wbtc_actors_minimum": 10,
            "selection_distinct_wbtc_actors_minimum": 5,
            "failure_action": "reject WCDR-2016 without repair or outcomes",
        },
        "controls": {
            "direction_flip": "exact primary clock with both sides reversed",
            "wbtc_only_contrarian": "valid WBTC states; side=-sign(wbtc_net_raw)",
            "usdc_only_direct": "valid USDC states; side=sign(usdc_net_raw)",
            "same_sign_direct": "equal nonzero signs; side=common sign",
            "stale_7d": "primary with both source cutoffs delayed seven days",
            "count_sign_consensus": (
                "primary plus amount/count net-sign agreement in both sources"
            ),
            "year_amount_permutation": (
                "deterministic within-source/event/year amount permutation"
            ),
            "deterministic_random_side": (
                "SHA-256 fixed side-count-matched primary-side permutation"
            ),
        },
        "economic_sequence": [
            "source-only support",
            "freeze strict evaluator",
            "train 2021-2022",
            "selection 2023 only after train pass",
            "immutable post-2023 source extension only after pre-2024 pass",
            "test 2024",
            "eval 2025",
            "recent 2026",
        ],
        "strict_economic_gates": {
            "absolute_return_positive": True,
            "cagr_to_strict_mdd_minimum": 3.0,
            "strict_mdd_pct_maximum": 15.0,
            "ten_bp_notional_side_stress_return_positive": True,
            "calendar_month_cluster_signflip_p_maximum": 0.10,
            "base_cost_notional_per_side": 0.0006,
            "stress_cost_notional_per_side": 0.0010,
            "realized_funding": True,
            "full_calendar_cagr": True,
            "strict_intratrade_high_water_mdd": True,
            "must_beat_wbtc_and_usdc_component_controls": True,
            "flip_stale_or_random_full_qualification_rejects": True,
        },
        "post_2023_extension": {
            "same_contracts_topics_confirmation_and_feature_policy": True,
            "new_bridges_assets_stablecoins_or_address_labels_allowed": False,
            "wbtc_open_api_dates_allowed": False,
        },
        "rllm_boundary": {
            "authorized_before_deterministic_train_and_selection_pass": False,
            "later_inputs": [
                "bucketed WBTC net/gross ratio",
                "bucketed USDC net/gross ratio",
                "event-count balances",
                "WBTC actor concentration",
                "state age",
                "current position",
                "time to exit",
            ],
            "later_actions": ["TRADE_FIXED_SIDE", "ABSTAIN"],
            "may_create_clock_reverse_side_or_change_hold": False,
            "reward_penalties": ["strict drawdown", "turnover"],
        },
        "stopping_rule": (
            "any source, support, integrity, or staged economic failure rejects "
            "WCDR-2016; repair requires a new candidate identity frozen before access"
        ),
    }


def build_preregistration(cfg: Config | None = None) -> dict[str, Any]:
    frozen_cfg = Config() if cfg is None else cfg
    mechanism_sha = _validate_regular_file(
        MECHANISM_DECISION,
        MECHANISM_DECISION_SHA256,
        "WCDR mechanism decision",
    )
    policy = policy_payload()
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "candidate": POLICY_ID,
        "config": asdict(frozen_cfg),
        "mechanism_decision": {
            "path": str(MECHANISM_DECISION),
            "sha256": mechanism_sha,
        },
        "preregistration_source": {
            "path": str(SCRIPT_PATH),
            "sha256": sha256_file(SCRIPT_PATH),
        },
        "source_bindings": source_bindings(),
        "policy": policy,
        "policy_hash": canonical_hash(policy),
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "performance_values_opened": False,
        "outcome_boundary": dict(EXPECTED_OUTCOME_BOUNDARY),
        "next_action": "build exact source-only WCDR-2016 support clocks",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate_preregistration(
    payload: Mapping[str, Any], *, verify_sources: bool = True
) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("WCDR preregistration canonical hash mismatch")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("WCDR preregistration protocol drift")
    if payload.get("candidate") != POLICY_ID:
        raise RuntimeError("WCDR preregistration candidate drift")
    if payload.get("policy") != policy_payload():
        raise RuntimeError("WCDR preregistration policy drift")
    if payload.get("policy_hash") != canonical_hash(policy_payload()):
        raise RuntimeError("WCDR policy hash drift")
    if payload.get("outcomes_opened") is not False:
        raise RuntimeError("WCDR preregistration opened outcomes")
    if payload.get("source_incidence_opened") is not False:
        raise RuntimeError("WCDR preregistration opened source incidence")
    if payload.get("outcome_boundary") != EXPECTED_OUTCOME_BOUNDARY:
        raise RuntimeError("WCDR preregistration outcome boundary drift")
    if verify_sources:
        config = payload.get("config")
        if not isinstance(config, Mapping):
            raise RuntimeError("WCDR preregistration config missing")
        expected = build_preregistration(Config(**dict(config)))
        if dict(payload) != expected:
            raise RuntimeError("WCDR preregistration binding drift")


def _protected_paths() -> set[Path]:
    return {
        _repository_path(SCRIPT_PATH),
        _repository_path(MECHANISM_DECISION),
        _repository_path(WBTC_SOURCE),
        _repository_path(WBTC_MANIFEST),
        _repository_path(WBTC_BUILDER),
        _repository_path(STABLECOIN_SOURCE),
        _repository_path(STABLECOIN_MANIFEST),
        _repository_path(STABLECOIN_BUILDER),
    }


def write_preregistration(cfg: Config | None = None) -> tuple[dict[str, Any], str]:
    frozen_cfg = Config() if cfg is None else cfg
    output = _repository_path(frozen_cfg.output)
    if output.suffix != ".json":
        raise ValueError("WCDR preregistration output must be JSON")
    if output in _protected_paths():
        raise ValueError("WCDR preregistration output aliases a protected input")
    payload = build_preregistration(frozen_cfg)
    validate_preregistration(payload)
    if output.exists() or output.is_symlink():
        if output.is_symlink() or not output.is_file():
            raise FileExistsError("WCDR preregistration output is not a regular file")
        existing = json.loads(output.read_text(encoding="utf-8"))
        validate_preregistration(existing)
        if existing != payload:
            raise RuntimeError("refusing to overwrite WCDR preregistration")
        return payload, "verified_existing"
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=output.parent, prefix=f".{output.name}.", delete=False
    ) as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return payload, "created"


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return Config(**vars(parser.parse_args()))


def main() -> None:
    payload, status = write_preregistration(parse_args())
    print(
        json.dumps(
            {
                "candidate": payload["candidate"],
                "manifest_hash": payload["manifest_hash"],
                "outcomes_opened": payload["outcomes_opened"],
                "source_incidence_opened": payload["source_incidence_opened"],
                "status": status,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
