"""Write the outcome-blind BLSR-288 singleton preregistration.

Only frozen manifest metadata and exact artifact byte hashes are inspected.
The confirmed-ledger CSV and comparator clocks are never decompressed or
parsed, and no BTC market or funding outcome is opened.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

from training import (
    preregister_fee_endpoint_topology_disagreement as source_contract,
)


POLICY_ID = "BLSR-288"
PROTOCOL_VERSION = "blockspace_load_settlement_relay_preregistration_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

SOURCE_MANIFEST = source_contract.SOURCE_MANIFEST
SOURCE_VALIDATOR = Path("training/preregister_fee_endpoint_topology_disagreement.py")
SOURCE_VALIDATOR_SHA256 = (
    "ae1329f0cd124787d822096e56dc3bc3ed05ccd2f2f6f0cb86f47e5cd766c413"
)
MECHANISM_DECISION = Path(
    "docs/blockspace-load-settlement-relay-mechanism-decision-2026-07-21.md"
)
MECHANISM_DECISION_SHA256 = (
    "dd7a5ddc67710b06537e6839c0519218508ed2190d1b799d548ad11f704a33f2"
)
PREREGISTRATION_SOURCE = Path(
    "training/preregister_blockspace_load_settlement_relay.py"
)
DEFAULT_OUTPUT = Path(
    "results/blockspace_load_settlement_relay_preregistration_2026-07-21.json"
)

COMPARATOR_BINDINGS: dict[str, dict[str, str]] = {
    "bate_288_primary_clock": {
        "path": "results/block_arrival_throughput_elasticity_clock_2026-07-20.csv",
        "sha256": "cd4fbd01c104bd969ca1c12a53b8da82dd0e9376990e233c286ff009a5115c02",
        "role": "directional_interval_clock",
    },
    "ufcp_1_primary_clock": {
        "path": "results/utxo_fee_clearing_polarity_primary_clock_2026-07-20.csv",
        "sha256": "8338c290d63b522531c8d55c8a79ba73cc13915c936733ec03ffcf6ab0e86c1b",
        "role": "directional_interval_clock",
    },
    "wctr_288_primary_clock": {
        "path": "results/witness_composition_transport_primary_clock_2026-07-20.csv.gz",
        "sha256": "7a6b56a3024d0d087322fad7b3229276c539b93374691cd2812af0630dc752b1",
        "role": "directional_interval_clock",
    },
    "fetd_288_preregistration": {
        "path": "results/fee_endpoint_topology_disagreement_preregistration_2026-07-20.json",
        "sha256": "2de820b6f78d0cd566f2750f91bfca8c092795ab93b81121326bdb067247e285",
        "role": "rebuild_contract",
    },
    "fetd_288_support": {
        "path": "results/fee_endpoint_topology_disagreement_support_2026-07-20.json",
        "sha256": "03ba910a314ba6efb647f6588dff603261d414e5114680ca33bdc27d59aed035",
        "role": "sealed_clock_commitment",
    },
    "fetd_288_support_builder": {
        "path": "training/build_fee_endpoint_topology_disagreement_support.py",
        "sha256": "1d1330415d6b22f0ebe32719dd6b5232cb4df28b08c2f0ee2942f15aa7c6f01d",
        "role": "deterministic_clock_rebuilder",
    },
    "prior_microstructure_bundle": {
        "path": "results/cdltr_prior_comparator_views_2026-07-21.csv.gz",
        "sha256": "bffdcf158d7d4e38db5794fb4761de528fb73b0b772ae950f3a087a93ab63f1a",
        "role": "mixed_directional_and_timestamp_clock_bundle",
    },
    "prior_microstructure_bundle_manifest": {
        "path": "results/cdltr_prior_comparator_views_manifest_2026-07-21.json",
        "sha256": "a795f384287f24200e00d2cc5a5721610bb5282d1b044b3a653a053190c44261",
        "role": "bundle_schema_and_provenance",
    },
}

OUTCOME_BOUNDARY = {
    "source_manifest_json_read": True,
    "source_artifact_bytes_hashed": True,
    "source_csv_values_read": 0,
    "comparator_artifact_bytes_hashed": True,
    "comparator_rows_read": 0,
    "blsr_feature_rows_derived": 0,
    "signal_incidence_rows_derived": 0,
    "market_rows_loaded": 0,
    "funding_rows_loaded": 0,
    "return_or_pnl_fields": 0,
    "post_2023_rows_loaded": 0,
    "network_calls": 0,
    "subprocess_calls": 0,
}

RESEARCH_SEQUENCE = {
    "support_first": "source-only train/selection incidence and novelty",
    "evaluator_second": "commit and hash-freeze strict evaluator",
    "outcomes": "train first; each later window only after exact pass",
    "sealed": "2024+ source and outcomes",
}

ARTIFACT_TOP_LEVEL_KEYS = frozenset(
    {
        "protocol_version",
        "policy_id",
        "config",
        "source_manifest",
        "source_validator",
        "mechanism_decision",
        "comparator_bindings",
        "policy",
        "policy_hash",
        "outcomes_opened",
        "outcome_boundary",
        "research_sequence",
        "preregistration_source",
        "manifest_hash",
    }
)

CONTROL_DEFINITIONS = {
    "fee_only": (
        "each fee_change with fee_magnitude_rank>=0.75 and nonzero sign; "
        "side=sign(fee_change); independent chronological nonoverlap"
    ),
    "endpoint_only": (
        "each endpoint_change with endpoint_magnitude_rank>=0.75 and nonzero "
        "sign; side=sign(endpoint_change); independent chronological nonoverlap"
    ),
    "same_packet_agreement": (
        "both changes significant and same-signed in one packet; side equals "
        "the common sign; independent chronological nonoverlap"
    ),
    "reverse_order_relay": (
        "significant endpoint change starts an episode; the first significant "
        "fee change in the next three packets must agree"
    ),
    "opposite_response_relay": (
        "exact primary fee onset and three-packet deadline, but emit only when "
        "the first significant endpoint response disagrees; side remains the "
        "fee-load sign"
    ),
    "one_packet_stale_response": (
        "shift the confirming endpoint state by one complete packet and apply "
        "it only at the later packet availability"
    ),
    "direction_flip": "exact primary entries/exits with every side multiplied by -1",
    "deterministic_random_side": (
        "exact primary entries/exits; SHA256('BLSR-288-random-side-20260721|' "
        "+ entry_time), first digest byte<128 LONG else SHORT"
    ),
    "one_bar_latency": (
        "shift primary entry and exit exactly 300 seconds; drop without "
        "replacement when the shifted trade leaves its original split"
    ),
}


@dataclass(frozen=True)
class Config:
    source_manifest: str = str(SOURCE_MANIFEST)
    preregistration_output: str = str(DEFAULT_OUTPUT)


def _repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPOSITORY_ROOT / candidate
    return candidate.resolve()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _repository_path(path).open("rb") as handle:
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


def _unique_object(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError(f"BLSR JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(
        _repository_path(path).read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
    )
    if not isinstance(payload, dict):
        raise RuntimeError("BLSR JSON must be an object")
    return payload


def _manifest_core(manifest: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key != "manifest_hash"}


def _validate_config(cfg: Config, *, require_new_output: bool) -> None:
    manifest = _repository_path(cfg.source_manifest)
    output = _repository_path(cfg.preregistration_output)
    if manifest != _repository_path(SOURCE_MANIFEST):
        raise RuntimeError("BLSR source manifest path differs from frozen source")
    if manifest.suffix != ".json" or output.suffix != ".json":
        raise ValueError("BLSR source manifest and preregistration must be JSON")
    protected = {
        manifest,
        _repository_path(source_contract.EXPECTED_SOURCE_OUTPUT),
        _repository_path(SOURCE_VALIDATOR),
        _repository_path(MECHANISM_DECISION),
        _repository_path(PREREGISTRATION_SOURCE),
        *(
            _repository_path(binding["path"])
            for binding in COMPARATOR_BINDINGS.values()
        ),
    }
    if output in protected:
        raise ValueError("BLSR preregistration output aliases a protected input")
    if require_new_output and output.exists():
        raise FileExistsError("BLSR preregistration is immutable")


def _validate_source_manifest(path: str | Path) -> dict[str, Any]:
    if sha256_file(SOURCE_VALIDATOR) != SOURCE_VALIDATOR_SHA256:
        raise RuntimeError("BLSR frozen source validator SHA drift")
    binding = source_contract._validate_source_manifest(path)
    if binding["source_output"]["path"] != str(source_contract.EXPECTED_SOURCE_OUTPUT):
        raise RuntimeError("BLSR source output path drift")
    if binding["source_output"]["sha256"] != (
        source_contract.EXPECTED_SOURCE_OUTPUT_SHA256
    ):
        raise RuntimeError("BLSR source output SHA drift")
    return binding


def _validate_comparators() -> dict[str, dict[str, str]]:
    validated: dict[str, dict[str, str]] = {}
    for name, binding in COMPARATOR_BINDINGS.items():
        path = _repository_path(binding["path"])
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"BLSR comparator {name} is missing or symlinked")
        if sha256_file(path) != binding["sha256"]:
            raise RuntimeError(f"BLSR comparator {name} SHA drift")
        validated[name] = dict(binding)
    return validated


def policy() -> dict[str, Any]:
    return {
        "policy_id": POLICY_ID,
        "singleton": True,
        "source_features": {
            "packet_blocks": 72,
            "packet_alignment": "packet_id=floor(height/72)",
            "first_complete_packet_start_height": 610_704,
            "last_complete_packet_end_height": 823_751,
            "complete_packet_count": 2_959,
            "edge_packet_policy": "drop incomplete first and last packets",
            "packet_weight": "sum(weight)",
            "packet_fees": "sum(total_fees)",
            "packet_endpoints": "sum(total_inputs+total_outputs)",
            "fee_pressure": "log(packet_fees/packet_weight)",
            "endpoint_density": "log(packet_endpoints/packet_weight)",
            "fee_change": "fee_pressure[t]-fee_pressure[t-1]",
            "endpoint_change": "endpoint_density[t]-endpoint_density[t-1]",
            "base_valid_feature_row": (
                "require t-1,t as consecutive valid absolute-height packets; "
                "each packet has exactly 72 contiguous linked blocks and "
                "positive finite packet_weight, packet_fees, packet_endpoints"
            ),
            "forbidden_primary_fields": [
                "utxo_set_change",
                "tx_count",
                "size",
                "mediantime",
                "market",
                "funding",
                "premium",
                "open_interest",
                "liquidation",
                "order_book",
                "return",
                "pnl",
            ],
            "numeric_contract": (
                "IEEE-754 binary64 and natural log; no epsilon, clipping, "
                "rounding, interpolation, forward fill, reassignment, or "
                "imputation; exact binary64 equality defines zero and ties"
            ),
        },
        "normalization": {
            "method": "strict-prior rolling empirical midrank of absolute change",
            "midrank_formula": (
                "(count(prior < current)+0.5*count(prior == current))/prior_count"
            ),
            "lookback_valid_packet_changes": 180,
            "minimum_prior_valid_packet_changes": 120,
            "window_selection": (
                "use the most recent 180 strict-prior base-valid changes when "
                "available; fewer than 120 makes current rank-unready"
            ),
            "current_row_excluded": True,
            "prior_available_at_strictly_before_current": True,
            "tie_equality": "exact IEEE-754 binary64 equality",
            "fee_magnitude_rank": "midrank(abs(fee_change))",
            "endpoint_magnitude_rank": "midrank(abs(endpoint_change))",
            "significance_boundary": 0.75,
            "parameter_grid": [],
        },
        "relay": {
            "processing_order": "strictly increasing packet_id",
            "availability_order": (
                "source_available_at must be strictly increasing with packet_id; "
                "tie or regression rejects the source clock"
            ),
            "onset": (
                "when inactive, first nonzero fee_change with "
                "fee_magnitude_rank>=0.75 starts load_sign=sign(fee_change)"
            ),
            "active_fee_shocks": "ignored until current episode resolves or expires",
            "deadline_packets_after_onset": 3,
            "first_response": (
                "first later nonzero endpoint_change with "
                "endpoint_magnitude_rank>=0.75 resolves the episode"
            ),
            "confirm": "sign(endpoint_change)==load_sign emits one candidate",
            "cancel": "opposite first significant endpoint response emits none",
            "expire": "no significant endpoint response by third later packet",
            "no_retry": True,
            "restart": "only the packet after the resolved/expired inspected packet",
            "side": "load_sign; positive LONG, negative SHORT",
        },
        "causal_availability": {
            "hash_linked_successors_after_packet_end": 6,
            "source_available_at": (
                "max header timestamp from packet start through h+6 + 48 hours"
            ),
            "availability_lag_seconds": 172_800,
            "ceil_5m": "((unix_seconds+299)//300)*300",
            "entry_time": "ceil_5m(confirming_source_available_at)+300 seconds",
            "entry_latency_seconds": 300,
            "already_aligned_rule": "aligned availability still receives 300s",
            "source_gap_action": "reject; never fill, reorder, or backdate",
        },
        "scheduling": {
            "candidate_order": ("entry_time,onset_packet_id,confirmation_packet_id"),
            "acceptance": "entry_time>=prior accepted exit; equal boundary allowed",
            "overlap_action": "suppress without priority or replacement",
            "global_nonoverlap": True,
            "split_containment": ("entry>=split_start and scheduled_exit<=split_end"),
        },
        "execution": {
            "bar_size": "5m",
            "entry": "next open exactly at entry_time",
            "hold_bars": 288,
            "hold_hours": 24,
            "scheduled_exit_time": "entry_time+86400 seconds",
            "interval": "[entry_time,scheduled_exit_time)",
            "notional_leverage": 0.5,
            "base_cost_bp_per_notional_per_side": 6,
            "stress_cost_bp_per_notional_per_side": 10,
            "funding": (
                "exact entry-inclusive/exit-exclusive marks at fixed entry quantity"
            ),
        },
        "calendar": {
            "warmup_source": "calendar 2020 only",
            "train": "[2021-01-01T00:00:00Z,2023-01-01T00:00:00Z)",
            "selection": "[2023-01-01T00:00:00Z,2024-01-01T00:00:00Z)",
            "sealed": "2024+",
            "fit_permission": "no fitted coefficient or threshold grid",
        },
        "support_gates": {
            "count_basis": (
                "accepted primary entries after relay, containment, and nonoverlap"
            ),
            "train_total_minimum": 80,
            "train_each_year_minimum": 30,
            "train_each_half_year_minimum": 12,
            "train_long_minimum": 24,
            "train_short_minimum": 24,
            "train_each_side_each_year_minimum": 8,
            "train_maximum_month_share": 0.20,
            "train_maximum_weekday_share": 0.25,
            "selection_total_minimum": 35,
            "selection_each_half_minimum": 14,
            "selection_each_quarter_minimum": 6,
            "selection_long_minimum": 12,
            "selection_short_minimum": 12,
            "selection_each_side_each_half_minimum": 4,
            "selection_maximum_month_share": 0.20,
            "selection_maximum_weekday_share": 0.25,
            "support_failure_action": "reject BLSR-288 before outcomes; no repair",
        },
        "novelty_gates": {
            "comparators": [
                "FETD-288",
                "BATE-288",
                "UFCP-1",
                "WCTR-288",
                "prior_microstructure_bundle",
            ],
            "exact_entry_timestamp_jaccard_maximum": 0.20,
            "candidate_one_to_one_within_six_hours_fraction_maximum": 0.35,
            "signed_occupied_exposure_absolute_pearson_maximum": 0.40,
            "exposure_grid": "full [2021-01-01,2024-01-01) UTC at 5m",
            "timestamp_only_rule": "omit only signed exposure correlation",
            "all_nonempty_comparators_must_pass": True,
            "comparator_removal_after_incidence": "forbidden",
        },
        "performance_gates": {
            "required_sequence": ["train", "selection", "2024", "2025", "2026"],
            "absolute_return_positive_each": True,
            "cagr_to_strict_mdd_minimum_each": 3.0,
            "strict_max_drawdown_maximum_each": 0.15,
            "weekly_cluster_sign_flip_p_maximum_each": 0.10,
            "mean_gross_bp_minimum_each": 30.0,
            "stress_absolute_return_positive_each": True,
            "one_bar_latency_absolute_return_positive_each": True,
            "positive_subperiods": ["2021", "2022", "2023H1", "2023H2"],
            "long_and_short_absolute_return_positive_train_selection": True,
            "mechanism_control_minimum_ratio_margin": 0.25,
            "mechanism_controls": [
                "fee_only",
                "endpoint_only",
                "same_packet_agreement",
                "reverse_order_relay",
                "opposite_response_relay",
                "one_packet_stale_response",
            ],
            "control_replacement": "forbidden under BLSR-288",
            "post_outcome_repair": "forbidden",
        },
        "strict_accounting": {
            "required_report_fields": [
                "absolute_return",
                "cagr",
                "strict_mdd",
                "cagr_to_strict_mdd",
                "trades",
                "long_trades",
                "short_trades",
                "calendar_clusters",
            ],
            "cagr_clock": "full declared wall-clock window including idle cash",
            "drawdown_high_water_mark": "global and pre-entry",
            "held_path_order": (
                "all favorable OHLC/funding-credit extremes before all adverse "
                "OHLC/funding-debit extremes"
            ),
            "costs": (
                "entry, scheduled exit, and hypothetical adverse liquidation costs"
            ),
            "cluster_test": (
                "one-sided weekly entry-cluster sign flip; 100000 draws; seed 20260721"
            ),
        },
        "controls": dict(CONTROL_DEFINITIONS),
        "rllm_boundary": (
            "only after deterministic standalone and orthogonality pass may a "
            "compact Gemma policy consume symbolic relay state to abstain or size"
        ),
        "promotion_boundary": (
            "pre-2024 pass required before separately frozen 2024+ source and "
            "outcome stages; live requires Bitcoin Core parity and 90 shadow days"
        ),
        "stopping_rule": (
            "stop permanently at first source/support/novelty/train/selection/"
            "test/eval/forward failure; no threshold, packet, rank, deadline, "
            "sign, availability, latency, hold, floor, comparator, or control repair"
        ),
    }


def _artifact_core(
    cfg: Config,
    source_binding: dict[str, Any],
    comparator_bindings: dict[str, dict[str, str]],
) -> dict[str, Any]:
    if sha256_file(MECHANISM_DECISION) != MECHANISM_DECISION_SHA256:
        raise RuntimeError("BLSR mechanism decision file drift")
    if sha256_file(SOURCE_VALIDATOR) != SOURCE_VALIDATOR_SHA256:
        raise RuntimeError("BLSR frozen source validator SHA drift")
    frozen_policy = policy()
    return {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "config": asdict(cfg),
        "source_manifest": source_binding,
        "source_validator": {
            "path": str(SOURCE_VALIDATOR),
            "sha256": SOURCE_VALIDATOR_SHA256,
        },
        "mechanism_decision": {
            "path": str(MECHANISM_DECISION),
            "sha256": MECHANISM_DECISION_SHA256,
        },
        "comparator_bindings": comparator_bindings,
        "policy": frozen_policy,
        "policy_hash": canonical_hash(frozen_policy),
        "outcomes_opened": False,
        "outcome_boundary": dict(OUTCOME_BOUNDARY),
        "research_sequence": dict(RESEARCH_SEQUENCE),
        "preregistration_source": {
            "path": str(PREREGISTRATION_SOURCE),
            "sha256": sha256_file(PREREGISTRATION_SOURCE),
        },
    }


def _temporary_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    )
    handle.close()
    return Path(handle.name)


def write_preregistration(cfg: Config) -> dict[str, Any]:
    _validate_config(cfg, require_new_output=True)
    source_binding = _validate_source_manifest(cfg.source_manifest)
    comparators = _validate_comparators()
    core = _artifact_core(cfg, source_binding, comparators)
    artifact = {**core, "manifest_hash": canonical_hash(core)}
    output = _repository_path(cfg.preregistration_output)
    temporary = _temporary_path(output)
    try:
        temporary.write_text(
            json.dumps(
                artifact,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.link(temporary, output)
        return artifact
    finally:
        temporary.unlink(missing_ok=True)


def load_preregistration(path: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    artifact = _read_json(path)
    if set(artifact) != ARTIFACT_TOP_LEVEL_KEYS:
        raise RuntimeError("BLSR preregistration top-level schema drift")
    core = _manifest_core(artifact)
    if canonical_hash(core) != artifact.get("manifest_hash"):
        raise RuntimeError("BLSR preregistration canonical hash mismatch")
    if artifact.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("BLSR preregistration protocol drift")
    if artifact.get("policy_id") != POLICY_ID:
        raise RuntimeError("BLSR preregistration policy ID drift")
    frozen_policy = policy()
    if artifact.get("policy") != frozen_policy:
        raise RuntimeError("BLSR preregistration policy drift")
    if artifact.get("policy_hash") != canonical_hash(frozen_policy):
        raise RuntimeError("BLSR preregistration policy hash drift")
    if artifact.get("outcomes_opened") is not False:
        raise RuntimeError("BLSR preregistration opened outcomes")
    if artifact.get("outcome_boundary") != OUTCOME_BOUNDARY:
        raise RuntimeError("BLSR preregistration outcome boundary drift")
    if artifact.get("research_sequence") != RESEARCH_SEQUENCE:
        raise RuntimeError("BLSR preregistration research sequence drift")

    expected_source = {
        "path": str(PREREGISTRATION_SOURCE),
        "sha256": sha256_file(PREREGISTRATION_SOURCE),
    }
    if artifact.get("preregistration_source") != expected_source:
        raise RuntimeError("BLSR preregistration source binding drift")
    expected_validator = {
        "path": str(SOURCE_VALIDATOR),
        "sha256": SOURCE_VALIDATOR_SHA256,
    }
    if artifact.get("source_validator") != expected_validator:
        raise RuntimeError("BLSR source validator binding drift")
    if sha256_file(SOURCE_VALIDATOR) != SOURCE_VALIDATOR_SHA256:
        raise RuntimeError("BLSR frozen source validator SHA drift")
    expected_decision = {
        "path": str(MECHANISM_DECISION),
        "sha256": MECHANISM_DECISION_SHA256,
    }
    if artifact.get("mechanism_decision") != expected_decision:
        raise RuntimeError("BLSR mechanism-decision binding drift")
    if sha256_file(MECHANISM_DECISION) != MECHANISM_DECISION_SHA256:
        raise RuntimeError("BLSR mechanism decision file drift")
    if artifact.get("comparator_bindings") != _validate_comparators():
        raise RuntimeError("BLSR comparator binding drift")

    raw_config = artifact.get("config")
    if not isinstance(raw_config, dict):
        raise RuntimeError("BLSR preregistration config missing")
    try:
        cfg = Config(**raw_config)
    except TypeError as exc:
        raise RuntimeError("BLSR preregistration config drift") from exc
    if raw_config != asdict(cfg):
        raise RuntimeError("BLSR preregistration config drift")
    _validate_config(cfg, require_new_output=False)
    if _repository_path(path) != _repository_path(cfg.preregistration_output):
        raise RuntimeError("BLSR preregistration output-path binding drift")
    if artifact.get("source_manifest") != _validate_source_manifest(
        cfg.source_manifest
    ):
        raise RuntimeError("BLSR preregistration source-manifest binding drift")
    return artifact


def parse_args(argv: list[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", default=Config.source_manifest)
    parser.add_argument(
        "--preregistration-output", default=Config.preregistration_output
    )
    return Config(**vars(parser.parse_args(argv)))


def main(argv: list[str] | None = None) -> int:
    artifact = write_preregistration(parse_args(argv))
    print(
        json.dumps(
            {
                "status": "created",
                "policy_id": artifact["policy_id"],
                "policy_hash": artifact["policy_hash"],
                "manifest_hash": artifact["manifest_hash"],
                "source_csv_values_read": 0,
                "comparator_rows_read": 0,
                "outcomes_opened": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
