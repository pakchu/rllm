"""Preregister BCTP-12H source sequencing before market or reward access."""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from numbers import Integral
from pathlib import Path
import tempfile
from typing import Any

from training import preregister_block_clearing_relational_topology as bcrt


POLICY_ID = "BCTP-12H"
PROTOCOL_VERSION = "block_clearing_target_position_mdp_preregistration_v1"
DEFAULT_OUTPUT = (
    "results/block_clearing_target_position_mdp_"
    "preregistration_2026-07-25.json"
)
SUPPORT_OUTPUT = (
    "results/block_clearing_target_position_mdp_support_2026-07-25.json"
)
SEQUENCE_OUTPUT = (
    "data/block_clearing_target_position_mdp_sequences_2020_2023.csv.gz"
)

BOUNDARY_DOCUMENT = (
    "docs/block-clearing-target-position-mdp-boundary-2026-07-25.md"
)
BOUNDARY_DOCUMENT_SHA256 = (
    "97f92a4b9e78fcdc50cb227f1a91c778e7dacec48799cda372c636cb5f58e16e"
)
BOUNDARY_COMMIT = "f994518e6d9ebdd696e5d8148350cc9a3c7034f1"

BCRT_PREREGISTRATION_SOURCE_SHA256 = (
    "e04fc7d16f550bf2c0cdde9a3359f079b2f233ede3dba315b41285f50e326e2b"
)
BCRT_SUPPORT_SOURCE_SHA256 = (
    "8a351be18a2f9b44a2ae8bdb48e5555e84393b704ff91b4c36801266e49f6a5e"
)
BCRT_PREREGISTRATION_ARTIFACT_SHA256 = (
    "322f91b41fce1aee06250a010d5a569557b83cc3f493ee3c47f5d6974aafe6a8"
)
BCRT_PREREGISTRATION_MANIFEST_HASH = (
    "c9f08196f5a25dd05320a2c7cf3fbf951403d10f2362e67e2b0169b03fec194f"
)
BCRT_SUPPORT_ARTIFACT_SHA256 = (
    "9ccccf7a3176fcf86baddacb65c11bbde78ea73ed7ab18d3594b0e6327567055"
)
BCRT_SUPPORT_MANIFEST_HASH = (
    "e2b2d7301d204043f2df33f4453da82112fb5db7bfb9aed66a74bee6ec76932b"
)
BCRT_RETIREMENT_DOCUMENT = "docs/bcrt-source-support-retirement-2026-07-24.md"
BCRT_RETIREMENT_DOCUMENT_SHA256 = (
    "0db179109af9e16f686758841eec01ee608804c58da4e1d4fd6c310cbc7ff8f3"
)

SEQUENCE_LABELS = ("S_MINUS_2", "S_MINUS_1", "S_0")
POSITION_TOKENS = ("POSITION_SHORT", "POSITION_FLAT", "POSITION_LONG")
TARGET_ACTIONS = ("TARGET_SHORT", "TARGET_FLAT", "TARGET_LONG")
SOURCE_TOKEN_COLUMNS = tuple(
    f"{label.lower()}__{token}"
    for label in SEQUENCE_LABELS
    for token in bcrt.TOKEN_COLUMNS
)
SOURCE_SEQUENCE_COLUMNS = (
    "sequence_id",
    "entry_time",
    "source_signal_id_m2",
    "source_signal_id_m1",
    "source_signal_id_s0",
    "source_signature",
    *SOURCE_TOKEN_COLUMNS,
)
FORBIDDEN_SOURCE_OUTPUT_FIELDS = (
    "position",
    "action",
    "side",
    "price",
    "return",
    "funding",
    "reward",
    "pnl",
    "cagr",
    "mdd",
    "rank",
    "raw_value",
)


@dataclass(frozen=True)
class Policy:
    policy_id: str = POLICY_ID
    sequence_states: int = 3
    source_bucket_seconds: int = 43_200
    confirmation_blocks: int = 288
    minimum_embargo_seconds: int = 172_800
    execution_bar_seconds: int = 300
    latency_bars: int = 1
    target_absolute_gross: float = 0.5
    base_cost_per_changed_notional: float = 0.0006
    stress_cost_per_changed_notional: float = 0.0010
    source_signature_share_max: float = 0.05
    minimum_actionable_states_per_development_year: int = 500
    random_seed: int = 20_260_725


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _utc_datetime(value: Any, *, field: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as error:
            raise ValueError(f"BCTP {field} is not an ISO timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError(f"BCTP {field} must be timezone aware")
    return parsed.astimezone(timezone.utc)


def _iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def validate_snapshot(tokens: Mapping[str, str]) -> dict[str, str]:
    """Validate one exact BCRT token snapshot without widening vocabulary."""

    return bcrt.validate_tokens(tokens)


def canonical_snapshot(tokens: Mapping[str, str]) -> str:
    validated = validate_snapshot(tokens)
    return " | ".join(
        f"{column.upper()}={validated[column]}"
        for column in bcrt.TOKEN_COLUMNS
    )


def source_sequence_signature(
    entry_times: Sequence[Any],
    snapshots: Sequence[Mapping[str, str]],
) -> str:
    """Hash three oldest-first source snapshots without exposing timestamps."""

    if len(entry_times) != Policy().sequence_states:
        raise ValueError("BCTP source sequence must contain exactly three times")
    if len(snapshots) != len(entry_times):
        raise ValueError("BCTP source sequence snapshot count differs")
    parsed = [
        _utc_datetime(value, field=f"sequence_time_{index}")
        for index, value in enumerate(entry_times)
    ]
    if any(right <= left for left, right in zip(parsed, parsed[1:])):
        raise ValueError("BCTP source sequence times must be strictly increasing")
    lines = [canonical_snapshot(tokens) for tokens in snapshots]
    return canonical_hash({"oldest_first_source_snapshots": lines})


def batch_actionable_releases(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Keep the latest bucket from each causal release batch.

    Earlier same-release rows remain source predecessors inside their already
    frozen BCRT token transition. They are not separate BCTP action times.
    """

    normalized: list[dict[str, Any]] = []
    identities: set[tuple[datetime, datetime, int, str]] = set()
    for source_row in rows:
        row = dict(source_row)
        entry = _utc_datetime(row.get("entry_time"), field="entry_time")
        bucket = _utc_datetime(row.get("bucket_start"), field="bucket_start")
        confirmation_height = row.get("confirmation_height")
        if (
            isinstance(confirmation_height, bool)
            or not isinstance(confirmation_height, Integral)
            or int(confirmation_height) <= 0
        ):
            raise ValueError(
                "BCTP confirmation_height must be a positive integer"
            )
        signal_id = str(row.get("signal_id", "")).strip()
        if not signal_id:
            raise ValueError("BCTP source signal_id must be nonempty")
        identity = (entry, bucket, int(confirmation_height), signal_id)
        if identity in identities:
            raise ValueError("BCTP duplicate same-release source identity")
        identities.add(identity)
        if bucket >= entry:
            raise ValueError("BCTP bucket_start must precede entry_time")
        token_mapping = {
            column: str(row.get(column, ""))
            for column in bcrt.TOKEN_COLUMNS
        }
        validate_snapshot(token_mapping)
        row["_entry"] = entry
        row["_bucket"] = bucket
        row["_confirmation_height"] = int(confirmation_height)
        normalized.append(row)

    normalized.sort(
        key=lambda row: (
            row["_entry"],
            row["_bucket"],
            int(row["_confirmation_height"]),
            str(row["signal_id"]),
        )
    )
    selected: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(normalized):
        entry = normalized[cursor]["_entry"]
        end = cursor + 1
        while end < len(normalized) and normalized[end]["_entry"] == entry:
            end += 1
        winner = max(
            normalized[cursor:end],
            key=lambda row: (
                row["_bucket"],
                int(row["_confirmation_height"]),
                str(row["signal_id"]),
            ),
        )
        clean = {
            key: value
            for key, value in winner.items()
            if not key.startswith("_")
        }
        clean["entry_time"] = _iso_z(entry)
        clean["bucket_start"] = _iso_z(winner["_bucket"])
        selected.append(clean)
        cursor = end

    entries = [
        _utc_datetime(row["entry_time"], field="entry_time")
        for row in selected
    ]
    if any(right <= left for left, right in zip(entries, entries[1:])):
        raise RuntimeError("BCTP batched releases are not strictly increasing")
    return selected


def build_source_sequences(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build source-only three-release sequences with two warm-up releases."""

    actionable = batch_actionable_releases(rows)
    output: list[dict[str, Any]] = []
    for current_index in range(Policy().sequence_states - 1, len(actionable)):
        window = actionable[
            current_index - Policy().sequence_states + 1 : current_index + 1
        ]
        times = [row["entry_time"] for row in window]
        snapshots = [
            {
                column: str(row[column])
                for column in bcrt.TOKEN_COLUMNS
            }
            for row in window
        ]
        signature = source_sequence_signature(times, snapshots)
        source_ids = [str(row["signal_id"]) for row in window]
        record: dict[str, Any] = {
            "sequence_id": canonical_hash(
                {
                    "policy_id": POLICY_ID,
                    "source_signal_ids": source_ids,
                }
            ),
            "entry_time": str(window[-1]["entry_time"]),
            "source_signal_id_m2": source_ids[0],
            "source_signal_id_m1": source_ids[1],
            "source_signal_id_s0": source_ids[2],
            "source_signature": signature,
        }
        for label, snapshot in zip(SEQUENCE_LABELS, snapshots):
            for token in bcrt.TOKEN_COLUMNS:
                record[f"{label.lower()}__{token}"] = snapshot[token]
        if tuple(record) != SOURCE_SEQUENCE_COLUMNS:
            raise RuntimeError("BCTP source sequence schema changed")
        output.append(record)
    return output


def _manifest_core() -> dict[str, Any]:
    policy = asdict(Policy())
    return {
        "protocol_version": PROTOCOL_VERSION,
        "policy": policy,
        "boundary": {
            "path": BOUNDARY_DOCUMENT,
            "sha256": BOUNDARY_DOCUMENT_SHA256,
            "commit": BOUNDARY_COMMIT,
        },
        "research_history_boundary": {
            "base_chain_family_outcomes_seen": True,
            "bcrt_source_values_and_token_marginals_seen": True,
            "bctp_sequences_seen": False,
            "bctp_rewards_seen": False,
            "bctp_model_outcomes_seen": False,
            "bctp_2023_market_outcomes_seen": False,
            "global_pristine_holdout_claimed": False,
            "claim_scope": "candidate-specific sequential target-position MDP",
        },
        "immutable_bcrt_representation": {
            "policy_id": bcrt.POLICY_ID,
            "preregistration_source": {
                "path": "training/preregister_block_clearing_relational_topology.py",
                "sha256": BCRT_PREREGISTRATION_SOURCE_SHA256,
            },
            "support_source": {
                "path": "training/build_block_clearing_relational_topology_support.py",
                "sha256": BCRT_SUPPORT_SOURCE_SHA256,
            },
            "preregistration_artifact": {
                "path": bcrt.DEFAULT_OUTPUT,
                "sha256": BCRT_PREREGISTRATION_ARTIFACT_SHA256,
                "manifest_hash": BCRT_PREREGISTRATION_MANIFEST_HASH,
            },
            "support_artifact": {
                "path": "results/block_clearing_relational_topology_support_2026-07-24.json",
                "sha256": BCRT_SUPPORT_ARTIFACT_SHA256,
                "manifest_hash": BCRT_SUPPORT_MANIFEST_HASH,
            },
            "retirement": {
                "path": BCRT_RETIREMENT_DOCUMENT,
                "sha256": BCRT_RETIREMENT_DOCUMENT_SHA256,
                "remains_terminal": True,
                "failed_gap_gate_not_changed": True,
            },
            "raw_source": {
                "path": bcrt.SOURCE,
                "sha256": bcrt.SOURCE_SHA256,
                "header_sha256": bcrt.SOURCE_HEADER_SHA256,
                "allowlist": list(bcrt.SOURCE_ALLOWLIST),
            },
            "source_manifest": {
                "path": bcrt.SOURCE_MANIFEST,
                "sha256": bcrt.SOURCE_MANIFEST_SHA256,
                "manifest_hash": bcrt.SOURCE_MANIFEST_HASH,
            },
            "reference": {
                "path": bcrt.REFERENCE,
                "sha256": bcrt.REFERENCE_SHA256,
                "header_sha256": bcrt.REFERENCE_HEADER_SHA256,
                "allowlist": list(bcrt.REFERENCE_ALLOWLIST),
            },
            "expected_replay_counts": {
                "formed_buckets": 2_918,
                "rank_complete_states": 2_792,
                "token_ready_states": 2_791,
            },
            "token_schema": [
                {"name": name, "vocabulary": list(vocabulary)}
                for name, vocabulary in bcrt.TOKEN_SCHEMA
            ],
        },
        "source_sequence_contract": {
            "sequence_labels_oldest_first": list(SEQUENCE_LABELS),
            "source_token_columns": list(SOURCE_TOKEN_COLUMNS),
            "output_columns": list(SOURCE_SEQUENCE_COLUMNS),
            "same_release_batch": (
                "latest bucket_start, then greatest confirmation_height, "
                "then lexical signal_id; duplicate full identities reject"
            ),
            "same_release_losers_are_actionable": False,
            "same_release_losers_remain_bcrt_source_predecessors": True,
            "warmup_actionable_releases": Policy().sequence_states - 1,
            "timestamps_in_signature": False,
            "position_in_source_support": False,
            "position_vocabulary_reserved_for_economics": list(POSITION_TOKENS),
            "target_actions_reserved_for_economics": list(TARGET_ACTIONS),
            "forbidden_output_fields": list(
                FORBIDDEN_SOURCE_OUTPUT_FIELDS
            ),
            "future_append_invariance_required": True,
        },
        "source_support_gates": {
            "exact_bcrt_replay_required": True,
            "development_years": [2020, 2021, 2022],
            "minimum_actionable_states_each_development_year": (
                Policy().minimum_actionable_states_per_development_year
            ),
            "active_months_2021": 12,
            "active_months_2022": 12,
            "maximum_exact_source_sequence_share": (
                Policy().source_signature_share_max
            ),
            "all_bcrt_train_2022_token_checks_must_remain_true": True,
            "calendar_boundary_gap": "report_only_non_boolean",
            "position_conditioned_concentration": (
                "deferred_until_policy_exists_non_source_gate"
            ),
            "failure_action": "retire_BCTP_12H_before_market_access",
        },
        "report_only_2023": {
            "incidence_emitted": True,
            "may_change_support_boolean": False,
            "may_select_or_repair": False,
            "unknown_vocabulary_action": "TARGET_FLAT",
        },
        "temporal_roles": {
            "algorithm_fit": ["2020-01-01T00:00:00Z", "2021-01-01T00:00:00Z"],
            "algorithm_transfer": [
                "2021-01-01T00:00:00Z",
                "2022-01-01T00:00:00Z",
            ],
            "final_cheap_fit": [
                "2020-01-01T00:00:00Z",
                "2022-01-01T00:00:00Z",
            ],
            "gemma_checkpoint_selection": [
                "2022-01-01T00:00:00Z",
                "2023-01-01T00:00:00Z",
            ],
            "untouched_candidate_eval": [
                "2023-01-01T00:00:00Z",
                "2024-01-01T00:00:00Z",
            ],
            "sealed_from": "2024-01-01T00:00:00Z",
            "2023_may_select": False,
        },
        "stage_authority": {
            "authorized": [
                "raw_source_integrity_replay",
                "bcrt_state_replay",
                "same_release_batching",
                "three_state_source_sequence",
                "source_sequence_support",
            ],
            "forbidden": [
                "market_read",
                "funding_read",
                "reward",
                "action_label",
                "policy_fit",
                "model_training",
                "economic_metric",
                "post_2023_numeric_source",
            ],
            "next_required_commit": (
                "source_support_implementation_and_synthetic_tests"
            ),
        },
        "outputs": {
            "preregistration": DEFAULT_OUTPUT,
            "source_sequences": SEQUENCE_OUTPUT,
            "support": SUPPORT_OUTPUT,
        },
        "outcome_boundary": {
            "new_raw_source_rows_decoded": 0,
            "bctp_sequences_built": 0,
            "bctp_actionable_incidence_opened": 0,
            "market_rows_decoded": 0,
            "funding_rows_decoded": 0,
            "future_return_rows_decoded": 0,
            "rewards_or_labels_created": 0,
            "models_trained": 0,
            "economic_metrics_computed": 0,
            "post_2023_source_rows_decoded": 0,
            "bctp_2023_market_outcomes_opened": False,
        },
    }


def build_manifest() -> dict[str, Any]:
    core = _manifest_core()
    return {**core, "manifest_hash": canonical_hash(core)}


def validate_manifest(payload: Mapping[str, Any]) -> None:
    expected = build_manifest()
    if dict(payload) != expected:
        raise ValueError("BCTP preregistration differs from frozen contract")
    if payload["manifest_hash"] != canonical_hash(_manifest_core()):
        raise ValueError("BCTP preregistration manifest hash mismatch")


def write_once(path: str | Path, payload: Mapping[str, Any]) -> str:
    target = Path(path)
    encoded = (
        json.dumps(
            dict(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if target.exists():
        if target.read_bytes() != encoded:
            raise RuntimeError(f"BCTP write-once artifact drift: {target}")
        return hashlib.sha256(encoded).hexdigest()
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=target.parent,
        prefix=f".{target.name}.",
        delete=False,
    ) as handle:
        handle.write(encoded)
        temporary = Path(handle.name)
    temporary.replace(target)
    return hashlib.sha256(encoded).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_manifest()
    validate_manifest(payload)
    digest = write_once(args.output, payload)
    print(
        json.dumps(
            {
                "decision": "PREREGISTERED",
                "manifest_hash": payload["manifest_hash"],
                "output": str(args.output),
                "sha256": digest,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
