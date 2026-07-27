#!/usr/bin/env python3
"""Preregister the report-only 2021 transfer of the sealed S6R1 policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

_BOOTSTRAP_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_BOOTSTRAP_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_REPO_ROOT))

from training import (
    preregister_psim_d8_rllm2_s4_semantic_fqi as s4,
)
from training import (
    preregister_psim_d8_rllm2_s5_direction_residual_ridge_fqi as s5,
)
from training import (
    preregister_psim_d8_rllm2_s6r1_action_mean_residual_ridge_fqi as s6r1,
)

REPO_ROOT = _BOOTSTRAP_REPO_ROOT
PROTOCOL_VERSION = (
    "psim_d8_rllm2_s7_2021_report_only_transfer_preregistration_v1"
)
STAGE_ID = "PSIM-D8-RLLM2-S7-2021-REPORT-ONLY-TRANSFER"
AS_OF_DATE = "2026-07-27"

PRIMARY_POLICY_ID = s6r1.PRIMARY_POLICY_ID
FAMILY_IDS = s6r1.COMBINED_2021_FAMILY_IDS
NONSEMANTIC_CONTROL_IDS = (
    "always_flat",
    "always_long",
    "always_short",
    "previous_target_persistence",
    "exact_redacted_payload_memory",
    "metadata_frontmatter_only",
    "path_section_diff_size_only",
    "cadence_revision_topology_only",
    "shuffled_eip_bip_daily_relation",
    "shuffled_old_new_pairing",
    "future_status_scrub",
    "ethereum_only",
    "bitcoin_only",
    "current_position_only",
    "masked_semantic_embedding",
    (
        "semantic_ridge_direction_residual_fqi_"
        "current_position_only"
    ),
    (
        "semantic_ridge_direction_residual_fqi_"
        "masked_semantic_embedding"
    ),
    (
        "semantic_ridge_direction_residual_fqi_"
        "metadata_frontmatter_only"
    ),
    f"{PRIMARY_POLICY_ID}_current_position_only",
    f"{PRIMARY_POLICY_ID}_masked_semantic_embedding",
    f"{PRIMARY_POLICY_ID}_metadata_frontmatter_only",
)

DEFAULT_OUTPUT = Path(
    "results/psim_d8_rllm2_s7_2021_report_only_transfer_"
    "preregistration_2026-07-27.json"
)
RUNNER_PATH = Path(
    "training/run_psim_d8_rllm2_s7_2021_report_only_transfer.py"
)
ATTEMPT_PATH = Path(
    "results/psim_d8_rllm2_s7_2021_report_only_transfer_"
    "attempt_2026-07-27.json"
)
RESULT_PATH = Path(
    "results/psim_d8_rllm2_s7_2021_report_only_transfer_"
    "result_2026-07-27.json"
)

S6R1_PASS_RECORD_COMMIT = "235db6bb9ca745826bcd308a0ea157d55551f0d2"
S6R1_PREREGISTRATION_SHA256 = (
    "a12db8dd9899a9b0e81f2e0ff80f1acaf7695019717cdfb4d14c7755d06576dc"
)
S6R1_PREREGISTRATION_MANIFEST_HASH = (
    "f527487ec53eefe7ee0ddcc931512e8f845c510d3566f4a17104082454c88aff"
)
S6R1_RUNNER_SHA256 = (
    "c791054d32b5f4e1891916bf4bce9beb070d9fdddeb759318393374068595016"
)
S6R1_ATTEMPT_SHA256 = (
    "f4669c0a37878351bd89ad0d0554f02b6509a15b3ab80e4dbf7f2def9616cbaa"
)
S6R1_ATTEMPT_HASH = (
    "b50b2c4314603931e03c3ae37b92a3cc4d1ed426784440761dad9c17c731cdd6"
)
S6R1_RESULT_SHA256 = (
    "020c86002df8348c497407b70e24d6e85d583c3734242c6bddb0b5764b260d2f"
)
S6R1_RESULT_HASH = (
    "63bec557d8e95a21becdad69c648f339d6f12510a28a679668a7bef3f4edd862"
)
S6R1_SCHEDULE_MANIFEST_SHA256 = (
    "816afbc7bca16df0313636194e4b0780bbe760cb7cd10f7944736c6968352644"
)
S6R1_SCHEDULE_MANIFEST_HASH = (
    "314298356bbf3bc94e394ae362ee4f1894fd8f07e6e1a0b47137dd785f78970e"
)
S6R1_BASE_SCHEDULES_SHA256 = (
    "f850b3f9e18e9d942b8279065512fa33e77a9493abbd4d927672dc025ab41971"
)
S6R1_DELAYED_SCHEDULE_SHA256 = (
    "b324347e72e08de8266dc39f7f800fae2a73a98129ab7092cb02b92893a02e28"
)
S4_BASE_SCHEDULES_SHA256 = (
    "2f5b04e2514ca8b328fa6a92a45cea2828225090afc2427a58b6b778655394ba"
)
S5_BASE_SCHEDULES_SHA256 = (
    "ed5ed208104242f571f4a07a53a815a1db2742957f43e9fcbecd9c18221a3833"
)
S6R1_PASS_DOCUMENT = Path(
    "docs/psim-d8-rllm2-s6r1-schedule-readiness-pass-2026-07-27.md"
)
S6R1_PASS_DOCUMENT_SHA256 = (
    "c5977fce9419db7a4827fb43b5c5c499facc2b2b810f3174faf4d01b32f7d696"
)

STAGE_SOURCE_ROOT = Path("data/bctp_stage_sources")
STAGE_SOURCE_MANIFEST = (
    STAGE_SOURCE_ROOT / "2021" / "source_manifest.json"
)
MARKET_PATH = STAGE_SOURCE_ROOT / "2021" / "bctp_market_2021.csv.gz"
FUNDING_PATH = (
    STAGE_SOURCE_ROOT / "2021" / "bctp_funding_2021.csv.gz"
)
STAGE_SOURCE_MANIFEST_SHA256 = (
    "1d12d8dad47eda810933ddce7ac2d911a7a4f85262ecc02e91e22df2488c6e2d"
)
STAGE_SOURCE_MANIFEST_HASH = (
    "d71ade5504ebe4d729bd9892afbceec0e03675f2ad197ac937d0a46cb0bd64c9"
)
MARKET_GZIP_SHA256 = (
    "a6b66b41aec8484f8ac30d22d3513ceb2cacda57260fd25f6c9456b59f119f6d"
)
FUNDING_GZIP_SHA256 = (
    "65d6bceacdb3655062e8e5f5ca95b2dd4a129607966d814bb6b787b5ad15901d"
)
STRICT_ECONOMICS_SHA256 = (
    "7db06b08ae6742d33b8a8b2cc2ebfb95b06743de397a8fc125ddd781e653c765"
)
STAGE_SOURCES_SHA256 = (
    "9a52bfde0c43f8fd1bc6ca18c117d2488499fa69a60c5202618f58b8a6620266"
)

BASE_COST_RATE = 0.0006
STRESS_COST_RATE = 0.0010
CALENDAR_START = "2021-01-01T00:00:00Z"
HALF_SPLIT = "2021-07-01T00:00:00Z"
CALENDAR_END = "2022-01-01T00:00:00Z"
MIN_RATIO = 1.0
MIN_NONFLAT_INTERVALS = 80
MIN_DIRECTION_SHARE = 0.20
FAMILYWISE_P_MAX_EXCLUSIVE = 0.25
STATISTICAL_DRAWS = 100_000
STATISTICAL_SEED = 20_260_725
STATISTICAL_BATCH_DRAWS = 2_000


def repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def canonical_bytes(payload: Any, *, pretty: bool = False) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8") + (b"\n" if pretty else b"")


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with repository_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_exact_json(
    path: str | Path,
    *,
    expected_sha256: str,
    self_hash_field: str,
    expected_self_hash: str,
) -> dict[str, Any]:
    target = repository_path(path)
    if target.is_symlink() or not target.is_file():
        raise RuntimeError(f"PSIM S7 evidence missing: {path}")
    if sha256_file(target) != expected_sha256:
        raise RuntimeError(f"PSIM S7 evidence changed: {path}")
    payload = json.loads(target.read_text(encoding="utf-8"))
    core = {
        key: value
        for key, value in payload.items()
        if key != self_hash_field
    }
    if (
        payload.get(self_hash_field) != expected_self_hash
        or payload.get(self_hash_field) != canonical_hash(core)
    ):
        raise RuntimeError(f"PSIM S7 evidence self hash changed: {path}")
    return payload


def validate_s6r1_schedule_pass() -> dict[str, Any]:
    registration = _read_exact_json(
        s6r1.DEFAULT_OUTPUT,
        expected_sha256=S6R1_PREREGISTRATION_SHA256,
        self_hash_field="manifest_hash",
        expected_self_hash=S6R1_PREREGISTRATION_MANIFEST_HASH,
    )
    attempt = _read_exact_json(
        s6r1.ATTEMPT_PATH,
        expected_sha256=S6R1_ATTEMPT_SHA256,
        self_hash_field="attempt_hash",
        expected_self_hash=S6R1_ATTEMPT_HASH,
    )
    result = _read_exact_json(
        s6r1.RESULT_PATH,
        expected_sha256=S6R1_RESULT_SHA256,
        self_hash_field="result_hash",
        expected_self_hash=S6R1_RESULT_HASH,
    )
    manifest = _read_exact_json(
        s6r1.SCHEDULE_MANIFEST_PATH,
        expected_sha256=S6R1_SCHEDULE_MANIFEST_SHA256,
        self_hash_field="manifest_hash",
        expected_self_hash=S6R1_SCHEDULE_MANIFEST_HASH,
    )
    for path, expected in (
        (s6r1.RUNNER_PATH, S6R1_RUNNER_SHA256),
        (s4.SCHEDULE_PATH, S4_BASE_SCHEDULES_SHA256),
        (s5.SCHEDULE_PATH, S5_BASE_SCHEDULES_SHA256),
        (s6r1.SCHEDULE_PATH, S6R1_BASE_SCHEDULES_SHA256),
        (s6r1.DELAYED_SCHEDULE_PATH, S6R1_DELAYED_SCHEDULE_SHA256),
        (S6R1_PASS_DOCUMENT, S6R1_PASS_DOCUMENT_SHA256),
    ):
        if sha256_file(path) != expected:
            raise RuntimeError(f"PSIM S7 predecessor artifact changed: {path}")
    readiness = result.get("schedule_readiness", {})
    boundary = result.get("access_boundary", {})
    if (
        registration != s6r1.build_preregistration()
        or result.get("decision") != "pass"
        or result.get("execution_commit")
        != "aee87ef7a543715b08a581f7c70a86547a39d135"
        or result.get("authorize_separate_2021_transfer_preregistration")
        is not True
        or result.get("authorize_2022_or_later_outcomes") is not False
        or readiness.get("passed") is not True
        or readiness.get("nonflat_target_rows") != 356
        or readiness.get("target_counts")
        != {
            "TARGET_FLAT": 9,
            "TARGET_LONG": 141,
            "TARGET_SHORT": 215,
        }
        or not all(readiness.get("gates", {}).values())
        or manifest.get("decision") != "pass"
        or manifest.get("schedule_readiness") != readiness
        or boundary.get("raw_market_or_funding_paths_read") != []
        or boundary.get("2021_market_rows_parsed") != 0
        or boundary.get("2021_funding_rows_parsed") != 0
        or boundary.get("2021_reward_rows_created") != 0
        or boundary.get("2021_economic_metrics_computed") != 0
        or boundary.get("2021_policy_specific_outcomes_opened") is not False
    ):
        raise RuntimeError("PSIM S7 predecessor pass boundary changed")
    return {
        "registration": registration,
        "attempt": attempt,
        "result": result,
        "manifest": manifest,
    }


def validate_stage_source_metadata() -> dict[str, Any]:
    target = repository_path(STAGE_SOURCE_MANIFEST)
    if (
        target.is_symlink()
        or not target.is_file()
        or sha256_file(target) != STAGE_SOURCE_MANIFEST_SHA256
    ):
        raise RuntimeError("PSIM S7 stage source manifest changed")
    payload = json.loads(target.read_text(encoding="utf-8"))
    core = dict(payload)
    self_hash = core.pop("manifest_hash", None)
    if (
        self_hash != STAGE_SOURCE_MANIFEST_HASH
        or self_hash != canonical_hash(core)
        or payload.get("stage") != "2021"
        or payload.get("start_inclusive") != CALENDAR_START
        or payload.get("end_exclusive") != CALENDAR_END
        or payload.get("expected_market_rows_5m") != 105_120
        or payload.get("expected_funding_rows_8h") != 1_095
        or payload.get("strategy_outcomes_calculated") is not False
        or payload.get("market_or_funding_parent_payload_bytes_hashed")
        is not False
        or payload.get("market", {}).get("gzip_sha256")
        != MARKET_GZIP_SHA256
        or payload.get("funding", {}).get("gzip_sha256")
        != FUNDING_GZIP_SHA256
        or not repository_path(MARKET_PATH).is_file()
        or not repository_path(FUNDING_PATH).is_file()
    ):
        raise RuntimeError("PSIM S7 stage source metadata boundary changed")
    return payload


def build_preregistration() -> dict[str, Any]:
    predecessor = validate_s6r1_schedule_pass()
    source = validate_stage_source_metadata()
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "candidate": {
            "id": STAGE_ID,
            "predecessor": s6r1.STAGE_ID,
            "role": "protocol_isolated_policy_specific_report_only_transfer",
            "primary_policy_id": PRIMARY_POLICY_ID,
            "single_promotable_primary": True,
            "selection_or_repair_from_2021_metrics": False,
            "globally_pristine_2021_claim": False,
            "success_is_live_promotion": False,
        },
        "predecessor_schedule_pass": {
            "terminal_record_commit": S6R1_PASS_RECORD_COMMIT,
            "preregistration_manifest_hash": (
                S6R1_PREREGISTRATION_MANIFEST_HASH
            ),
            "attempt_hash": S6R1_ATTEMPT_HASH,
            "result_hash": S6R1_RESULT_HASH,
            "schedule_manifest_hash": S6R1_SCHEDULE_MANIFEST_HASH,
            "target_counts": predecessor["result"][
                "schedule_readiness"
            ]["target_counts"],
            "economics_opened": False,
        },
        "frozen_schedule_family": {
            "family_ids": list(FAMILY_IDS),
            "family_count": len(FAMILY_IDS),
            "s4_base": {
                "path": s4.SCHEDULE_PATH.as_posix(),
                "sha256": S4_BASE_SCHEDULES_SHA256,
                "policy_count": len(s4.POLICY_FAMILY_IDS),
            },
            "s5_base": {
                "path": s5.SCHEDULE_PATH.as_posix(),
                "sha256": S5_BASE_SCHEDULES_SHA256,
                "policy_count": len(s5.POLICY_FAMILY_IDS),
            },
            "s6r1_base": {
                "path": s6r1.SCHEDULE_PATH.as_posix(),
                "sha256": S6R1_BASE_SCHEDULES_SHA256,
                "policy_count": len(s6r1.POLICY_FAMILY_IDS),
            },
            "s6r1_delayed_primary": {
                "path": s6r1.DELAYED_SCHEDULE_PATH.as_posix(),
                "sha256": S6R1_DELAYED_SCHEDULE_SHA256,
                "policy_count": 1,
            },
            "nonsemantic_control_ids": list(NONSEMANTIC_CONTROL_IDS),
        },
        "frozen_2021_outcome_source": {
            "manifest": {
                "path": STAGE_SOURCE_MANIFEST.as_posix(),
                "sha256": STAGE_SOURCE_MANIFEST_SHA256,
                "manifest_hash": STAGE_SOURCE_MANIFEST_HASH,
            },
            "market": {
                "path": MARKET_PATH.as_posix(),
                "expected_gzip_sha256": MARKET_GZIP_SHA256,
                "expected_rows": source["expected_market_rows_5m"],
            },
            "funding": {
                "path": FUNDING_PATH.as_posix(),
                "expected_gzip_sha256": FUNDING_GZIP_SHA256,
                "expected_rows": source["expected_funding_rows_8h"],
            },
            "payload_bytes_hashed_at_preregistration": False,
            "numeric_rows_parsed_at_preregistration": 0,
        },
        "economic_contract": {
            "strict_economics_module_sha256": STRICT_ECONOMICS_SHA256,
            "stage_sources_module_sha256": STAGE_SOURCES_SHA256,
            "calendar_start": CALENDAR_START,
            "calendar_end": CALENDAR_END,
            "full_calendar_cagr_includes_flat_periods": True,
            "target_absolute_gross": 0.5,
            "base_cost_rate": BASE_COST_RATE,
            "stress_cost_rate": STRESS_COST_RATE,
            "funding_boundary_rule": "min(0,old_cash,new_cash)",
            "strict_mdd": (
                "global pre-entry high-water plus favorable-then-adverse "
                "held 5m OHLC path and terminal flatten"
            ),
            "trade_counts_reported": [
                "directional_entries_including_flips",
                "all_target_changes_including_terminal_flatten",
            ],
        },
        "fixed_robustness_contract": {
            "delayed_schedule": "exact sealed +5m schedule",
            "first_half": [CALENDAR_START, HALF_SPLIT],
            "second_half": [HALF_SPLIT, CALENDAR_END],
            "both_half_absolute_returns_positive": True,
            "stress_absolute_return_positive": True,
            "delayed_absolute_return_positive": True,
        },
        "familywise_inference": {
            "family_ids": list(FAMILY_IDS),
            "cluster": "monday_00_utc_half_open_week",
            "alternative": "one_sided_positive_studentized_mean",
            "method": "shared_sign_rademacher_max_stat",
            "draws": STATISTICAL_DRAWS,
            "seed": STATISTICAL_SEED,
            "batch_draws": STATISTICAL_BATCH_DRAWS,
            "p_max_strictly_below": FAMILYWISE_P_MAX_EXCLUSIVE,
        },
        "pass_gate": {
            "base_absolute_return_positive": True,
            "stress_absolute_return_positive": True,
            "delayed_absolute_return_positive": True,
            "first_half_absolute_return_positive": True,
            "second_half_absolute_return_positive": True,
            "base_cagr_to_strict_mdd_minimum": MIN_RATIO,
            "minimum_nonflat_intervals": MIN_NONFLAT_INTERVALS,
            "minimum_long_share": MIN_DIRECTION_SHARE,
            "minimum_short_share": MIN_DIRECTION_SHARE,
            "primary_must_beat_strongest_nonsemantic_control_on": [
                "absolute_return",
                "cagr_to_strict_mdd",
            ],
            "action_code_permutation_schedule_identity": True,
            "familywise_p_max_strictly_below": (
                FAMILYWISE_P_MAX_EXCLUSIVE
            ),
            "all_checks_required": True,
        },
        "execution_contract": {
            "runner": RUNNER_PATH.as_posix(),
            "attempt": ATTEMPT_PATH.as_posix(),
            "result": RESULT_PATH.as_posix(),
            "clean_head_equals_origin_main_required": True,
            "runner_committed_and_hash_bound_before_execution": True,
            "attempt_written_before_market_or_funding_open_or_read": True,
            "attempt_written_before_market_or_funding_payload_hash_or_parse": (
                True
            ),
            "fixed_paths_no_output_override": True,
            "write_once": True,
            "result_published_last": True,
            "no_2022_or_later_outcome_access": True,
            "no_model_load_or_forward": True,
            "no_selection_or_repair_after_result": True,
        },
        "access_boundary": {
            "raw_market_or_funding_paths_opened_or_read_before_attempt": [],
            "market_or_funding_payload_bytes_hashed": False,
            "market_rows_parsed": 0,
            "funding_rows_parsed": 0,
            "economic_metric_sets_computed": 0,
            "2022_or_later_outcomes_opened": False,
            "model_loaded": False,
            "model_forwards_started": 0,
        },
        "next_authorized_step": (
            "IMPLEMENT_REVIEW_COMMIT_AND_PUSH_S7_RUNNER_THEN_EXECUTE_"
            "THE_SINGLE_REPORT_ONLY_2021_TRANSFER"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def write_preregistration(
    path: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    target = repository_path(path)
    payload = build_preregistration()
    raw = canonical_bytes(payload, pretty=True)
    if target.exists():
        if (
            target.is_symlink()
            or not target.is_file()
            or target.read_bytes() != raw
        ):
            raise RuntimeError("PSIM S7 preregistration drift")
        return payload
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = write_preregistration(args.output)
    print(
        json.dumps(
            {
                "candidate": payload["candidate"]["id"],
                "manifest_hash": payload["manifest_hash"],
                "next_authorized_step": payload["next_authorized_step"],
                "output": repository_path(args.output).as_posix(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
