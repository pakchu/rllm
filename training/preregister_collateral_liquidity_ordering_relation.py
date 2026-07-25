"""Preregister the source- and outcome-blind CLOR-D1 policy."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
POLICY_ID = "CLOR-D1"
PROTOCOL_VERSION = "clor_d1_preregistration_v1"
PRODUCER_SCRIPT = (
    "training/preregister_collateral_liquidity_ordering_relation.py"
)
DEFAULT_OUTPUT = (
    "results/collateral_liquidity_ordering_relation_preregistration_"
    "2026-07-25.json"
)

SELECTION_DOCUMENT = "docs/post-cefs-d2-alpha-mechanism-audit-2026-07-25.md"
SELECTION_COMMIT = "006ff4286913c90ef766ce4ba2a563b12b6ec6c0"
SELECTION_SHA256 = (
    "8013f07934a4ef2000e69ba274be06f84142d360e62296b83ca1c2c160930717"
)
BOUNDARY_DOCUMENT = (
    "docs/collateral-liquidity-ordering-relation-target-policy-boundary-"
    "2026-07-25.md"
)
BOUNDARY_COMMIT = "c82dccf1aea20e4f71e9b676e3d3f22b00b92e77"
BOUNDARY_SHA256 = (
    "bc537d568701215e72199e632fcc196724927d68348f847e42a47776d248f9df"
)
COMMON_WINDOW_DOCUMENT = (
    "docs/novelty-comparator-common-window-policy-2026-07-23.md"
)
COMMON_WINDOW_COMMIT = "26c37a88d2286bd6bfe535c00f8d48009ac08dd5"
COMMON_WINDOW_SHA256 = (
    "928bce6e04fb34001478b4b4ea84156580b661c88a0f0338065a891c009bd580"
)

GIT_EXECUTABLE = "/usr/bin/git"
GIT_EXECUTABLE_SHA256 = (
    "2a8c18fbf43da9f692d75474c72bea9dfd796c260b0f3dfe456376abc3bbd668"
)

TREASURY_PATH = (
    "data/us_treasury_auction_demand_2016_2023/"
    "us_treasury_nominal_original_auctions_2016_2023.csv.gz"
)
TREASURY_COMMIT = "e13ed6edec7569a2711a23ccc3d3a573fa53e7bb"
TREASURY_SHA256 = (
    "34a19163630c015a4f9d2671c95ca7cf7cc8a8ada024b3ef985405704fe0e4c1"
)
TREASURY_HEADER_SHA256 = (
    "4f7eab19bebc30f60ded1f6520ee54e2418bc05ef86dde632cad7762e4abf5bf"
)
TREASURY_MANIFEST = (
    "data/us_treasury_auction_demand_2016_2023/build_manifest.json"
)
TREASURY_MANIFEST_COMMIT = TREASURY_COMMIT
TREASURY_MANIFEST_SHA256 = (
    "6da6a3848e89c3418efcbf0d836fda34b537a2da87a8777b74670f3912ad94f2"
)

SOMA_OPERATIONS_PATH = (
    "data/new_york_fed_securities_lending_2019_2023/"
    "new_york_fed_securities_lending_operations_2019_2023.csv.gz"
)
SOMA_DETAILS_PATH = (
    "data/new_york_fed_securities_lending_2019_2023/"
    "new_york_fed_securities_lending_details_2019_2023.csv.gz"
)
SOMA_COMMIT = "8946140b32929b393978357ac36bf102e469e316"
SOMA_OPERATIONS_SHA256 = (
    "99eb8c37c05417789dfad7452c7b2ddc5b6b640078b87451f1c945158af77906"
)
SOMA_OPERATIONS_HEADER_SHA256 = (
    "c0d63795e5e53cef816c50472c6941069cb018f30ad1f745f250daa0fa6b9200"
)
SOMA_DETAILS_SHA256 = (
    "27178d8738cb50c4e6c13f1e5940fcfdf4009e6979b006c42fb86fb399d0716d"
)
SOMA_DETAILS_HEADER_SHA256 = (
    "9f4d54dff4b9c9f0c47c0a85e0bf245276e5a3cb764b3c084017f679586b76dd"
)
SOMA_MANIFEST = (
    "data/new_york_fed_securities_lending_2019_2023/build_manifest.json"
)
SOMA_MANIFEST_COMMIT = SOMA_COMMIT
SOMA_MANIFEST_SHA256 = (
    "58b9eb56728065d919978b8969e9bbb4bcb291f723a290d22045fe2ca3da2019"
)

OFR_PATH = (
    "data/ofr_repo_preliminary_2019_2023/"
    "ofr_repo_preliminary_observations_2019_2023.csv.gz"
)
OFR_COMMIT = "22f9119f606cdcbe5f7f882f43401138b94d1871"
OFR_SHA256 = (
    "6bb24b9a3d6ed6d266b0c885a9d9888bcccb3045df66d37d83031af66c6dc02a"
)
OFR_HEADER_SHA256 = (
    "d477472ebf9510be24bb4c596c2e795e468a3a3e774f8ba2279ba91fa44e36c4"
)
OFR_MANIFEST = "data/ofr_repo_preliminary_2019_2023/build_manifest.json"
OFR_MANIFEST_COMMIT = OFR_COMMIT
OFR_MANIFEST_SHA256 = (
    "f937f567e1789ecb39a2b84d6288b2cbab931da4e9f1f4e51addea4b3423b705"
)

SOURCE_BINDINGS = {
    "treasury": {
        "path": TREASURY_PATH,
        "commit": TREASURY_COMMIT,
        "sha256": TREASURY_SHA256,
        "header_sha256": TREASURY_HEADER_SHA256,
        "manifest_path": TREASURY_MANIFEST,
        "manifest_commit": TREASURY_MANIFEST_COMMIT,
        "manifest_sha256": TREASURY_MANIFEST_SHA256,
    },
    "soma_operations": {
        "path": SOMA_OPERATIONS_PATH,
        "commit": SOMA_COMMIT,
        "sha256": SOMA_OPERATIONS_SHA256,
        "header_sha256": SOMA_OPERATIONS_HEADER_SHA256,
        "manifest_path": SOMA_MANIFEST,
        "manifest_commit": SOMA_MANIFEST_COMMIT,
        "manifest_sha256": SOMA_MANIFEST_SHA256,
    },
    "soma_details": {
        "path": SOMA_DETAILS_PATH,
        "commit": SOMA_COMMIT,
        "sha256": SOMA_DETAILS_SHA256,
        "header_sha256": SOMA_DETAILS_HEADER_SHA256,
        "manifest_path": SOMA_MANIFEST,
        "manifest_commit": SOMA_MANIFEST_COMMIT,
        "manifest_sha256": SOMA_MANIFEST_SHA256,
    },
    "ofr": {
        "path": OFR_PATH,
        "commit": OFR_COMMIT,
        "sha256": OFR_SHA256,
        "header_sha256": OFR_HEADER_SHA256,
        "manifest_path": OFR_MANIFEST,
        "manifest_commit": OFR_MANIFEST_COMMIT,
        "manifest_sha256": OFR_MANIFEST_SHA256,
    },
}

PREDECESSOR_BINDINGS = {
    "TADI": {
        "path": (
            "results/treasury_auction_demand_impulse_preregistered_clock_"
            "2026-07-17.csv.gz"
        ),
        "commit": "cd15d7fa20f08a95c0e2f74fc75f60ad9e9c7130",
        "sha256": (
            "9bb416413a0cfee5a5ebbdb73032e5889735e88098eaa1dc264b6d224fa489f6"
        ),
    },
    "TASCC": {
        "path": (
            "data/treasury_auction_settlement_collision_carry_2020_2023/"
            "tascc72_support_clocks_2020_2023.csv.gz"
        ),
        "commit": "b39b72aa07c864dbdffee38749465d92c9b41389",
        "sha256": (
            "0333ba7f523d86a310e76ac51c15e4d273a1f4fb3e98f5e48dad530ac3696de4"
        ),
    },
    "SLCS": {
        "path": (
            "results/soma_lending_collateral_scarcity_clocks_"
            "2026-07-23.csv.gz"
        ),
        "commit": "4f59185c5073549f7ec792b301f80fb6248151d9",
        "sha256": (
            "b3fe0dc8c895f9a8974cdf08b5bed9d58ff693b8aca7ed59c224627de930a948"
        ),
    },
    "SCAF": {
        "path": "data/soma_collateral_allocation_fracture_clocks_2020_2023.csv.gz",
        "commit": "34a2c386acc60b8dbf2e215049d798596e7d8166",
        "sha256": (
            "64e07005d70442bfa7a110b1e6bea9802ee94be16d95f6e7db9228f4790a28e6"
        ),
    },
    "RVFC": {
        "path": (
            "results/ofr_repo_venue_fragmentation_consensus_clocks_"
            "2026-07-23.csv.gz"
        ),
        "commit": "c15cfa5e5d9afcf53cf6d2514fcae337872e928f",
        "sha256": (
            "b2d251b147feecc98def94610834c7b11f078f35e580f6e7a950fc55bfe0724e"
        ),
    },
    "RMSR": {
        "path": (
            "results/ofr_repo_mix_shock_resolution_race_clocks_"
            "2026-07-23.csv.gz"
        ),
        "commit": "819fe5edb42354a249f9fa5eebb51d00ad354367",
        "sha256": (
            "bed2f522bd68fd389cb0af3d655980999324c0e6adb4f1dc027c6b6ea2cc5ae6"
        ),
    },
    "RCRE": {
        "path": (
            "results/ofr_repo_collateral_routing_efficiency_clocks_"
            "2026-07-23.csv.gz"
        ),
        "commit": "075580d2d5632ac2c20ec934d2643a7792a52eb3",
        "sha256": (
            "cbe4e5f6fc52b66062abbf931e46ea4aa0d1f3c0157ffd365d0638aa573c2826"
        ),
    },
    "DMSH": {
        "path": (
            "results/ofr_dvp_maturity_stock_flow_handoff_clocks_"
            "2026-07-23.csv.gz"
        ),
        "commit": "c419dc3f5223d7ebea9f1ae3ae1565ba80b0100d",
        "sha256": (
            "0cfb881b4e3a0123111eeab904eba7bee074767b9c1315f74e7bddf54e3371c3"
        ),
    },
}
PREDECESSOR_PARSERS = {
    "TADI": {
        "header_sha256": (
            "1e798fc8c8cf2dc5c66e40aaa631a2fa20a8f4368b9551290686d92dada55046"
        ),
        "required_columns": [
            "entry_time",
            "scheduled_exit_time",
            "side",
            "clock_mode",
        ],
        "filter": {"clock_mode": "primary"},
        "interval_columns": ["entry_time", "scheduled_exit_time", "side"],
    },
    "TASCC": {
        "header_sha256": (
            "c2e97f5b9d7726dc5174ad4bff9e6af962f01e56d8d232236236afd3b82e1f3a"
        ),
        "required_columns": [
            "candidate",
            "control",
            "entry_time",
            "exit_time",
            "side",
        ],
        "filter": {
            "candidate": "TASCC-72-SOURCE-FAMILY-SEEN",
            "control": "primary",
        },
        "interval_columns": ["entry_time", "exit_time", "side"],
    },
    "SLCS": {
        "header_sha256": (
            "45a24e800b79a30047ffeb5f45c69cf4817262e57b0af1cf5e046332536e5e94"
        ),
        "required_columns": ["control", "entry_time", "exit_time", "side"],
        "filter": {"control": "primary"},
        "interval_columns": ["entry_time", "exit_time", "side"],
    },
    "SCAF": {
        "header_sha256": (
            "770965eb9e07bbca6f6b3f3c3165fe5c04301ef6573da86dacd161582cfa8c8f"
        ),
        "required_columns": ["control", "entry_time", "exit_time", "side"],
        "filter": {"control": "primary"},
        "interval_columns": ["entry_time", "exit_time", "side"],
    },
    "RVFC": {
        "header_sha256": (
            "93d6771691150abdb0f571460afa837a2c8e582ec8fafbf2f3203657a9801782"
        ),
        "required_columns": ["control", "entry_time", "exit_time", "side"],
        "filter": {"control": "primary"},
        "interval_columns": ["entry_time", "exit_time", "side"],
    },
    "RMSR": {
        "header_sha256": (
            "3053df0fdaaf4ab8015d36403f52f927d78e55301fead996ff02ca6cd4bf1660"
        ),
        "required_columns": ["control", "entry_time", "exit_time", "side"],
        "filter": {"control": "primary"},
        "interval_columns": ["entry_time", "exit_time", "side"],
    },
    "RCRE": {
        "header_sha256": (
            "ae1d29dc71aaf5149a77432f22626736026c10eda680cef47472d7b5a1348638"
        ),
        "required_columns": ["control", "entry_time", "exit_time", "side"],
        "filter": {"control": "primary"},
        "interval_columns": ["entry_time", "exit_time", "side"],
    },
    "DMSH": {
        "header_sha256": (
            "d8d2cb1cf0ba29c686b7abe9415a9fc9785fe81f24ef0af6516038665d7ec3bb"
        ),
        "required_columns": ["control", "entry_time", "exit_time", "side"],
        "filter": {"control": "primary"},
        "interval_columns": ["entry_time", "exit_time", "side"],
    },
}

TREASURY_ALLOWLIST = (
    "auction_date",
    "result_available_at_utc",
    "original_security_term",
    "competitive_accepted_usd",
    "primary_dealer_accepted_usd",
    "direct_bidder_accepted_usd",
    "indirect_bidder_accepted_usd",
    "source_complete",
)
TREASURY_PHYSICAL_HEADER = (
    "auction_date",
    "result_available_at_utc",
    "security_type",
    "original_security_term",
    "cusip",
    "bid_to_cover_ratio",
    "competitive_accepted_usd",
    "primary_dealer_accepted_usd",
    "direct_bidder_accepted_usd",
    "indirect_bidder_accepted_usd",
    "indirect_competitive_share",
    "closing_time_competitive_et",
    "updated_timestamp_et",
    "competitive_results_pdf_url",
    "competitive_results_xml_url",
    "source_complete",
)
SOMA_OPERATION_ALLOWLIST = (
    "operation_id",
    "operation_date",
    "available_at_utc",
    "total_par_submitted",
    "total_par_accepted",
)
SOMA_OPERATION_PHYSICAL_HEADER = (
    "operation_id",
    "operation_date",
    "settlement_date",
    "maturity_date",
    "release_time_et",
    "close_time_et",
    "last_updated_et",
    "available_at_utc",
    "note",
    "total_par_submitted",
    "total_par_accepted",
    "total_par_extended",
)
SOMA_DETAIL_ALLOWLIST = (
    "operation_id",
    "operation_date",
    "available_at_utc",
    "par_submitted",
    "par_accepted",
)
SOMA_DETAIL_PHYSICAL_HEADER = (
    "operation_id",
    "operation_date",
    "available_at_utc",
    "cusip",
    "security_description",
    "par_submitted",
    "par_accepted",
    "weighted_average_rate",
    "soma_holdings",
    "theoretical_available_to_borrow",
    "actual_available_to_borrow",
    "outstanding_loans",
)
OFR_ALLOWLIST = (
    "mnemonic",
    "observation_date",
    "available_at_utc",
    "value",
    "disclosure_edit",
)
OFR_PHYSICAL_HEADER = (
    "mnemonic",
    "observation_date",
    "available_at_utc",
    "value",
    "disclosure_edit",
    "segment",
    "measure",
    "subset",
    "series_name",
)
TREASURY_TERM_ORDER = (
    "2-Year",
    "3-Year",
    "5-Year",
    "7-Year",
    "10-Year",
    "20-Year",
    "30-Year",
)
OFR_MNEMONICS = (
    "REPO-DVP_AR_TOT-P",
    "REPO-GCF_AR_TOT-P",
    "REPO-TRIV1_AR_TOT-P",
    "REPO-DVP_TV_TOT-P",
    "REPO-GCF_TV_TOT-P",
    "REPO-TRIV1_TV_TOT-P",
)
CONTROL_IDS = (
    "treasury_bidder_label_rotation",
    "soma_one_batch_stale",
    "ofr_venue_label_rotation",
    "one_merged_update_stale",
    "within_year_source_time_reverse",
    "deterministic_random_relations",
    "future_append",
)
RELATION_FALSIFICATION_CONTROL_IDS = CONTROL_IDS[:-1]
APPEND_INVARIANCE_CONTROL_ID = CONTROL_IDS[-1]
CONTROL_DEFINITIONS = {
    "treasury_bidder_label_rotation": {
        "mapping": {"P": "D", "D": "I", "I": "P"},
        "simultaneous": True,
        "preserves": [
            "term_order",
            "equality_groups",
            "batch_timing",
            "all_other_fields",
        ],
    },
    "soma_one_batch_stale": {
        "rule": "prior_emitted_valid_SOMA_transition_at_current_SOMA_update",
        "first_emitted_transition": "unchanged",
    },
    "ofr_venue_label_rotation": {
        "mapping": {"DVP": "GCF", "GCF": "TRIV1", "TRIV1": "DVP"},
        "simultaneous": True,
        "preserves": [
            "equality_groups",
            "batch_timing",
            "all_other_fields",
        ],
    },
    "one_merged_update_stale": {
        "first_valid_line": "unchanged",
        "later_line_fields": "immediately_preceding_valid_primary_line",
        "execution_time": "current",
    },
    "within_year_source_time_reverse": {
        "partition": ["source", "UTC_calendar_year"],
        "operation": "reverse_valid_emitted_tokens_across_original_update_slots",
        "preserves": [
            "availability",
            "execution_group",
            "update_membership",
            "validity",
            "non_updating_source_timing",
        ],
        "recompute_modified_carried_state_until_next_source_update": True,
    },
    "deterministic_random_relations": {
        "key": (
            "CLOR-D1|deterministic_random_relations|<SOURCE>|"
            "<execution_time>|<field>"
        ),
        "digest_selection": "uint64_big_endian(first_8_SHA256_bytes)%vocabulary_size",
        "vocabulary_order": "ASCII_lexicographic",
        "step_vocabulary": ["DOWN", "EQUAL", "UP"],
        "weak_order_vocabulary": (
            "all_serializations_from_contiguous_three-label_rank_vectors;"
            "rank_zero_highest;ties_equals;descending_groups_greater_than;"
            "fixed_tie_label_order"
        ),
        "tie_label_orders": {
            "TREASURY": ["P", "D", "I"],
            "OFR": ["DVP", "GCF", "TRIV1"],
        },
        "field_keys": {
            "TREASURY": "term",
            "SOMA": [
                "submitted_step",
                "accepted_step",
                "coverage_step",
            ],
            "OFR": ["rate_order", "volume_order"],
        },
        "execution_time_format": "canonical_whole_second_UTC",
        "treasury_each_retained_term_independent": True,
        "preserves": [
            "primary_schedule",
            "validity",
            "term_membership",
            "update_membership",
        ],
    },
    "future_append": {
        "mode": "append_invariance_only",
        "synthetic_batches": [
            {
                "source": "TREASURY",
                "available_at_utc": "2024-01-02T00:00:00Z",
                "token": "2-Year:P>D>I",
            },
            {
                "source": "SOMA",
                "available_at_utc": "2024-01-02T00:05:00Z",
                "token": "UP,UP,UP",
            },
            {
                "source": "OFR",
                "available_at_utc": "2024-01-02T00:10:00Z",
                "token": "DVP>GCF>TRIV1,TRIV1>GCF>DVP",
            },
        ],
        "pre_2024_required_identical": [
            "primary_line",
            "validity",
            "sequence",
            "sequence_hash",
            "action_only_expiry",
        ],
        "minimum_changed_sequence_fraction": None,
        "hash_distinct_required": False,
    },
}
ACTION_SPACE = ("TARGET_LONG", "TARGET_FLAT", "TARGET_SHORT")
POSITION_SPACE = ("LONG", "FLAT", "SHORT")
PROMPT_INSTRUCTION = (
    "You manage one BTC perpetual target from causal collateral-release symbols.\n"
    "Use only ordering, transition, and update sequence. Do not invent numbers,\n"
    "dates, rules, or explanations. Return exactly one target token."
)
CANONICAL_LINE_GRAMMAR = (
    "UPDATED=<TREASURY,SOMA,OFR ordered subset>;"
    "TREASURY=<token>;"
    "SOMA=<submitted_step,accepted_step,coverage_step>;"
    "OFR=<rate_order,volume_order>"
)
ABLATION_LINE_GRAMMAR = (
    "UPDATED=<NONE_or_TREASURY,SOMA,OFR_ordered_remaining_subset>;"
    "TREASURY=<token|MASKED>;"
    "SOMA=<submitted_step,accepted_step,coverage_step|MASKED>;"
    "OFR=<rate_order,volume_order|MASKED>"
)
PROMPT_TEMPLATE = (
    f"{PROMPT_INSTRUCTION}\n\n"
    + "\n".join(f"STATE_{index:02d} <line>" for index in range(1, 13))
    + "\nCURRENT_POSITION=<LONG|FLAT|SHORT>"
    + "\nVALID_TARGETS=TARGET_LONG|TARGET_FLAT|TARGET_SHORT"
    + "\nTARGET="
)
FORBIDDEN_COUNTER_NAMES = (
    "comparator_action_rows_opened",
    "funding_rows_opened",
    "future_return_rows_built",
    "joint_state_rows_built",
    "market_rows_opened",
    "model_rows_built",
    "pnl_cagr_mdd_values_computed",
    "post_2023_source_value_rows_opened",
    "reward_rows_built",
    "selected_action_rows_built",
    "trade_rows_built",
)
TERMINAL_ACTIONS = {
    "failure": "retire_clor_d1_unchanged_before_outcomes",
    "pass": "authorize_clor_d1_economic_rllm_evaluator_freeze_only",
}


def repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        raise RuntimeError("CLOR-D1 repository path must be relative")
    resolved = (REPOSITORY_ROOT / candidate).resolve(strict=False)
    try:
        resolved.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as error:
        raise RuntimeError("CLOR-D1 repository path escaped root") from error
    return resolved


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(repository_path(path).read_bytes()).hexdigest()


def sha256_absolute_file(path: str | Path) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise RuntimeError("CLOR-D1 system path must be absolute")
    return hashlib.sha256(candidate.read_bytes()).hexdigest()


def canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def _git_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("GIT_")
    }
    environment.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_PAGER": "cat",
            "HOME": "/nonexistent-clor-d1-home",
            "LC_ALL": "C",
            "XDG_CONFIG_HOME": "/nonexistent-clor-d1-xdg",
        }
    )
    return environment


def _run_git(
    *args: str,
    check: bool = True,
    text: bool = True,
) -> subprocess.CompletedProcess[Any]:
    return subprocess.run(
        (GIT_EXECUTABLE, *args),
        cwd=REPOSITORY_ROOT,
        env=_git_environment(),
        check=check,
        capture_output=True,
        text=text,
    )


def _git_output(*args: str) -> str:
    completed = _run_git(*args)
    return completed.stdout.strip()


def _assert_committed(path: str, *, expected_commit: str | None = None) -> str:
    _git_output("ls-files", "--error-unmatch", "--", path)
    for args in (
        ("diff", "--quiet", "--", path),
        ("diff", "--cached", "--quiet", "--", path),
    ):
        completed = _run_git(*args, check=False)
        if completed.returncode:
            raise RuntimeError(f"CLOR-D1 frozen artifact is dirty: {path}")
    commit = _git_output("log", "-1", "--format=%H", "--", path)
    if expected_commit is not None and commit != expected_commit:
        raise RuntimeError(f"CLOR-D1 frozen artifact commit mismatch: {path}")
    return commit


def _git_blob_sha256(commit: str, path: str) -> str:
    completed = _run_git("show", f"{commit}:{path}", text=False)
    return hashlib.sha256(completed.stdout).hexdigest()


def validate_runtime_authority() -> dict[str, str]:
    path_value = os.environ.get("PATH")
    git_path = Path(GIT_EXECUTABLE)
    if not path_value:
        raise RuntimeError("CLOR-D1 PATH is missing")
    if "/usr/bin" not in path_value.split(os.pathsep):
        raise RuntimeError("CLOR-D1 PATH lacks exact /usr/bin component")
    if git_path.is_symlink() or not git_path.is_file():
        raise RuntimeError("CLOR-D1 Git executable is not a regular file")
    if sha256_absolute_file(git_path) != GIT_EXECUTABLE_SHA256:
        raise RuntimeError("CLOR-D1 Git executable hash mismatch")
    version = _git_output("--version")
    if not version.startswith("git version "):
        raise RuntimeError("CLOR-D1 Git version probe failed")
    top_level = Path(_git_output("rev-parse", "--show-toplevel")).resolve()
    if top_level != REPOSITORY_ROOT.resolve():
        raise RuntimeError("CLOR-D1 Git top-level authority mismatch")
    if _git_output("rev-parse", "--is-inside-work-tree") != "true":
        raise RuntimeError("CLOR-D1 Git worktree authority mismatch")
    return {
        "path": GIT_EXECUTABLE,
        "path_component": "/usr/bin",
        "sha256": GIT_EXECUTABLE_SHA256,
        "version": version,
        "top_level_matches_repository_root": True,
        "ambient_git_environment": "removed_before_every_git_subprocess",
        "system_and_user_git_config": "disabled",
    }


def _binding(path: str, commit: str, digest: str) -> dict[str, str]:
    return {"path": path, "commit": commit, "sha256": digest}


def _validate_binding(path: str, commit: str, digest: str) -> None:
    if _assert_committed(path, expected_commit=commit) != commit:
        raise RuntimeError(f"CLOR-D1 binding commit mismatch: {path}")
    if sha256_file(path) != digest:
        raise RuntimeError(f"CLOR-D1 binding file hash mismatch: {path}")
    if _git_blob_sha256(commit, path) != digest:
        raise RuntimeError(f"CLOR-D1 binding blob hash mismatch: {path}")


def csv_header_bytes(path: str | Path) -> bytes:
    source = repository_path(path)
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rb") as stream:
        header = stream.readline()
    if not header.endswith(b"\n") or b"\n" in header[:-1] or b"\r" in header:
        raise RuntimeError(f"CLOR-D1 noncanonical CSV header: {path}")
    return header


def csv_header(path: str | Path) -> tuple[str, ...]:
    raw = csv_header_bytes(path).decode("utf-8")
    columns = tuple(next(csv.reader([raw.rstrip("\n")])))
    if not columns or any(not column for column in columns):
        raise RuntimeError(f"CLOR-D1 empty CSV header field: {path}")
    if len(columns) != len(set(columns)):
        raise RuntimeError(f"CLOR-D1 duplicate CSV header field: {path}")
    return columns


def _validate_csv_header(
    path: str,
    expected_header: tuple[str, ...],
    expected_sha256: str,
) -> None:
    if csv_header(path) != expected_header:
        raise RuntimeError(f"CLOR-D1 physical header mismatch: {path}")
    if hashlib.sha256(csv_header_bytes(path)).hexdigest() != expected_sha256:
        raise RuntimeError(f"CLOR-D1 physical header hash mismatch: {path}")


def _require_false_or_zero(
    payload: Mapping[str, Any],
    keys: tuple[str, ...],
    *,
    label: str,
) -> None:
    for key in keys:
        if payload.get(key) not in (False, 0, [], {}):
            raise RuntimeError(f"CLOR-D1 {label} boundary opened: {key}")


def _require_all_true(
    payload: Mapping[str, Any],
    expected_keys: tuple[str, ...],
    *,
    label: str,
) -> None:
    if set(payload) != set(expected_keys) or any(
        payload[key] is not True for key in expected_keys
    ):
        raise RuntimeError(f"CLOR-D1 {label} checks mismatch")


def _validate_source_manifests() -> None:
    treasury = json.loads(repository_path(TREASURY_MANIFEST).read_text())
    soma = json.loads(repository_path(SOMA_MANIFEST).read_text())
    ofr = json.loads(repository_path(OFR_MANIFEST).read_text())
    _validate_csv_header(
        TREASURY_PATH,
        TREASURY_PHYSICAL_HEADER,
        TREASURY_HEADER_SHA256,
    )
    _validate_csv_header(
        SOMA_OPERATIONS_PATH,
        SOMA_OPERATION_PHYSICAL_HEADER,
        SOMA_OPERATIONS_HEADER_SHA256,
    )
    _validate_csv_header(
        SOMA_DETAILS_PATH,
        SOMA_DETAIL_PHYSICAL_HEADER,
        SOMA_DETAILS_HEADER_SHA256,
    )
    _validate_csv_header(OFR_PATH, OFR_PHYSICAL_HEADER, OFR_HEADER_SHA256)
    if (
        treasury.get("output") != TREASURY_PATH
        or treasury.get("output_sha256") != TREASURY_SHA256
        or treasury.get("columns") != list(TREASURY_PHYSICAL_HEADER)
        or treasury.get("schema_version") != 1
        or treasury.get("config", {}).get("start_date") != "2016-01-01"
        or treasury.get("config", {}).get("end_date") != "2023-12-31"
    ):
        raise RuntimeError("CLOR-D1 Treasury manifest output mismatch")
    _require_false_or_zero(
        treasury.get("protocol", {}),
        ("crypto_market_fields_opened", "future_source_rows_used", "outcomes_opened"),
        label="Treasury manifest",
    )
    if (
        soma.get("operations", {}).get("path") != SOMA_OPERATIONS_PATH
        or soma.get("operations", {}).get("sha256") != SOMA_OPERATIONS_SHA256
        or soma.get("operations", {}).get("columns")
        != list(SOMA_OPERATION_PHYSICAL_HEADER)
        or soma.get("details", {}).get("path") != SOMA_DETAILS_PATH
        or soma.get("details", {}).get("sha256") != SOMA_DETAILS_SHA256
        or soma.get("details", {}).get("columns")
        != list(SOMA_DETAIL_PHYSICAL_HEADER)
        or soma.get("source_window") != ["2019-01-01", "2023-12-31"]
    ):
        raise RuntimeError("CLOR-D1 SOMA manifest output mismatch")
    _require_all_true(
        soma.get("source_checks", {}),
        (
            "all_dates_inside_frozen_window",
            "availability_not_before_next_utc_midnight",
            "operation_cusips_unique",
            "operation_detail_totals_reconciled",
            "operation_ids_unique",
            "unknown_fields_rejected",
        ),
        label="SOMA manifest",
    )
    _require_false_or_zero(
        soma.get("research_boundary", {}),
        (
            "btc_market_rows_read",
            "candidate_features_computed",
            "candidate_incidence_opened",
            "funding_rows_read",
            "pnl_cagr_mdd_opened",
            "return_rows_read",
        ),
        label="SOMA manifest",
    )
    if (
        ofr.get("observations", {}).get("path") != OFR_PATH
        or ofr.get("observations", {}).get("sha256") != OFR_SHA256
        or ofr.get("observations", {}).get("columns")
        != list(OFR_PHYSICAL_HEADER)
        or ofr.get("source_window") != ["2019-01-01", "2023-12-31"]
        or ofr.get("availability_policy")
        != {
            "observation_lag_elapsed_days": 8,
            "preliminary_feed_floor_utc": "2020-09-10T00:00:00+00:00",
        }
    ):
        raise RuntimeError("CLOR-D1 OFR manifest output mismatch")
    _require_all_true(
        ofr.get("source_checks", {}),
        (
            "all_series_daily",
            "all_series_preliminary",
            "availability_matches_frozen_clock",
            "dates_inside_frozen_window",
            "final_or_asof_rows_read_zero",
            "measures_recognized",
            "out_of_window_disclosures_not_normalized",
            "preliminary_final_definitions_correspond",
            "prepublication_rows_not_backdated",
            "segments_recognized",
            "series_dates_unique",
            "source_decision_hash_matches",
            "unknown_envelope_fields_rejected",
        ),
        label="OFR manifest",
    )
    _require_false_or_zero(
        ofr.get("research_boundary", {}),
        (
            "btc_market_rows_read",
            "candidate_features_computed",
            "candidate_incidence_opened",
            "final_source_rows_read",
            "funding_rows_read",
            "pnl_cagr_mdd_opened",
            "return_rows_read",
        ),
        label="OFR manifest",
    )
    for allowlist, physical, label in (
        (TREASURY_ALLOWLIST, TREASURY_PHYSICAL_HEADER, "Treasury"),
        (SOMA_OPERATION_ALLOWLIST, SOMA_OPERATION_PHYSICAL_HEADER, "SOMA operations"),
        (SOMA_DETAIL_ALLOWLIST, SOMA_DETAIL_PHYSICAL_HEADER, "SOMA details"),
        (OFR_ALLOWLIST, OFR_PHYSICAL_HEADER, "OFR"),
    ):
        if len(allowlist) != len(set(allowlist)) or not set(allowlist).issubset(
            physical
        ):
            raise RuntimeError(f"CLOR-D1 {label} allowlist mismatch")


def _validate_predecessor_headers() -> None:
    for name, binding in PREDECESSOR_BINDINGS.items():
        parser = PREDECESSOR_PARSERS[name]
        observed = csv_header(binding["path"])
        if (
            hashlib.sha256(csv_header_bytes(binding["path"])).hexdigest()
            != parser["header_sha256"]
            or not set(parser["required_columns"]).issubset(observed)
            or not set(parser["filter"]).issubset(observed)
            or not set(parser["interval_columns"]).issubset(observed)
        ):
            raise RuntimeError(f"CLOR-D1 predecessor header mismatch: {name}")


def validate_frozen_authority() -> dict[str, Any]:
    runtime = validate_runtime_authority()
    _validate_binding(SELECTION_DOCUMENT, SELECTION_COMMIT, SELECTION_SHA256)
    _validate_binding(BOUNDARY_DOCUMENT, BOUNDARY_COMMIT, BOUNDARY_SHA256)
    _validate_binding(
        COMMON_WINDOW_DOCUMENT,
        COMMON_WINDOW_COMMIT,
        COMMON_WINDOW_SHA256,
    )
    validated_manifests: set[tuple[str, str, str]] = set()
    for item in SOURCE_BINDINGS.values():
        _validate_binding(item["path"], item["commit"], item["sha256"])
        manifest = (
            item["manifest_path"],
            item["manifest_commit"],
            item["manifest_sha256"],
        )
        if manifest not in validated_manifests:
            _validate_binding(*manifest)
            validated_manifests.add(manifest)
    for item in PREDECESSOR_BINDINGS.values():
        _validate_binding(item["path"], item["commit"], item["sha256"])
    _validate_source_manifests()
    _validate_predecessor_headers()
    return {
        "runtime": runtime,
        "selection": _binding(
            SELECTION_DOCUMENT,
            SELECTION_COMMIT,
            SELECTION_SHA256,
        ),
        "boundary": _binding(
            BOUNDARY_DOCUMENT,
            BOUNDARY_COMMIT,
            BOUNDARY_SHA256,
        ),
        "common_window_policy": _binding(
            COMMON_WINDOW_DOCUMENT,
            COMMON_WINDOW_COMMIT,
            COMMON_WINDOW_SHA256,
        ),
        "sources": {
            name: dict(binding)
            for name, binding in sorted(SOURCE_BINDINGS.items())
        },
        "predecessors": {
            name: {
                **dict(binding),
                "parser": dict(PREDECESSOR_PARSERS[name]),
            }
            for name, binding in sorted(PREDECESSOR_BINDINGS.items())
        },
    }


def producer_binding() -> dict[str, str]:
    commit = _assert_committed(PRODUCER_SCRIPT)
    return _binding(PRODUCER_SCRIPT, commit, sha256_file(PRODUCER_SCRIPT))


def scientific_contract() -> dict[str, Any]:
    return {
        "decision": {
            "policy_id": POLICY_ID,
            "phase": "source_only_preregistration",
            "architecture": "dense_causal_relation_sequence_target_policy",
            "claim": (
                "exact_joint_cards_actions_and_outcomes_unopened_only"
            ),
            "source_incidence_informed": True,
            "independent_or_pristine_claim": False,
            "predecessor_gzip_headers_decoded": True,
            "predecessor_value_or_action_rows_decoded": 0,
            "joint_state_rows_decoded": 0,
        },
        "sources": {
            "treasury": {
                **dict(SOURCE_BINDINGS["treasury"]),
                "allowlist": list(TREASURY_ALLOWLIST),
                "physical_header": list(TREASURY_PHYSICAL_HEADER),
                "term_order": list(TREASURY_TERM_ORDER),
                "forbidden": [
                    "bid_to_cover_ratio",
                    "indirect_competitive_share",
                    "cusip",
                    "closing_time_competitive_et",
                    "updated_timestamp_et",
                    "competitive_results_pdf_url",
                    "competitive_results_xml_url",
                ],
            },
            "soma_operations": {
                **dict(SOURCE_BINDINGS["soma_operations"]),
                "allowlist": list(SOMA_OPERATION_ALLOWLIST),
                "physical_header": list(SOMA_OPERATION_PHYSICAL_HEADER),
            },
            "soma_details": {
                **dict(SOURCE_BINDINGS["soma_details"]),
                "allowlist": list(SOMA_DETAIL_ALLOWLIST),
                "physical_header": list(SOMA_DETAIL_PHYSICAL_HEADER),
                "forbidden": [
                    "cusip",
                    "security_description",
                    "weighted_average_rate",
                    "soma_holdings",
                    "theoretical_available_to_borrow",
                    "actual_available_to_borrow",
                    "outstanding_loans",
                ],
            },
            "ofr": {
                **dict(SOURCE_BINDINGS["ofr"]),
                "allowlist": list(OFR_ALLOWLIST),
                "physical_header": list(OFR_PHYSICAL_HEADER),
                "mnemonics": list(OFR_MNEMONICS),
                "forbidden": [
                    "final_vintage",
                    "TRI",
                    "collateral_subdivisions",
                    "maturity_buckets",
                    "nulls",
                    "interpolation",
                    "forward_fill",
                ],
            },
        },
        "parsing": {
            "source_numeric_arithmetic": "exact_rational_only",
            "binary_floating_point_source_arithmetic": False,
            "timestamps": "timezone_aware_utc",
            "post_2023_source_values": False,
            "wider_then_drop_column_loading": False,
        },
        "primitives": {
            "treasury": {
                "batch_key": "exact_result_available_at_utc",
                "row_token": "exact_weak_order_of_P_D_I",
                "batch_order": "frozen_term_order",
                "scalar_score_or_side": False,
            },
            "soma": {
                "batch_key": "exact_available_at_utc",
                "totals": ["submitted", "accepted", "accepted/submitted"],
                "transition_levels": ["UP", "DOWN", "EQUAL"],
                "reference": "immediately_previous_complete_strictly_earlier_batch",
                "invalidity_resets": True,
                "rank_threshold_vote_or_side": False,
            },
            "ofr": {
                "batch_key": "exact_available_at_utc",
                "decision_row": "greatest_complete_observation_date_in_batch",
                "rate_order_labels": ["DVP", "GCF", "TRIV1"],
                "volume_order_labels": ["DVP", "GCF", "TRIV1"],
                "dispersion_hhi_rank_threshold_product_or_side": False,
            },
        },
        "clock": {
            "execution_time": "ceil_to_5m(available_at_utc)+5m",
            "same_execution_group_source_order": [
                "TREASURY",
                "SOMA",
                "OFR",
            ],
            "freshness_elapsed_days": {
                "TREASURY": 14,
                "SOMA": 4,
                "OFR": 4,
            },
            "invalid_line_action": "TARGET_FLAT",
            "invalid_line_resets_sequence": True,
            "future_rows_may_move_prior_clock": False,
            "maximum_target_age_elapsed_hours": 72,
            "expiry_base": "last_valid_model_decision_execution_time",
            "same_timestamp_order": "source_group_then_old_expiry",
            "fresh_decision_cancels_old_expiry": True,
            "expiry_transition": {
                "action": "TARGET_FLAT",
                "source_line_emitted": False,
                "sequence_changed": False,
                "model_invoked": False,
                "durable_position_changed_only": True,
                "already_flat": "no_op",
            },
        },
        "sequence_language": {
            "length": 12,
            "first_decision_valid_line_number": 12,
            "decision_uses_current_line": True,
            "line_fields": ["UPDATED", "TREASURY", "SOMA", "OFR"],
            "canonical_line_grammar": CANONICAL_LINE_GRAMMAR,
            "updated_member_order": ["TREASURY", "SOMA", "OFR"],
            "prompt_instruction": PROMPT_INSTRUCTION,
            "prompt_template": PROMPT_TEMPLATE,
            "encoding": "UTF-8_ASCII_subset",
            "newline": "LF",
            "terminal_newline": False,
            "action_space": list(ACTION_SPACE),
            "position_space": list(POSITION_SPACE),
            "malformed_output": "TARGET_FLAT",
            "current_position_included": True,
            "raw_numbers_dates_ids_prices_returns_rewards_splits": False,
            "free_text_or_hidden_reasoning": False,
        },
        "splits": {
            "warmup_end": "2020-09-10T00:00:00Z",
            "TRAIN": [
                "2020-09-10T00:00:00Z",
                "2022-01-01T00:00:00Z",
            ],
            "TEST": [
                "2022-01-01T00:00:00Z",
                "2023-01-01T00:00:00Z",
            ],
            "EVAL": [
                "2023-01-01T00:00:00Z",
                "2024-01-01T00:00:00Z",
            ],
            "sealed_from": "2024-01-01T00:00:00Z",
            "sequence_and_position_reset_each_split": True,
        },
        "support_gates": {
            "minimum_model_decisions": {
                "TRAIN": 450,
                "TEST": 180,
                "EVAL": 180,
            },
            "minimum_updated_decisions": {
                "TREASURY": {"TRAIN": 40, "TEST": 20, "EVAL": 20},
                "SOMA": {"TRAIN": 200, "TEST": 90, "EVAL": 90},
                "OFR": {"TRAIN": 200, "TEST": 90, "EVAL": 90},
            },
            "maximum_decision_gap_elapsed_days": 10,
            "minimum_2020_post_floor_decisions": 30,
            "minimum_2021_quarter_decisions": 50,
            "minimum_test_eval_quarter_decisions": 40,
            "primitive_minimum_levels_each_split": 2,
            "primitive_maximum_level_share": 0.95,
            "maximum_state_line_signature_share": 0.25,
            "minimum_unique_sequence_hashes": {
                "TRAIN": 150,
                "TEST": 70,
                "EVAL": 70,
            },
            "relation_control_minimum_changed_sequence_fraction": 0.10,
            "relation_controls": list(RELATION_FALSIFICATION_CONTROL_IDS),
            "append_invariance_control": APPEND_INVARIANCE_CONTROL_ID,
            "append_invariance_requires_pre_2024_byte_identity": True,
            "append_invariance_changed_hash_floor": None,
            "first_failure_stops": True,
        },
        "controls": {
            "ordered_ids": list(CONTROL_IDS),
            "definitions": {
                name: dict(CONTROL_DEFINITIONS[name]) for name in CONTROL_IDS
            },
            "same_primary_clock_except_future_append": True,
            "may_replace_primary": False,
        },
        "predecessor_and_ablation_gates": {
            "bindings": {
                name: {
                    **dict(binding),
                    "parser": dict(PREDECESSOR_PARSERS[name]),
                }
                for name, binding in sorted(PREDECESSOR_BINDINGS.items())
            },
            "common_window_policy": _binding(
                COMMON_WINDOW_DOCUMENT,
                COMMON_WINDOW_COMMIT,
                COMMON_WINDOW_SHA256,
            ),
            "comparison_window": [
                "2020-09-10T00:00:00Z",
                "2023-01-01T00:00:00Z",
            ],
            "clor_interval": {
                "entry": "durable_target_change_to_LONG_or_SHORT",
                "reversal_is_entry": True,
                "repeated_target_is_entry": False,
                "exit": [
                    "next_target_change",
                    "safety_flat",
                    "72h_expiry",
                    "split_end",
                ],
                "interval": "entry_inclusive_exit_exclusive",
            },
            "exact_entry_key": "canonical_entry_timestamp_signless",
            "exact_entry_jaccard": "intersection_size/union_size",
            "empty_entry_set": "fail",
            "exact_entry_jaccard_max": 0.35,
            "one_to_one_tolerance_hours": 6,
            "one_to_one_side": "ignored_for_gate",
            "one_to_one_algorithm": (
                "sort_left(entry_time,row_identity);for_each_left_choose_"
                "unmatched_right_min(abs_delta,right_entry,right_identity);"
                "run_independently_both_directions"
            ),
            "row_identity": (
                "SHA256(CLOR-D1|comparison|artifact_sha256|group|"
                "entry|exit|side)"
            ),
            "both_matched_fractions_max": 0.50,
            "exposure_grid": (
                "every_5m_UTC_interval_start_over_complete_common_window"
            ),
            "clor_exposure": {"LONG": 1, "FLAT": 0, "SHORT": -1},
            "predecessor_exposure": (
                "side_when_entry_time<=t<exit_time_else_zero"
            ),
            "joint_zero_grid_cells_included": True,
            "absolute_exposure_correlation_max": 0.40,
            "ablations": ["no_Treasury", "no_SOMA", "no_OFR"],
            "ablation_mask": {
                "canonical_line_grammar": ABLATION_LINE_GRAMMAR,
                "omitted_source_field": "MASKED",
                "exactly_one_masked_source_field": True,
                "remove_omitted_source_from_UPDATED": True,
                "empty_UPDATED": "NONE",
                "NONE_and_MASKED_legal_only_in_ablations": True,
                "ablation_id_in_prompt": False,
                "validity_and_schedule": "full_primary_preserved",
                "freshness_recomputed": False,
                "position": "separately_trained_ablation_durable_position",
            },
            "source_support_uses_primary_nonempty_UPDATED_only": True,
            "ablations_may_affect_or_rescue_source_support": False,
            "ablation_identical": [
                "seeds",
                "budgets",
                "reward",
                "checkpoint_rules",
                "execution_schedule",
                "split_resets",
                "model_decision_timestamps",
                "72h_expiry",
                "sequence_length",
            ],
            "ablation_target_change_fraction": (
                "unequal_target_tokens/all_identical_model_decision_timestamps"
            ),
            "ablation_minimum_target_change_fraction": 0.10,
            "full_test_absolute_return_positive": True,
            "full_test_cagr_to_strict_mdd_positive": True,
            "full_test_ratio_margin_over_each_ablation": 0.25,
            "missing_ambiguous_or_clipped_comparator_fails": True,
        },
        "economic_rllm_boundary": {
            "authorized_now": False,
            "one_compact_gemma_family_model": True,
            "analyzer_trader_dual_model": False,
            "train_updates_only": True,
            "test_checkpoint_selection_only": True,
            "eval_one_shot": True,
            "cheap_baselines_required": True,
            "exact_model_training_reward_and_simulator_freeze_required": True,
        },
        "live_boundary": {
            "research_only": True,
            "live_target": "TARGET_FLAT",
            "forward_raw_capture_required": True,
            "revision_alarms_required": True,
            "shadow_parity_required": True,
        },
        "forbidden_access": {
            name: 0 for name in FORBIDDEN_COUNTER_NAMES
        },
        "terminal_actions": dict(TERMINAL_ACTIONS),
    }


def manifest_core(producer: Mapping[str, str]) -> dict[str, Any]:
    contract = scientific_contract()
    authority = validate_frozen_authority()
    authority["producer"] = dict(producer)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "authority": authority,
        "scientific_contract": contract,
        "scientific_contract_hash": canonical_hash(contract),
        "terminal_actions": dict(TERMINAL_ACTIONS),
        "forbidden_access": {
            name: 0 for name in FORBIDDEN_COUNTER_NAMES
        },
        "source_rows_parsed": 0,
        "source_values_opened": False,
        "joint_state_rows_built": 0,
        "outcomes_opened": False,
    }


def build_manifest() -> dict[str, Any]:
    core = manifest_core(producer_binding())
    return {**core, "manifest_hash": canonical_hash(core)}


def _validate_manifest_structure(payload: Mapping[str, Any]) -> None:
    expected_fields = {
        "protocol_version",
        "policy_id",
        "authority",
        "scientific_contract",
        "scientific_contract_hash",
        "terminal_actions",
        "forbidden_access",
        "source_rows_parsed",
        "source_values_opened",
        "joint_state_rows_built",
        "outcomes_opened",
        "manifest_hash",
    }
    if set(payload) != expected_fields:
        raise RuntimeError("CLOR-D1 preregistration fields mismatch")
    core = {
        key: value for key, value in payload.items() if key != "manifest_hash"
    }
    if payload["manifest_hash"] != canonical_hash(core):
        raise RuntimeError("CLOR-D1 preregistration manifest hash mismatch")
    forbidden = payload["forbidden_access"]
    if (
        payload["protocol_version"] != PROTOCOL_VERSION
        or payload["policy_id"] != POLICY_ID
        or payload["scientific_contract"] != scientific_contract()
        or payload["scientific_contract_hash"]
        != canonical_hash(payload["scientific_contract"])
        or payload["terminal_actions"] != TERMINAL_ACTIONS
        or payload["source_rows_parsed"] != 0
        or payload["source_values_opened"] is not False
        or payload["joint_state_rows_built"] != 0
        or payload["outcomes_opened"] is not False
        or not isinstance(forbidden, dict)
        or set(forbidden) != set(FORBIDDEN_COUNTER_NAMES)
        or any(forbidden.values())
    ):
        raise RuntimeError("CLOR-D1 preregistration invariant mismatch")
    authority = payload["authority"]
    if not isinstance(authority, dict):
        raise RuntimeError("CLOR-D1 authority is not a mapping")
    producer = authority.get("producer")
    if (
        not isinstance(producer, dict)
        or set(producer) != {"path", "commit", "sha256"}
        or producer.get("path") != PRODUCER_SCRIPT
        or not isinstance(producer.get("commit"), str)
        or len(producer["commit"]) != 40
        or any(c not in "0123456789abcdef" for c in producer["commit"])
        or not isinstance(producer.get("sha256"), str)
        or len(producer["sha256"]) != 64
        or any(c not in "0123456789abcdef" for c in producer["sha256"])
    ):
        raise RuntimeError("CLOR-D1 producer binding grammar mismatch")
    expected_authority = validate_frozen_authority()
    observed_authority = dict(authority)
    observed_authority.pop("producer")
    if observed_authority != expected_authority:
        raise RuntimeError("CLOR-D1 preregistration authority mismatch")


def validate_manifest(payload: Mapping[str, Any]) -> None:
    _validate_manifest_structure(payload)
    _validate_sealed_producer(payload)


def _validate_sealed_producer(payload: Mapping[str, Any]) -> None:
    producer = payload["authority"]["producer"]
    try:
        sealed = _run_git(
            "show",
            f"{producer['commit']}:{PRODUCER_SCRIPT}",
            text=False,
        ).stdout
    except subprocess.CalledProcessError as error:
        raise RuntimeError("CLOR-D1 producer commit is unreadable") from error
    if hashlib.sha256(sealed).hexdigest() != producer["sha256"]:
        raise RuntimeError("CLOR-D1 sealed producer bytes mismatch")


def _encoded_json(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _relative_output_path(path: str | Path) -> Path:
    candidate = Path(path)
    if (
        candidate.is_absolute()
        or not candidate.parts
        or any(part in ("", ".", "..") for part in candidate.parts)
    ):
        raise RuntimeError("CLOR-D1 output path is unsafe")
    repository_path(candidate)
    return candidate


def _open_output_parent(path: str | Path) -> tuple[int, str]:
    relative = _relative_output_path(path)
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(REPOSITORY_ROOT, flags)
    try:
        for part in relative.parent.parts:
            try:
                next_descriptor = os.open(part, flags, dir_fd=descriptor)
            except OSError as error:
                raise RuntimeError(
                    "CLOR-D1 output parent is missing or unsafe"
                ) from error
            os.close(descriptor)
            descriptor = next_descriptor
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, relative.name


def _output_entry_exists(path: str | Path) -> bool:
    parent_descriptor, name = _open_output_parent(path)
    try:
        try:
            metadata = os.stat(
                name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return False
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError("CLOR-D1 output is not a regular file")
        return True
    finally:
        os.close(parent_descriptor)


def _read_existing_output(parent_descriptor: int, name: str) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError("CLOR-D1 output is not a regular file")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            return stream.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _write_once_bytes(path: str | Path, encoded: bytes) -> str:
    parent_descriptor, name = _open_output_parent(path)
    try:
        try:
            existing = _read_existing_output(parent_descriptor, name)
        except FileNotFoundError:
            existing = None
        if existing is not None:
            if existing != encoded:
                raise RuntimeError("CLOR-D1 preregistration artifact drift")
            return hashlib.sha256(encoded).hexdigest()
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(
                name,
                flags,
                0o644,
                dir_fd=parent_descriptor,
            )
        except FileExistsError:
            concurrent = _read_existing_output(parent_descriptor, name)
            if concurrent != encoded:
                raise RuntimeError(
                    "CLOR-D1 concurrent preregistration artifact drift"
                )
            return hashlib.sha256(encoded).hexdigest()
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            try:
                os.unlink(name, dir_fd=parent_descriptor)
                os.fsync(parent_descriptor)
            except FileNotFoundError:
                pass
            raise
        os.fsync(parent_descriptor)
        return hashlib.sha256(encoded).hexdigest()
    finally:
        os.close(parent_descriptor)


def write_once(path: str | Path, payload: Mapping[str, Any]) -> str:
    if str(path) != DEFAULT_OUTPUT:
        raise RuntimeError("CLOR-D1 preregistration path is frozen")
    validate_manifest(payload)
    encoded = _encoded_json(payload)
    current = producer_binding()
    if current != payload["authority"]["producer"]:
        raise RuntimeError("CLOR-D1 current producer differs from manifest")
    head = _git_output("rev-parse", "HEAD")
    exists = _output_entry_exists(path)
    if exists:
        ancestry = _run_git(
            "merge-base",
            "--is-ancestor",
            current["commit"],
            head,
            check=False,
        )
        if ancestry.returncode != 0:
            raise RuntimeError(
                "CLOR-D1 existing artifact producer is not a HEAD ancestor"
            )
    elif head != current["commit"]:
        raise RuntimeError(
            "CLOR-D1 artifact creation requires sealed producer HEAD"
        )
    return _write_once_bytes(path, encoded)


def create(path: str = DEFAULT_OUTPUT) -> dict[str, Any]:
    payload = build_manifest()
    write_once(path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = create(args.output)
    print(
        json.dumps(
            {
                "path": args.output,
                "policy_id": payload["policy_id"],
                "manifest_hash": payload["manifest_hash"],
                "source_values_opened": payload["source_values_opened"],
                "outcomes_opened": payload["outcomes_opened"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
