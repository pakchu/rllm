"""Freeze DCLB-864 before decoding joint source incidence or market outcomes.

This stage validates immutable file hashes and reads CSV header lines only. It
does not parse an H.4.1, ON RRP, H.8, comparator, BTC, funding, return, or PnL
value row.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import csv
import errno
import gzip
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    "results/dollar_collateral_liquidity_bank_relay_"
    "preregistration_2026-07-24.json"
)

BOUNDARY_DOCUMENT = (
    "docs/dollar-collateral-liquidity-bank-relay-boundary-2026-07-24.md"
)
BOUNDARY_DOCUMENT_SHA256 = (
    "fed61a096acf0186f153f0cc4e939cec39652fa68eba1fe5afd480121572bf24"
)
MECHANISM_DOCUMENT = (
    "docs/dollar-collateral-liquidity-bank-relay-"
    "mechanism-decision-2026-07-24.md"
)
MECHANISM_DOCUMENT_SHA256 = (
    "71fff458151cc74ae16ec48d1a8d17a91a7b899a0d99715e55c98628e1a9d686"
)
COMMON_WINDOW_POLICY = (
    "docs/novelty-comparator-common-window-policy-2026-07-23.md"
)
COMMON_WINDOW_POLICY_SHA256 = (
    "928bce6e04fb34001478b4b4ea84156580b661c88a0f0338065a891c009bd580"
)

H41_SOURCE = (
    "data/federal_reserve_h41_net_liquidity_2018_2023/"
    "federal_reserve_h41_net_liquidity_2018-01-04_2023-12-28.csv.gz"
)
H41_SOURCE_SHA256 = (
    "224883dad01b9d7f17d52eb87f3d7ef9890c8dd055a6c36577a534d2afe69621"
)
H41_HEADER_SHA256 = (
    "4bd522eddda52fefa94c9722f6015596fcde80769c59441046bc0438e1d314d9"
)
H41_MANIFEST = (
    "data/federal_reserve_h41_net_liquidity_2018_2023/build_manifest.json"
)
H41_MANIFEST_SHA256 = (
    "1ec212a85de0e49c5a0c2d35b8b22be86eb7d62989f7a0098be1bb1274b2a99b"
)
H41_AUDIT = (
    "docs/federal-reserve-h41-net-liquidity-source-audit-2026-07-17.md"
)
H41_AUDIT_SHA256 = (
    "fec3e11fa5112aff97b338c6404b3202cfa38c02391debaeee3c18ff3b3b23c1"
)
H41_ALLOWLIST = (
    "release_date",
    "observation_date",
    "available_at_utc",
    "net_liquidity_usd_millions",
)

RRP_SOURCE = (
    "data/new_york_fed_overnight_rrp_2018_2023/"
    "new_york_fed_overnight_rrp_2018-01-01_2023-12-31.csv.gz"
)
RRP_SOURCE_SHA256 = (
    "49f67ed44b7eb81fd35c17a8209cf14d6a8019d7e9f77fce8c343d1a7fb66b27"
)
RRP_HEADER_SHA256 = (
    "81a388d6e36c5e84c166b5fe111d3766ea5c6b56ac83895ed3541a6c05a01e9c"
)
RRP_MANIFEST = "data/new_york_fed_overnight_rrp_2018_2023/build_manifest.json"
RRP_MANIFEST_SHA256 = (
    "4f87e2219da71c94832c8708086ba01387efc145e3488b62cd3b3d07c62d8fee"
)
RRP_AUDIT = "docs/new-york-fed-overnight-rrp-source-audit-2026-07-17.md"
RRP_AUDIT_SHA256 = (
    "329db1cf886bfbceb0a048b1c44c59378af717ddd9731e5e26fd09e14ada8d23"
)
RRP_ALLOWLIST = (
    "operation_date",
    "result_available_at_utc",
    "total_amount_accepted_usd",
    "source_complete",
    "quarantine_reason",
)

H8_SOURCE = (
    "data/fed_h8_deposit_migration_2017_2023/"
    "fed_h8_deposit_migration_2017_2023.csv.gz"
)
H8_SOURCE_SHA256 = (
    "c8d1bfb0bbd13ef6d35f09ad7367ef8d2d5bb28981376223b735746ade68a572"
)
H8_HEADER_SHA256 = (
    "b9c20c15035b90266cb47b8465922fde5f1062c3634050b0f006ee6263b978e8"
)
H8_MANIFEST = "data/fed_h8_deposit_migration_2017_2023/build_manifest.json"
H8_MANIFEST_SHA256 = (
    "1f0a194e628ab9c44c23fc4a923145dcf89a62bface745cc36872eeee919eda9"
)
H8_AUDIT = "docs/fed-h8-deposit-migration-source-audit-2026-07-18.md"
H8_AUDIT_SHA256 = (
    "a022ee840b3db977030de057890c2c57cd0a45029c879a3b37d648a49e063d3c"
)
H8_ALLOWLIST = (
    "release_date",
    "release_time_utc",
    "release_weekday",
    "sa_large_other_deposits_prior",
    "sa_large_other_deposits_latest",
    "sa_small_other_deposits_prior",
    "sa_small_other_deposits_latest",
    "sa_small_borrowings_prior",
    "sa_small_borrowings_latest",
    "sa_small_cash_assets_prior",
    "sa_small_cash_assets_latest",
    "nsa_large_other_deposits_prior",
    "nsa_large_other_deposits_latest",
    "nsa_small_other_deposits_prior",
    "nsa_small_other_deposits_latest",
    "nsa_small_borrowings_prior",
    "nsa_small_borrowings_latest",
    "nsa_small_cash_assets_prior",
    "nsa_small_cash_assets_latest",
)

CONTROL_ORDER = (
    "primary",
    "h41_only",
    "rrp_interval_only",
    "h8_only",
    "macro_concordant_only",
    "macro_discordant_only",
    "bank_supports_only",
    "bank_opposes_only",
    "stale_h41_one_release",
    "stale_rrp_one_interval",
    "exact_direction_flip",
    "deterministic_random_side",
    "one_h8_release_execution_delay",
    "nsa_h8",
)


@dataclass(frozen=True)
class Policy:
    policy_id: str = "DCLB-864"
    h41_prior_deltas: int = 104
    rrp_prior_interval_deltas: int = 13
    h8_prior_component_observations: int = 104
    h8_decision_local_time: str = "17:00:00 America/New_York"
    entry_local_time: str = "17:05:00 America/New_York"
    bar_minutes: int = 5
    hold_bars: int = 864
    hold_elapsed_minutes: int = 4_320
    gross_exposure: float = 0.50
    random_namespace: str = "DCLB-864"


def _repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    raw = str(path)
    if raw.startswith("~") or candidate.is_absolute() or ".." in candidate.parts:
        raise RuntimeError("DCLB-864 dependency path must be repository-relative")
    root = REPOSITORY_ROOT.resolve(strict=True)
    current = REPOSITORY_ROOT
    for part in candidate.parts:
        current /= part
        if current.is_symlink():
            raise RuntimeError("DCLB-864 dependency path contains a symlink")
        if not current.exists():
            break
    target = REPOSITORY_ROOT / candidate
    try:
        target.resolve(strict=False).relative_to(root)
    except ValueError as error:
        raise RuntimeError("DCLB-864 dependency path escapes repository") from error
    return target


def _output_relative_path(path: str | Path) -> Path:
    candidate = Path(path)
    raw = str(path)
    if (
        raw.startswith("~")
        or candidate.is_absolute()
        or ".." in candidate.parts
        or candidate.name in {"", ".", ".."}
    ):
        raise RuntimeError("DCLB-864 output path must be repository-relative")
    return candidate


def _open_output_parent(candidate: Path) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    try:
        current_fd = os.open(REPOSITORY_ROOT, flags)
    except OSError as error:
        raise RuntimeError("DCLB-864 repository root is not a safe directory") from error
    try:
        for part in candidate.parent.parts:
            try:
                next_fd = os.open(part, flags, dir_fd=current_fd)
            except OSError as error:
                raise RuntimeError(
                    "DCLB-864 output parent is missing, non-directory, or symlinked"
                ) from error
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except Exception:
        os.close(current_fd)
        raise


def _read_regular_at(directory_fd: int, name: str) -> bytes:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except FileNotFoundError:
        raise
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise RuntimeError("DCLB-864 output is symlinked or invalid") from error
        raise
    try:
        mode = os.fstat(descriptor).st_mode
        if not stat.S_ISREG(mode):
            raise RuntimeError("DCLB-864 output is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: str | Path) -> str:
    candidate = Path(path)
    target = candidate if candidate.is_absolute() else _repository_path(candidate)
    return _sha256_path(target)


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def csv_header_bytes(path: str | Path) -> bytes:
    target = _repository_path(path)
    opener = gzip.open if target.suffix == ".gz" else open
    with opener(target, "rb") as handle:
        header = handle.readline()
    if not header.endswith(b"\n") or b"\n" in header[:-1]:
        raise RuntimeError(f"DCLB-864 CSV header is not one LF line: {path}")
    return header


def csv_header(path: str | Path) -> list[str]:
    header = csv_header_bytes(path).decode("utf-8")
    columns = next(csv.reader([header.rstrip("\n")]))
    if len(columns) != len(set(columns)):
        raise RuntimeError(f"DCLB-864 CSV header has duplicate columns: {path}")
    return columns


def sha256_csv_header(path: str | Path) -> str:
    return hashlib.sha256(csv_header_bytes(path)).hexdigest()


def source_contracts() -> dict[str, dict[str, Any]]:
    return {
        "h41": {
            "path": H41_SOURCE,
            "sha256": H41_SOURCE_SHA256,
            "header_sha256": H41_HEADER_SHA256,
            "manifest": H41_MANIFEST,
            "manifest_sha256": H41_MANIFEST_SHA256,
            "audit": H41_AUDIT,
            "audit_sha256": H41_AUDIT_SHA256,
            "allowlist": list(H41_ALLOWLIST),
            "loader": "pandas.read_csv(usecols=exact_allowlist)",
            "numeric_rule": "finite and strictly positive",
        },
        "rrp": {
            "path": RRP_SOURCE,
            "sha256": RRP_SOURCE_SHA256,
            "header_sha256": RRP_HEADER_SHA256,
            "manifest": RRP_MANIFEST,
            "manifest_sha256": RRP_MANIFEST_SHA256,
            "audit": RRP_AUDIT,
            "audit_sha256": RRP_AUDIT_SHA256,
            "allowlist": list(RRP_ALLOWLIST),
            "loader": "pandas.read_csv(usecols=exact_allowlist)",
            "numeric_rule": (
                "amount blank iff source_complete=false with nonempty "
                "quarantine_reason; otherwise finite and nonnegative"
            ),
        },
        "h8": {
            "path": H8_SOURCE,
            "sha256": H8_SOURCE_SHA256,
            "header_sha256": H8_HEADER_SHA256,
            "manifest": H8_MANIFEST,
            "manifest_sha256": H8_MANIFEST_SHA256,
            "audit": H8_AUDIT,
            "audit_sha256": H8_AUDIT_SHA256,
            "allowlist": list(H8_ALLOWLIST),
            "loader": "pandas.read_csv(usecols=exact_allowlist)",
            "numeric_rule": "retained SA and NSA levels finite and strictly positive",
        },
    }


def _comparator_parser(
    *,
    group_columns: list[str],
    entry_column: str = "entry_time",
    exit_column: str = "exit_time",
    side_column: str = "side",
) -> dict[str, Any]:
    return {
        "read_csv": (
            "pandas.read_csv(usecols=exact_usecols,dtype='string',"
            "keep_default_na=False,na_filter=False)"
        ),
        "group_columns": group_columns,
        "group_filter": (
            "exact UTF-8 string equality on every frozen filter field; "
            "no strip, case-fold, regex, coercion, or substring matching"
        ),
        "entry_column": entry_column,
        "exit_column": exit_column,
        "timestamp_parser": (
            "require RFC3339 timezone suffix Z or +/-HH:MM, then "
            "pandas.to_datetime(errors='raise',utc=True)"
        ),
        "side_column": side_column,
        "side_mapping": {"LONG": 1, "SHORT": -1},
        "unknown_or_blank_side": "fail",
        "raw_validation": (
            "parse all artifact rows before containment; selected group "
            "requires unique entry, exit>entry, and chronological nonoverlap"
        ),
    }


def comparator_contracts() -> list[dict[str, Any]]:
    return [
        {
            "id": "FLCC",
            "path": (
                "results/federal_liquidity_component_concordance_"
                "preregistered_clock_2026-07-17.csv.gz"
            ),
            "sha256": (
                "7ebb0450422d9265e46c596e0b6415b6a8816c66f5e0cbb9ccda14ca6cb4c67c"
            ),
            "header_sha256": (
                "38d354aa0f63efa58bf5181cdc8cdecc4d9fc2f1a6eda8df8933c95f79cffdb7"
            ),
            "usecols": [
                "candidate_id",
                "clock_name",
                "entry_time",
                "exit_time",
                "side",
            ],
            "parser": _comparator_parser(
                group_columns=["candidate_id", "clock_name"]
            ),
            "groups": [
                {
                    "filter": {
                        "candidate_id": "FLCC-H4-Q60",
                        "clock_name": "primary",
                    },
                    "minimum_contained_rows": 90,
                },
                {
                    "filter": {
                        "candidate_id": "FLCC-H4-Q65",
                        "clock_name": "primary",
                    },
                    "minimum_contained_rows": 90,
                },
                {
                    "filter": {
                        "candidate_id": "FLCC-H8-Q60",
                        "clock_name": "primary",
                    },
                    "minimum_contained_rows": 90,
                },
                {
                    "filter": {
                        "candidate_id": "FLCC-H8-Q65",
                        "clock_name": "primary",
                    },
                    "minimum_contained_rows": 90,
                },
            ],
            "clock_family": "asynchronous",
        },
        {
            "id": "ORFR",
            "path": "results/overnight_rrp_flow_release_clocks_2026-07-17.csv.gz",
            "sha256": (
                "7242d9870627dfc0cf067ff87d9664a1576dd374cb8985e927b40f15d1e3d480"
            ),
            "header_sha256": (
                "3a45759e0b14eeef01ddfb5146ca03515a562846412c98fa1ca1aca7e285528e"
            ),
            "usecols": ["control", "entry_time", "exit_time", "side"],
            "parser": _comparator_parser(group_columns=["control"]),
            "groups": [
                {
                    "filter": {"control": "primary"},
                    "minimum_contained_rows": 150,
                },
                {
                    "filter": {"control": "one_day_delta_tail"},
                    "minimum_contained_rows": 150,
                },
                {
                    "filter": {"control": "one_release_delay"},
                    "minimum_contained_rows": 150,
                },
            ],
            "clock_family": "asynchronous",
        },
        {
            "id": "ORPB",
            "path": (
                "results/overnight_rrp_participant_breadth_"
                "support_clocks_2026-07-21.csv.gz"
            ),
            "sha256": (
                "ef21323229801f11557e0c2d9d4465f7d58b13569552d656d64fdb7d440622ed"
            ),
            "header_sha256": (
                "257ee9b477b9c62e9c287d03269d813c3a8b4b6286d836ab61ed5a925a2fd3f4"
            ),
            "usecols": [
                "candidate_id",
                "control",
                "entry_time",
                "exit_time",
                "side",
            ],
            "parser": _comparator_parser(
                group_columns=["candidate_id", "control"]
            ),
            "groups": [
                {
                    "filter": {
                        "candidate_id": "ORPB-21",
                        "control": "primary",
                    },
                    "minimum_contained_rows": 180,
                },
            ],
            "clock_family": "asynchronous",
        },
        {
            "id": "H8DM",
            "path": (
                "results/fed_h8_deposit_migration_"
                "preregistered_clock_2026-07-18.csv.gz"
            ),
            "sha256": (
                "20405f79b86861adcc784c81223baae1c40fdf3c73edda339578471a6a6d1b40"
            ),
            "header_sha256": (
                "58dd787ede642429260f05ca2bc0918a22f2a83eb778686de49b279d8a1cf8b3"
            ),
            "usecols": ["clock_mode", "entry_time", "exit_time", "side"],
            "parser": _comparator_parser(group_columns=["clock_mode"]),
            "groups": [
                {
                    "filter": {"clock_mode": "primary"},
                    "minimum_contained_rows": 90,
                },
            ],
            "clock_family": "same_h8_anchor",
        },
        {
            "id": "BDRC",
            "path": (
                "results/bank_deposit_secured_repo_"
                "concordance_clocks_2026-07-20.csv.gz"
            ),
            "sha256": (
                "1ff3a6075e3ceff928e1dd19d05880dbe9dbab0e07d79b853146d7b4c8f6cabc"
            ),
            "header_sha256": (
                "ef3cd7e042ff592bd4747ecd9bbf47b66cc7ab587db60a834efb502a11c7a605"
            ),
            "usecols": ["clock_name", "entry_time", "exit_time", "side"],
            "parser": _comparator_parser(group_columns=["clock_name"]),
            "groups": [
                {
                    "filter": {"clock_name": "primary"},
                    "minimum_contained_rows": 50,
                },
                {
                    "filter": {"clock_name": "h8_only"},
                    "minimum_contained_rows": 120,
                },
            ],
            "clock_family": "same_h8_anchor",
        },
    ]


def frozen_dependencies() -> dict[str, str]:
    dependencies = {
        BOUNDARY_DOCUMENT: BOUNDARY_DOCUMENT_SHA256,
        MECHANISM_DOCUMENT: MECHANISM_DOCUMENT_SHA256,
        COMMON_WINDOW_POLICY: COMMON_WINDOW_POLICY_SHA256,
    }
    for contract in source_contracts().values():
        dependencies[contract["path"]] = contract["sha256"]
        dependencies[contract["manifest"]] = contract["manifest_sha256"]
        dependencies[contract["audit"]] = contract["audit_sha256"]
    for contract in comparator_contracts():
        dependencies[contract["path"]] = contract["sha256"]
    return dependencies


def validate_frozen_dependencies() -> None:
    for path, expected in frozen_dependencies().items():
        target = _repository_path(path)
        if not target.is_file():
            raise RuntimeError(f"DCLB-864 frozen dependency missing: {path}")
        if _sha256_path(target) != expected:
            raise RuntimeError(f"DCLB-864 frozen dependency changed: {path}")
    for contract in source_contracts().values():
        path = contract["path"]
        if sha256_csv_header(path) != contract["header_sha256"]:
            raise RuntimeError(f"DCLB-864 source header changed: {path}")
        header = csv_header(path)
        if not set(contract["allowlist"]).issubset(header):
            raise RuntimeError(f"DCLB-864 source allowlist missing: {path}")
    for contract in comparator_contracts():
        path = contract["path"]
        if sha256_csv_header(path) != contract["header_sha256"]:
            raise RuntimeError(
                f"DCLB-864 comparator header changed: {contract['id']}"
            )
        header = csv_header(path)
        if not set(contract["usecols"]).issubset(header):
            raise RuntimeError(
                f"DCLB-864 comparator usecols missing: {contract['id']}"
            )


def _core_manifest() -> dict[str, Any]:
    policy = Policy()
    return {
        "protocol_version": (
            "dollar_collateral_liquidity_bank_relay_preregistration_v1"
        ),
        "policy": asdict(policy),
        "research_history_boundary": {
            "h41_source_and_family_outcomes_seen": True,
            "rrp_source_and_family_outcomes_seen": True,
            "h8_source_and_family_outcomes_seen": True,
            "cxrt_source_support_failure_seen": True,
            "exact_dclb_joint_state_or_incidence_seen": False,
            "exact_dclb_comparator_overlap_seen": False,
            "exact_dclb_market_outcomes_seen": False,
            "global_pristine_holdout_claimed": False,
        },
        "frozen_documents": {
            "boundary": {
                "path": BOUNDARY_DOCUMENT,
                "sha256": BOUNDARY_DOCUMENT_SHA256,
            },
            "mechanism": {
                "path": MECHANISM_DOCUMENT,
                "sha256": MECHANISM_DOCUMENT_SHA256,
            },
            "common_window_policy": {
                "path": COMMON_WINDOW_POLICY,
                "sha256": COMMON_WINDOW_POLICY_SHA256,
            },
        },
        "source_contracts": source_contracts(),
        "source_algebra": {
            "h41": {
                "feature": "log(current_net_liquidity/prior_net_liquidity)",
                "strict_prior_midrank_count": 104,
                "numerator": "2*count(prior<x)+count(prior==x)",
                "center": "h41_num-104",
                "first_rankable_delta": 105,
                "freshness": (
                    "previous_h8_decision < available_at_utc "
                    "<= current_h8_decision"
                ),
                "reused_release_forbidden": True,
            },
            "rrp": {
                "interval": (
                    "previous_h8_decision < result_available_at_utc "
                    "<= current_h8_decision"
                ),
                "complete_operation_count_range": [3, 7],
                "level": "log1p(mean(accepted_usd)/1_000_000_000)",
                "delta": "current_complete_level-prior_complete_level",
                "strict_prior_midrank_count": 13,
                "numerator": "2*count(prior<x)+count(prior==x)",
                "relief_center": "-(rrp_num-13)",
                "first_rankable_post_reset_delta": 14,
                "quarantine": (
                    "incomplete interval emits no level/delta and resets "
                    "rank history; no bridge"
                ),
            },
            "h8": {
                "anchor": "every archived release including irregular weekdays",
                "exclusions": [
                    "2020-10-02",
                    "2023-03-31",
                    "2023-06-30",
                    "2023-12-15",
                ],
                "components": [
                    "large_minus_small_other_deposit_log_change_bp",
                    "small_borrowings_log_change_bp",
                    "negative_small_cash_asset_log_change_bp",
                ],
                "robust_z_prior_observations": 104,
                "robust_scale": "1.4826*MAD",
                "validity": (
                    "nonzero mean stress with at least two component signs "
                    "matching composite sign"
                ),
                "sa_primary_nsa_control_only": True,
            },
            "macro": {
                "integer": "13*h41_center_num-104*rrp_center_num",
                "side_sign": "sign(macro_integer)",
                "side": "LONG iff +1; SHORT iff -1",
                "bank_relation": (
                    "BANK_SUPPORTS iff h8_relief_sign==side_sign; "
                    "otherwise BANK_OPPOSES"
                ),
                "neutral_macro_ineligible": True,
            },
        },
        "execution_contract": {
            "decision": "H8 release date 17:00:00 America/New_York",
            "entry": "H8 release date 17:05:00 America/New_York",
            "exit": "entry_utc + 4,320 elapsed minutes",
            "exposure": "[entry_utc, exit_utc)",
            "dst_wall_clock_normalization": False,
            "instrument": "Binance USD-M BTCUSDT perpetual",
            "global_nonoverlap_before_split": True,
            "entry_equal_previous_exit": "accepted",
            "split_containment_after_reservation": True,
            "excluded_release_originates_state": False,
            "excluded_release_may_host_prior_delayed_state": True,
        },
        "source_only_controls": {
            "ordered": list(CONTROL_ORDER),
            "all_required": True,
            "independently_reserved": [
                "primary",
                "h41_only",
                "rrp_interval_only",
                "h8_only",
                "macro_concordant_only",
                "macro_discordant_only",
                "bank_supports_only",
                "bank_opposes_only",
                "stale_h41_one_release",
                "stale_rrp_one_interval",
                "one_h8_release_execution_delay",
                "nsa_h8",
            ],
            "same_primary_timestamps": [
                "exact_direction_flip",
                "deterministic_random_side",
            ],
            "stale_h41": "immediately preceding emitted available H41 rank",
            "stale_rrp": (
                "immediately preceding emitted rank in same post-quarantine "
                "segment"
            ),
            "random_side": (
                "UTF8 SHA256('DCLB-864|YYYY-MM-DDTHH:MM:SSZ'); "
                "LONG iff first digest byte<128"
            ),
            "delayed": (
                "prior valid raw state enters next archived H8 release at "
                "17:05 ET, then global reservation repeats"
            ),
            "nsa_h8": (
                "primary macro side and clock with exact NSA H8 validity "
                "and relation replay"
            ),
        },
        "source_support_gate": {
            "train": {
                "window": [
                    "2020-01-01T00:00:00Z",
                    "2023-01-01T00:00:00Z",
                ],
                "events_min": 75,
                "each_year_events_min": 12,
                "active_months_min": 24,
                "each_side_share_min": 0.20,
                "maximum_month_share": 0.12,
                "maximum_quarter_share": 0.24,
                "maximum_entry_gap_days": 60.0,
                "maximum_same_side_run": 12,
            },
            "selection": {
                "window": [
                    "2023-01-01T00:00:00Z",
                    "2024-01-01T00:00:00Z",
                ],
                "events_min": 20,
                "each_half_events_min": 7,
                "each_quarter_events_min": 2,
                "active_months_min": 8,
                "each_side_share_min": 0.20,
                "maximum_month_share": 0.25,
                "maximum_entry_gap_days": 75.0,
                "maximum_same_side_run": 10,
            },
            "composition_each_split": {
                "bank_supports_share_min": 0.20,
                "bank_opposes_share_min": 0.20,
                "macro_concordant_share_min": 0.20,
                "macro_discordant_share_min": 0.20,
                "weak_share_min": 0.15,
                "strong_share_min": 0.15,
                "two_of_three_share_min": 0.10,
                "three_of_three_share_min": 0.10,
                "h41_only_same_side_reproduction_max": 0.85,
                "rrp_only_same_side_reproduction_max": 0.85,
                "each_stale_same_side_reproduction_max": 0.85,
                "random_same_side_reproduction_max": 0.60,
            },
            "every_required_control_nonempty_each_split": True,
            "undefined_or_empty_denominator": "fail",
            "failure_action": (
                "retire DCLB-864 unchanged before comparator rows or outcomes"
            ),
        },
        "novelty_contract": {
            "common_window_policy_path": COMMON_WINDOW_POLICY,
            "common_window_policy_sha256": COMMON_WINDOW_POLICY_SHA256,
            "prospective_policy_motivation": {
                "prior_cross_boundary_comparator_timing_row_seen": True,
                "disclosed_fact": (
                    "a valid prior RMSR comparator trade entered in late "
                    "2023 and exited in early 2024"
                ),
                "dclb_source_incidence_or_overlap_opened_when_disclosed": False,
                "dclb_market_outcomes_opened_when_disclosed": False,
                "effect": (
                    "bind universal full-containment accounting; do not "
                    "clip, repair, or reinterpret the prior row"
                ),
            },
            "window": [
                "2020-01-01T00:00:00Z",
                "2024-01-01T00:00:00Z",
            ],
            "full_interval_containment": True,
            "raw_validation_before_filter": True,
            "crossing_interval": "report and exclude whole",
            "complete_five_minute_grid": True,
            "groups_compared_separately": True,
            "comparators": comparator_contracts(),
            "same_h8_anchor_thresholds": {
                "exact_entry_jaccard_max": 0.60,
                "same_entry_same_side_reproduction_max": 0.75,
                "absolute_signed_occupancy_pearson_max": 0.65,
            },
            "asynchronous_thresholds": {
                "exact_entry_jaccard_max": 0.20,
                "six_hour_one_to_one_jaccard_max": 0.35,
                "absolute_signed_occupancy_pearson_max": 0.45,
            },
            "seven_calendar_day_one_to_one_jaccard": "report_only",
            "undefined_correlation": "fail",
            "failure_action": "retire DCLB-864 unchanged before outcomes",
        },
        "live_fail_flat_contract": {
            "expected_publication_calendars_predeclared": True,
            "missing_late_schema_integrity_availability_or_quarantine_mismatch": (
                "no source update and no event"
            ),
            "stale_carry_alternate_endpoint_fill_or_imputation": False,
            "retrieval_time_and_response_sha256_append_only_ledger": True,
            "source_migration": (
                "separate prospective commit and shadow replay; no change "
                "to already scheduled event"
            ),
        },
        "economic_rllm_sequence": {
            "source_support_and_novelty_before_market": True,
            "separate_committed_evaluator_required": True,
            "roles": {
                "source_warmup": "2017-2019",
                "rllm_fit": "2020-2021",
                "inner_test": "2022",
                "sealed_eval": "2023",
                "post_2023": "separately audited source extension required",
            },
            "fixed_economics": {
                "gross_exposure": 0.50,
                "base_cost_bp_per_notional_side": 6.0,
                "stress_cost_bp_per_notional_side": 10.0,
                "realized_funding": "[entry,exit)",
                "full_calendar_cagr_includes_idle_cash": True,
                "strict_mdd_includes_pre_entry_hwm_and_intratrade_extremes": True,
                "stops_take_profit_early_exit_or_overlap": False,
            },
            "sealed_eval_qualification": {
                "base_and_stress_absolute_return_positive": True,
                "each_half_absolute_return_positive": True,
                "cagr_to_strict_mdd_min": 3.0,
                "strict_mdd_max": 0.15,
                "executed_trades_min": 12,
                "each_side_trades_min": 4,
                "each_half_trades_min": 4,
                "mean_gross_underlying_bp_strictly_above": 20.0,
                "weekly_cluster_sign_flip_p_max": 0.10,
            },
        },
        "rllm_boundary": {
            "action_space": ["TRADE_FIXED_SIDE", "ABSTAIN"],
            "allowed_tokens": [
                "fixed_side",
                "h41_direction_and_transition",
                "rrp_interval_direction_and_transition",
                "macro_concordance_dominance_and_strength",
                "h8_relief_stress_and_component_agreement",
                "bank_support_relation",
                "source_age_count_and_validity_buckets",
                "current_position_state",
            ],
            "forbidden": [
                "raw_levels_deltas_zscores_ranks_or_rank_numerators",
                "date_timestamp_weekday_release_or_row_identity",
                "source_url_or_hash",
                "BTC_price_return_funding_future_path_label_reward_or_PnL",
                "CAGR_MDD_or_split_identity",
                "candidate_creation_side_reversal_hold_leverage_or_time_choice",
            ],
            "model_checkpoint_reward_and_seed_freeze_before_fit": True,
            "no_2022_checkpoint_choice_from_2023": True,
            "prompt_reveals_outcome_summary": False,
        },
        "strict_sequence": {
            "stop_at_first_failure": True,
            "no_parameter_repair": True,
            "stages": [
                "mechanism_commit",
                "write_once_preregistration_commit",
                "source_support_evaluator_commit",
                "source_support",
                "comparator_novelty",
                "economic_rllm_evaluator_commit",
                "fit_2020_2021",
                "inner_test_2022",
                "sealed_eval_2023",
                "post_2023_source_extension",
            ],
        },
        "evidence_boundary": {
            "source_value_rows_read": 0,
            "source_feature_rows_derived": 0,
            "joint_source_incidence_rows_derived": 0,
            "comparator_event_rows_read": 0,
            "btc_market_rows_loaded": 0,
            "funding_rows_loaded": 0,
            "return_rows_loaded": 0,
            "return_or_pnl_fields_read": 0,
            "post_2023_rows_read": 0,
            "network_calls": 0,
            "subprocess_calls": 0,
        },
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "source_rows_decoded": False,
        "comparator_rows_decoded": False,
    }


def build_manifest() -> dict[str, Any]:
    core = _core_manifest()
    return {**core, "manifest_hash": canonical_hash(core)}


def validate_manifest(payload: dict[str, Any]) -> None:
    expected = build_manifest()
    if payload != expected:
        raise RuntimeError("DCLB-864 manifest core differs from code")
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("DCLB-864 manifest hash mismatch")
    for field in (
        "outcomes_opened",
        "source_incidence_opened",
        "source_rows_decoded",
        "comparator_rows_decoded",
    ):
        if payload.get(field) is not False:
            raise RuntimeError("DCLB-864 evidence boundary opened")
    if any(payload["evidence_boundary"].values()):
        raise RuntimeError("DCLB-864 preregistration decoded forbidden evidence")


def _canonical_manifest_text() -> str:
    return (
        json.dumps(
            build_manifest(),
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )


def write_once(path: str | Path, payload: dict[str, Any]) -> str:
    validate_frozen_dependencies()
    validate_manifest(payload)
    expected = _canonical_manifest_text().encode("utf-8")
    output = _output_relative_path(path)
    directory_fd = _open_output_parent(output)
    temporary_name = (
        f".{output.name}.{os.getpid()}.{secrets.token_hex(12)}.tmp"
    )
    temporary_created = False
    try:
        try:
            actual = _read_regular_at(directory_fd, output.name)
        except FileNotFoundError:
            actual = None
        if actual is not None:
            if hashlib.sha256(actual).digest() != hashlib.sha256(expected).digest():
                raise RuntimeError("DCLB-864 existing manifest hash mismatch")
            if actual != expected:
                raise RuntimeError("DCLB-864 noncanonical existing manifest")
            return "verified_existing"

        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(
            temporary_name,
            flags,
            0o600,
            dir_fd=directory_fd,
        )
        temporary_created = True
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(expected)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(
                temporary_name,
                output.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            winner = _read_regular_at(directory_fd, output.name)
            if winner != expected:
                raise RuntimeError("DCLB-864 manifest race drift")
            return "verified_existing"
        os.fsync(directory_fd)
        return "created"
    finally:
        if temporary_created:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
            os.fsync(directory_fd)
        os.close(directory_fd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_manifest()
    status = write_once(args.output, payload)
    print(
        json.dumps(
            {
                "status": status,
                "output": args.output,
                "manifest_hash": payload["manifest_hash"],
                "outcomes_opened": False,
                "source_incidence_opened": False,
                "source_rows_decoded": False,
                "comparator_rows_decoded": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
