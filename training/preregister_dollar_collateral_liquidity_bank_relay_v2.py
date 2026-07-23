"""Freeze DCLB-864 preregistration v2 after a pre-incidence token amendment.

The immutable v1 artifact remains the base. This module adds only the
control-only exact macro-balance relation and binds the amendment document.
It still reads hashes and CSV headers only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
from typing import Any

from training import preregister_dollar_collateral_liquidity_bank_relay as v1


REPOSITORY_ROOT = v1.REPOSITORY_ROOT
DEFAULT_OUTPUT = (
    "results/dollar_collateral_liquidity_bank_relay_"
    "preregistration_v2_2026-07-24.json"
)
BASE_PREREGISTRATION = v1.DEFAULT_OUTPUT
BASE_PREREGISTRATION_SHA256 = (
    "0947513376f36991a1cf4e5dc0a2aae7417f246dd2698f82d19f6dfe09bc67ec"
)
BASE_MANIFEST_HASH = (
    "da77218e89ee588f901d719ea9944fad1cfd5f1b88712f679dc1e2de5a3a9e4d"
)
BASE_BUILDER = "training/preregister_dollar_collateral_liquidity_bank_relay.py"
BASE_BUILDER_SHA256 = (
    "852b4225adfcb2ca1c00f4c923e39674aec36245f1d3eda4c8bd6c07683dab99"
)
AMENDMENT_DOCUMENT = (
    "docs/dclb-control-macro-balance-preincidence-amendment-2026-07-24.md"
)
AMENDMENT_DOCUMENT_SHA256 = (
    "dd1b20bdd45759fa5e99a32eb7d1509f350584e7fa425475e3bf2e4001a787aa"
)

Policy = v1.Policy
CONTROL_ORDER = v1.CONTROL_ORDER
H41_SOURCE = v1.H41_SOURCE
H41_HEADER_SHA256 = v1.H41_HEADER_SHA256
H41_ALLOWLIST = v1.H41_ALLOWLIST
RRP_SOURCE = v1.RRP_SOURCE
RRP_HEADER_SHA256 = v1.RRP_HEADER_SHA256
RRP_ALLOWLIST = v1.RRP_ALLOWLIST
H8_SOURCE = v1.H8_SOURCE
H8_HEADER_SHA256 = v1.H8_HEADER_SHA256
H8_ALLOWLIST = v1.H8_ALLOWLIST
COMMON_WINDOW_POLICY = v1.COMMON_WINDOW_POLICY
COMMON_WINDOW_POLICY_SHA256 = v1.COMMON_WINDOW_POLICY_SHA256

sha256_file = v1.sha256_file
canonical_hash = v1.canonical_hash
csv_header_bytes = v1.csv_header_bytes
csv_header = v1.csv_header
sha256_csv_header = v1.sha256_csv_header
source_contracts = v1.source_contracts
comparator_contracts = v1.comparator_contracts
_comparator_parser = v1._comparator_parser


def frozen_dependencies() -> dict[str, str]:
    return {
        **v1.frozen_dependencies(),
        BASE_PREREGISTRATION: BASE_PREREGISTRATION_SHA256,
        BASE_BUILDER: BASE_BUILDER_SHA256,
        AMENDMENT_DOCUMENT: AMENDMENT_DOCUMENT_SHA256,
    }


def validate_frozen_dependencies() -> None:
    v1.validate_frozen_dependencies()
    for path, expected in (
        (BASE_PREREGISTRATION, BASE_PREREGISTRATION_SHA256),
        (BASE_BUILDER, BASE_BUILDER_SHA256),
        (AMENDMENT_DOCUMENT, AMENDMENT_DOCUMENT_SHA256),
    ):
        if sha256_file(path) != expected:
            raise RuntimeError(f"DCLB-864 v2 frozen dependency changed: {path}")
    payload = json.loads(
        Path(BASE_PREREGISTRATION).read_text(encoding="utf-8")
    )
    v1.validate_manifest(payload)
    if payload["manifest_hash"] != BASE_MANIFEST_HASH:
        raise RuntimeError("DCLB-864 v1 base manifest hash drift")


def _core_manifest() -> dict[str, Any]:
    base = v1.build_manifest()
    core = json.loads(
        json.dumps(
            {key: value for key, value in base.items() if key != "manifest_hash"},
            sort_keys=True,
        )
    )
    core["protocol_version"] = (
        "dollar_collateral_liquidity_bank_relay_preregistration_v2"
    )
    core["supersedes_without_mutating"] = {
        "base_preregistration": {
            "path": BASE_PREREGISTRATION,
            "sha256": BASE_PREREGISTRATION_SHA256,
            "manifest_hash": BASE_MANIFEST_HASH,
        },
        "base_builder": {
            "path": BASE_BUILDER,
            "sha256": BASE_BUILDER_SHA256,
        },
        "reason": (
            "static pre-incidence discovery of an unnamed exact macro "
            "balance on component-only control rows"
        ),
        "source_or_comparator_rows_opened": 0,
        "market_or_outcome_rows_opened": 0,
    }
    core["frozen_documents"]["control_macro_balance_amendment"] = {
        "path": AMENDMENT_DOCUMENT,
        "sha256": AMENDMENT_DOCUMENT_SHA256,
    }
    core["research_history_boundary"][
        "control_macro_balance_gap_found_by_static_review"
    ] = True
    core["research_history_boundary"][
        "source_or_comparator_rows_opened_before_v2"
    ] = False
    core["source_algebra"]["macro"]["control_only_balanced_relation"] = {
        "token": "MACRO_BALANCED_OPPOSITION",
        "condition": (
            "h41_sign and rrp_relief_sign nonzero/opposite and "
            "macro_integer==0"
        ),
        "allowed_controls": [
            "h41_only",
            "rrp_interval_only",
            "h8_only",
        ],
        "primary_eligible": False,
        "primary_rllm_token_allowed": False,
    }
    core["strict_sequence"]["stages"] = [
        "mechanism_commit",
        "write_once_preregistration_v1_commit",
        "preincidence_control_token_amendment_commit",
        "write_once_preregistration_v2_commit",
        "source_support_evaluator_commit",
        "source_support",
        "comparator_novelty",
        "economic_rllm_evaluator_commit",
        "fit_2020_2021",
        "inner_test_2022",
        "sealed_eval_2023",
        "post_2023_source_extension",
    ]
    return core


def build_manifest() -> dict[str, Any]:
    core = _core_manifest()
    return {**core, "manifest_hash": canonical_hash(core)}


def validate_manifest(payload: dict[str, Any]) -> None:
    if payload != build_manifest():
        raise RuntimeError("DCLB-864 v2 manifest core differs from code")
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("DCLB-864 v2 manifest hash mismatch")
    for field in (
        "outcomes_opened",
        "source_incidence_opened",
        "source_rows_decoded",
        "comparator_rows_decoded",
    ):
        if payload.get(field) is not False:
            raise RuntimeError("DCLB-864 v2 evidence boundary opened")
    if any(payload["evidence_boundary"].values()):
        raise RuntimeError("DCLB-864 v2 decoded forbidden evidence")


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
    output = v1._output_relative_path(path)
    directory_fd = v1._open_output_parent(output)
    temporary_name = (
        f".{output.name}.{os.getpid()}.{secrets.token_hex(12)}.tmp"
    )
    temporary_created = False
    try:
        try:
            actual = v1._read_regular_at(directory_fd, output.name)
        except FileNotFoundError:
            actual = None
        if actual is not None:
            if hashlib.sha256(actual).digest() != hashlib.sha256(expected).digest():
                raise RuntimeError("DCLB-864 v2 existing manifest hash mismatch")
            if actual != expected:
                raise RuntimeError("DCLB-864 v2 noncanonical existing manifest")
            return "verified_existing"
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
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
            if v1._read_regular_at(directory_fd, output.name) != expected:
                raise RuntimeError("DCLB-864 v2 manifest race drift")
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
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
