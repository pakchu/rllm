"""Freeze LAMB-21 before decoding joint source incidence or market outcomes.

This command validates immutable bytes and physical CSV headers only. It does
not parse a source value row, execution bar, funding row, future return,
reward, model example, trade, or portfolio outcome.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
POLICY_ID = "LAMB-21"
PROTOCOL_VERSION = "liquidity_aware_microstructure_braid_preregistration_v1"
DEFAULT_OUTPUT = (
    "results/liquidity_aware_microstructure_braid_"
    "preregistration_2026-07-25.json"
)
PRODUCER_SCRIPT = "training/preregister_liquidity_aware_microstructure_braid.py"
SEALED_PRODUCER_COMMIT = "32f97c8d74e2598c9858da32b7eb203b690da0b4"
SEALED_PRODUCER_SCRIPT_SHA256 = (
    "1fb3b7f39fe418e9c160a3035cbb63a8f65cb72119a49d36c93fc5528c37e10c"
)

AUDIT_DOCUMENT = "docs/post-tracer-alpha-mechanism-audit-2026-07-25.md"
AUDIT_DOCUMENT_SHA256 = (
    "7394cd096d92b5469eb625605faaa8f53c49fc486b921269a1b2da0b08afbf9e"
)
BOUNDARY_DOCUMENT = (
    "docs/liquidity-aware-microstructure-braid-boundary-2026-07-25.md"
)
BOUNDARY_DOCUMENT_SHA256 = (
    "a412b3e7dcfad625e0cdccd3a1941055bc5e104f4693b21ddf24b4fa1aa7c654"
)
BOUNDARY_COMMIT = "b0a6838236091cfdb6caf4c001c9901458e43f03"

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
H41_PHYSICAL_HEADER = tuple(
    "release_date,observation_date,available_at_utc,"
    "total_assets_usd_millions,treasury_general_account_usd_millions,"
    "reverse_repurchase_agreements_usd_millions,"
    "net_liquidity_usd_millions,source_format,source_url,source_sha256".split(
        ","
    )
)
H41_ALLOWLIST = (
    "release_date",
    "observation_date",
    "available_at_utc",
    "net_liquidity_usd_millions",
)
H41_MANIFEST = (
    "data/federal_reserve_h41_net_liquidity_2018_2023/build_manifest.json"
)
H41_MANIFEST_SHA256 = (
    "1ec212a85de0e49c5a0c2d35b8b22be86eb7d62989f7a0098be1bb1274b2a99b"
)
H41_BUILDER = "training/build_federal_reserve_h41_net_liquidity.py"
H41_BUILDER_SHA256 = (
    "822ab0602549d71f50834da4abf13c5a81dd6af7e58147b9b37aa9940355dc7d"
)
H41_AUDIT = "docs/federal-reserve-h41-net-liquidity-source-audit-2026-07-17.md"
H41_AUDIT_SHA256 = (
    "fec3e11fa5112aff97b338c6404b3202cfa38c02391debaeee3c18ff3b3b23c1"
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
RRP_PHYSICAL_HEADER = tuple(
    "operation_id,operation_date,settlement_date,maturity_date,close_time_et,"
    "result_available_at_utc,last_updated_et,total_amount_submitted_usd,"
    "total_amount_accepted_usd,participating_counterparties,"
    "accepted_counterparties,source_complete,quarantine_reason".split(",")
)
RRP_ALLOWLIST = (
    "operation_date",
    "result_available_at_utc",
    "total_amount_accepted_usd",
    "participating_counterparties",
    "accepted_counterparties",
    "source_complete",
    "quarantine_reason",
)
RRP_MANIFEST = "data/new_york_fed_overnight_rrp_2018_2023/build_manifest.json"
RRP_MANIFEST_SHA256 = (
    "4f87e2219da71c94832c8708086ba01387efc145e3488b62cd3b3d07c62d8fee"
)
RRP_BUILDER = "training/build_new_york_fed_overnight_rrp.py"
RRP_BUILDER_SHA256 = (
    "0567157dde18b1c6ccfb37b669ceead521360f23dd0b73033fccc08e37c0d42c"
)
RRP_AUDIT = "docs/new-york-fed-overnight-rrp-source-audit-2026-07-17.md"
RRP_AUDIT_SHA256 = (
    "329db1cf886bfbceb0a048b1c44c59378af717ddd9731e5e26fd09e14ada8d23"
)

LATTICE_SOURCE = (
    "data/binance_um_quantity_lattice_btc_2020_2023/"
    "BTCUSDT_quantity_lattice_5m_2020-01-01_2023-12-31.csv.gz"
)
LATTICE_SOURCE_SHA256 = (
    "3ca945f134115fc7b58086405fd881db3e3b70087bd9da54ffc293f6b658072e"
)
LATTICE_HEADER_SHA256 = (
    "1021675e0998dfaf49a13d46af7365b0762719917817d8719c2d8d99116f47ed"
)
LATTICE_PHYSICAL_HEADER = tuple(
    "date,source_observed,source_complete,source_gap_day,"
    "verified_zero_volume_empty,post_gap_quarantine,agg_trade_count,"
    "total_quantity_mbtc,total_signed_quantity_mbtc,coarse_event_count,"
    "coarse_quantity_mbtc,coarse_signed_quantity_mbtc,medium_event_count,"
    "medium_quantity_mbtc,medium_signed_quantity_mbtc,fine_event_count,"
    "fine_quantity_mbtc,fine_signed_quantity_mbtc,coarse_quantity_share,"
    "coarse_coherence,fine_signed_share,coarse_side,cohort_opposition,"
    "qlcd_score".split(",")
)
LATTICE_ALLOWLIST = (
    "date",
    "source_observed",
    "source_complete",
    "source_gap_day",
    "verified_zero_volume_empty",
    "post_gap_quarantine",
    "agg_trade_count",
    "total_quantity_mbtc",
    "coarse_quantity_mbtc",
    "coarse_signed_quantity_mbtc",
    "fine_quantity_mbtc",
    "fine_signed_quantity_mbtc",
)
LATTICE_MANIFEST = (
    "data/binance_um_quantity_lattice_btc_2020_2023/build_manifest.json"
)
LATTICE_MANIFEST_SHA256 = (
    "bcdf89924f54a5b97d4219749c2094d2a4c08d8473a37bc5367d9b8e5791284f"
)
LATTICE_TRANSFORM = "preprocessing/quantity_lattice_cohort.py"
LATTICE_TRANSFORM_SHA256 = (
    "8e7503dfb518bdd6515d255dc1ae4a1ac8b47cc078f39bee24cd2d52561a8e1b"
)
LATTICE_HISTORY = (
    "docs/quantity-lattice-cohort-disagreement-train-result-2026-07-20.md"
)
LATTICE_HISTORY_SHA256 = (
    "2256387b0e1b71c125c71fea0a4785d1ba4ba0d67249db15a6c01cea959cc162"
)

CASCADE_SOURCE = (
    "data/binance_um_same_millisecond_cascade_btc_2020_2023/"
    "BTCUSDT_same_millisecond_5m_2020-01-01_2023-12-31.csv.gz"
)
CASCADE_SOURCE_SHA256 = (
    "8fa03b0d7f58db9d0ba6c889e99ce87ba668f55a3c7f0ab5638a374c4584bfd1"
)
CASCADE_HEADER_SHA256 = (
    "1af1937bd53b900960f12c73af0701a86990fcb688d6727c9715c9330b1f6090"
)
CASCADE_PHYSICAL_HEADER = tuple(
    "date,source_observed,source_complete,source_gap_day,"
    "verified_zero_volume_empty,post_gap_quarantine,first_transact_time_ms,"
    "last_transact_time_ms,agg_trade_count,underlying_trade_count,"
    "millisecond_group_count,collision_group_count,first_price,last_price,"
    "quote_notional,signed_quote_notional,collision_quote_notional,"
    "collision_notional_share,max_ms_transact_time,max_ms_event_count,"
    "max_ms_underlying_trade_count,max_ms_quote_notional,"
    "max_ms_signed_quote_notional,max_ms_notional_share,max_ms_coherence,"
    "max_ms_side,max_ms_pre_group_price,max_ms_first_price,max_ms_last_price,"
    "max_ms_sweep_bp,max_ms_score".split(",")
)
CASCADE_ALLOWLIST = (
    "date",
    "source_observed",
    "source_complete",
    "source_gap_day",
    "verified_zero_volume_empty",
    "post_gap_quarantine",
    "first_transact_time_ms",
    "last_transact_time_ms",
    "agg_trade_count",
    "first_price",
    "last_price",
    "quote_notional",
    "collision_quote_notional",
    "max_ms_quote_notional",
    "max_ms_signed_quote_notional",
)
CASCADE_MANIFEST = (
    "data/binance_um_same_millisecond_cascade_btc_2020_2023/"
    "build_manifest.json"
)
CASCADE_MANIFEST_SHA256 = (
    "e6ba3fbf74bc9bc1a7c1b35873e9ff430e5bc0a7b7edcc7e082f3f397362c805"
)
CASCADE_TRANSFORM = "preprocessing/same_millisecond_cascade.py"
CASCADE_TRANSFORM_SHA256 = (
    "cfc1c1c587236e1458465955c133b240a6c4f4748c2e7260519e9cdbea3a16de"
)
CASCADE_HISTORY = "docs/same-millisecond-cascade-support-rejection-2026-07-20.md"
CASCADE_HISTORY_SHA256 = (
    "6168d578b19afa4aa215882ed73e1126631c8e9b617b713112b021fc173381c0"
)

TRACER_RETIREMENT = "docs/tracer4-source-support-retirement-2026-07-25.md"
TRACER_RETIREMENT_SHA256 = (
    "07f0501f826e2a722e770c616ce5c2698851e7bdcc0cc17d7958c8628bb20135"
)
DCLB_RESULT = "docs/dclb-source-support-result-2026-07-24.md"
DCLB_RESULT_SHA256 = (
    "7ce83d3d0aaec48133ddcac85c6f3ba738225629ffa129f4702d0a2d2176b5eb"
)

TOKEN_SCHEMA: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("h41_impulse", ("H41_EXPANDS", "H41_CONTRACTS", "H41_FLAT")),
    ("rrp_impulse", ("RRP_RELEASES", "RRP_DRAINS", "RRP_FLAT")),
    (
        "macro_sponsorship",
        (
            "LIQUIDITY_SUPPORTS",
            "LIQUIDITY_RESTRICTS",
            "MACRO_SPLIT",
            "MACRO_NEUTRAL",
        ),
    ),
    ("macro_age", ("BOTH_FRESH", "H41_AGING", "RRP_AGING", "BOTH_AGING")),
    (
        "lattice_relation",
        (
            "COHORTS_BUY",
            "COHORTS_SELL",
            "COARSE_BUY_FINE_SELL",
            "COARSE_SELL_FINE_BUY",
            "LATTICE_NEUTRAL",
        ),
    ),
    (
        "lattice_concentration",
        ("COARSE_DOMINANT", "FINE_DOMINANT", "LATTICE_MIXED"),
    ),
    (
        "cascade_impact",
        (
            "CASCADE_BUY_FOLLOWTHROUGH",
            "CASCADE_BUY_ABSORBED",
            "CASCADE_SELL_FOLLOWTHROUGH",
            "CASCADE_SELL_ABSORBED",
            "CASCADE_NEUTRAL",
        ),
    ),
    (
        "cascade_intensity",
        ("CASCADE_BROAD", "CASCADE_LOCAL", "CASCADE_MIXED"),
    ),
    (
        "micro_braid",
        (
            "MICRO_CONFIRMS_BUY",
            "MICRO_CONFIRMS_SELL",
            "LATTICE_BUY_CASCADE_SELL",
            "LATTICE_SELL_CASCADE_BUY",
            "MICRO_NEUTRAL",
        ),
    ),
    (
        "macro_transition",
        (
            "SUPPORT_PERSISTS",
            "RESTRICTION_PERSISTS",
            "ROTATES_TO_SUPPORT",
            "ROTATES_TO_RESTRICTION",
            "MACRO_TRANSITION_MIXED",
        ),
    ),
    (
        "micro_transition",
        (
            "BUY_PRESSURE_PERSISTS",
            "SELL_PRESSURE_PERSISTS",
            "PRESSURE_FLIPS",
            "PRESSURE_DISSIPATES",
            "MICRO_TRANSITION_MIXED",
        ),
    ),
)
TOKEN_COLUMNS = tuple(name for name, _ in TOKEN_SCHEMA)
SAFETY_TOKENS = (
    "SOURCE_INVALID",
    "RRP_FLAT",
    "MACRO_NEUTRAL",
    "BOTH_AGING",
    "LATTICE_NEUTRAL",
    "LATTICE_MIXED",
    "CASCADE_NEUTRAL",
    "CASCADE_MIXED",
    "MICRO_NEUTRAL",
    "MACRO_TRANSITION_MIXED",
    "MICRO_TRANSITION_MIXED",
)
CONTROL_IDS = (
    "h41_stale_one_release",
    "rrp_stale_one_operation",
    "lattice_cohort_swap",
    "cascade_delay_37",
    "macro_relation_mask",
)
ACTION_SPACE = ("TARGET_SHORT", "TARGET_FLAT", "TARGET_LONG")
FORBIDDEN_SUPPORT_FIELDS = (
    "execution_open",
    "execution_high",
    "execution_low",
    "execution_close",
    "funding_rate",
    "future_return",
    "label",
    "action",
    "target",
    "reward",
    "model_prediction",
    "trade",
    "pnl",
    "cagr",
    "mdd",
    "strict_mdd",
    "portfolio_weight",
)


@dataclass(frozen=True)
class Policy:
    policy_id: str = POLICY_ID
    boundary_hours_utc: tuple[int, ...] = (0, 8, 16)
    source_window_minutes: int = 480
    micro_rows_per_boundary: int = 96
    decision_delay_minutes: int = 5
    execution_delay_minutes: int = 10
    sequence_lines: int = 21
    h41_max_age_days: int = 10
    rrp_max_age_days: int = 5
    h41_fresh_days: int = 4
    rrp_fresh_days: int = 2
    rank_history_max: int = 270
    rank_history_min: int = 180
    rank_quantiles: tuple[float, float] = (0.33, 0.67)
    source_join_min: float = 0.99
    core_valid_min: float = 0.95
    forced_flat_max: float = 0.08
    category_support_min: float = 0.03
    category_share_max: float = 0.94
    distinct_signatures_min: int = 120
    signature_share_max: float = 0.10
    jsd_max: float = 0.30
    cascade_control_delay_rows: int = 37


def _repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    if (
        str(path).startswith("~")
        or candidate.is_absolute()
        or ".." in candidate.parts
        or candidate.name in {"", ".", ".."}
    ):
        raise RuntimeError("LAMB dependency path must be repository-relative")
    root = REPOSITORY_ROOT.resolve(strict=True)
    current = REPOSITORY_ROOT
    for part in candidate.parts:
        current /= part
        if current.is_symlink():
            raise RuntimeError("LAMB dependency path contains a symlink")
    target = REPOSITORY_ROOT / candidate
    try:
        target.resolve(strict=True).relative_to(root)
    except (FileNotFoundError, ValueError) as error:
        raise RuntimeError(
            "LAMB dependency is missing or escapes repository"
        ) from error
    if not target.is_file():
        raise RuntimeError("LAMB dependency is not a regular file")
    return target


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _repository_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_header_bytes(path: str | Path) -> bytes:
    target = _repository_path(path)
    if target.suffix == ".gz":
        with gzip.open(target, "rb") as handle:
            header = handle.readline()
    else:
        with target.open("rb") as handle:
            header = handle.readline()
    if not header.endswith(b"\n") or b"\r" in header or b"\x00" in header:
        raise RuntimeError("LAMB physical CSV header is malformed")
    return header


def csv_header(path: str | Path) -> tuple[str, ...]:
    try:
        decoded = csv_header_bytes(path).decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError("LAMB physical CSV header is not UTF-8") from error
    return tuple(decoded.removesuffix("\n").split(","))


def sha256_csv_header(path: str | Path) -> str:
    return hashlib.sha256(csv_header_bytes(path)).hexdigest()


def gzip_mtime(path: str | Path) -> int:
    raw = _repository_path(path).read_bytes()[:10]
    if len(raw) != 10 or raw[:2] != b"\x1f\x8b":
        raise RuntimeError("LAMB source is not gzip")
    return int.from_bytes(raw[4:8], "little")


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def jsonable(payload: Any) -> Any:
    return json.loads(
        json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
    )


def _git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def producer_binding() -> dict[str, Any]:
    return {
        "script": PRODUCER_SCRIPT,
        "script_sha256": SEALED_PRODUCER_SCRIPT_SHA256,
        "script_tracked": True,
        "script_clean_at_generation": True,
        "script_commit": SEALED_PRODUCER_COMMIT,
        "head_at_generation": SEALED_PRODUCER_COMMIT,
        "uncommitted_producer": False,
    }


def frozen_dependencies() -> dict[str, str]:
    return {
        AUDIT_DOCUMENT: AUDIT_DOCUMENT_SHA256,
        BOUNDARY_DOCUMENT: BOUNDARY_DOCUMENT_SHA256,
        H41_SOURCE: H41_SOURCE_SHA256,
        H41_MANIFEST: H41_MANIFEST_SHA256,
        H41_BUILDER: H41_BUILDER_SHA256,
        H41_AUDIT: H41_AUDIT_SHA256,
        RRP_SOURCE: RRP_SOURCE_SHA256,
        RRP_MANIFEST: RRP_MANIFEST_SHA256,
        RRP_BUILDER: RRP_BUILDER_SHA256,
        RRP_AUDIT: RRP_AUDIT_SHA256,
        LATTICE_SOURCE: LATTICE_SOURCE_SHA256,
        LATTICE_MANIFEST: LATTICE_MANIFEST_SHA256,
        LATTICE_TRANSFORM: LATTICE_TRANSFORM_SHA256,
        LATTICE_HISTORY: LATTICE_HISTORY_SHA256,
        CASCADE_SOURCE: CASCADE_SOURCE_SHA256,
        CASCADE_MANIFEST: CASCADE_MANIFEST_SHA256,
        CASCADE_TRANSFORM: CASCADE_TRANSFORM_SHA256,
        CASCADE_HISTORY: CASCADE_HISTORY_SHA256,
        TRACER_RETIREMENT: TRACER_RETIREMENT_SHA256,
        DCLB_RESULT: DCLB_RESULT_SHA256,
    }


def source_contracts() -> dict[str, dict[str, Any]]:
    contracts = {
        "h41": {
            "path": H41_SOURCE,
            "sha256": H41_SOURCE_SHA256,
            "physical_header": list(H41_PHYSICAL_HEADER),
            "physical_header_sha256": H41_HEADER_SHA256,
            "allowlist": list(H41_ALLOWLIST),
            "manifest": H41_MANIFEST,
            "manifest_sha256": H41_MANIFEST_SHA256,
            "availability": "available_at_utc <= B; age <= 10 elapsed days",
            "predecessor": "immediately prior physical release only",
        },
        "rrp": {
            "path": RRP_SOURCE,
            "sha256": RRP_SOURCE_SHA256,
            "physical_header": list(RRP_PHYSICAL_HEADER),
            "physical_header_sha256": RRP_HEADER_SHA256,
            "allowlist": list(RRP_ALLOWLIST),
            "manifest": RRP_MANIFEST,
            "manifest_sha256": RRP_MANIFEST_SHA256,
            "availability": (
                "result_available_at_utc <= B; age <= 5 elapsed days; "
                "source_complete=true; quarantine_reason blank"
            ),
            "predecessor": (
                "immediately prior physical operation in same complete segment"
            ),
        },
        "lattice": {
            "path": LATTICE_SOURCE,
            "sha256": LATTICE_SOURCE_SHA256,
            "physical_header": list(LATTICE_PHYSICAL_HEADER),
            "physical_header_sha256": LATTICE_HEADER_SHA256,
            "allowlist": list(LATTICE_ALLOWLIST),
            "manifest": LATTICE_MANIFEST,
            "manifest_sha256": LATTICE_MANIFEST_SHA256,
            "excluded_columns": ["qlcd_score"],
            "cohort_definition": {
                "coarse": "quantity_mbtc % 100 == 0",
                "medium": "quantity_mbtc % 10 == 0 and not coarse",
                "fine": "all remaining exact 1 mBTC increments",
            },
        },
        "cascade": {
            "path": CASCADE_SOURCE,
            "sha256": CASCADE_SOURCE_SHA256,
            "physical_header": list(CASCADE_PHYSICAL_HEADER),
            "physical_header_sha256": CASCADE_HEADER_SHA256,
            "allowlist": list(CASCADE_ALLOWLIST),
            "manifest": CASCADE_MANIFEST,
            "manifest_sha256": CASCADE_MANIFEST_SHA256,
            "excluded_columns": ["max_ms_score"],
        },
    }
    for contract in contracts.values():
        contract["loader"] = "pandas.read_csv(usecols=exact_allowlist)"
        contract["load_all_then_drop_forbidden"] = False
        contract["projection_order_is_frozen"] = True
    return contracts


def validate_frozen_dependencies() -> None:
    for path, expected in frozen_dependencies().items():
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(f"LAMB frozen dependency hash drift: {path}")
    header_specs = (
        (H41_SOURCE, H41_PHYSICAL_HEADER, H41_HEADER_SHA256, H41_ALLOWLIST),
        (RRP_SOURCE, RRP_PHYSICAL_HEADER, RRP_HEADER_SHA256, RRP_ALLOWLIST),
        (
            LATTICE_SOURCE,
            LATTICE_PHYSICAL_HEADER,
            LATTICE_HEADER_SHA256,
            LATTICE_ALLOWLIST,
        ),
        (
            CASCADE_SOURCE,
            CASCADE_PHYSICAL_HEADER,
            CASCADE_HEADER_SHA256,
            CASCADE_ALLOWLIST,
        ),
    )
    for path, physical, digest, allowlist in header_specs:
        observed = csv_header(path)
        if observed != physical:
            raise RuntimeError(f"LAMB physical header drift: {path}")
        if sha256_csv_header(path) != digest:
            raise RuntimeError(f"LAMB physical header hash drift: {path}")
        if len(set(allowlist)) != len(allowlist) or not set(allowlist).issubset(
            observed
        ):
            raise RuntimeError(f"LAMB source allowlist drift: {path}")
        if gzip_mtime(path) != 0:
            raise RuntimeError(f"LAMB source gzip mtime drift: {path}")


def assert_boundary_committed() -> None:
    for document in (AUDIT_DOCUMENT, BOUNDARY_DOCUMENT):
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", document],
            cwd=REPOSITORY_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if tracked.returncode:
            raise RuntimeError(f"LAMB boundary dependency is untracked: {document}")
        clean = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", document],
            cwd=REPOSITORY_ROOT,
            check=False,
        )
        if clean.returncode:
            raise RuntimeError(
                f"LAMB boundary dependency differs from HEAD: {document}"
            )
        latest = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", document],
            cwd=REPOSITORY_ROOT,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        if latest != BOUNDARY_COMMIT:
            raise RuntimeError(f"LAMB boundary commit drift: {document}")


def build_manifest() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "producer": producer_binding(),
        "policy": jsonable(asdict(Policy())),
        "boundary": {
            "audit_document": AUDIT_DOCUMENT,
            "audit_sha256": AUDIT_DOCUMENT_SHA256,
            "boundary_document": BOUNDARY_DOCUMENT,
            "boundary_sha256": BOUNDARY_DOCUMENT_SHA256,
            "commit": BOUNDARY_COMMIT,
        },
        "research_history": {
            "global_pristine_holdout_claimed": False,
            "component_source_incidence_seen": True,
            "component_market_outcomes_seen": True,
            "exact_lamb_joint_incidence_seen": False,
            "exact_lamb_market_outcomes_seen": False,
            "qlcd_primitive_reuse_disclosed": True,
            "qlcd_policy_reused": False,
            "smcc_policy_reused": False,
            "dclb_policy_reused": False,
            "tracer_policy_reused": False,
        },
        "source_contracts": source_contracts(),
        "clock": {
            "boundaries_utc": ["00:00:00", "08:00:00", "16:00:00"],
            "micro_window": "[B-8h,B)",
            "micro_rows_each_source": 96,
            "micro_completion": "date+5m <= B",
            "macro_asof": "<= B",
            "state_complete": "B",
            "decision": "B+5m",
            "execution": "B+10m at USD-M five-minute open",
            "next_execution": "B+8h+10m",
            "target_persists_between_decisions": True,
            "invalid_or_unready_target": "TARGET_FLAT",
            "wall_clock_time_compressed": False,
        },
        "rank_contract": {
            "ranked_primitives": [
                "coarse_share",
                "coarse_coherence",
                "fine_conviction",
                "collision_share",
                "cascade_share",
                "cascade_coherence",
            ],
            "strictly_prior": True,
            "maximum_prior_valid_boundaries": 270,
            "minimum_prior_valid_boundaries": 180,
            "quantiles": [0.33, 0.67],
            "tie_rule": "LOW x<=q33; MID q33<x<=q67; HIGH x>q67",
            "macro_ranked": False,
            "invalid_boundary_enters_reference": False,
        },
        "tokens": {
            "schema": [
                {"name": name, "vocabulary": list(vocabulary)}
                for name, vocabulary in TOKEN_SCHEMA
            ],
            "columns": list(TOKEN_COLUMNS),
            "sequence_lines": 21,
            "sequence_calendar_span_days": 7,
            "safety_tokens": list(SAFETY_TOKENS),
            "position_tokens": ["SHORT", "FLAT", "LONG"],
            "action_space": list(ACTION_SPACE),
            "invalid_output_action": "TARGET_FLAT",
            "raw_numeric_prompt_fields": 0,
        },
        "controls": {
            "ordered": list(CONTROL_IDS),
            "cascade_delay": (
                "37 prior five-minute positions inside UTC month; "
                "first 37 positions control-invalid"
            ),
            "macro_mask": (
                "mask four macro fields on every history line and recompute "
                "macro_transition=MIXED"
            ),
            "independent_rebuild": True,
            "may_replace_primary": False,
        },
        "source_support_gates": {
            "all_conjunctive": True,
            "source_join_min_each_year": 0.99,
            "core_valid_min_each_year": 0.95,
            "sequence_ready_min": {
                "2020": 750,
                "2021": 1_000,
                "2022": 1_000,
                "2023": 1_000,
            },
            "quarter_ready_min_after_warmup": 225,
            "forced_flat_max_each_full_post_warmup_quarter": 0.08,
            "category_support_min_each_year": 0.03,
            "minimum_supported_categories_per_field_each_year": 2,
            "category_share_max_each_year": 0.94,
            "macro_support_and_restrict_min_each_year": 0.05,
            "micro_buy_and_sell_min_each_year": 0.10,
            "cascade_follow_and_absorb_min_each_year": 0.075,
            "distinct_signatures_min_each_year": 120,
            "signature_share_max_each_year": 0.10,
            "adjacent_year_jsd_max": 0.30,
            "diversity_denominator": (
                "sequence-ready core-valid states in named UTC year; "
                "safety and current_position excluded"
            ),
            "all_controls_distinct": True,
            "append_replay_byte_identical": True,
            "forbidden_counter_max": 0,
            "failure_action": "retire LAMB-21 unchanged before rewards",
        },
        "stage_authority": {
            "authorized": [
                "physical_header_validation",
                "exact_source_projection",
                "causal_source_join",
                "primitive_construction",
                "strict_prior_rank",
                "token_support",
                "source_only_controls",
                "append_replay",
            ],
            "forbidden": [
                "execution_market",
                "funding",
                "future_return",
                "reward",
                "model_training",
                "checkpoint_selection",
                "trade",
                "economic_evaluation",
            ],
        },
        "contingent_economic_sequence": {
            "requires_support_pass": True,
            "reward_and_model_protocol_committed_before_reward": True,
            "2020_2022": "development/history",
            "2023": "candidate-specific transfer, not globally pristine",
            "2024": "sealed historical annual test after source extension",
            "2025": "sealed historical annual eval after unchanged 2024 pass",
            "2026_ytd": (
                "sealed recent confirmation after unchanged 2024 and 2025 passes"
            ),
            "historical_not_realtime_prospective": True,
            "live_claim_requires_forward_shadow_or_live_interval": True,
            "full_three_calendar_year_claim_before_2026_12_31": False,
            "minimum_base_cagr_to_strict_mdd": 3.0,
            "maximum_strict_mdd": 0.15,
            "minimum_stress_and_delay_ratio": 2.5,
            "minimum_nonflat_intervals": 120,
            "minimum_each_direction_share": 0.20,
            "maximum_familywise_p": 0.10,
            "lattice_only_and_no_lattice_killer_baselines": True,
        },
        "forbidden_support_fields": list(FORBIDDEN_SUPPORT_FIELDS),
        "evidence_boundary": {
            "source_value_rows_decoded": 0,
            "joint_state_rows_built": 0,
            "execution_market_rows_opened": 0,
            "funding_rows_opened": 0,
            "future_return_rows_opened": 0,
            "reward_rows_built": 0,
            "model_rows_built": 0,
            "trades_built": 0,
            "pnl_values_computed": 0,
            "cagr_values_computed": 0,
            "mdd_values_computed": 0,
            "post_2023_source_rows_opened": 0,
        },
        "support_outputs": {
            "token_support": "data/lamb21_source_support/token_support.csv.gz",
            "support_report": "results/lamb21_source_support_2026-07-25.json",
        },
    }
    payload["manifest_hash"] = canonical_hash(payload)
    return payload


def validate_manifest(payload: Mapping[str, Any]) -> None:
    expected = jsonable(build_manifest())
    normalized = jsonable(dict(payload))
    if normalized != expected:
        raise ValueError("LAMB preregistration manifest differs from frozen code")
    if tuple(normalized["tokens"]["columns"]) != TOKEN_COLUMNS:
        raise ValueError("LAMB token schema drift")
    if tuple(normalized["controls"]["ordered"]) != CONTROL_IDS:
        raise ValueError("LAMB control order drift")
    if tuple(normalized["tokens"]["action_space"]) != ACTION_SPACE:
        raise ValueError("LAMB action space drift")
    if any(value != 0 for value in normalized["evidence_boundary"].values()):
        raise ValueError("LAMB preregistration opened forbidden evidence")
    token_names = {name.lower() for name in TOKEN_COLUMNS}
    if token_names.intersection(FORBIDDEN_SUPPORT_FIELDS):
        raise ValueError("LAMB forbidden field entered token schema")
    core = {
        key: value for key, value in normalized.items() if key != "manifest_hash"
    }
    if normalized["manifest_hash"] != canonical_hash(core):
        raise ValueError("LAMB preregistration manifest hash drift")


def assert_producer_committed(*, creating: bool) -> None:
    sealed_bytes = subprocess.run(
        ["git", "show", f"{SEALED_PRODUCER_COMMIT}:{PRODUCER_SCRIPT}"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=True,
    ).stdout
    if hashlib.sha256(sealed_bytes).hexdigest() != SEALED_PRODUCER_SCRIPT_SHA256:
        raise RuntimeError("LAMB sealed producer commit no longer reproduces its hash")
    if not creating:
        return
    current_head = _git_output("rev-parse", "HEAD")
    if (
        current_head != SEALED_PRODUCER_COMMIT
        or sha256_file(PRODUCER_SCRIPT) != SEALED_PRODUCER_SCRIPT_SHA256
    ):
        raise RuntimeError(
            "LAMB missing artifact can only be created by the sealed producer HEAD"
        )


def _output_path(path: str | Path) -> Path:
    candidate = Path(path)
    if (
        str(path).startswith("~")
        or candidate.is_absolute()
        or ".." in candidate.parts
        or candidate.name in {"", ".", ".."}
    ):
        raise RuntimeError("LAMB output path must be repository-relative")
    target = REPOSITORY_ROOT / candidate
    root = REPOSITORY_ROOT.resolve(strict=True)
    try:
        target.resolve(strict=False).relative_to(root)
    except ValueError as error:
        raise RuntimeError("LAMB output path escapes repository") from error
    current = REPOSITORY_ROOT
    for part in candidate.parent.parts:
        current /= part
        if current.is_symlink():
            raise RuntimeError("LAMB output parent contains a symlink")
    return target


def write_once(path: str | Path, payload: Mapping[str, Any]) -> str:
    target = _output_path(path)
    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if target.exists():
        if target.is_symlink() or not target.is_file():
            raise RuntimeError("LAMB write-once target is not a regular file")
        if target.read_bytes() != encoded:
            raise RuntimeError(f"LAMB write-once artifact drift: {target}")
        return "verified_existing"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.is_symlink():
        raise RuntimeError("LAMB output parent contains a symlink")
    try:
        descriptor = os.open(
            target,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o644,
        )
    except FileExistsError:
        if target.read_bytes() != encoded:
            raise RuntimeError(f"LAMB write-once artifact drift: {target}")
        return "verified_existing"
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        target.unlink(missing_ok=True)
        raise
    return "created"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    assert_boundary_committed()
    output_exists = _output_path(args.output).exists()
    assert_producer_committed(creating=not output_exists)
    validate_frozen_dependencies()
    payload = build_manifest()
    validate_manifest(payload)
    status = write_once(args.output, payload)
    print(
        json.dumps(
            {
                "output": args.output,
                "status": status,
                "sha256": sha256_file(args.output),
                "manifest_hash": payload["manifest_hash"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
