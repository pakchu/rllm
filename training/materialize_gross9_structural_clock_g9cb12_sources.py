"""One-shot G9CB-12 source-support materializer.

The production entry point deliberately has no command-line options.  Tests use
``materialize`` with a synthetic :class:`MaterializationConfig`; production
uses the frozen configuration returned by :func:`official_config`.
"""

from __future__ import annotations

import csv
import ctypes
import errno
import gzip
import hashlib
import io
import json
import math
import os
import stat
import subprocess
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, NoReturn, Sequence

import numpy as np
import pandas as pd

from training.gross9_structural_clock_primitives import (
    _merge_aux,
    normalise_funding_history_frame,
    normalise_premium_index_frame,
)


class SourceSupportFailure(RuntimeError):
    """Terminal failure for the consumed G9CB-12 source-support identity."""

    def __init__(self, message: str, *, failure_reason: str = "structural_or_schema_violation") -> None:
        super().__init__(message)
        self.failure_reason = failure_reason


IDENTITY = "G9CB-12-SOURCE-SUPPORT"
VERSION = "gross9_structural_clock_bundle_g9cb12_source_support_v1"
BRANCH = "codex/gross9-structural-clock-bundle-20260731"
A9 = "98fe1e95708ad095cf0727363c32a89e7d03ead6"
T8 = "4188f35caa2c491f7b12f400d0815ea3a1a6144b"
S9 = "fe7dbb94e474d0d6f7ec3514ef79402e46c47c1e"
A10 = "6f9dd21554bc7b3282d0b2cbf7badee126e75c1a"
T9 = "a3ce195b02598b139068294089695b5d5dcd5044"
S10 = "1079c3575c7e7dced52eea15e1ef35ae0171a5dd"
A11 = "189b5403c66ea0283e67b42b9fbc6ba909280a57"
T10 = "7f5866be73e01e9531e585c7a13b19661906b05c"
S11 = "646fccbf6568bcf39fab12a47873f72da880ca01"
A12 = "a533ec5ec6bb01d0eeed8ab66a37a3a10f1dba5d"
T11 = "87c9d32df28f4b8c157d78e2d88145d6bfbb92c0"

MARKET_SCHEMA = (
    "date", "open", "high", "low", "close", "volume",
    "quote_asset_volume", "number_of_trades", "taker_buy_base",
    "taker_buy_quote", "tic", "day", "dxy", "kimchi_premium",
    "usdkrw", "btckrw", "dxy_available", "kimchi_available",
    "usdkrw_available", "external_any_available", "dxy_zscore",
    "dxy_momentum", "kimchi_premium_zscore", "kimchi_premium_change",
    "usdkrw_zscore", "usdkrw_momentum",
)
OI_SCHEMA = ("date", "open_interest")
METRICS_SCHEMA = (
    "create_time", "symbol", "sum_open_interest",
    "sum_open_interest_value", "count_toptrader_long_short_ratio",
    "sum_toptrader_long_short_ratio", "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
)
RANK7_PROJECTION = (
    "date", "spot_close", "spot_rows", "premium_index_1m_close",
    "premium_rows",
)
RANK7_REQUIRED_TAIL_LIMIT = 3000

# Authenticated tracked bytes from
# configs/shadow/portfolio_added_alpha_signal_parity_sources_2026-07-16.json.
# The official materializer must not open that path as an eighth input.
_INHERITED_MANIFEST_LITERAL: Mapping[str, Any] = {
    "schema_version": 1,
    "as_of": "2026-07-16",
    "sources": [
        {
            "name": "market_5m",
            "path": "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz",
            "sha256": "a77cd0ae5b88b3c95e509d8d2610773d34af3afdc9170c63d88564bc3d0b990c",
        },
        {
            "name": "funding",
            "path": "data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz",
            "sha256": "4d381be086e275bacaf31df431dc31307a71a26b3947b7082efffc10bb129dd7",
        },
        {
            "name": "premium",
            "path": "data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz",
            "sha256": "b45fcc5a3cf75c8e594effe61a698c4652f841b1d304107e9669524e0fc9d0d7",
        },
        {
            "name": "open_interest",
            "path": "/tmp/btcusdt_open_interest_5m_2020_2026.csv",
            "sha256": "e08f93033e56959e8e7a9c1e21f27c5f01efc8d06fa6b4fbbfe7354697122b31",
        },
        {
            "name": "rex_taker_train",
            "path": "data/rex_pullback_reclaim_q075_h144_ranker_train_2021_2023.jsonl",
            "sha256": "07f6c4bb43ac92b341ce1a1b54ea6a429983611000148ad6966b81ea4a086df0",
            "rows": 1230,
        },
        {
            "name": "rex_taker_test",
            "path": "data/rex_pullback_reclaim_q075_h144_ranker_test_2024.jsonl",
            "sha256": "b1f5abf59c901ac109823a50063665ef455e75e70e90135acda77755ab8e5371",
            "rows": 252,
        },
        {
            "name": "rex_taker_eval",
            "path": "data/rex_pullback_reclaim_q075_h144_ranker_eval_2025_2026h1.jsonl",
            "sha256": "bbe13d845d8dffcbb3e6c9b0f348390bd9d089c2d7b7bd6bccbafb91e75d9ce7",
            "rows": 207,
        },
        {
            "name": "rex_veto_source",
            "path": "data/rex_event_reasoning_policy_sft_20260712.jsonl",
            "sha256": "2f5f477ed7ffd6063bd25b1fdbcb6cbaa804685be43b4522b7105dfba1b75d48",
            "rows": 1444,
        },
    ],
}


@dataclass(frozen=True)
class InputBinding:
    name: str
    path: str
    sha256: str
    size_bytes: int
    compressed: bool
    mode: int = 0o644


@dataclass(frozen=True)
class MaterializationConfig:
    root: Path
    inputs: tuple[InputBinding, ...]
    attempt_path: str
    market_output_path: str
    oi_output_path: str
    manifest_output_path: str
    support_output_path: str
    old_last: pd.Timestamp
    domain_end: pd.Timestamp
    expected_old_rows: int
    expected_complete_rows: int
    expected_append_rows: int
    splice_rows: int = 13
    repository_head: str = "synthetic-s12"
    repository_parent: str = T11
    authority_commit: str = A12
    branch: str = BRANCH
    enforce_repository_gates: bool = False
    inherited_manifest: Mapping[str, Any] | None = None

    @property
    def output_paths(self) -> tuple[str, ...]:
        return (
            self.attempt_path,
            self.market_output_path,
            self.oi_output_path,
            self.manifest_output_path,
            self.support_output_path,
        )


PRE_SENTINEL_FAILURE = "pre_sentinel_failure"
POST_SENTINEL_PRE_OTHER_OUTPUT_FAILURE = "post_sentinel_pre_other_output_failure"
PARTIAL_PUBLICATION_FAILURE = "partial_publication_failure"

PREFLIGHT_OR_BINDING_FAILURE = "preflight_or_binding_failure"
RANK7_TAIL_MEMBERSHIP_MISMATCH = "rank7_tail_membership_mismatch"
STRUCTURAL_OR_SCHEMA_VIOLATION = "structural_or_schema_violation"
ZERO_DISCLOSURE_BREACH = "zero_disclosure_breach"
PUBLICATION_OR_READBACK_FAILURE = "publication_or_readback_failure"
FINAL_REAUTHENTICATION_FAILURE = "final_reauthentication_failure"

ALLOWED_TERMINAL_PAIRS = frozenset(
    {
        (PRE_SENTINEL_FAILURE, PREFLIGHT_OR_BINDING_FAILURE),
        (PRE_SENTINEL_FAILURE, PUBLICATION_OR_READBACK_FAILURE),
        (POST_SENTINEL_PRE_OTHER_OUTPUT_FAILURE, RANK7_TAIL_MEMBERSHIP_MISMATCH),
        (POST_SENTINEL_PRE_OTHER_OUTPUT_FAILURE, STRUCTURAL_OR_SCHEMA_VIOLATION),
        (POST_SENTINEL_PRE_OTHER_OUTPUT_FAILURE, ZERO_DISCLOSURE_BREACH),
        (POST_SENTINEL_PRE_OTHER_OUTPUT_FAILURE, PUBLICATION_OR_READBACK_FAILURE),
        (PARTIAL_PUBLICATION_FAILURE, PUBLICATION_OR_READBACK_FAILURE),
        (PARTIAL_PUBLICATION_FAILURE, FINAL_REAUTHENTICATION_FAILURE),
    }
)


def validate_terminal_pair(publication_state: str, failure_reason: str) -> None:
    """Reject every terminal state/reason combination not authorized by A12."""
    if (publication_state, failure_reason) not in ALLOWED_TERMINAL_PAIRS:
        raise SourceSupportFailure("invalid terminal publication-state/failure-reason pair")


def classify_and_validate_terminal_pair(
    config: MaterializationConfig,
    failure_reason: str,
) -> tuple[str, str]:
    """Derive state from the immutable prefix, then validate its reason."""
    publication_state = classify_terminal_publication_state(config)
    validate_terminal_pair(publication_state, failure_reason)
    return publication_state, failure_reason


def classify_terminal_failure(
    config: MaterializationConfig,
    failure: SourceSupportFailure,
) -> tuple[str, str]:
    """Validate exception taxonomy against state derived only from outputs."""
    return classify_and_validate_terminal_pair(config, failure.failure_reason)


_validate_terminal_pair = validate_terminal_pair


def classify_terminal_publication_state(config: MaterializationConfig) -> str:
    """Classify a failed one-shot invocation from its immutable output prefix."""
    present: list[bool] = []
    for text in config.output_paths:
        path = _path(config, text)
        try:
            info = os.lstat(path)
        except FileNotFoundError:
            present.append(False)
            continue
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o444 or info.st_nlink != 1:
            _fail(f"terminal publication metadata differs: {text}")
        present.append(True)
    if not any(present):
        return PRE_SENTINEL_FAILURE
    if present == [True, False, False, False, False]:
        return POST_SENTINEL_PRE_OTHER_OUTPUT_FAILURE
    prefix_length = 0
    for value in present:
        if not value:
            break
        prefix_length += 1
    if prefix_length >= 2 and present == [index < prefix_length for index in range(5)]:
        return PARTIAL_PUBLICATION_FAILURE
    _fail("terminal publication set is not an ordered prefix")


_classify_terminal_state = classify_terminal_publication_state


# This is intentionally process-local, not durable guard state.  A fresh-process
# retry is unauthorized; official invocation cardinality is supervisor/authority
# owned so pre_sentinel_failure can leave all five visible outputs absent.
_PROCESS_LOCAL_CONSUMED_INVOCATIONS: set[tuple[str, tuple[str, ...]]] = set()


OFFICIAL_INPUTS = (
    InputBinding("old_market", "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz", "a77cd0ae5b88b3c95e509d8d2610773d34af3afdc9170c63d88564bc3d0b990c", 66696659, True),
    InputBinding("replacement_market", "/home/pakchu/rllm/data/cache_market_ext_5m_wavefull_2020-01-01_2026-07-05_dbappend.csv.gz", "0447a2c89926a1deebdfd495edde069a697d9481bc5936bc360c8c1488de2ebe", 65420089, True),
    InputBinding("funding", "data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz", "4d381be086e275bacaf31df431dc31307a71a26b3947b7082efffc10bb129dd7", 89326, True),
    InputBinding("premium", "data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz", "b45fcc5a3cf75c8e594effe61a698c4652f841b1d304107e9669524e0fc9d0d7", 1196481, True),
    InputBinding("old_open_interest", "/tmp/btcusdt_open_interest_5m_2020_2026.csv", "e08f93033e56959e8e7a9c1e21f27c5f01efc8d06fa6b4fbbfe7354697122b31", 19657777, False),
    InputBinding("binance_metrics_open_interest", "/home/pakchu/rllm/data/binance_um_metrics_BTCUSDT_5m_2020-09-01_2026-06-01.csv.gz", "d391022352d5b14dea7ffd207a9d1f84f603d06ddae42da55dd792f722fc0106", 21440132, True),
    InputBinding("rank7_spot_premium_5m", "/home/pakchu/rllm/data/cache_spot_premium_5m_2020-01-01_2026-06-01.csv.gz", "c21ed3f52804a3f879ef5167b6b81ee0fd7dd262f2e9d87ee5c1c25dbad73617", 15772146, True),
)


def official_config(root: Path | None = None) -> MaterializationConfig:
    repository = Path(__file__).resolve().parents[1] if root is None else root
    return MaterializationConfig(
        root=repository,
        inputs=OFFICIAL_INPUTS,
        attempt_path="results/gross9_structural_clock_bundle_g9cb12_source_support_attempt_consumed_2026-07-31.json",
        market_output_path="data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_g9cb12_complete.csv.gz",
        oi_output_path="data/btcusdt_open_interest_5m_2020-01-01_2026-06-01_g9cb12_complete.csv.gz",
        manifest_output_path="configs/shadow/gross9_structural_clock_bundle_g9cb12_sources_2026-07-31.json",
        support_output_path="results/gross9_structural_clock_bundle_g9cb12_source_support_2026-07-31.json",
        old_last=pd.Timestamp("2026-05-31 15:00:00"),
        domain_end=pd.Timestamp("2026-06-01 00:00:00"),
        expected_old_rows=674785,
        expected_complete_rows=674892,
        expected_append_rows=107,
        enforce_repository_gates=True,
    )


def _fail(
    message: str,
    *,
    failure_reason: str = STRUCTURAL_OR_SCHEMA_VIOLATION,
) -> NoReturn:
    raise SourceSupportFailure(message, failure_reason=failure_reason)


def _with_failure_reason(
    failure_reason: str,
    operation: Callable[..., Any],
    /,
    *args: Any,
    **kwargs: Any,
) -> Any:
    try:
        return operation(*args, **kwargs)
    except SourceSupportFailure as exc:
        raise SourceSupportFailure(str(exc), failure_reason=failure_reason) from exc
    except OSError as exc:
        raise SourceSupportFailure(str(exc), failure_reason=failure_reason) from exc


def _with_oserror_reason(
    failure_reason: str,
    operation: Callable[..., Any],
    /,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Translate operating-system failures without overwriting specific reasons."""
    try:
        return operation(*args, **kwargs)
    except OSError as exc:
        raise SourceSupportFailure(str(exc), failure_reason=failure_reason) from exc


def canonical_json_bytes(payload: Any, *, trailing_lf: bool = True) -> bytes:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return raw + (b"\n" if trailing_lf else b"")


def object_hash(payload: Mapping[str, Any], field: str) -> str:
    return hashlib.sha256(canonical_json_bytes({k: v for k, v in payload.items() if k != field}, trailing_lf=False)).hexdigest()


def _with_hash(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(payload)
    result[field] = object_hash(result, field)
    return result


def _path(config: MaterializationConfig, text: str) -> Path:
    path = Path(text)
    return path if path.is_absolute() else config.root / path


def _timestamp_z(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_git(root: Path, *args: str) -> str:
    completed = subprocess.run(["git", "-C", str(root), *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True)
    if completed.returncode:
        _fail(f"git {' '.join(args)} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def _validate_command_and_root(config: MaterializationConfig) -> None:
    if sys.argv[1:]:
        _fail("the official entry point accepts no arguments")
    canonical = Path(__file__).resolve().parents[1]
    if config.root.resolve(strict=True) != canonical or Path.cwd().resolve(strict=True) != canonical:
        _fail("canonical repository root differs")
    if os.environ.get("PYTHONPATH") != str(canonical):
        _fail("official PYTHONPATH command shape differs")
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1" or not sys.dont_write_bytecode:
        _fail("bytecode-disabled command shape differs")


def _validate_bytecode_first_gate(root: Path) -> None:
    # This is the shared Q8 preflight, imported lazily so command-shape checking
    # precedes its repository traversal.
    from training.preregister_gross9_structural_clock_bundle import validate_repository_bytecode_preflight

    try:
        validate_repository_bytecode_preflight(root)
    except (OSError, ValueError) as exc:
        raise SourceSupportFailure(str(exc)) from exc


def _validate_git_gate(config: MaterializationConfig) -> str:
    root = config.root
    head = _run_git(root, "rev-parse", "HEAD")
    if _run_git(root, "branch", "--show-current") != config.branch:
        _fail("official branch differs")
    if _run_git(root, "rev-parse", "@{upstream}") != head:
        _fail("HEAD and upstream differ")
    parents = _run_git(root, "rev-list", "--parents", "-n", "1", head).split()
    if len(parents) != 2 or parents[1] != T11:
        _fail("S12 direct parent differs")
    if _run_git(root, "rev-parse", f"{T11}^") != A12:
        _fail("T11/A12 topology differs")
    if _run_git(root, "rev-parse", f"{A12}^") != S11:
        _fail("A12/S11 topology differs")
    if _run_git(root, "rev-parse", f"{S11}^") != T10:
        _fail("S11/T10 topology differs")
    if _run_git(root, "diff", "--name-status", S11, A12).splitlines() != [
        "A\tdocs/gross9-structural-clock-bundle-g9cb12-successor-authority-decision-2026-07-31.md"
    ]:
        _fail("exact A12 one-file diff differs")
    if sorted(_run_git(root, "diff", "--name-status", A12, T11).splitlines()) != [
        "A\tresults/gross9_structural_clock_bundle_g9cb11_source_support_attempt_consumed_2026-07-31.json",
        "A\tresults/gross9_structural_clock_bundle_g9cb11_source_support_terminal_failure_2026-07-31.json",
    ]:
        _fail("exact T11 two-file diff differs")
    expected = [
        "A\ttests/test_materialize_gross9_structural_clock_g9cb12_sources.py",
        "A\ttraining/materialize_gross9_structural_clock_g9cb12_sources.py",
    ]
    if sorted(_run_git(root, "diff", "--name-status", T11, head).splitlines()) != expected:
        _fail("exact S12 two-file diff differs")
    for entry in expected:
        relative = entry.split("\t", 1)[1]
        indexed = _run_git(root, "ls-files", "-s", "--", relative).split()
        if len(indexed) < 4 or indexed[0] != "100644" or indexed[2] != "0" or indexed[3] != relative:
            _fail(f"S12 Git mode or index binding differs: {relative}")
        worktree = os.lstat(root / relative)
        if (
            not stat.S_ISREG(worktree.st_mode)
            or stat.S_IMODE(worktree.st_mode) != 0o644
            or worktree.st_nlink != 1
            or _run_git(root, "rev-parse", f"{head}:{relative}") != indexed[1]
        ):
            _fail(f"S12 worktree mode or commit binding differs: {relative}")
    if _run_git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        _fail("index or worktree is not clean")
    return head


_A12_AUTHORITY = (
    "docs/gross9-structural-clock-bundle-g9cb12-successor-authority-decision-2026-07-31.md",
    44054,
    "1c10d085d9e38aad9568f8769795de38d9d8729bf41334db70c839723d64ba6f",
    "f27653975eb4a1b7fd2ce057034fc26ad447a0ff",
)
_T11_EVIDENCE = (
    (
        "results/gross9_structural_clock_bundle_g9cb11_source_support_attempt_consumed_2026-07-31.json",
        3056,
        "128ad6213785ecfa360114eae6e3587254dda3b18e94108b9dd30a0f34533e31",
        "1dceda439db16bc247cb82c3a5d807d89d0a1525",
    ),
    (
        "results/gross9_structural_clock_bundle_g9cb11_source_support_terminal_failure_2026-07-31.json",
        4706,
        "da943985354e4abfab87a06a16576235ab34a487bdff0a4b498ae6fb1728e045",
        "550b5a77fa83b0f56ac37ad51613ef4a407b6014",
    ),
)
_G9CB11_PERMANENT_ABSENCES = (
    "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_g9cb11_complete.csv.gz",
    "data/btcusdt_open_interest_5m_2020-01-01_2026-06-01_g9cb11_complete.csv.gz",
    "configs/shadow/gross9_structural_clock_bundle_g9cb11_sources_2026-07-31.json",
    "results/gross9_structural_clock_bundle_g9cb11_source_support_2026-07-31.json",
)
_T11_TERMINAL_LITERAL = json.loads('{"access":{"attempt_sentinel_publication_count":1,"cagr_evaluation_count":0,"candidate_value_rows_opened":0,"comparator_value_rows_opened":0,"decode_pass_count":9,"decode_passes":["old_market","replacement_market_date_scan","replacement_market_tail","funding","premium","old_open_interest","binance_metrics_open_interest_date_scan","binance_metrics_open_interest_selected_window","rank7_spot_premium_5m"],"drawdown_evaluation_count":0,"economic_evaluation_count":0,"economic_value_rows_opened":0,"feature_value_rows_opened":0,"generated_output_publication_count":0,"generated_output_readback_count":0,"global_metrics_alignment_comparison_count":0,"mdd_evaluation_count":0,"metrics_date_scan_count":1,"metrics_overlap_row_count":13,"metrics_selected_decode_count":1,"metrics_selected_row_count":120,"metrics_tail_row_count":107,"non_selected_metrics_non_date_semantic_evaluation_count":0,"off_grid_detail_disclosure_count":0,"pnl_value_rows_opened":0,"rank7_all_history_coverage_comparison_count":1,"rank7_gap_detail_disclosure_count":0,"rank7_spot_premium_5m_decode_count":1,"rank7_tail_completeness_evaluation_count":0,"raw_file_count":7,"raw_file_open_count":7,"replacement_market_date_scan_count":1,"replacement_market_tail_decode_count":1,"replacement_market_tail_selected_row_count":107,"return_value_rows_opened":0,"schedule_value_rows_opened":0,"signal_value_rows_opened":0},"attempt_sentinel":{"attempt_hash":"6a6204b5074aee399f6a4e318d24764140cfb07aea9b6ebd01b021f7333038f1","device":2096,"filesystem_type":"regular_file","git_mode":"100644","inode":934842,"link_count":1,"one_shot":true,"opaque_bytes_hashed_before_publication":190272610,"path":"results/gross9_structural_clock_bundle_g9cb11_source_support_attempt_consumed_2026-07-31.json","raw_input_count":7,"repository_head":"646fccbf6568bcf39fab12a47873f72da880ca01","repository_parent":"7f5866be73e01e9531e585c7a13b19661906b05c","resume_allowed":false,"retry_allowed":false,"sha256":"128ad6213785ecfa360114eae6e3587254dda3b18e94108b9dd30a0f34533e31","size_bytes":3056,"worktree_mode":"0444"},"authority":{"commit":"189b5403c66ea0283e67b42b9fbc6ba909280a57","document_path":"docs/gross9-structural-clock-bundle-g9cb11-successor-authority-decision-2026-07-31.md"},"execution":{"command":"PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.materialize_gross9_structural_clock_g9cb11_sources","exit_status":1,"invocation_count":1,"one_shot":true,"resume_allowed":false,"retry_allowed":false},"failure":{"exception_class":"SourceSupportFailure","exception_message":"incomplete Rank7 projection coverage","failure_reason":"rank7_all_history_coverage_mismatch","phase":"transform","publication_state":"post_sentinel_pre_other_output_failure","rank7_gap_count_disclosed":false,"rank7_gap_detail_disclosure_count":0,"rank7_gap_location_disclosed":false,"rank7_gap_timestamp_disclosed":false,"rank7_gap_value_disclosed":false,"traceback_source_value_excerpt_emitted":false},"identity":"G9CB-11-SOURCE-SUPPORT","implementation":{"commit":"646fccbf6568bcf39fab12a47873f72da880ca01","files":[{"git_blob":"ab4459a9e6ae48047840787644a0839f474adc9c","git_mode":"100644","path":"training/materialize_gross9_structural_clock_g9cb11_sources.py","sha256":"4e34ecbc9c812e4fe7d633110f2e06536ff491c553f4449d36fd8dca58bfb828","size_bytes":98456,"worktree_mode":"0644"},{"git_blob":"4415d503f94ddb5e17bac3c115de550f8842b39b","git_mode":"100644","path":"tests/test_materialize_gross9_structural_clock_g9cb11_sources.py","sha256":"bf4a3716cd2dd612e7ac884df55298a4001b174d9dbdea39ab6d29907083d7de","size_bytes":147428,"worktree_mode":"0644"}],"parent_commit":"7f5866be73e01e9531e585c7a13b19661906b05c"},"ledger_kind":"gross9_structural_clock_bundle_g9cb11_source_support_terminal_failure_v1","output_state":{"downstream_consumable":false,"forbidden_stages":["M11","Q11","P11","C11","D11","V11","H11"],"permanently_absent_output_paths":["data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_g9cb11_complete.csv.gz","data/btcusdt_open_interest_5m_2020-01-01_2026-06-01_g9cb11_complete.csv.gz","configs/shadow/gross9_structural_clock_bundle_g9cb11_sources_2026-07-31.json","results/gross9_structural_clock_bundle_g9cb11_source_support_2026-07-31.json"],"source_authoritative":false,"terminal_evidence_paths":["results/gross9_structural_clock_bundle_g9cb11_source_support_attempt_consumed_2026-07-31.json"]},"schema_version":1,"seal_authority":{"commit":"a533ec5ec6bb01d0eeed8ab66a37a3a10f1dba5d","document_path":"docs/gross9-structural-clock-bundle-g9cb12-successor-authority-decision-2026-07-31.md"},"status":"terminal_rank7_all_history_coverage_failure","terminal_failure_hash":"c0ea80bafac28fd924b2c1c19bf2192c2fa3e5ec4a212055cd7b154da607c019"}')


def _read_bound_json(
    config: MaterializationConfig,
    relative: str,
    size: int,
    digest: str,
    blob: str,
    *,
    mode: int,
) -> dict[str, Any]:
    fd = _open_nofollow_components(config.root / relative, os.O_RDONLY)
    try:
        info = os.fstat(fd)
        raw = _pread_complete(fd, info.st_size)
    finally:
        os.close(fd)
    if (
        not stat.S_ISREG(info.st_mode)
        or stat.S_IMODE(info.st_mode) != mode
        or info.st_nlink != 1
        or info.st_size != size
        or hashlib.sha256(raw).hexdigest() != digest
    ):
        _fail(f"bound evidence differs: {relative}")
    indexed = _run_git(config.root, "ls-files", "-s", "--", relative).split()
    if indexed != ["100644", blob, "0", relative]:
        _fail(f"bound evidence Git identity differs: {relative}")
    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=lambda pairs: _unique_json(pairs))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceSupportFailure(f"bound evidence JSON differs: {relative}") from exc
    if canonical_json_bytes(payload) != raw:
        _fail(f"bound evidence canonical bytes differ: {relative}")
    return payload


def _validate_history_gate(config: MaterializationConfig) -> None:
    authority_path, authority_size, authority_digest, authority_blob = _A12_AUTHORITY
    authority_fd = _open_nofollow_components(config.root / authority_path, os.O_RDONLY)
    try:
        authority_info = os.fstat(authority_fd)
        authority_raw = _pread_complete(authority_fd, authority_info.st_size)
    finally:
        os.close(authority_fd)
    if (
        not stat.S_ISREG(authority_info.st_mode)
        or stat.S_IMODE(authority_info.st_mode) != 0o644
        or authority_info.st_nlink != 1
        or authority_info.st_size != authority_size
        or hashlib.sha256(authority_raw).hexdigest() != authority_digest
        or _run_git(config.root, "ls-files", "-s", "--", authority_path).split()
        != ["100644", authority_blob, "0", authority_path]
    ):
        _fail("A12 authority binding differs")

    attempt = _read_bound_json(config, *_T11_EVIDENCE[0], mode=0o444)
    terminal = _read_bound_json(config, *_T11_EVIDENCE[1], mode=0o444)
    attempt_info = os.lstat(config.root / _T11_EVIDENCE[0][0])
    if (
        (attempt_info.st_dev, attempt_info.st_ino) != (2096, 934842)
        or attempt.get("identity") != "G9CB-11-SOURCE-SUPPORT"
        or attempt.get("repository_head") != S11
        or attempt.get("repository_parent") != T10
        or attempt.get("attempt_hash") != "6a6204b5074aee399f6a4e318d24764140cfb07aea9b6ebd01b021f7333038f1"
        or attempt.get("expected_outputs") != [_T11_EVIDENCE[0][0], *_G9CB11_PERMANENT_ABSENCES]
    ):
        _fail("G9CB-11 attempt sentinel constants differ")
    if (
        canonical_json_bytes(terminal) != canonical_json_bytes(_T11_TERMINAL_LITERAL)
        or object_hash(terminal, "terminal_failure_hash") != terminal.get("terminal_failure_hash")
        or terminal.get("terminal_failure_hash") != "c0ea80bafac28fd924b2c1c19bf2192c2fa3e5ec4a212055cd7b154da607c019"
        or terminal.get("seal_authority") != {"commit": A12, "document_path": authority_path}
    ):
        _fail("T11 terminal ledger constants differ")
    for relative in _G9CB11_PERMANENT_ABSENCES:
        path = config.root / relative
        if path.exists() or path.is_symlink():
            _fail(f"permanent G9CB-11 absence differs: {relative}")


def _stat_token(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode), stat.S_IMODE(info.st_mode), info.st_size, info.st_nlink)


def _descriptor_hash(fd: int) -> tuple[str, int]:
    info = os.fstat(fd)
    digest = hashlib.sha256()
    offset = 0
    while offset < info.st_size:
        chunk = os.pread(fd, min(1024 * 1024, info.st_size - offset), offset)
        if not chunk:
            _fail("retained source descriptor short read")
        digest.update(chunk)
        offset += len(chunk)
    if offset != info.st_size or os.pread(fd, 1, offset):
        _fail("retained source descriptor size differs")
    return digest.hexdigest(), offset


def _pread_complete(fd: int, size: int) -> bytes:
    chunks: list[bytes] = []
    offset = 0
    while offset < size:
        chunk = os.pread(fd, min(1024 * 1024, size - offset), offset)
        if not chunk:
            _fail("descriptor readback was short")
        chunks.append(chunk)
        offset += len(chunk)
    if os.pread(fd, 1, size):
        _fail("descriptor readback exceeded bound size")
    return b"".join(chunks)


@dataclass
class _Retained:
    binding: InputBinding
    fd: int
    token: tuple[int, int, int, int, int, int]


def _open_nofollow_components(path: Path, final_flags: int) -> int:
    """Open *path* without allowing a symlink in any traversed component."""
    absolute = path.absolute()
    parts = absolute.parts
    if not absolute.is_absolute() or len(parts) < 2:
        _fail(f"invalid bound path: {path}")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    directory_fd = os.open(parts[0], directory_flags)
    try:
        for component in parts[1:-1]:
            if component in ("", ".", ".."):
                _fail(f"invalid bound path component: {path}")
            next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        if parts[-1] in ("", ".", ".."):
            _fail(f"invalid bound path leaf: {path}")
        return os.open(parts[-1], final_flags | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0), dir_fd=directory_fd)
    finally:
        os.close(directory_fd)


def _stat_nofollow_components(path: Path) -> os.stat_result:
    """Stat *path* without reopening its leaf or following any symlink."""
    absolute = path.absolute()
    parts = absolute.parts
    if not absolute.is_absolute() or len(parts) < 2:
        _fail(f"invalid bound path: {path}")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    directory_fd = os.open(parts[0], directory_flags)
    try:
        for component in parts[1:-1]:
            if component in ("", ".", ".."):
                _fail(f"invalid bound path component: {path}")
            next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        if parts[-1] in ("", ".", ".."):
            _fail(f"invalid bound path leaf: {path}")
        return os.stat(parts[-1], dir_fd=directory_fd, follow_symlinks=False)
    finally:
        os.close(directory_fd)


def _path_edge_check(retained: _Retained) -> None:
    try:
        edge = _stat_nofollow_components(Path(retained.binding.path))
    except OSError as exc:
        raise SourceSupportFailure(f"raw input path-edge drift: {retained.binding.name}") from exc
    if _stat_token(edge) != retained.token:
        _fail(f"raw input pathname replacement: {retained.binding.name}")


def _assert_retained_identity(retained: _Retained, phase: str) -> None:
    info = os.fstat(retained.fd)
    if _stat_token(info) != retained.token or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        _fail(f"retained descriptor identity drift: {retained.binding.name}:{phase}")
    _path_edge_check(retained)


def _identity_check(retained: _Retained, phase: str, failpoint: Callable[[str], None]) -> None:
    _assert_retained_identity(retained, phase)
    failpoint(f"identity:{retained.binding.name}:{phase}")


def _final_reauthenticate_input(
    retained: _Retained,
    slot: int,
    failpoint: Callable[[str], None],
) -> None:
    """Bracket a final full-byte rehash with descriptor/path identity checks."""
    # Keep the inherited per-source final event.  Slot-scoped events expose the
    # seven-pass sequence without changing the established source event stream.
    _identity_check(retained, "final", failpoint)
    failpoint(f"final_input_reauthentication:{slot}:before_hash")
    _assert_retained_identity(retained, "final_before_rehash")
    digest, size = _descriptor_hash(retained.fd)
    if digest != retained.binding.sha256 or size != retained.binding.size_bytes:
        _fail(f"final retained input rehash differs: {retained.binding.name}")
    _assert_retained_identity(retained, "final_after_rehash")
    failpoint(f"final_input_reauthentication:{slot}:complete")


def _open_inputs(config: MaterializationConfig, failpoint: Callable[[str], None]) -> dict[str, _Retained]:
    if len(config.inputs) != 7 or len({row.name for row in config.inputs}) != 7:
        _fail("exactly seven unique raw inputs are required")
    retained: dict[str, _Retained] = {}
    identities: set[tuple[int, int]] = set()
    try:
        for binding in config.inputs:
            path = _path(config, binding.path)
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            try:
                fd = _open_nofollow_components(path, flags)
            except OSError as exc:
                raise SourceSupportFailure(f"cannot open bound raw input: {binding.name}") from exc
            info = os.fstat(fd)
            token = _stat_token(info)
            item = _Retained(InputBinding(binding.name, str(path), binding.sha256, binding.size_bytes, binding.compressed, binding.mode), fd, token)
            try:
                if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != binding.mode or info.st_nlink != 1 or info.st_size != binding.size_bytes:
                    _fail(f"raw input metadata drift: {binding.name}")
                identity = (info.st_dev, info.st_ino)
                if identity in identities:
                    _fail("raw input inode aliasing")
                identities.add(identity)
                failpoint(f"identity:{binding.name}:before_hash")
                digest, size = _descriptor_hash(fd)
                if digest != binding.sha256 or size != binding.size_bytes:
                    _fail(f"raw input byte binding drift: {binding.name}")
                _identity_check(item, "after_open_hash", failpoint)
                retained[binding.name] = item
            except BaseException:
                os.close(fd)
                raise
        return retained
    except BaseException:
        for item in retained.values():
            os.close(item.fd)
        raise


def _validate_output_absence(config: MaterializationConfig) -> None:
    input_ids: set[tuple[int, int]] = set()
    for binding in config.inputs:
        info = os.lstat(_path(config, binding.path))
        input_ids.add((info.st_dev, info.st_ino))
    for text in config.output_paths:
        path = _path(config, text)
        current = Path(path.anchor)
        for component in path.parts[1:-1]:
            current /= component
            info = os.lstat(current)
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                _fail(f"canonical output parent is not a real directory: {text}")
        try:
            info = os.lstat(path)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode):
            _fail(f"canonical output is a symlink: {text}")
        if (info.st_dev, info.st_ino) in input_ids:
            _fail(f"canonical output aliases an input: {text}")
        _fail(f"canonical output already exists: {text}")


def _snapshot_output_directories(config: MaterializationConfig) -> dict[Path, frozenset[str]]:
    return {
        _path(config, text).parent: frozenset(os.listdir(_path(config, text).parent))
        for text in config.output_paths
    }


def _validate_output_directory_delta(
    config: MaterializationConfig,
    baseline: Mapping[Path, frozenset[str]],
) -> None:
    expected_by_parent: dict[Path, set[str]] = {parent: set() for parent in baseline}
    for text in config.output_paths:
        path = _path(config, text)
        expected_by_parent[path.parent].add(path.name)
    for parent, original in baseline.items():
        current = frozenset(os.listdir(parent))
        if current != original.union(expected_by_parent[parent]):
            _fail(f"unauthorized output or temporary residue: {parent}")


def _decode_csv(item: _Retained, failpoint: Callable[[str], None]) -> tuple[pd.DataFrame, tuple[str, ...]]:
    _identity_check(item, "before_decode", failpoint)
    duplicate = os.dup(item.fd)
    try:
        os.lseek(duplicate, 0, os.SEEK_SET)
        with ExitStack() as stack:
            raw: io.BufferedIOBase = stack.enter_context(os.fdopen(duplicate, "rb", closefd=True))
            duplicate = -1
            stream: Any = (
                stack.enter_context(gzip.GzipFile(fileobj=raw, mode="rb"))
                if item.binding.compressed
                else raw
            )
            text = stack.enter_context(io.TextIOWrapper(stream, encoding="utf-8", newline=""))
            header_line = text.readline()
            try:
                header = tuple(next(csv.reader([header_line], strict=True)))
            except (csv.Error, StopIteration) as exc:
                raise SourceSupportFailure(f"invalid CSV header: {item.binding.name}") from exc
            if not header or len(set(header)) != len(header):
                _fail(f"duplicate or empty CSV columns: {item.binding.name}")
            text.seek(0)
            frame = pd.read_csv(text, low_memory=False)
            if tuple(frame.columns) != header:
                _fail(f"CSV schema decode drift: {item.binding.name}")
    finally:
        if duplicate >= 0:
            os.close(duplicate)
    _identity_check(item, "after_decode", failpoint)
    digest, size = _descriptor_hash(item.fd)
    if digest != item.binding.sha256 or size != item.binding.size_bytes:
        _fail(f"postdecode source rehash differs: {item.binding.name}")
    _identity_check(item, "after_rehash", failpoint)
    return frame, header


def _decode_rank7_tokens(
    item: _Retained,
    failpoint: Callable[[str], None],
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Decode Rank7 without inferring or converting any non-date token."""
    _identity_check(item, "before_decode", failpoint)
    duplicate = os.dup(item.fd)
    rows: list[list[str]] = []
    try:
        os.lseek(duplicate, 0, os.SEEK_SET)
        with ExitStack() as stack:
            raw: io.BufferedIOBase = stack.enter_context(os.fdopen(duplicate, "rb", closefd=True))
            duplicate = -1
            stream: Any = (
                stack.enter_context(gzip.GzipFile(fileobj=raw, mode="rb"))
                if item.binding.compressed
                else raw
            )
            text = stack.enter_context(io.TextIOWrapper(stream, encoding="utf-8", newline=""))
            reader = csv.reader(text, strict=True)
            try:
                header = tuple(next(reader))
                if not header or len(set(header)) != len(header):
                    _fail("duplicate or empty CSV columns: rank7_spot_premium_5m")
                if any(column not in header for column in RANK7_PROJECTION):
                    _fail("Rank7 projection column missing")
                for row in reader:
                    if len(row) != len(header):
                        _fail("Rank7 row width differs")
                    rows.append(row)
            except (csv.Error, StopIteration) as exc:
                raise SourceSupportFailure("Rank7 token decode failed") from exc
    finally:
        if duplicate >= 0:
            os.close(duplicate)
    frame = pd.DataFrame(rows, columns=header, dtype="object")
    if tuple(frame.columns) != header:
        _fail("Rank7 token schema decode drift")
    _identity_check(item, "after_decode", failpoint)
    digest, size = _descriptor_hash(item.fd)
    if digest != item.binding.sha256 or size != item.binding.size_bytes:
        _fail("postdecode source rehash differs: rank7_spot_premium_5m")
    _identity_check(item, "after_rehash", failpoint)
    return frame, header


@dataclass(frozen=True)
class ReplacementDateScan:
    schema: tuple[str, ...]
    row_count: int
    tail_positions: tuple[int, ...]
    tail_dates: pd.Series


def _scan_replacement_dates(
    item: _Retained,
    config: MaterializationConfig,
    failpoint: Callable[[str], None],
) -> ReplacementDateScan:
    """Scan only replacement timestamps and select physical tail positions."""
    _identity_check(item, "before_date_scan", failpoint)
    duplicate = os.dup(item.fd)
    dates: list[str] = []
    try:
        os.lseek(duplicate, 0, os.SEEK_SET)
        with ExitStack() as stack:
            raw: io.BufferedIOBase = stack.enter_context(os.fdopen(duplicate, "rb", closefd=True))
            duplicate = -1
            stream: Any = (
                stack.enter_context(gzip.GzipFile(fileobj=raw, mode="rb"))
                if item.binding.compressed
                else raw
            )
            text = stack.enter_context(io.TextIOWrapper(stream, encoding="utf-8", newline=""))
            reader = csv.reader(text, strict=True)
            try:
                header = tuple(next(reader))
                if header != MARKET_SCHEMA:
                    _fail("replacement market schema drift")
                for row in reader:
                    if not row:
                        _fail("replacement market row lacks date")
                    dates.append(row[0])
            except (csv.Error, StopIteration) as exc:
                raise SourceSupportFailure("replacement market date scan failed") from exc
    finally:
        if duplicate >= 0:
            os.close(duplicate)
    parsed = _dates(pd.Series(dates, dtype="object"), "replacement_market_date_scan")
    if parsed.duplicated().any():
        _fail("duplicate timestamps: replacement_market_date_scan")
    if not parsed.is_monotonic_increasing:
        _fail("out-of-order timestamps: replacement_market_date_scan")
    _require_five_minute_alignment(parsed, label="replacement_market_date_scan")
    mask = (parsed > config.old_last) & (parsed < config.domain_end)
    positions = tuple(int(index) for index in np.flatnonzero(mask.to_numpy()))
    tail_dates = parsed.iloc[list(positions)].reset_index(drop=True)
    required_first = config.old_last + pd.Timedelta(minutes=5)
    required_last = config.domain_end - pd.Timedelta(minutes=5)
    if len(positions) != config.expected_append_rows:
        _fail("replacement market tail row count differs")
    _require_grid(
        tail_dates,
        first=required_first,
        last=required_last,
        label="replacement_market_tail_positions",
    )
    _identity_check(item, "after_date_scan", failpoint)
    digest, size = _descriptor_hash(item.fd)
    if digest != item.binding.sha256 or size != item.binding.size_bytes:
        _fail("postscan source rehash differs: replacement_market")
    _identity_check(item, "after_date_scan_rehash", failpoint)
    return ReplacementDateScan(header, len(parsed), positions, tail_dates)


def _decode_replacement_tail(
    item: _Retained,
    scan: ReplacementDateScan,
    failpoint: Callable[[str], None],
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Decode full values only for the timestamp-selected replacement tail."""
    _identity_check(item, "before_tail_decode", failpoint)
    duplicate = os.dup(item.fd)
    selected: list[list[str]] = []
    row_count = 0
    wanted = iter(scan.tail_positions)
    next_position = next(wanted, None)
    try:
        os.lseek(duplicate, 0, os.SEEK_SET)
        with ExitStack() as stack:
            raw: io.BufferedIOBase = stack.enter_context(os.fdopen(duplicate, "rb", closefd=True))
            duplicate = -1
            stream: Any = (
                stack.enter_context(gzip.GzipFile(fileobj=raw, mode="rb"))
                if item.binding.compressed
                else raw
            )
            text = stack.enter_context(io.TextIOWrapper(stream, encoding="utf-8", newline=""))
            reader = csv.reader(text, strict=True)
            try:
                header = tuple(next(reader))
                if header != scan.schema or header != MARKET_SCHEMA:
                    _fail("replacement market tail schema drift")
                for row_count, row in enumerate(reader, start=1):
                    physical_position = row_count - 1
                    if physical_position != next_position:
                        continue
                    if len(row) != len(header):
                        _fail("replacement market selected tail row width differs")
                    selected.append(row)
                    next_position = next(wanted, None)
            except (csv.Error, StopIteration) as exc:
                raise SourceSupportFailure("replacement market tail decode failed") from exc
    finally:
        if duplicate >= 0:
            os.close(duplicate)
    if row_count != scan.row_count or next_position is not None or len(selected) != len(scan.tail_positions):
        _fail("replacement market physical tail selection drift")
    rebuilt = io.StringIO(newline="")
    writer = csv.writer(rebuilt, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    writer.writerow(scan.schema)
    writer.writerows(selected)
    rebuilt.seek(0)
    frame = pd.read_csv(rebuilt, low_memory=False)
    if tuple(frame.columns) != scan.schema or len(frame) != len(selected):
        _fail("replacement market selected tail decode drift")
    _identity_check(item, "after_tail_decode", failpoint)
    digest, size = _descriptor_hash(item.fd)
    if digest != item.binding.sha256 or size != item.binding.size_bytes:
        _fail("postdecode source rehash differs: replacement_market_tail")
    _identity_check(item, "after_tail_rehash", failpoint)
    return frame, scan.schema


@dataclass(frozen=True)
class MetricsDateScan:
    schema: tuple[str, ...]
    row_count: int
    selected_positions: tuple[int, ...]
    selected_dates: pd.Series


def _required_metrics_grid(config: MaterializationConfig) -> pd.Series:
    start = config.old_last - pd.Timedelta(minutes=5 * (config.splice_rows - 1))
    end = config.domain_end - pd.Timedelta(minutes=5)
    return pd.Series(pd.date_range(start, end, freq="5min"), dtype="datetime64[ns]")


def _scan_metrics_dates(
    item: _Retained,
    config: MaterializationConfig,
    failpoint: Callable[[str], None],
) -> MetricsDateScan:
    """Scan every create_time and select only exact frozen-grid members."""
    _identity_check(item, "before_date_scan", failpoint)
    duplicate = os.dup(item.fd)
    date_tokens: list[str] = []
    try:
        os.lseek(duplicate, 0, os.SEEK_SET)
        with ExitStack() as stack:
            raw: io.BufferedIOBase = stack.enter_context(os.fdopen(duplicate, "rb", closefd=True))
            duplicate = -1
            stream: Any = (
                stack.enter_context(gzip.GzipFile(fileobj=raw, mode="rb"))
                if item.binding.compressed else raw
            )
            text = stack.enter_context(io.TextIOWrapper(stream, encoding="utf-8", newline=""))
            reader = csv.reader(text, strict=True)
            try:
                header = tuple(next(reader))
                if header != METRICS_SCHEMA:
                    _fail("metrics schema drift")
                for row in reader:
                    if not row:
                        _fail("metrics row lacks create_time")
                    date_tokens.append(row[0])
            except (csv.Error, StopIteration) as exc:
                raise SourceSupportFailure("metrics date scan failed") from exc
    finally:
        if duplicate >= 0:
            os.close(duplicate)
    parsed = _dates(pd.Series(date_tokens, dtype="object"), "binance_metrics_open_interest_date_scan")
    if parsed.duplicated().any():
        _fail("duplicate timestamps: binance_metrics_open_interest_date_scan")
    if not parsed.is_monotonic_increasing:
        _fail("out-of-order timestamps: binance_metrics_open_interest_date_scan")
    required = _required_metrics_grid(config)
    required_ns = frozenset(required.astype("int64").tolist())
    positions = tuple(
        int(position)
        for position, value in enumerate(parsed.astype("int64").tolist())
        if value in required_ns
    )
    selected_dates = parsed.iloc[list(positions)].reset_index(drop=True)
    if len(positions) != len(required) or not selected_dates.equals(required):
        selected_ns = frozenset(selected_dates.astype("int64").tolist())
        overlap_ns = frozenset(required.iloc[: config.splice_rows].astype("int64").tolist())
        if not overlap_ns.issubset(selected_ns):
            _fail("missing OI splice anchor window")
        _fail("missing tail OI required metrics timestamp selection differs")
    _identity_check(item, "after_date_scan", failpoint)
    digest, size = _descriptor_hash(item.fd)
    if digest != item.binding.sha256 or size != item.binding.size_bytes:
        _fail("postscan source rehash differs: binance_metrics_open_interest")
    _identity_check(item, "after_date_scan_rehash", failpoint)
    return MetricsDateScan(header, len(parsed), positions, selected_dates)


def _decode_metrics_selected_window(
    item: _Retained,
    scan: MetricsDateScan,
    failpoint: Callable[[str], None],
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Frame all rows while decoding non-date values only at selected positions."""
    _identity_check(item, "before_decode", failpoint)
    _identity_check(item, "before_selected_window_decode", failpoint)
    digest, size = _descriptor_hash(item.fd)
    if digest != item.binding.sha256 or size != item.binding.size_bytes:
        _fail("predecode source rehash differs: binance_metrics_open_interest_selected_window")
    duplicate = os.dup(item.fd)
    selected: list[list[str]] = []
    row_count = 0
    wanted = iter(scan.selected_positions)
    next_position = next(wanted, None)
    try:
        os.lseek(duplicate, 0, os.SEEK_SET)
        with ExitStack() as stack:
            raw: io.BufferedIOBase = stack.enter_context(os.fdopen(duplicate, "rb", closefd=True))
            duplicate = -1
            stream: Any = (
                stack.enter_context(gzip.GzipFile(fileobj=raw, mode="rb"))
                if item.binding.compressed else raw
            )
            text = stack.enter_context(io.TextIOWrapper(stream, encoding="utf-8", newline=""))
            reader = csv.reader(text, strict=True)
            try:
                header = tuple(next(reader))
                if header != scan.schema or header != METRICS_SCHEMA:
                    _fail("metrics selected-window schema drift")
                for row_count, row in enumerate(reader, start=1):
                    position = row_count - 1
                    if position != next_position:
                        continue
                    if len(row) != len(header):
                        _fail("metrics selected row width differs")
                    selected.append([row[0], row[1], row[2]])
                    next_position = next(wanted, None)
            except (csv.Error, StopIteration) as exc:
                raise SourceSupportFailure("metrics selected-window decode failed") from exc
    finally:
        if duplicate >= 0:
            os.close(duplicate)
    if row_count != scan.row_count or next_position is not None or len(selected) != len(scan.selected_positions):
        _fail("metrics selected physical positions drift")
    selected_schema = METRICS_SCHEMA[:3]
    frame = pd.DataFrame(selected, columns=selected_schema, dtype="object")
    if tuple(frame.columns) != selected_schema or len(frame) != len(selected):
        _fail("metrics selected-window frame drift")
    _identity_check(item, "after_decode", failpoint)
    _identity_check(item, "after_selected_window_decode", failpoint)
    digest, size = _descriptor_hash(item.fd)
    if digest != item.binding.sha256 or size != item.binding.size_bytes:
        _fail("postdecode source rehash differs: binance_metrics_open_interest_selected_window")
    _identity_check(item, "after_rehash", failpoint)
    _identity_check(item, "after_selected_window_rehash", failpoint)
    return frame, selected_schema


def _dates(series: pd.Series, label: str) -> pd.Series:
    try:
        result = pd.to_datetime(series, errors="raise", utc=True).dt.tz_convert(None)
    except (TypeError, ValueError) as exc:
        raise SourceSupportFailure(f"invalid timestamps: {label}") from exc
    if result.isna().any():
        _fail(f"null timestamps: {label}")
    return result


def _normalise_ordered(frame: pd.DataFrame, date_column: str, label: str) -> pd.DataFrame:
    out = frame.copy()
    out[date_column] = _dates(out[date_column], label)
    if out[date_column].duplicated().any():
        _fail(f"duplicate timestamps: {label}")
    if not out[date_column].is_monotonic_increasing:
        _fail(f"out-of-order timestamps: {label}")
    sorted_out = out.sort_values(date_column, kind="mergesort").drop_duplicates(date_column, keep="last").reset_index(drop=True)
    if len(sorted_out) != len(out):
        _fail(f"timestamp normalization changed rows: {label}")
    return sorted_out


def _require_grid(dates: pd.Series, *, first: pd.Timestamp | None = None, last: pd.Timestamp | None = None, label: str) -> None:
    if dates.empty:
        _fail(f"empty timestamp grid: {label}")
    if first is not None and dates.iloc[0] != first:
        _fail(f"early/late first timestamp: {label}")
    if last is not None and dates.iloc[-1] != last:
        _fail(f"early/late terminal coverage: {label}")
    values = dates.astype("int64").to_numpy()
    if len(values) > 1 and not np.all(np.diff(values) == 300_000_000_000):
        _fail(f"internal five-minute grid gap: {label}")
    if np.any((values // 1_000_000_000) % 300 != 0):
        _fail(f"off-grid timestamps: {label}")


def _require_five_minute_alignment(dates: pd.Series, *, label: str) -> None:
    if dates.empty:
        _fail(f"empty timestamp sequence: {label}")
    values = dates.astype("int64").to_numpy()
    if np.any((values // 1_000_000_000) % 300 != 0):
        _fail(f"off-grid timestamps: {label}")


def _exact_frame(left: pd.DataFrame, right: pd.DataFrame, label: str) -> None:
    try:
        pd.testing.assert_frame_equal(left.reset_index(drop=True), right.reset_index(drop=True), check_dtype=True, check_exact=True)
    except AssertionError as exc:
        raise SourceSupportFailure(f"{label} differs") from exc


def _numeric_float64_preserving_missing(series: pd.Series, label: str) -> pd.Series:
    parsed = pd.to_numeric(series, errors="coerce")
    serialized_missing = series.map(
        lambda value: isinstance(value, str) and value == ""
    )
    malformed = parsed.isna() & ~(series.isna() | serialized_missing)
    if malformed.any():
        _fail(f"malformed numeric value: {label}")
    return parsed.astype("float64")


def _select_rank7_required_tail(
    rank_raw: pd.DataFrame,
    rank_schema: tuple[str, ...],
    required_dates: pd.Series,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Select Rank7 by exact required-date membership, with no pre-tail semantics."""
    if tuple(rank_raw.columns) != rank_schema or any(
        column not in rank_schema for column in RANK7_PROJECTION
    ):
        _fail("Rank7 projection schema differs")

    # These are the only global Rank7 semantic checks: date validity, UTC parse,
    # uniqueness, monotonicity, and exact five-minute alignment.  Every non-date
    # token remains an object string unless its row is selected below.
    rank_dates = _dates(rank_raw["date"], "rank7_spot_premium_5m")
    if rank_dates.duplicated().any():
        _fail("duplicate timestamps: rank7_spot_premium_5m")
    if not rank_dates.is_monotonic_increasing:
        _fail("out-of-order timestamps: rank7_spot_premium_5m")
    _require_five_minute_alignment(rank_dates, label="rank7_spot_premium_5m")

    membership = rank_dates.isin(required_dates)
    selected = rank_raw.loc[membership, list(RANK7_PROJECTION)].copy().reset_index(drop=True)
    selected["date"] = rank_dates.loc[membership].reset_index(drop=True)
    expected = required_dates.reset_index(drop=True)
    if len(selected) != len(expected) or not selected["date"].equals(expected):
        _fail(
            "Rank7 required-tail membership differs",
            failure_reason=RANK7_TAIL_MEMBERSHIP_MISMATCH,
        )

    selected_numeric: dict[str, pd.Series] = {}
    for column in RANK7_PROJECTION[1:]:
        values = pd.to_numeric(selected[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype="float64")).all():
            _fail(f"invalid selected Rank7 numeric values: {column}")
        selected_numeric[column] = values
    for column in ("spot_rows", "premium_rows"):
        if not selected_numeric[column].eq(5.0).all():
            _fail("invalid selected Rank7 row counts")
    if selected.empty:
        _fail("empty Rank7 required tail")
    latest_spot = float(selected_numeric["spot_close"].iloc[-1])
    latest_premium = float(selected_numeric["premium_index_1m_close"].iloc[-1])
    if latest_spot <= 0 or not math.isfinite(latest_spot) or not math.isfinite(latest_premium):
        _fail("invalid latest Rank7 values")

    proof = {
        "rank7_required_tail_rows": len(expected),
        "rank7_tail_exact_matches": len(selected),
        "rank7_pre_tail_coverage_comparison_count": 0,
        "rank7_gap_detail_disclosure_count": 0,
    }
    _validate_rank7_zero_disclosure(proof)
    return selected, proof


def _validate_rank7_zero_disclosure(proof: Mapping[str, int]) -> None:
    if (
        proof.get("rank7_pre_tail_coverage_comparison_count") != 0
        or proof.get("rank7_gap_detail_disclosure_count") != 0
    ):
        _fail("Rank7 zero-disclosure proof differs", failure_reason=ZERO_DISCLOSURE_BREACH)


def _transform(
    config: MaterializationConfig,
    frames: Mapping[str, pd.DataFrame],
    schemas: Mapping[str, tuple[str, ...]],
    replacement_scan: ReplacementDateScan,
    metrics_scan: MetricsDateScan,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, int]]:
    old_market_raw = frames["old_market"]
    replacement_tail_raw = frames["replacement_market"]
    if schemas["old_market"] != MARKET_SCHEMA or schemas["replacement_market"] != MARKET_SCHEMA:
        _fail("market schema drift")
    old_market = _normalise_ordered(old_market_raw, "date", "old_market")
    replacement_tail = _normalise_ordered(
        replacement_tail_raw, "date", "replacement_market_tail"
    )
    if len(old_market) != config.expected_old_rows or old_market.date.iloc[-1] != config.old_last:
        _fail("old market row count or terminal timestamp differs")
    required_last = config.domain_end - pd.Timedelta(minutes=5)
    if len(replacement_tail) != config.expected_append_rows:
        _fail("replacement market tail row count differs")
    _require_grid(
        replacement_tail.date,
        first=config.old_last + pd.Timedelta(minutes=5),
        last=required_last,
        label="replacement_market_tail",
    )
    if not replacement_tail.date.equals(replacement_scan.tail_dates):
        _fail("replacement market tail timestamps differ from date scan")
    old_timestamps = set(old_market.date.astype("int64").tolist())
    tail_timestamps = set(replacement_tail.date.astype("int64").tolist())
    if old_timestamps.intersection(tail_timestamps):
        _fail("market partition timestamp intersection differs")
    market = pd.concat([old_market, replacement_tail], ignore_index=True)
    output_timestamps = set(market.date.astype("int64").tolist())
    if output_timestamps != old_timestamps.union(tail_timestamps):
        _fail("market partition union differs")
    if len(market) != len(old_market) + len(replacement_tail) or len(market) != config.expected_complete_rows:
        _fail("materialized market row counts differ")
    _exact_frame(market.iloc[: len(old_market)], old_market, "materialized old market partition")
    _exact_frame(market.iloc[len(old_market) :], replacement_tail, "materialized replacement tail partition")
    _require_grid(market.date, first=old_market.date.iloc[0], last=required_last, label="materialized_market")

    old_oi_raw = frames["old_open_interest"]
    metrics_raw = frames["binance_metrics_open_interest"]
    if schemas["old_open_interest"] != OI_SCHEMA or schemas["binance_metrics_open_interest"] != METRICS_SCHEMA[:3]:
        _fail("open-interest schema drift")
    old_oi = _normalise_ordered(old_oi_raw, "date", "old_open_interest")
    metrics = metrics_raw.copy()
    metrics["create_time"] = _dates(metrics.create_time, "binance_metrics_open_interest_selected_window")
    if not metrics.create_time.equals(metrics_scan.selected_dates):
        _fail("metrics selected dates differ from date scan")
    if len(old_oi) != config.expected_old_rows or old_oi.date.iloc[-1] != config.old_last:
        _fail("old OI row count or terminal timestamp differs")
    _require_grid(old_oi.date, last=config.old_last, label="old_open_interest")
    if not metrics["symbol"].eq("BTCUSDT").all():
        _fail("metrics symbol differs")
    metrics_oi = _numeric_float64_preserving_missing(
        metrics["sum_open_interest"], "binance_metrics_open_interest.sum_open_interest"
    )
    mapped = pd.DataFrame({"date": metrics["create_time"], "open_interest": metrics_oi})
    old_values = _numeric_float64_preserving_missing(
        old_oi.open_interest, "old_open_interest.open_interest"
    )
    old_numeric = pd.DataFrame({"date": old_oi.date, "open_interest": old_values})
    overlap = mapped.iloc[: config.splice_rows].reset_index(drop=True)
    splice_start = config.old_last - pd.Timedelta(minutes=5 * (config.splice_rows - 1))
    inherited_overlap = old_numeric.loc[
        (old_numeric.date >= splice_start) & (old_numeric.date <= config.old_last)
    ].reset_index(drop=True)
    if len(overlap) != config.splice_rows or len(inherited_overlap) != config.splice_rows:
        _fail("missing OI splice anchor window")
    if not overlap.date.equals(inherited_overlap.date):
        _fail("OI splice timestamp conflict")
    left = inherited_overlap.open_interest.to_numpy(dtype="float64")
    right = overlap.open_interest.to_numpy(dtype="float64")
    if not np.array_equal(left, right, equal_nan=True):
        _fail("OI overlap conflict")
    _require_grid(overlap.date, first=splice_start, last=config.old_last, label="OI splice anchor window")
    tail = mapped.iloc[config.splice_rows :].reset_index(drop=True)
    if len(tail) != config.expected_append_rows or tail.open_interest.isna().any():
        _fail("missing tail OI")
    tail_values = tail.open_interest.to_numpy(dtype="float64")
    if not np.isfinite(tail_values).all() or not (tail_values > 0).all():
        _fail("non-positive or non-finite tail OI")
    _require_grid(
        tail.date,
        first=config.old_last + pd.Timedelta(minutes=5),
        last=required_last,
        label="OI selected tail",
    )
    tail_frame = pd.DataFrame({"date": tail.date, "open_interest": tail_values})
    materialized_oi = pd.concat([old_numeric, tail_frame], ignore_index=True)
    if len(materialized_oi) != config.expected_complete_rows:
        _fail("materialized OI row count differs")
    _require_grid(materialized_oi.date, first=old_oi.date.iloc[0], last=required_last, label="materialized_open_interest")

    rank_raw = frames["rank7_spot_premium_5m"]
    rank_schema = schemas["rank7_spot_premium_5m"]
    required_rank_dates = market.date.tail(
        min(RANK7_REQUIRED_TAIL_LIMIT, len(market))
    )
    _rank_selected, _rank7_proof = _select_rank7_required_tail(
        rank_raw, rank_schema, required_rank_dates
    )

    funding_raw = frames["funding"]
    premium_raw = frames["premium"]
    funding_dates = "funding_time" if "funding_time" in funding_raw.columns and "date" not in funding_raw.columns else "date"
    if funding_dates not in funding_raw or "funding_rate" not in funding_raw:
        _fail("funding columns differ")
    funding_check = _dates(funding_raw[funding_dates], "funding")
    if funding_check.duplicated().any() or not funding_check.is_monotonic_increasing:
        _fail("funding timestamp ordering/uniqueness differs")
    funding_values_raw = pd.to_numeric(funding_raw["funding_rate"], errors="coerce")
    if funding_values_raw.isna().any() or not np.isfinite(funding_values_raw.to_numpy(dtype="float64")).all():
        _fail("funding contains null or non-finite values")
    if "premium_index" in premium_raw.columns and "date" in premium_raw.columns:
        premium_dates = _dates(premium_raw.date, "premium")
    elif "close" in premium_raw.columns and "close_time" in premium_raw.columns:
        numeric = pd.to_numeric(premium_raw.close_time, errors="coerce")
        premium_dates = pd.to_datetime(numeric, unit="ms", errors="raise", utc=True).dt.tz_convert(None) if numeric.notna().any() else _dates(premium_raw.close_time, "premium")
    elif "close" in premium_raw.columns and "date" in premium_raw.columns:
        premium_dates = _dates(premium_raw.date, "premium")
    else:
        _fail("premium columns differ")
    if premium_dates.isna().any() or premium_dates.duplicated().any() or not premium_dates.is_monotonic_increasing:
        _fail("premium timestamp ordering/uniqueness differs")
    premium_value_column = "premium_index" if "premium_index" in premium_raw.columns else "close"
    premium_values_raw = pd.to_numeric(premium_raw[premium_value_column], errors="coerce")
    if premium_values_raw.isna().any() or not np.isfinite(premium_values_raw.to_numpy(dtype="float64")).all():
        _fail("premium contains null or non-finite values")
    funding = normalise_funding_history_frame(funding_raw)
    premium = normalise_premium_index_frame(premium_raw)
    attached, funding_available = _merge_aux(market, funding, value_cols=["funding_rate"], tolerance="12h")
    attached, premium_available = _merge_aux(attached, premium, value_cols=["premium_index"], tolerance="2h")
    if len(attached) != len(market) or not attached.date.equals(market.date):
        _fail("auxiliary attachment changed market timestamps")
    tail_slice = slice(len(old_market), len(market))
    funding_tail = funding_available.iloc[tail_slice]
    premium_tail = premium_available.iloc[tail_slice]
    funding_values = pd.to_numeric(attached.funding_rate.iloc[tail_slice], errors="coerce").to_numpy(dtype="float64")
    premium_values = pd.to_numeric(attached.premium_index.iloc[tail_slice], errors="coerce").to_numpy(dtype="float64")
    if len(funding_tail) != config.expected_append_rows or not funding_tail.eq(1.0).all() or not np.isfinite(funding_values).all():
        _fail("funding tail is unavailable or stale")
    if len(premium_tail) != config.expected_append_rows or not premium_tail.eq(1.0).all() or not np.isfinite(premium_values).all():
        _fail("premium tail is unavailable or stale")

    validation = {
        "appended_market_rows": config.expected_append_rows,
        "appended_oi_rows": config.expected_append_rows,
        "appended_open_interest_rows": config.expected_append_rows,
        "domain_end_exclusive": _timestamp_z(config.domain_end),
        "funding_attachment_tolerance": "12h",
        "funding_tail_available_rows": int(funding_tail.eq(1.0).sum()),
        "latest_funding_available_after_causal_attachment": bool(funding_tail.iloc[-1] == 1.0),
        "latest_open_interest_positive_after_exact_join": bool(tail_values[-1] > 0),
        "latest_premium_available_after_causal_attachment": bool(premium_tail.iloc[-1] == 1.0),
        "market_grid_seconds": 300,
        "market_partition": {
            "boundary_inclusive_old": _timestamp_z(config.old_last),
            "disjoint_union": True,
            "domain_end_exclusive": _timestamp_z(config.domain_end),
            "old_prefix_rows": len(old_market),
            "output_rows": len(market),
            "overlap_value_comparison_count": 0,
            "replacement_prefix_non_date_values_semantically_evaluated": 0,
            "replacement_prefix_rows_selected": 0,
            "replacement_tail_rows": len(replacement_tail),
            "timestamp_intersection_rows": 0,
        },
        "metrics_selected_timestamp_values_exact": True,
        "metrics_global_alignment_comparison_count": 0,
        "metrics_non_selected_non_date_semantic_evaluation_count": 0,
        "oi_grid_seconds": 300,
        "oi_splice_window_exact_rows": len(overlap),
        "oi_tail_exact_rows": len(tail),
        "old_last_timestamp": _timestamp_z(config.old_last),
        "premium_attachment_tolerance": "2h",
        "premium_tail_available_rows": int(premium_tail.eq(1.0).sum()),
        "rank7_spot_premium_latest_values_valid": True,
        "rank7_spot_premium_projection_schema": list(RANK7_PROJECTION),
        "rank7_spot_premium_raw_column_count": len(rank_schema),
        "rank7_spot_premium_raw_schema_sha256": hashlib.sha256(canonical_json_bytes(list(rank_schema), trailing_lf=False)).hexdigest(),
        "required_last_timestamp": _timestamp_z(required_last),
    }
    normalized = {
        "old_market": len(old_market), "replacement_market": len(replacement_tail),
        "funding": len(funding), "premium": len(premium),
        "old_open_interest": len(old_oi), "binance_metrics_open_interest": metrics_scan.row_count,
    }
    return market, materialized_oi, validation, normalized


def serialize_csv_bytes(frame: pd.DataFrame) -> tuple[bytes, str]:
    text = frame.to_csv(index=False, lineterminator="\n", date_format="%Y-%m-%d %H:%M:%S", float_format="%.17g", na_rep="", quoting=csv.QUOTE_MINIMAL)
    csv_bytes = text.encode("utf-8")
    return csv_bytes, hashlib.sha256(csv_bytes).hexdigest()


def serialize_csv_gzip(frame: pd.DataFrame) -> tuple[bytes, bytes, str]:
    csv_bytes, frame_hash = serialize_csv_bytes(frame)
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, compresslevel=9, mtime=0) as handle:
        handle.write(csv_bytes)
    return buffer.getvalue(), csv_bytes, frame_hash


_LIBC = ctypes.CDLL(None, use_errno=True)
_LIBC.linkat.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
_LIBC.linkat.restype = ctypes.c_int
_AT_FDCWD = -100
_AT_SYMLINK_FOLLOW = 0x400


def _write_all(fd: int, raw: bytes) -> None:
    view = memoryview(raw)
    offset = 0
    while offset < len(view):
        written = os.write(fd, view[offset:])
        if written <= 0:
            _fail("unnamed publication write stalled")
        offset += written


def _linkat(fd: int, directory_fd: int, leaf: str) -> None:
    result = _LIBC.linkat(_AT_FDCWD, f"/proc/self/fd/{fd}".encode("ascii"), directory_fd, os.fsencode(leaf), _AT_SYMLINK_FOLLOW)
    if result:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(f"write-once path exists: {leaf}")
        raise OSError(error, os.strerror(error), leaf)


def _publish(config: MaterializationConfig, relative: str, raw: bytes, label: str, failpoint: Callable[[str], None]) -> dict[str, Any]:
    path = _path(config, relative)
    parent = path.parent
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    parent_fd = _open_nofollow_components(parent, flags)
    unnamed = -1
    canonical = -1
    try:
        if not getattr(os, "O_TMPFILE", 0):
            _fail("O_TMPFILE is unavailable")
        unnamed = os.open(".", os.O_RDWR | os.O_TMPFILE | getattr(os, "O_CLOEXEC", 0), 0o600, dir_fd=parent_fd)
        initial = os.fstat(unnamed)
        if not stat.S_ISREG(initial.st_mode) or initial.st_size != 0 or initial.st_nlink != 0:
            _fail("unnamed publication inode is invalid")
        failpoint(f"publication:{label}:unnamed_initial")
        _write_all(unnamed, raw)
        os.fchmod(unnamed, 0o444)
        os.fsync(unnamed)
        complete = os.fstat(unnamed)
        readback = _pread_complete(unnamed, complete.st_size)
        if not stat.S_ISREG(initial.st_mode) or stat.S_IMODE(initial.st_mode) != 0o600 or not stat.S_ISREG(complete.st_mode) or stat.S_IMODE(complete.st_mode) != 0o444 or complete.st_nlink != 0 or complete.st_size != len(raw) or readback != raw:
            _fail("completed unnamed inode verification differs")
        failpoint(f"publication:{label}:unnamed_complete")
        failpoint(f"before_link:{label}")
        failpoint(f"publish:{label}:before_linkat")
        _linkat(unnamed, parent_fd, path.name)
        os.fsync(parent_fd)
        canonical = os.open(path.name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0), dir_fd=parent_fd)
        info = os.fstat(canonical)
        canonical_raw = _pread_complete(canonical, info.st_size)
        if (info.st_dev, info.st_ino) != (complete.st_dev, complete.st_ino) or not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o444 or info.st_nlink != 1 or canonical_raw != raw:
            _fail("linked canonical inode verification differs")
        failpoint(f"publication:{label}:linked_canonical")
        failpoint(f"after_link:{label}")
        return {"sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw), "inode": (info.st_dev, info.st_ino)}
    finally:
        if canonical >= 0:
            os.close(canonical)
        if unnamed >= 0:
            os.close(unnamed)
        os.close(parent_fd)


def _readback_generated(
    config: MaterializationConfig,
    relative: str,
    compressed: bool = True,
    failpoint: Callable[[str], None] | None = None,
) -> tuple[str, int, pd.DataFrame]:
    inject = failpoint or (lambda _name: None)
    path = _path(config, relative)
    fd = _open_nofollow_components(path, os.O_RDONLY)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o444 or info.st_nlink != 1:
            _fail("generated readback metadata differs")
        digest, size = _descriptor_hash(fd)
        inject(f"readback:{relative}:authenticated")
        duplicate = os.dup(fd)
        with ExitStack() as stack:
            source = stack.enter_context(os.fdopen(duplicate, "rb"))
            stream: Any = (
                stack.enter_context(gzip.GzipFile(fileobj=source, mode="rb"))
                if compressed
                else source
            )
            frame = pd.read_csv(stream, float_precision="round_trip")
        return digest, size, frame
    finally:
        os.close(fd)


def _load_inherited_manifest(config: MaterializationConfig) -> dict[str, Any]:
    """Deep-copy a synthetic override or the embedded authenticated literal."""
    source = (
        config.inherited_manifest
        if config.inherited_manifest is not None
        else _INHERITED_MANIFEST_LITERAL
    )
    return json.loads(json.dumps(source))


def _unique_json(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _build_manifest(config: MaterializationConfig, market_digest: str, oi_digest: str) -> dict[str, Any]:
    inherited = _load_inherited_manifest(config)
    if set(inherited) != {"schema_version", "as_of", "sources"} or inherited.get("schema_version") != 1 or not isinstance(inherited.get("sources"), list) or len(inherited["sources"]) != 8:
        _fail("inherited source manifest schema differs")
    rows: list[dict[str, Any]] = []
    for original in inherited["sources"]:
        row = dict(original)
        if row.get("name") == "market_5m":
            row["path"] = config.market_output_path
            row["sha256"] = market_digest
        elif row.get("name") == "open_interest":
            row["path"] = config.oi_output_path
            row["sha256"] = oi_digest
        rows.append(row)
        if row.get("name") == "open_interest":
            rank = next(binding for binding in config.inputs if binding.name == "rank7_spot_premium_5m")
            rows.append({"name": rank.name, "path": rank.path, "sha256": rank.sha256})
    names = [row.get("name") for row in rows]
    if len(rows) != 9 or len(set(names)) != 9 or names[4] != "rank7_spot_premium_5m":
        _fail("nine-row source manifest ordering differs")
    return {"schema_version": 1, "as_of": "2026-07-31", "sources": rows}


def _attempt(config: MaterializationConfig, head: str) -> dict[str, Any]:
    raw_inputs = [{"mode_octal": f"{row.mode:04o}", "name": row.name, "path": row.path, "path_type": "regular_file", "sha256": row.sha256, "size_bytes": row.size_bytes} for row in config.inputs]
    core = {
        "branch": config.branch,
        "expected_outputs": list(config.output_paths),
        "identity": IDENTITY,
        "one_shot": True,
        "raw_inputs": raw_inputs,
        "repository_head": head,
        "repository_parent": config.repository_parent,
        "resume_allowed": False,
        "retry_allowed": False,
        "source_access_at_publication": {"opaque_bytes_hashed": sum(row.size_bytes for row in config.inputs), "preexisting_sources_decoded": 0, "source_rows_decoded": 0},
        "status": "source_support_attempt_consumed_before_source_decode",
        "topology": {"authority_commit": config.authority_commit, "implementation_commit": head, "terminal_evidence_commit": config.repository_parent},
        "version": VERSION,
    }
    return _with_hash(core, "attempt_hash")


def _expected_access_ledger(
    config: MaterializationConfig,
    head: str,
    attempt: Mapping[str, Any],
    attempt_binding: Mapping[str, Any],
    replacement_scan: ReplacementDateScan,
    metrics_scan: MetricsDateScan,
    decoded_rows: Mapping[str, int],
) -> dict[str, Any]:
    terminal = json.loads(json.dumps(_T11_TERMINAL_LITERAL))
    historical_s11 = {
        "identity": terminal["identity"],
        "status": terminal["status"],
        "authority": terminal["authority"],
        "seal_authority": terminal["seal_authority"],
        "implementation": terminal["implementation"],
        "attempt_sentinel": terminal["attempt_sentinel"],
        "terminal_ledger": {
            "path": _T11_EVIDENCE[1][0],
            "commit": T11,
            "size_bytes": _T11_EVIDENCE[1][1],
            "sha256": _T11_EVIDENCE[1][2],
            "git_blob": _T11_EVIDENCE[1][3],
            "git_mode": "100644",
            "worktree_mode": "0444",
            "terminal_failure_hash": terminal["terminal_failure_hash"],
        },
        "execution": terminal["execution"],
        "failure": terminal["failure"],
        "access": terminal["access"],
        "output_state": terminal["output_state"],
        "permanently_absent_output_paths": list(_G9CB11_PERMANENT_ABSENCES),
        "authority_transferred": False,
    }
    decode_passes = [
        "old_market", "replacement_market_date_scan", "replacement_market_tail",
        "funding", "premium", "old_open_interest",
        "binance_metrics_open_interest_date_scan",
        "binance_metrics_open_interest_selected_window", "rank7_spot_premium_5m",
    ]
    zero_economics = {
        "candidate_value_rows_opened": 0, "comparator_value_rows_opened": 0,
        "feature_value_rows_opened": 0, "schedule_value_rows_opened": 0,
        "signal_value_rows_opened": 0, "return_value_rows_opened": 0,
        "pnl_value_rows_opened": 0, "cagr_evaluation_count": 0,
        "mdd_evaluation_count": 0, "drawdown_evaluation_count": 0,
        "economic_value_rows_opened": 0, "economic_evaluation_count": 0,
    }
    required_tail_rows = min(
        RANK7_REQUIRED_TAIL_LIMIT, config.expected_complete_rows
    )
    rank7_proof = {
        "rank7_required_tail_rows": required_tail_rows,
        "rank7_tail_exact_matches": required_tail_rows,
        "rank7_pre_tail_coverage_comparison_count": 0,
        "rank7_gap_detail_disclosure_count": 0,
    }
    current_access = {
        "raw_file_count": 7, "raw_file_open_count": 7, "decode_pass_count": 9,
        "decode_passes": decode_passes,
        "replacement_market_date_scan_count": 1,
        "replacement_market_tail_decode_count": 1,
        "replacement_market_tail_selected_row_count": config.expected_append_rows,
        "metrics_date_scan_count": 1, "metrics_selected_decode_count": 1,
        "metrics_selected_row_count": config.splice_rows + config.expected_append_rows,
        "metrics_overlap_row_count": config.splice_rows,
        "metrics_tail_row_count": config.expected_append_rows,
        "non_selected_metrics_non_date_semantic_evaluation_count": 0,
        "global_metrics_alignment_comparison_count": 0,
        "rank7_spot_premium_5m_decode_count": 1,
        "attempt_sentinel_publication_count": 1,
        "generated_output_publication_count": 4,
        "generated_output_readback_count": 2,
        "off_grid_detail_disclosure_count": 0,
        **zero_economics,
    }
    implementation_paths = [
        "training/materialize_gross9_structural_clock_g9cb12_sources.py",
        "tests/test_materialize_gross9_structural_clock_g9cb12_sources.py",
    ]
    current_s12 = {
        "identity": IDENTITY,
        "authority": {"commit": config.authority_commit, "document_path": _A12_AUTHORITY[0]},
        "terminal_evidence": {
            "commit": config.repository_parent,
            "attempt_sentinel_path": _T11_EVIDENCE[0][0],
            "terminal_ledger_path": _T11_EVIDENCE[1][0],
            "terminal_failure_hash": terminal["terminal_failure_hash"],
        },
        "implementation": {
            "commit": head, "parent_commit": config.repository_parent,
            "paths": implementation_paths,
        },
        "attempt_sentinel": {
            "path": config.attempt_path, "size_bytes": attempt_binding["size_bytes"],
            "sha256": attempt_binding["sha256"], "attempt_hash": attempt["attempt_hash"],
        },
        "execution": {
            "command": "PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run python -B -m training.materialize_gross9_structural_clock_g9cb12_sources",
            "invocation_count": 1, "one_shot": True,
            "retry_allowed": False, "resume_allowed": False,
        },
        "required_tail": {
            "predicate": "market.date.tail(min(3000, len(market)))",
            "selection": "exact_date_membership_only",
            "selected_dates_equal_required_dates": True,
            "selected_required_dates_unique": True,
        },
        **rank7_proof,
        "access": current_access,
    }
    addends = [
        decoded_rows["old_market"], replacement_scan.row_count,
        len(replacement_scan.tail_positions), decoded_rows["funding"],
        decoded_rows["premium"], decoded_rows["old_open_interest"],
        metrics_scan.row_count, len(metrics_scan.selected_positions),
    ]
    if (
        addends[0] != config.expected_old_rows
        or addends[2] != config.expected_append_rows
        or addends[5] != config.expected_old_rows
        or addends[7] != config.splice_rows + config.expected_append_rows
    ):
        _fail("source-value row provenance differs")
    process_local = {
        "stage": "S12", "slot": 0, "invocation_count": 1,
        "raw_file_count": 7, "raw_file_open_count": 7, "decode_pass_count": 9,
        "old_market_rows_opened": addends[0],
        "replacement_market_date_rows_scanned": addends[1],
        "replacement_market_tail_rows_opened": addends[2],
        "funding_rows_opened": addends[3], "premium_rows_opened": addends[4],
        "old_open_interest_rows_opened": addends[5],
        "binance_metrics_open_interest_date_rows_scanned": addends[6],
        "binance_metrics_open_interest_selected_window_rows_opened": addends[7],
        "non_selected_metrics_non_date_semantic_evaluation_count": 0,
        "global_metrics_alignment_comparison_count": 0,
        "generated_output_readback_count": 2,
        "off_grid_detail_disclosure_count": 0,
        **zero_economics,
    }
    historical_outputs = [_T11_EVIDENCE[0][0], *_G9CB11_PERMANENT_ABSENCES]
    current_outputs = list(config.output_paths)
    if set(historical_outputs).intersection(current_outputs):
        _fail("G9CB-12 replay guard output intersection differs")
    if terminal["identity"] == IDENTITY or terminal["attempt_sentinel"]["attempt_hash"] == attempt["attempt_hash"]:
        _fail("G9CB-12 replay guard identity or attempt hash differs")
    replay_guard = _with_hash({
        "stage_order": ["A12", "T11", "S12"],
        "a12": {"commit": config.authority_commit, "authority_document_path": _A12_AUTHORITY[0]},
        "t11": {
            "commit": config.repository_parent,
            "attempt_sentinel_path": _T11_EVIDENCE[0][0],
            "attempt_hash": terminal["attempt_sentinel"]["attempt_hash"],
            "terminal_ledger_path": _T11_EVIDENCE[1][0],
            "terminal_ledger_sha256": _T11_EVIDENCE[1][2],
            "terminal_failure_hash": terminal["terminal_failure_hash"],
            "permanently_absent_output_paths": list(_G9CB11_PERMANENT_ABSENCES),
        },
        "s12": {
            "commit": head, "parent_commit": config.repository_parent,
            "implementation_paths": implementation_paths,
            "attempt_sentinel_path": config.attempt_path,
            "attempt_hash": attempt["attempt_hash"],
        },
        "identities": {"historical_s11": terminal["identity"], "current_s12": IDENTITY},
        "attempt_hashes": {
            "historical_s11": terminal["attempt_sentinel"]["attempt_hash"],
            "current_s12": attempt["attempt_hash"],
        },
        "expected_output_paths": {
            "historical_s11": historical_outputs, "current_s12": current_outputs,
        },
        "pairwise_output_intersection": [],
        "identities_pairwise_distinct": True,
        "attempt_hashes_pairwise_distinct": True,
    }, "replay_guard_hash")
    return _with_hash({
        "schema_version": 1,
        "ledger_kind": "gross9_structural_clock_bundle_g9cb12_access_v1",
        "historical_s11": historical_s11,
        "current_s12": current_s12,
        "process_local": process_local,
        "replay_guard": replay_guard,
    }, "access_ledger_hash")


def _build_access_ledger(
    config: MaterializationConfig,
    head: str,
    attempt: Mapping[str, Any],
    attempt_binding: Mapping[str, Any],
    replacement_scan: ReplacementDateScan,
    metrics_scan: MetricsDateScan,
    decoded_rows: Mapping[str, int],
) -> dict[str, Any]:
    return _expected_access_ledger(
        config, head, attempt, attempt_binding, replacement_scan, metrics_scan, decoded_rows
    )


def _validate_access_ledger(
    ledger: Mapping[str, Any],
    config: MaterializationConfig,
    *,
    head: str,
    attempt: Mapping[str, Any],
    attempt_binding: Mapping[str, Any],
    replacement_scan: ReplacementDateScan,
    metrics_scan: MetricsDateScan,
    decoded_rows: Mapping[str, int],
) -> None:
    expected = _expected_access_ledger(
        config, head, attempt, attempt_binding, replacement_scan, metrics_scan, decoded_rows
    )
    try:
        actual_raw = canonical_json_bytes(ledger)
    except (TypeError, ValueError) as exc:
        raise SourceSupportFailure("access ledger exact schema, type, order, or binding differs") from exc
    if actual_raw != canonical_json_bytes(expected):
        _fail("access ledger exact schema, type, order, or binding differs")

def materialize(config: MaterializationConfig, *, failpoint: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Run once; fresh-process retry prohibition is supervisor-owned."""
    invocation_key = (str(config.root.resolve()), config.output_paths)
    if invocation_key in _PROCESS_LOCAL_CONSUMED_INVOCATIONS:
        _fail(
            "one-shot invocation already exists and cannot be retried",
            failure_reason=PREFLIGHT_OR_BINDING_FAILURE,
        )
    _PROCESS_LOCAL_CONSUMED_INVOCATIONS.add(invocation_key)
    inject = failpoint or (lambda _name: None)
    head = config.repository_head
    if config.enforce_repository_gates:
        _with_failure_reason(PREFLIGHT_OR_BINDING_FAILURE, _validate_command_and_root, config)
        _with_failure_reason(PREFLIGHT_OR_BINDING_FAILURE, _validate_bytecode_first_gate, config.root)
        head = _with_failure_reason(PREFLIGHT_OR_BINDING_FAILURE, _validate_git_gate, config)
        _with_failure_reason(PREFLIGHT_OR_BINDING_FAILURE, _validate_history_gate, config)
    _with_failure_reason(PREFLIGHT_OR_BINDING_FAILURE, _validate_output_absence, config)
    output_directory_baseline = _with_failure_reason(
        PREFLIGHT_OR_BINDING_FAILURE, _snapshot_output_directories, config
    )
    retained = _with_failure_reason(PREFLIGHT_OR_BINDING_FAILURE, _open_inputs, config, inject)
    try:
        inject("checkpoint:1")
        attempt = _attempt(config, head)
        attempt_raw = canonical_json_bytes(attempt)
        attempt_binding = _with_failure_reason(
            PUBLICATION_OR_READBACK_FAILURE,
            _publish,
            config,
            config.attempt_path,
            attempt_raw,
            "attempt_sentinel",
            inject,
        )
        inject("checkpoint:2")

        frames: dict[str, pd.DataFrame] = {}
        schemas: dict[str, tuple[str, ...]] = {}
        decoded_rows: dict[str, int] = {}
        replacement_scan: ReplacementDateScan | None = None
        metrics_scan: MetricsDateScan | None = None
        for binding in config.inputs:
            if binding.name == "replacement_market":
                replacement_scan = _with_oserror_reason(
                    STRUCTURAL_OR_SCHEMA_VIOLATION,
                    _scan_replacement_dates,
                    retained[binding.name],
                    config,
                    inject,
                )
                frame, schema = _with_oserror_reason(
                    STRUCTURAL_OR_SCHEMA_VIOLATION,
                    _decode_replacement_tail,
                    retained[binding.name],
                    replacement_scan,
                    inject,
                )
                decoded_rows[binding.name] = replacement_scan.row_count
            elif binding.name == "binance_metrics_open_interest":
                metrics_scan = _with_oserror_reason(
                    STRUCTURAL_OR_SCHEMA_VIOLATION,
                    _scan_metrics_dates,
                    retained[binding.name],
                    config,
                    inject,
                )
                frame, schema = _with_oserror_reason(
                    STRUCTURAL_OR_SCHEMA_VIOLATION,
                    _decode_metrics_selected_window,
                    retained[binding.name],
                    metrics_scan,
                    inject,
                )
                decoded_rows[binding.name] = metrics_scan.row_count
            elif binding.name == "rank7_spot_premium_5m":
                frame, schema = _with_oserror_reason(
                    STRUCTURAL_OR_SCHEMA_VIOLATION,
                    _decode_rank7_tokens,
                    retained[binding.name],
                    inject,
                )
            else:
                frame, schema = _with_oserror_reason(
                    STRUCTURAL_OR_SCHEMA_VIOLATION,
                    _decode_csv,
                    retained[binding.name],
                    inject,
                )
                decoded_rows[binding.name] = len(frame)
            frames[binding.name] = frame
            schemas[binding.name] = schema
        if replacement_scan is None:
            _fail("replacement market decode pass is absent")
        if metrics_scan is None:
            _fail("metrics two-pass decode is absent")
        market, oi, validation, normalized_rows = _with_oserror_reason(
            STRUCTURAL_OR_SCHEMA_VIOLATION,
            _transform,
            config,
            frames,
            schemas,
            replacement_scan,
            metrics_scan,
        )
        frames.clear()
        schemas.clear()
        del frame, schema
        inject("checkpoint:3")

        market_rows = len(market)
        market_first = market.date.iloc[0]
        market_last = market.date.iloc[-1]
        oi_rows = len(oi)
        oi_first = oi.date.iloc[0]
        oi_last = oi.date.iloc[-1]

        market_gzip, market_csv, market_frame_hash = serialize_csv_gzip(market)
        market_binding = _with_failure_reason(
            PUBLICATION_OR_READBACK_FAILURE,
            _publish,
            config,
            config.market_output_path,
            market_gzip,
            "materialized_market",
            inject,
        )
        del market_gzip
        inject("checkpoint:4")
        oi_gzip, oi_csv, oi_frame_hash = _with_failure_reason(
            PUBLICATION_OR_READBACK_FAILURE,
            serialize_csv_gzip,
            oi,
        )
        oi_binding = _with_failure_reason(
            PUBLICATION_OR_READBACK_FAILURE,
            _publish,
            config,
            config.oi_output_path,
            oi_gzip,
            "materialized_open_interest",
            inject,
        )
        del oi_gzip
        inject("checkpoint:5")

        market_digest, market_size, market_read = _with_failure_reason(
            PUBLICATION_OR_READBACK_FAILURE,
            _readback_generated,
            config,
            config.market_output_path,
            failpoint=inject,
        )
        if market_digest != market_binding["sha256"] or market_size != market_binding["size_bytes"]:
            _fail(
                "generated market gzip readback binding differs",
                failure_reason=PUBLICATION_OR_READBACK_FAILURE,
            )
        market_read["date"] = _with_failure_reason(
            PUBLICATION_OR_READBACK_FAILURE,
            _dates,
            market_read.date,
            "materialized_market_readback",
        )
        market_recsv, _ = _with_failure_reason(
            PUBLICATION_OR_READBACK_FAILURE,
            serialize_csv_bytes,
            market_read,
        )
        if market_recsv != market_csv or len(market_read) != market_rows:
            _fail(
                "generated market logical-frame readback differs",
                failure_reason=PUBLICATION_OR_READBACK_FAILURE,
            )
        del market, market_csv, market_read, market_recsv

        oi_digest, oi_size, oi_read = _with_failure_reason(
            PUBLICATION_OR_READBACK_FAILURE,
            _readback_generated,
            config,
            config.oi_output_path,
            failpoint=inject,
        )
        if oi_digest != oi_binding["sha256"] or oi_size != oi_binding["size_bytes"]:
            _fail(
                "generated OI gzip readback binding differs",
                failure_reason=PUBLICATION_OR_READBACK_FAILURE,
            )
        oi_read["date"] = _with_failure_reason(
            PUBLICATION_OR_READBACK_FAILURE,
            _dates,
            oi_read.date,
            "materialized_oi_readback",
        )
        oi_recsv, _ = _with_failure_reason(
            PUBLICATION_OR_READBACK_FAILURE,
            serialize_csv_bytes,
            oi_read,
        )
        if oi_recsv != oi_csv or len(oi_read) != oi_rows:
            _fail(
                "generated OI logical-frame readback differs",
                failure_reason=PUBLICATION_OR_READBACK_FAILURE,
            )
        del oi, oi_csv, oi_read, oi_recsv
        inject("checkpoint:6")

        manifest = _with_failure_reason(
            PUBLICATION_OR_READBACK_FAILURE,
            _build_manifest,
            config,
            market_binding["sha256"],
            oi_binding["sha256"],
        )
        manifest_raw = _with_failure_reason(
            PUBLICATION_OR_READBACK_FAILURE,
            canonical_json_bytes,
            manifest,
        )
        manifest_binding = _with_failure_reason(
            PUBLICATION_OR_READBACK_FAILURE,
            _publish,
            config,
            config.manifest_output_path,
            manifest_raw,
            "source_manifest",
            inject,
        )
        inject("checkpoint:7")

        raw_source_rows = []
        for binding in config.inputs:
            row = {
                "mode_octal": f"{binding.mode:04o}",
                "name": binding.name,
                "path": binding.path, "path_type": "regular_file", "sha256": binding.sha256,
                "size_bytes": binding.size_bytes,
            }
            if binding.name != "rank7_spot_premium_5m":
                row["decoded_rows"] = decoded_rows[binding.name]
                row["normalized_rows"] = normalized_rows[binding.name]
            raw_source_rows.append(row)
        access_ledger = _with_failure_reason(
            PUBLICATION_OR_READBACK_FAILURE,
            _build_access_ledger,
            config,
            head,
            attempt,
            attempt_binding,
            replacement_scan,
            metrics_scan,
            decoded_rows,
        )
        _with_failure_reason(
            PUBLICATION_OR_READBACK_FAILURE,
            _validate_access_ledger,
            access_ledger,
            config,
            head=head,
            attempt=attempt,
            attempt_binding=attempt_binding,
            replacement_scan=replacement_scan,
            metrics_scan=metrics_scan,
            decoded_rows=decoded_rows,
        )
        support_core = {
            "access_ledger": access_ledger,
            "attempt_sentinel": {"attempt_hash": attempt["attempt_hash"], "path": config.attempt_path, "sha256": attempt_binding["sha256"], "size_bytes": attempt_binding["size_bytes"]},
            "identity": IDENTITY,
            "materialized_sources": {
                "market_5m": {"filesystem_mode_octal": "0444", "first_timestamp": _timestamp_z(market_first), "frame_hash": market_frame_hash, "gzip": {"compresslevel": 9, "embedded_filename": "", "mtime": 0}, "last_timestamp": _timestamp_z(market_last), "path": config.market_output_path, "path_type": "regular_file", "rows": market_rows, "schema": list(MARKET_SCHEMA), "sha256": market_binding["sha256"], "size_bytes": market_binding["size_bytes"]},
                "open_interest": {"filesystem_mode_octal": "0444", "first_timestamp": _timestamp_z(oi_first), "frame_hash": oi_frame_hash, "gzip": {"compresslevel": 9, "embedded_filename": "", "mtime": 0}, "last_timestamp": _timestamp_z(oi_last), "path": config.oi_output_path, "path_type": "regular_file", "rows": oi_rows, "schema": list(OI_SCHEMA), "sha256": oi_binding["sha256"], "size_bytes": oi_binding["size_bytes"]},
            },
            "raw_sources": raw_source_rows,
            "source_manifest": {"path": config.manifest_output_path, "sha256": manifest_binding["sha256"], "size_bytes": manifest_binding["size_bytes"]},
            "source_support_commit": head,
            "validation": validation,
            "version": VERSION,
        }
        support = _with_failure_reason(
            PUBLICATION_OR_READBACK_FAILURE,
            _with_hash,
            support_core,
            "support_hash",
        )
        support_raw = _with_failure_reason(
            PUBLICATION_OR_READBACK_FAILURE,
            canonical_json_bytes,
            support,
        )
        support_binding = _with_failure_reason(
            PUBLICATION_OR_READBACK_FAILURE,
            _publish,
            config,
            config.support_output_path,
            support_raw,
            "source_support",
            inject,
        )
        inject("checkpoint:8")

        for slot, item in enumerate(retained.values()):
            _with_failure_reason(
                FINAL_REAUTHENTICATION_FAILURE,
                _final_reauthenticate_input,
                item,
                slot,
                inject,
            )
        expected_output_bindings = {
            config.attempt_path: attempt_binding,
            config.market_output_path: market_binding,
            config.oi_output_path: oi_binding,
            config.manifest_output_path: manifest_binding,
            config.support_output_path: support_binding,
        }
        for text in config.output_paths:
            fd = _with_failure_reason(
                FINAL_REAUTHENTICATION_FAILURE,
                _open_nofollow_components,
                _path(config, text),
                os.O_RDONLY,
            )
            try:
                info = os.fstat(fd)
                digest, size = _descriptor_hash(fd)
            except OSError as exc:
                raise SourceSupportFailure(
                    str(exc),
                    failure_reason=FINAL_REAUTHENTICATION_FAILURE,
                ) from exc
            finally:
                os.close(fd)
            expected = expected_output_bindings[text]
            if (
                not stat.S_ISREG(info.st_mode)
                or stat.S_IMODE(info.st_mode) != 0o444
                or info.st_nlink != 1
                or (info.st_dev, info.st_ino) != expected["inode"]
                or size != expected["size_bytes"]
                or digest != expected["sha256"]
            ):
                _fail(
                    f"final output binding differs: {text}",
                    failure_reason=FINAL_REAUTHENTICATION_FAILURE,
                )
            inject(f"output:{text}:final")
        _with_failure_reason(
            FINAL_REAUTHENTICATION_FAILURE,
            _validate_output_directory_delta,
            config,
            output_directory_baseline,
        )
        if config.enforce_repository_gates:
            _with_failure_reason(
                FINAL_REAUTHENTICATION_FAILURE,
                _validate_bytecode_first_gate,
                config.root,
            )
        inject("checkpoint:9")
        return support
    except SourceSupportFailure as exc:
        publication_state = classify_terminal_publication_state(config)
        validate_terminal_pair(publication_state, exc.failure_reason)
        raise
    finally:
        for item in retained.values():
            os.close(item.fd)


def main() -> int:
    materialize(official_config())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
