"""Build the frozen pre-replay ESDI-288 Ethereum source artifacts.

The production path is intentionally one-shot: two fixed Ethereum mainnet
transports are queried once per JSON-RPC request, no checkpoint is written,
and every disagreement is terminal.  This module does not open market data,
comparators, labels, returns, or outcomes.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, localcontext
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import ssl
import subprocess
import tempfile
from typing import Any, Iterable, Mapping, Protocol, Sequence
import urllib.request

from training import preregister_ethereum_settlement_demand_impulse as prereg


POLICY_ID = "ESDI-288"
PROTOCOL_VERSION = "ethereum_settlement_demand_impulse_source_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = Path(
    "training/build_ethereum_settlement_demand_impulse_source.py"
)
PREREGISTER_EXECUTABLE_PATH = Path(
    "training/preregister_ethereum_settlement_demand_impulse.py"
)
GROSS9_ADAPTER_EXTRA_CLOSURE_PATHS = (
    Path("training/alpha_feature_correlation_report.py"),
    Path("training/audit_fresh_kimchi_orthogonal_alpha.py"),
    Path("training/audit_gross9_fixed_candidate_state_substitution.py"),
    Path("training/audit_gross9_oi_pullback_marginal.py"),
    Path("training/audit_gross9_pullback_premium_overheat_marginal.py"),
    Path("training/audit_rank7_fresh_kimchi_fixed_portfolio.py"),
    Path("training/audit_rex8640_usdkrw_gate.py"),
    Path("training/audit_stable_ensemble_conditional_pullback_alpha.py"),
    Path("training/compare_expanding_extratrees_rank7_refit_cadence_pre2025.py"),
    Path("training/economic_action_backtest.py"),
    Path("training/evaluate_expanding_extratrees_rank7_refit_cadence_oos.py"),
    Path("training/evaluate_expanding_extratrees_top10_oos.py"),
    Path("training/evaluate_stable_ensemble_conditional_pullback_oos.py"),
    Path("training/evaluate_volume_wave_portfolio_combo.py"),
    Path("training/freeze_stable_ensemble_conditional_pullback_alpha.py"),
    Path("training/portfolio_opt_added_alpha_update.py"),
    Path("training/portfolio_opt_all_discovered_alpha_gross10.py"),
    Path("training/portfolio_opt_combined_rex_new_alpha.py"),
    Path("training/portfolio_with_dynamic_exit_sleeves.py"),
    Path("training/search_calendar_oi_funding_alpha.py"),
    Path("training/search_lowcorr_macro_alpha.py"),
    Path("training/search_path_memory_bidirectional_alpha.py"),
    Path("training/search_portfolio_gross6_cost6bp_mdd20_with_dynamic.py"),
    Path("training/search_pullback_premium_overheat_state_machine_alpha.py"),
    Path("training/search_stable_ensemble_conditional_pullback_alpha.py"),
    Path("training/search_vpin_formulaic_alpha.py"),
    Path("training/select_expanding_extratrees_top10_pre2025.py"),
    Path("training/state_model_top10_ensemble.py"),
)
PROTOCOL_PATHS = (
    BUILDER_PATH,
    PREREGISTER_EXECUTABLE_PATH,
    *GROSS9_ADAPTER_EXTRA_CLOSURE_PATHS,
    Path("training/evaluate_ethereum_settlement_demand_impulse_economics.py"),
    Path("training/evaluate_ethereum_settlement_demand_impulse_novelty.py"),
    Path("training/evaluate_ethereum_settlement_demand_impulse_source_support.py"),
    Path("tests/test_build_ethereum_settlement_demand_impulse_source.py"),
    Path("tests/test_evaluate_ethereum_settlement_demand_impulse_economics.py"),
    Path("tests/test_evaluate_ethereum_settlement_demand_impulse_novelty.py"),
    Path("tests/test_evaluate_ethereum_settlement_demand_impulse_source_support.py"),
)
PREREGISTRATION_PATH = Path(
    "results/ethereum_settlement_demand_impulse_preregistration_2026-07-30.json"
)
PREREGISTRATION_SHA256 = (
    "2a481fc60044d3d468340457d50f92a91f2a52184a464e1a91badfb418bbcaba"
)
PREREGISTRATION_MANIFEST_HASH = (
    "d5279f95cc7b92757aa77ecbbc5835d8b1cc4ce34f5a81d6f279abdcf2fcfe8a"
)

TRANSPORTS = (
    "https://eth-mainnet.public.blastapi.io",
    "https://eth.merkle.io",
)
FROZEN_BOUNDARIES = (
    {
        "utc": "2023-01-01T00:00:00Z",
        "first_block_at_or_after": 16_308_190,
        "hash": "0x53dd35d982c984441b3b613919d64dbbf131063d0f85804d77f93f190fa5e106",
    },
    {
        "utc": "2023-06-01T00:00:00Z",
        "first_block_at_or_after": 17_382_266,
        "hash": "0xe0ef11cab4909c80599087b4ffb0bf1e92b1affcc72abc3b802f20a9d5d21096",
    },
    {
        "utc": "2025-01-01T00:00:00Z",
        "first_block_at_or_after": 21_525_891,
        "hash": "0x9512042c5c38145528389a91bd3d63193a1f48fb45d6a3b144ad2d833331fc4c",
    },
    {
        "utc": "2026-01-01T00:00:00Z",
        "first_block_at_or_after": 24_136_053,
        "hash": "0x53e1c0caa885383824d39dc57c0692ea20e971ade409553c4a8031e90f44c516",
    },
    {
        "utc": "2026-06-01T00:00:00Z",
        "first_block_at_or_after": 25_218_798,
        "hash": "0x55f8fdbda40a23cd51a9a2bffba625317ed15d9d1cdc2128c7643bf66e2a906e",
    },
)
RPC_METHODS = ("eth_chainId", "eth_getBlockByNumber", "eth_feeHistory")
CHAIN_ID = 1
REQUEST_CHUNK_BLOCKS = 1_024
REQUEST_COUNT = 8_698
FIRST_REQUESTED_BLOCK = 16_311_600
LAST_RETAINED_BLOCK = 25_217_999
LAST_REQUESTED_BLOCK = 25_218_351
TERMINAL_PADDING_BLOCKS = 352
PRE_2026_06_BOUNDARY_BLOCK = 25_218_798
FIRST_EPOCH_ID = 4_531
LAST_EPOCH_ID = 7_004
EPOCH_SIZE_BLOCKS = 3_600
EPOCH_COUNT = 2_474
CONFIRMATION_BLOCKS = 64
LAST_CONFIRMATION_BLOCK = 25_218_063
BOUNDARY_HEADER_REQUESTS = 2 * len(FROZEN_BOUNDARIES)
EPOCH_HEADER_REQUESTS = 2 * EPOCH_COUNT
TOTAL_RPC_REQUESTS_PER_TRANSPORT = (
    1 + BOUNDARY_HEADER_REQUESTS + 1 + REQUEST_COUNT + EPOCH_HEADER_REQUESTS
)
GAS_RATIO_DECIMAL_PRECISION = 80

SOURCE_OUTPUT_DIRECTORY = Path("data/ethereum_settlement_demand_impulse")
DEFAULT_RAW_OUTPUT = (
    SOURCE_OUTPUT_DIRECTORY / "ethereum_fee_history_chunks_2023_2026.ndjson.gz"
)
DEFAULT_EPOCH_OUTPUT = (
    SOURCE_OUTPUT_DIRECTORY
    / "ethereum_settlement_demand_impulse_epochs_2023_2026.csv.gz"
)
DEFAULT_MANIFEST_OUTPUT = (
    SOURCE_OUTPUT_DIRECTORY
    / "ethereum_settlement_demand_impulse_source_manifest_2026-07-30.json"
)
REPLAY_CLAIM_PATH = Path(
    "results/ethereum_settlement_demand_impulse_source_replay_claim_2026-07-30.json"
)
PRODUCTION_TIMEOUT_SECONDS = 60.0

EPOCH_COLUMNS = (
    "epoch_id",
    "start_block",
    "end_block",
    "end_block_hash",
    "end_block_timestamp_utc",
    "confirmation_block",
    "confirmation_block_hash",
    "available_at_utc",
    "median_base_fee_wei_x2",
    "base_fee_vector_sha256",
    "mean_gas_used_ratio_decimal",
)

_QUANTITY_RE = re.compile(r"0x(?:0|[1-9a-f][0-9a-f]*)\Z")
_HASH_RE = re.compile(r"0x[0-9a-f]{64}\Z")
_DECIMAL_RE = re.compile(r"(?:0(?:\.[0-9]+)?|1(?:\.0+)?)\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_GIT_OBJECT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_FEE_HISTORY_REQUIRED_FIELDS = frozenset(
    {"oldestBlock", "baseFeePerGas", "gasUsedRatio"}
)


class TerminalSourceFailure(RuntimeError):
    """A terminal failure of the frozen one-shot source replay."""


class RpcLike(Protocol):
    url: str

    def call(self, method: str, params: Sequence[Any]) -> Any:
        """Issue exactly one JSON-RPC request."""


@dataclass(frozen=True)
class _SyntheticConfig:
    raw_output: str
    epoch_output: str
    manifest_output: str


@dataclass(frozen=True)
class Header:
    number: int
    block_hash: str
    parent_hash: str
    timestamp: int


@dataclass(frozen=True)
class FeeHistoryChunk:
    oldest_block: int
    base_fees: tuple[int, ...]
    gas_ratios: tuple[Decimal, ...]
    gas_ratio_strings: tuple[str, ...]


def _repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (
        REPOSITORY_ROOT / candidate
    ).resolve()


def _repository_candidate(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _repository_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(payload: Any, *, trailing_lf: bool = False) -> bytes:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return encoded + (b"\n" if trailing_lf else b"")


def _git_command(
    *arguments: str, allow_failure: bool = False
) -> subprocess.CompletedProcess[bytes]:
    completed = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0 and not allow_failure:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise TerminalSourceFailure(
            f"protocol seal git command failed: {' '.join(arguments)}: {detail}"
        )
    return completed


def _git_text(*arguments: str) -> str:
    return _git_command(*arguments).stdout.decode("ascii").strip()


def _protocol_seal_core(
    git_commit: str | None, files: Mapping[str, Mapping[str, str | None]]
) -> dict[str, Any]:
    return {
        "protocol_version": "ethereum_settlement_demand_impulse_protocol_seal_v1",
        "policy_id": POLICY_ID,
        "git_commit": git_commit,
        "protocol_paths": [path.as_posix() for path in PROTOCOL_PATHS],
        "files": {path: dict(binding) for path, binding in files.items()},
    }


def current_protocol_seal(require_committed: bool = True) -> dict[str, Any]:
    """Seal every executable protocol dependency against its exact HEAD blob."""

    commit_result = _git_command(
        "rev-parse", "--verify", "HEAD", allow_failure=not require_committed
    )
    git_commit = (
        commit_result.stdout.decode("ascii").strip()
        if commit_result.returncode == 0
        else None
    )
    if require_committed and (
        git_commit is None or _GIT_OBJECT_RE.fullmatch(git_commit) is None
    ):
        raise TerminalSourceFailure("protocol seal has no valid HEAD commit")

    files: dict[str, dict[str, str | None]] = {}
    for relative in PROTOCOL_PATHS:
        relative_text = relative.as_posix()
        candidate = _repository_candidate(relative)
        actual_path = candidate.resolve()
        if (
            not actual_path.is_file()
            or candidate.is_symlink()
            or not actual_path.is_relative_to(REPOSITORY_ROOT.resolve())
        ):
            raise TerminalSourceFailure(
                f"protocol path is not a regular repository file: {relative_text}"
            )
        actual = actual_path.read_bytes()
        tracked = _git_command(
            "ls-files", "--error-unmatch", "--", relative_text, allow_failure=True
        ).returncode == 0
        if not tracked:
            if require_committed:
                raise TerminalSourceFailure(
                    f"protocol path is not tracked: {relative_text}"
                )
            files[relative_text] = {
                "git_blob": None,
                "sha256": _sha256_bytes(actual),
            }
            continue

        index_diff = _git_command(
            "diff",
            "--cached",
            "--quiet",
            "HEAD",
            "--",
            relative_text,
            allow_failure=True,
        )
        worktree_diff = _git_command(
            "diff",
            "--quiet",
            "--",
            relative_text,
            allow_failure=True,
        )
        if index_diff.returncode != 0 or worktree_diff.returncode != 0:
            raise TerminalSourceFailure(
                f"protocol path is not HEAD-clean: {relative_text}"
            )
        head_result = _git_command(
            "show", f"HEAD:{relative_text}", allow_failure=not require_committed
        )
        if head_result.returncode != 0:
            if require_committed:
                raise TerminalSourceFailure(
                    f"protocol path is absent from HEAD: {relative_text}"
                )
            files[relative_text] = {
                "git_blob": None,
                "sha256": _sha256_bytes(actual),
            }
            continue
        head = head_result.stdout
        if actual != head:
            raise TerminalSourceFailure(
                f"protocol path is not HEAD-clean: {relative_text}"
            )
        actual_sha256 = _sha256_bytes(actual)
        head_sha256 = _sha256_bytes(head)
        if actual_sha256 != head_sha256:
            raise TerminalSourceFailure(
                f"protocol path SHA256 differs from HEAD bytes: {relative_text}"
            )
        git_blob = _git_text("rev-parse", f"HEAD:{relative_text}")
        if _GIT_OBJECT_RE.fullmatch(git_blob) is None:
            raise TerminalSourceFailure(
                f"protocol path has an invalid git blob: {relative_text}"
            )
        files[relative_text] = {
            "git_blob": git_blob,
            "sha256": actual_sha256,
        }

    core = _protocol_seal_core(git_commit, files)
    return {
        **core,
        "seal_hash": _sha256_bytes(_canonical_json_bytes(core)),
    }


def validate_protocol_seal(
    recorded: Mapping[str, Any],
    *,
    require_committed: bool = True,
    require_recorded_commit_ancestor: bool = True,
) -> dict[str, Any]:
    """Validate ancestry and prove every current protocol blob is unchanged."""

    if not isinstance(recorded, Mapping):
        raise TerminalSourceFailure("recorded protocol seal is not an object")
    if set(recorded) != {
        "protocol_version",
        "policy_id",
        "git_commit",
        "protocol_paths",
        "files",
        "seal_hash",
    }:
        raise TerminalSourceFailure("recorded protocol seal schema differs")
    if (
        recorded.get("protocol_version")
        != "ethereum_settlement_demand_impulse_protocol_seal_v1"
        or recorded.get("policy_id") != POLICY_ID
    ):
        raise TerminalSourceFailure("recorded protocol seal identity differs")
    core = {key: value for key, value in recorded.items() if key != "seal_hash"}
    if (
        recorded.get("seal_hash")
        != _sha256_bytes(_canonical_json_bytes(core))
    ):
        raise TerminalSourceFailure("recorded protocol seal hash differs")
    if recorded.get("protocol_paths") != [
        path.as_posix() for path in PROTOCOL_PATHS
    ]:
        raise TerminalSourceFailure("recorded protocol paths differ")
    recorded_commit = recorded.get("git_commit")
    if (
        not isinstance(recorded_commit, str)
        or _GIT_OBJECT_RE.fullmatch(recorded_commit) is None
    ):
        raise TerminalSourceFailure("recorded protocol commit is invalid")
    recorded_files = recorded.get("files")
    expected_paths = {path.as_posix() for path in PROTOCOL_PATHS}
    if not isinstance(recorded_files, Mapping) or set(recorded_files) != expected_paths:
        raise TerminalSourceFailure("recorded protocol file set differs")
    for path, binding in recorded_files.items():
        if not isinstance(binding, Mapping) or set(binding) != {
            "git_blob",
            "sha256",
        }:
            raise TerminalSourceFailure(f"recorded protocol binding differs: {path}")
        git_blob = binding.get("git_blob")
        sha256 = binding.get("sha256")
        if (
            not isinstance(git_blob, str)
            or _GIT_OBJECT_RE.fullmatch(git_blob) is None
            or not isinstance(sha256, str)
            or _SHA256_RE.fullmatch(sha256) is None
        ):
            raise TerminalSourceFailure(
                f"recorded protocol hashes are invalid: {path}"
            )
        recorded_blob = _git_text("rev-parse", f"{recorded_commit}:{path}")
        if recorded_blob != git_blob:
            raise TerminalSourceFailure(
                f"recorded protocol commit/blob binding differs: {path}"
            )

    if require_recorded_commit_ancestor:
        ancestry = _git_command(
            "merge-base",
            "--is-ancestor",
            recorded_commit,
            "HEAD",
            allow_failure=True,
        )
        if ancestry.returncode != 0:
            raise TerminalSourceFailure(
                "recorded evaluator commit is not an ancestor of HEAD"
            )

    current = current_protocol_seal(require_committed=require_committed)
    if current["files"] != recorded_files:
        raise TerminalSourceFailure(
            "current protocol blobs are not unchanged from recorded seal"
        )
    return current


def _unique_object(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TerminalSourceFailure(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _parse_quantity(value: Any, label: str) -> int:
    if not isinstance(value, str) or _QUANTITY_RE.fullmatch(value) is None:
        raise TerminalSourceFailure(f"{label} is not a canonical quantity")
    return int(value, 16)


def _parse_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise TerminalSourceFailure(f"{label} is not a canonical 32-byte hash")
    return value


def _canonical_decimal(value: Decimal) -> str:
    if not value.is_finite():
        raise TerminalSourceFailure("gas-used ratio is not finite")
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def _parse_gas_ratio(value: Any) -> tuple[Decimal, str]:
    if isinstance(value, bool) or isinstance(value, float):
        raise TerminalSourceFailure(
            "gas-used ratio must be parsed without binary floating point"
        )
    if isinstance(value, int):
        raw = str(value)
    elif isinstance(value, Decimal):
        raw = format(value, "f")
    elif isinstance(value, str):
        raw = value
    else:
        raise TerminalSourceFailure("gas-used ratio has an invalid type")
    if _DECIMAL_RE.fullmatch(raw) is None:
        raise TerminalSourceFailure("gas-used ratio is not canonical or in [0,1]")
    try:
        parsed = Decimal(raw)
    except InvalidOperation as exc:
        raise TerminalSourceFailure("gas-used ratio is invalid") from exc
    canonical = _canonical_decimal(parsed)
    return parsed, canonical


class HttpJsonRpcClient:
    """Strict one-attempt HTTPS JSON-RPC client with redirects disabled."""

    def __init__(self, url: str, *, timeout_seconds: float = 60.0) -> None:
        if url not in TRANSPORTS:
            raise ValueError("ESDI-288 RPC URL is not one of the frozen transports")
        if timeout_seconds <= 0:
            raise ValueError("ESDI-288 timeout must be positive")
        self.url = url
        self.timeout_seconds = timeout_seconds
        self._request_id = 0
        context = ssl.create_default_context()
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            _RejectRedirects(),
            urllib.request.HTTPSHandler(context=context),
        )

    def call(self, method: str, params: Sequence[Any]) -> Any:
        if method not in RPC_METHODS:
            raise TerminalSourceFailure(f"unfrozen RPC method {method!r}")
        self._request_id += 1
        payload = _canonical_json_bytes(
            {
                "jsonrpc": "2.0",
                "id": self._request_id,
                "method": method,
                "params": list(params),
            }
        )
        request = urllib.request.Request(
            self.url,
            data=payload,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "rllm-esdi-288-source/1",
            },
            method="POST",
        )
        try:
            with self._opener.open(
                request, timeout=self.timeout_seconds
            ) as response:
                if response.status != 200:
                    raise TerminalSourceFailure(
                        f"{self.url} returned HTTP {response.status}"
                    )
                body = response.read()
        except TerminalSourceFailure:
            raise
        except Exception as exc:
            raise TerminalSourceFailure(
                f"single RPC attempt failed for {self.url} {method}"
            ) from exc
        try:
            decoded = json.loads(
                body.decode("utf-8"),
                parse_float=Decimal,
                parse_int=Decimal,
                object_pairs_hook=_unique_object,
            )
        except Exception as exc:
            raise TerminalSourceFailure("RPC response is not strict UTF-8 JSON") from exc
        if not isinstance(decoded, dict):
            raise TerminalSourceFailure("RPC response is not an object")
        if decoded.get("jsonrpc") != "2.0":
            raise TerminalSourceFailure("RPC response version differs")
        if decoded.get("id") != Decimal(self._request_id):
            raise TerminalSourceFailure("RPC response id differs")
        if "error" in decoded:
            raise TerminalSourceFailure(f"RPC returned an error for {method}")
        if set(decoded) != {"jsonrpc", "id", "result"}:
            raise TerminalSourceFailure("RPC response fields differ")
        return decoded["result"]


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Mapping[str, str],
        newurl: str,
    ) -> None:
        raise TerminalSourceFailure("RPC redirect is forbidden")


def validate_preregistration(path: str | Path = PREREGISTRATION_PATH) -> dict[str, Any]:
    resolved = _repository_path(path)
    if resolved != _repository_path(PREREGISTRATION_PATH):
        raise TerminalSourceFailure("preregistration path differs from frozen path")
    raw = resolved.read_bytes()
    if _sha256_bytes(raw) != PREREGISTRATION_SHA256:
        raise TerminalSourceFailure("preregistration file SHA256 differs")
    payload = json.loads(
        raw.decode("utf-8"), object_pairs_hook=_unique_object
    )
    if not isinstance(payload, dict):
        raise TerminalSourceFailure("preregistration is not an object")
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if _sha256_bytes(_canonical_json_bytes(core)) != PREREGISTRATION_MANIFEST_HASH:
        raise TerminalSourceFailure("preregistration canonical hash differs")
    if payload.get("manifest_hash") != PREREGISTRATION_MANIFEST_HASH:
        raise TerminalSourceFailure("preregistration manifest hash differs")
    try:
        preregistered_sha256 = payload["frozen_preregistration"][
            "repository_identity"
        ]["sha256"][PREREGISTER_EXECUTABLE_PATH.as_posix()]
    except (KeyError, TypeError) as exc:
        raise TerminalSourceFailure(
            "preregistration executable binding is absent"
        ) from exc
    current_preregister_sha256 = _sha256_file(PREREGISTER_EXECUTABLE_PATH)
    if (
        not isinstance(preregistered_sha256, str)
        or _SHA256_RE.fullmatch(preregistered_sha256) is None
        or preregistered_sha256 != current_preregister_sha256
    ):
        raise TerminalSourceFailure(
            "preregistration executable bytes differ from preregistered SHA256"
        )
    source = payload.get("source")
    expected_source = {
        "chain_id": CHAIN_ID,
        "request_chunk_blocks": REQUEST_CHUNK_BLOCKS,
        "rpc_attempts_per_request": 1,
        "rpc_retry_backoff_or_resume": False,
        "provider_fallback_after_values_open": False,
        "dual_replay_exact_agreement_required": True,
        "epoch_size_blocks": EPOCH_SIZE_BLOCKS,
        "first_epoch_id": FIRST_EPOCH_ID,
        "last_epoch_id": LAST_EPOCH_ID,
        "epoch_count": EPOCH_COUNT,
        "first_source_block": FIRST_REQUESTED_BLOCK,
        "last_source_block": LAST_RETAINED_BLOCK,
        "confirmation_blocks_after_end": CONFIRMATION_BLOCKS,
    }
    if not isinstance(source, dict) or any(
        source.get(key) != value for key, value in expected_source.items()
    ):
        raise TerminalSourceFailure("preregistration frozen source contract differs")
    if source.get("boundaries") != list(FROZEN_BOUNDARIES):
        raise TerminalSourceFailure("preregistration boundaries differ")
    return payload


def normalize_header(raw: Any, *, expected_number: int | None = None) -> Header:
    if not isinstance(raw, dict):
        raise TerminalSourceFailure("block header is not an object")
    number = _parse_quantity(raw.get("number"), "block number")
    if expected_number is not None and number != expected_number:
        raise TerminalSourceFailure("block header number differs")
    block_hash = _parse_hash(raw.get("hash"), "block hash")
    parent_hash = _parse_hash(raw.get("parentHash"), "parent hash")
    timestamp = _parse_quantity(raw.get("timestamp"), "block timestamp")
    return Header(number, block_hash, parent_hash, timestamp)


def _paired_call(
    clients: Sequence[RpcLike], method: str, params: Sequence[Any]
) -> tuple[Any, Any]:
    if len(clients) != 2 or tuple(client.url for client in clients) != TRANSPORTS:
        raise TerminalSourceFailure("exactly the two frozen transports are required")
    left = clients[0].call(method, params)
    right = clients[1].call(method, params)
    return left, right


def validate_chain_ids(clients: Sequence[RpcLike]) -> None:
    left, right = _paired_call(clients, "eth_chainId", ())
    left_id = _parse_quantity(left, "primary chain id")
    right_id = _parse_quantity(right, "verification chain id")
    if left != right or left_id != CHAIN_ID or right_id != CHAIN_ID:
        raise TerminalSourceFailure("transports do not agree on Ethereum mainnet")


def _header_pair(
    clients: Sequence[RpcLike], selector: str, *, expected_number: int | None = None
) -> Header:
    left_raw, right_raw = _paired_call(
        clients, "eth_getBlockByNumber", (selector, False)
    )
    left = normalize_header(left_raw, expected_number=expected_number)
    right = normalize_header(right_raw, expected_number=expected_number)
    if left != right:
        raise TerminalSourceFailure("provider block headers disagree")
    return left


def validate_boundaries(clients: Sequence[RpcLike]) -> list[dict[str, Any]]:
    audits: list[dict[str, Any]] = []
    for boundary in FROZEN_BOUNDARIES:
        number = boundary["first_block_at_or_after"]
        previous = _header_pair(clients, hex(number - 1), expected_number=number - 1)
        current = _header_pair(clients, hex(number), expected_number=number)
        threshold = int(
            datetime.fromisoformat(boundary["utc"].replace("Z", "+00:00")).timestamp()
        )
        if current.block_hash != boundary["hash"]:
            raise TerminalSourceFailure("frozen boundary hash differs")
        if current.parent_hash != previous.block_hash:
            raise TerminalSourceFailure("frozen boundary parent relation differs")
        if not previous.timestamp < threshold <= current.timestamp:
            raise TerminalSourceFailure("frozen boundary timestamp relation differs")
        audits.append(
            {
                "utc": boundary["utc"],
                "first_block_at_or_after": number,
                "previous_block": number - 1,
                "previous_timestamp_before_boundary": True,
                "current_timestamp_at_or_after_boundary": True,
                "parent_relation_exact": True,
                "hash_exact": True,
            }
        )
    return audits


def validate_common_finalized_head(clients: Sequence[RpcLike]) -> Header:
    header = _header_pair(clients, "finalized")
    if header.number < LAST_CONFIRMATION_BLOCK:
        raise TerminalSourceFailure("common finalized head is below required history")
    return header


def normalize_fee_history(
    raw: Any, *, expected_start: int, block_count: int = REQUEST_CHUNK_BLOCKS
) -> FeeHistoryChunk:
    if not isinstance(raw, dict):
        raise TerminalSourceFailure("eth_feeHistory result is not an object")
    if not _FEE_HISTORY_REQUIRED_FIELDS.issubset(raw):
        raise TerminalSourceFailure("eth_feeHistory result fields differ")
    oldest = _parse_quantity(raw["oldestBlock"], "fee-history oldest block")
    if oldest != expected_start:
        raise TerminalSourceFailure("fee-history oldest block differs")
    base_raw = raw["baseFeePerGas"]
    gas_raw = raw["gasUsedRatio"]
    if not isinstance(base_raw, list) or len(base_raw) != block_count + 1:
        raise TerminalSourceFailure("fee-history base-fee subsection is shortened")
    if not isinstance(gas_raw, list) or len(gas_raw) != block_count:
        raise TerminalSourceFailure("fee-history gas-ratio subsection is shortened")
    base_fees = tuple(
        _parse_quantity(value, "base fee") for value in base_raw
    )
    if any(value <= 0 or value >= 2**256 for value in base_fees):
        raise TerminalSourceFailure("base fee is not a positive uint256")
    parsed_ratios = tuple(_parse_gas_ratio(value) for value in gas_raw)
    return FeeHistoryChunk(
        oldest_block=oldest,
        base_fees=base_fees,
        gas_ratios=tuple(value for value, _ in parsed_ratios),
        gas_ratio_strings=tuple(value for _, value in parsed_ratios),
    )


def _fee_history_pair(
    clients: Sequence[RpcLike], *, start: int, block_count: int
) -> FeeHistoryChunk:
    end = start + block_count - 1
    params = (hex(block_count), hex(end), [])
    left_raw, right_raw = _paired_call(clients, "eth_feeHistory", params)
    left = normalize_fee_history(
        left_raw, expected_start=start, block_count=block_count
    )
    right = normalize_fee_history(
        right_raw, expected_start=start, block_count=block_count
    )
    if left != right:
        raise TerminalSourceFailure("provider fee-history responses disagree")
    return left


def validate_next_base_fee_overlap(
    previous_next_base_fee: int | None, chunk: FeeHistoryChunk
) -> int:
    if (
        previous_next_base_fee is not None
        and previous_next_base_fee != chunk.base_fees[0]
    ):
        raise TerminalSourceFailure("adjacent next-base-fee overlap differs")
    return chunk.base_fees[-1]


def _utc(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def build_epoch_row(
    *,
    epoch_id: int,
    base_fees: Sequence[int],
    gas_ratios: Sequence[Decimal],
    end_header: Header,
    confirmation_header: Header,
) -> dict[str, Any]:
    start, end, confirmation = prereg.epoch_blocks(epoch_id)
    if (
        len(base_fees) != EPOCH_SIZE_BLOCKS
        or len(gas_ratios) != EPOCH_SIZE_BLOCKS
    ):
        raise TerminalSourceFailure("normalized epoch is not exactly 3,600 blocks")
    if end_header.number != end or confirmation_header.number != confirmation:
        raise TerminalSourceFailure("epoch header number differs")
    if end_header.timestamp >= confirmation_header.timestamp:
        raise TerminalSourceFailure("epoch confirmation timestamp is not later")
    if any(
        not isinstance(value, Decimal) or not value.is_finite()
        for value in gas_ratios
    ):
        raise TerminalSourceFailure("epoch gas ratios are not exact Decimal values")
    with localcontext() as context:
        context.prec = GAS_RATIO_DECIMAL_PRECISION
        mean_ratio = sum(gas_ratios, Decimal(0)) / Decimal(EPOCH_SIZE_BLOCKS)
    return {
        "epoch_id": epoch_id,
        "start_block": start,
        "end_block": end,
        "end_block_hash": end_header.block_hash,
        "end_block_timestamp_utc": _utc(end_header.timestamp),
        "confirmation_block": confirmation,
        "confirmation_block_hash": confirmation_header.block_hash,
        "available_at_utc": _utc(confirmation_header.timestamp),
        "median_base_fee_wei_x2": prereg.median2(base_fees),
        "base_fee_vector_sha256": prereg.base_fee_vector_sha256(base_fees),
        "mean_gas_used_ratio_decimal": _canonical_decimal(mean_ratio),
    }


def _deterministic_gzip(payload: bytes) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(
        fileobj=buffer, mode="wb", filename="", compresslevel=9, mtime=0
    ) as stream:
        stream.write(payload)
    return buffer.getvalue()


def _csv_gzip(rows: Sequence[Mapping[str, Any]]) -> bytes:
    text = io.StringIO(newline="")
    writer = csv.DictWriter(
        text, fieldnames=EPOCH_COLUMNS, lineterminator="\n", extrasaction="raise"
    )
    writer.writeheader()
    writer.writerows(rows)
    return _deterministic_gzip(text.getvalue().encode("utf-8"))


def _temporary_path(final: Path) -> Path:
    final.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{final.name}.", suffix=".tmp", dir=final.parent, delete=False
    )
    handle.close()
    return Path(handle.name)


def _preflight_outputs(paths: Sequence[Path]) -> None:
    if len(set(paths)) != len(paths):
        raise TerminalSourceFailure("source output paths must be distinct")
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.parent.is_symlink() or not path.parent.is_dir():
            raise TerminalSourceFailure("source output parent is not a real directory")
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"write-once output already exists: {path}")


def atomic_publish(staged: Mapping[Path, Path]) -> None:
    """Publish all staged files write-once, rolling back this attempt on error."""

    finals = tuple(staged)
    _preflight_outputs(finals)
    published: list[Path] = []
    try:
        for final, temporary in staged.items():
            os.link(temporary, final)
            published.append(final)
    except Exception:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _durable_write_once(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o444,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("durable staged write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _canonical_output_paths() -> tuple[Path, Path, Path]:
    return tuple(
        _repository_candidate(path)
        for path in (
            DEFAULT_RAW_OUTPUT,
            DEFAULT_EPOCH_OUTPUT,
            DEFAULT_MANIFEST_OUTPUT,
        )
    )


def _canonical_output_directory() -> Path:
    return _repository_candidate(SOURCE_OUTPUT_DIRECTORY)


def _production_stage_directory(claim_hash: str) -> Path:
    if _SHA256_RE.fullmatch(claim_hash) is None:
        raise TerminalSourceFailure("production claim hash is invalid")
    final_directory = _canonical_output_directory()
    return final_directory.parent / (
        f".{final_directory.name}.stage-{claim_hash}"
    )


def _production_stage_paths(claim_hash: str) -> tuple[Path, Path, Path]:
    directory = _production_stage_directory(claim_hash)
    return tuple(
        directory / path.name
        for path in (
            DEFAULT_RAW_OUTPUT,
            DEFAULT_EPOCH_OUTPUT,
            DEFAULT_MANIFEST_OUTPUT,
        )
    )


def _prepare_production_stage(claim_hash: str) -> tuple[Path, Path, Path]:
    directory = _production_stage_directory(claim_hash)
    directory.parent.mkdir(parents=True, exist_ok=True)
    if directory.parent.is_symlink() or not directory.parent.is_dir():
        raise TerminalSourceFailure(
            "production source parent is not a real directory"
        )
    final_directory = _canonical_output_directory()
    if final_directory.exists() or final_directory.is_symlink():
        raise TerminalSourceFailure(
            "canonical source directory exists before atomic publication"
        )
    try:
        directory.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise TerminalSourceFailure(
            "production durable stage already exists and replay cannot resume"
        ) from exc
    _fsync_directory(directory.parent)
    return _production_stage_paths(claim_hash)


def _assert_production_replay_available() -> None:
    blockers = (
        _repository_candidate(REPLAY_CLAIM_PATH),
        _canonical_output_directory(),
    )
    for path in blockers:
        if path.exists() or path.is_symlink():
            raise FileExistsError(
                "ESDI-288 canonical claim/output already exists; "
                "production replay is forbidden forever"
            )


def _production_claim_payload(
    pre_replay_protocol_seal: Mapping[str, Any],
) -> dict[str, Any]:
    core = {
        "protocol_version": (
            "ethereum_settlement_demand_impulse_source_replay_claim_v1"
        ),
        "policy_id": POLICY_ID,
        "status": "claimed_before_first_rpc",
        "one_shot": True,
        "retry_backoff_fallback_or_resume": False,
        "preregistration": {
            "path": PREREGISTRATION_PATH.as_posix(),
            "sha256": PREREGISTRATION_SHA256,
            "manifest_hash": PREREGISTRATION_MANIFEST_HASH,
        },
        "transports": list(TRANSPORTS),
        "canonical_outputs": [
            DEFAULT_RAW_OUTPUT.as_posix(),
            DEFAULT_EPOCH_OUTPUT.as_posix(),
            DEFAULT_MANIFEST_OUTPUT.as_posix(),
        ],
        "pre_replay_protocol_seal": dict(pre_replay_protocol_seal),
    }
    return {
        **core,
        "claim_hash": _sha256_bytes(_canonical_json_bytes(core)),
    }


def _create_production_claim(
    pre_replay_protocol_seal: Mapping[str, Any],
) -> dict[str, Any]:
    _assert_production_replay_available()
    claim_path = _repository_path(REPLAY_CLAIM_PATH)
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _production_claim_payload(pre_replay_protocol_seal)
    raw = _canonical_json_bytes(payload, trailing_lf=True)
    temporary = _temporary_path(claim_path)
    try:
        temporary.write_bytes(raw)
        temporary.chmod(0o444)
        try:
            os.link(temporary, claim_path)
            _fsync_file(claim_path)
            _fsync_directory(claim_path.parent)
        except FileExistsError as exc:
            raise FileExistsError(
                "ESDI-288 production claim already exists; replay is "
                "forbidden forever"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": REPLAY_CLAIM_PATH.as_posix(),
        "sha256": _sha256_bytes(raw),
        "claim_hash": payload["claim_hash"],
    }


def _load_production_claim() -> tuple[dict[str, Any], dict[str, Any]]:
    claim_path = _repository_path(REPLAY_CLAIM_PATH)
    if (
        not claim_path.is_file()
        or claim_path.is_symlink()
    ):
        raise TerminalSourceFailure("production replay claim is absent or invalid")
    raw = claim_path.read_bytes()
    try:
        payload = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_unique_object
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TerminalSourceFailure("production replay claim is invalid") from exc
    if not isinstance(payload, dict):
        raise TerminalSourceFailure("production replay claim is not an object")
    seal = payload.get("pre_replay_protocol_seal")
    if not isinstance(seal, Mapping) or payload != _production_claim_payload(seal):
        raise TerminalSourceFailure("production replay claim bytes differ")
    binding = {
        "path": REPLAY_CLAIM_PATH.as_posix(),
        "sha256": _sha256_bytes(raw),
        "claim_hash": payload["claim_hash"],
    }
    return payload, binding


def _load_and_validate_generation(
    paths: Sequence[Path],
    *,
    claim_binding: Mapping[str, Any],
    pre_replay_protocol_seal: Mapping[str, Any],
) -> dict[str, Any]:
    if len(paths) != 3:
        raise AssertionError("generation must contain raw, epoch, and manifest")
    raw_path, epoch_path, manifest_path = paths
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise TerminalSourceFailure(
                "complete valid source generation is absent"
            )
    manifest_raw = manifest_path.read_bytes()
    try:
        manifest = json.loads(
            manifest_raw.decode("utf-8"), object_pairs_hook=_unique_object
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TerminalSourceFailure("source manifest is invalid") from exc
    if not isinstance(manifest, dict):
        raise TerminalSourceFailure("source manifest is not an object")
    core = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    if (
        set(manifest) != set(core) | {"manifest_hash"}
        or manifest.get("manifest_hash")
        != _sha256_bytes(_canonical_json_bytes(core))
        or manifest_raw != _canonical_json_bytes(manifest, trailing_lf=True)
    ):
        raise TerminalSourceFailure("source manifest cryptographic binding differs")
    if (
        manifest.get("protocol_version") != PROTOCOL_VERSION
        or manifest.get("policy_id") != POLICY_ID
        or manifest.get("status") != "complete_outcome_blind_source_replay"
        or manifest.get("claim") != dict(claim_binding)
        or manifest.get("pre_replay_protocol_seal")
        != dict(pre_replay_protocol_seal)
    ):
        raise TerminalSourceFailure("source manifest generation binding differs")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise TerminalSourceFailure("source manifest outputs are invalid")
    expected_output_paths = {
        "raw_chunks": DEFAULT_RAW_OUTPUT.as_posix(),
        "normalized_epochs": DEFAULT_EPOCH_OUTPUT.as_posix(),
    }
    for path, key in ((raw_path, "raw_chunks"), (epoch_path, "normalized_epochs")):
        binding = outputs.get(key)
        if (
            not isinstance(binding, dict)
            or binding.get("path") != expected_output_paths[key]
            or binding.get("bytes") != path.stat().st_size
            or binding.get("sha256") != _sha256_file(path)
        ):
            raise TerminalSourceFailure(
                f"source {key} cryptographic binding differs"
            )
    manifest_binding = outputs.get("manifest")
    if (
        not isinstance(manifest_binding, dict)
        or manifest_binding.get("path") != DEFAULT_MANIFEST_OUTPUT.as_posix()
    ):
        raise TerminalSourceFailure("source manifest output path differs")
    return manifest


def _validate_generation_directory(
    directory: Path,
    *,
    claim_binding: Mapping[str, Any],
    pre_replay_protocol_seal: Mapping[str, Any],
) -> dict[str, Any]:
    if directory.is_symlink() or not directory.is_dir():
        raise TerminalSourceFailure("source generation directory is unsafe")
    expected_names = {
        DEFAULT_RAW_OUTPUT.name,
        DEFAULT_EPOCH_OUTPUT.name,
        DEFAULT_MANIFEST_OUTPUT.name,
    }
    try:
        actual_names = {path.name for path in directory.iterdir()}
    except OSError as exc:
        raise TerminalSourceFailure(
            "source generation directory cannot be inspected"
        ) from exc
    if actual_names != expected_names:
        raise TerminalSourceFailure(
            "source generation directory inventory differs"
        )
    paths = tuple(
        directory / path.name
        for path in (
            DEFAULT_RAW_OUTPUT,
            DEFAULT_EPOCH_OUTPUT,
            DEFAULT_MANIFEST_OUTPUT,
        )
    )
    return _load_and_validate_generation(
        paths,
        claim_binding=claim_binding,
        pre_replay_protocol_seal=pre_replay_protocol_seal,
    )


def _publish_staged_generation(
    *,
    staged_paths: Sequence[Path],
    claim_binding: Mapping[str, Any],
    pre_replay_protocol_seal: Mapping[str, Any],
) -> dict[str, Any]:
    claim_hash = str(claim_binding.get("claim_hash", ""))
    stage_directory = _production_stage_directory(claim_hash)
    expected_staged_paths = _production_stage_paths(claim_hash)
    if tuple(staged_paths) != expected_staged_paths:
        raise TerminalSourceFailure("production stage path inventory differs")
    manifest = _validate_generation_directory(
        stage_directory,
        claim_binding=claim_binding,
        pre_replay_protocol_seal=pre_replay_protocol_seal,
    )
    for staged in expected_staged_paths:
        staged.chmod(0o444)
        _fsync_file(staged)
    stage_directory.chmod(0o555)
    _fsync_directory(stage_directory)
    final_directory = _canonical_output_directory()
    if final_directory.exists() or final_directory.is_symlink():
        raise TerminalSourceFailure(
            "canonical source directory exists before atomic publication"
        )
    stage_identity = stage_directory.stat()
    os.rename(stage_directory, final_directory)
    _fsync_directory(final_directory.parent)
    published_identity = final_directory.stat()
    if (
        published_identity.st_dev != stage_identity.st_dev
        or published_identity.st_ino != stage_identity.st_ino
    ):
        raise TerminalSourceFailure(
            "atomic source directory publication identity differs"
        )
    canonical_manifest = _validate_generation_directory(
        final_directory,
        claim_binding=claim_binding,
        pre_replay_protocol_seal=pre_replay_protocol_seal,
    )
    if canonical_manifest != manifest:
        raise TerminalSourceFailure(
            "published canonical manifest differs from staged generation"
        )
    return canonical_manifest


def _load_atomically_published_generation() -> dict[str, Any]:
    payload, claim_binding = _load_production_claim()
    seal = payload["pre_replay_protocol_seal"]
    validate_preregistration()
    validate_protocol_seal(seal)
    stage_directory = _production_stage_directory(
        str(claim_binding["claim_hash"])
    )
    if stage_directory.exists() or stage_directory.is_symlink():
        raise TerminalSourceFailure(
            "claim exists with an unpublished durable stage; replay and "
            "stage recovery are forbidden"
        )
    final_directory = _canonical_output_directory()
    if not final_directory.exists() and not final_directory.is_symlink():
        raise TerminalSourceFailure(
            "claim exists without an atomically published source generation; "
            "replay is forbidden"
        )
    return _validate_generation_directory(
        final_directory,
        claim_binding=claim_binding,
        pre_replay_protocol_seal=seal,
    )


def _assert_synthetic_outputs(cfg: _SyntheticConfig) -> None:
    synthetic_paths = {
        _repository_path(cfg.raw_output),
        _repository_path(cfg.epoch_output),
        _repository_path(cfg.manifest_output),
    }
    forbidden = {
        *_canonical_output_paths(),
        _repository_path(REPLAY_CLAIM_PATH),
    }
    if synthetic_paths & forbidden:
        raise TerminalSourceFailure(
            "synthetic source helper cannot publish a canonical artifact path"
        )


def _synthetic_bindings(
    cfg: _SyntheticConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    seal_core = {
        "protocol_version": (
            "ethereum_settlement_demand_impulse_synthetic_protocol_seal_v1"
        ),
        "policy_id": POLICY_ID,
        "mode": "synthetic_only",
        "protocol_paths": [path.as_posix() for path in PROTOCOL_PATHS],
    }
    seal = {
        **seal_core,
        "seal_hash": _sha256_bytes(_canonical_json_bytes(seal_core)),
    }
    claim_core = {
        "mode": "synthetic_unpublished",
        "outputs": [cfg.raw_output, cfg.epoch_output, cfg.manifest_output],
        "pre_replay_protocol_seal_hash": seal["seal_hash"],
    }
    claim = {
        "path": None,
        "sha256": _sha256_bytes(_canonical_json_bytes(claim_core)),
        "synthetic_unpublished": True,
    }
    return seal, claim


def _manifest(
    *,
    cfg: _SyntheticConfig,
    raw_size: int,
    raw_sha256: str,
    epoch_bytes: bytes,
    finalized: Header,
    boundary_audit: Sequence[Mapping[str, Any]],
    builder_sha256: str,
    claim_binding: Mapping[str, Any],
    pre_replay_protocol_seal: Mapping[str, Any],
) -> dict[str, Any]:
    raw_path = _repository_path(cfg.raw_output)
    epoch_path = _repository_path(cfg.epoch_output)
    manifest_path = _repository_path(cfg.manifest_output)
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "status": "complete_outcome_blind_source_replay",
        "claim": dict(claim_binding),
        "pre_replay_protocol_seal": dict(pre_replay_protocol_seal),
        "preregistration": {
            "path": str(PREREGISTRATION_PATH),
            "sha256": PREREGISTRATION_SHA256,
            "manifest_hash": PREREGISTRATION_MANIFEST_HASH,
        },
        "source_builder": {
            "path": str(BUILDER_PATH),
            "sha256": builder_sha256,
        },
        "transports": list(TRANSPORTS),
        "rpc": {
            "methods": list(RPC_METHODS),
            "attempts_per_request": 1,
            "retry": False,
            "backoff": False,
            "fallback": False,
            "resume": False,
            "request_chunk_blocks": REQUEST_CHUNK_BLOCKS,
            "fee_history_requests_per_transport": REQUEST_COUNT,
            "boundary_header_requests_per_transport": BOUNDARY_HEADER_REQUESTS,
            "finalized_header_requests_per_transport": 1,
            "epoch_and_confirmation_header_requests_per_transport": (
                EPOCH_HEADER_REQUESTS
            ),
            "total_requests_per_transport": TOTAL_RPC_REQUESTS_PER_TRANSPORT,
        },
        "range": {
            "first_requested_block": FIRST_REQUESTED_BLOCK,
            "last_requested_block": LAST_REQUESTED_BLOCK,
            "last_retained_block": LAST_RETAINED_BLOCK,
            "terminal_padding_blocks_requested": TERMINAL_PADDING_BLOCKS,
            "terminal_padding_first_block": LAST_RETAINED_BLOCK + 1,
            "terminal_padding_last_block": LAST_REQUESTED_BLOCK,
            "terminal_padding_disposition": "discarded_before_epoch_normalization",
            "terminal_padding_entered_normalized_epochs": 0,
            "last_request_before_frozen_2026_06_boundary": (
                LAST_REQUESTED_BLOCK < PRE_2026_06_BOUNDARY_BLOCK
            ),
        },
        "epochs": {
            "first_epoch_id": FIRST_EPOCH_ID,
            "last_epoch_id": LAST_EPOCH_ID,
            "epoch_size_blocks": EPOCH_SIZE_BLOCKS,
            "rows": EPOCH_COUNT,
            "confirmation_blocks_after_end": CONFIRMATION_BLOCKS,
            "base_fee_vector_sha256_implementation": (
                "training.preregister_ethereum_settlement_demand_impulse."
                "base_fee_vector_sha256"
            ),
            "gas_ratio_arithmetic": "decimal",
            "gas_ratio_decimal_precision": GAS_RATIO_DECIMAL_PRECISION,
        },
        "validation": {
            "chain_id": CHAIN_ID,
            "boundary_audit": list(boundary_audit),
            "common_finalized_head": finalized.number,
            "common_finalized_head_hash": finalized.block_hash,
            "common_finalized_head_at_or_after_last_confirmation": True,
            "dual_provider_response_differences": 0,
            "shortened_responses": 0,
            "next_base_fee_overlap_differences": 0,
            "epoch_end_header_differences": 0,
            "confirmation_header_differences": 0,
        },
        "outputs": {
            "raw_chunks": {
                "path": str(raw_path.relative_to(REPOSITORY_ROOT))
                if raw_path.is_relative_to(REPOSITORY_ROOT)
                else str(raw_path),
                "format": "deterministic gzip NDJSON",
                "rows": REQUEST_COUNT,
                "bytes": raw_size,
                "sha256": raw_sha256,
            },
            "normalized_epochs": {
                "path": str(epoch_path.relative_to(REPOSITORY_ROOT))
                if epoch_path.is_relative_to(REPOSITORY_ROOT)
                else str(epoch_path),
                "format": "deterministic gzip CSV",
                "rows": EPOCH_COUNT,
                "columns": list(EPOCH_COLUMNS),
                "bytes": len(epoch_bytes),
                "sha256": _sha256_bytes(epoch_bytes),
            },
            "manifest": {
                "path": str(manifest_path.relative_to(REPOSITORY_ROOT))
                if manifest_path.is_relative_to(REPOSITORY_ROOT)
                else str(manifest_path),
            },
        },
        "outcome_boundary": {
            "ethereum_source_values_opened": True,
            "btc_market_rows_opened": 0,
            "comparator_rows_opened": 0,
            "funding_rows_opened": 0,
            "return_or_pnl_rows_opened": 0,
            "outcomes_opened": False,
        },
    }
    return {
        **core,
        "manifest_hash": _sha256_bytes(_canonical_json_bytes(core)),
    }


def _execute_source_replay(
    cfg: _SyntheticConfig,
    *,
    clients: Sequence[RpcLike],
    claim_binding: Mapping[str, Any],
    pre_replay_protocol_seal: Mapping[str, Any],
    production: bool,
) -> dict[str, Any]:
    """Execute a pre-authorized replay into already-selected output paths."""

    if production:
        selected = tuple(
            _repository_path(path)
            for path in (
                cfg.raw_output,
                cfg.epoch_output,
                cfg.manifest_output,
            )
        )
        if selected != _canonical_output_paths():
            raise TerminalSourceFailure(
                "production replay outputs differ from canonical paths"
            )
        if tuple(client.url for client in clients) != TRANSPORTS or any(
            not isinstance(client, HttpJsonRpcClient) for client in clients
        ):
            raise TerminalSourceFailure(
                "production replay requires the real frozen clients"
            )
        claim_path = _repository_path(REPLAY_CLAIM_PATH)
        if (
            claim_binding.get("path") != REPLAY_CLAIM_PATH.as_posix()
            or not claim_path.is_file()
            or _sha256_file(claim_path) != claim_binding.get("sha256")
        ):
            raise TerminalSourceFailure(
                "production replay claim is absent or differs"
            )
    else:
        _assert_synthetic_outputs(cfg)
    if REQUEST_COUNT * REQUEST_CHUNK_BLOCKS != (
        LAST_REQUESTED_BLOCK - FIRST_REQUESTED_BLOCK + 1
    ):
        raise AssertionError("ESDI-288 request geometry is internally inconsistent")
    if LAST_REQUESTED_BLOCK - LAST_RETAINED_BLOCK != TERMINAL_PADDING_BLOCKS:
        raise AssertionError("ESDI-288 terminal padding is internally inconsistent")
    output_paths = tuple(
        _repository_candidate(path)
        for path in (cfg.raw_output, cfg.epoch_output, cfg.manifest_output)
    )
    if production:
        final_directory = _canonical_output_directory()
        if final_directory.exists() or final_directory.is_symlink():
            raise TerminalSourceFailure(
                "canonical source directory exists before replay"
            )
    else:
        _preflight_outputs(output_paths)
    working_paths: dict[Path, Path] = {}
    clean_working_paths = not production
    try:
        if production:
            staged_paths = _prepare_production_stage(
                str(claim_binding.get("claim_hash"))
            )
            working_paths = dict(zip(output_paths, staged_paths))
        else:
            for path in output_paths:
                working_paths[path] = _temporary_path(path)
        if tuple(client.url for client in clients) != TRANSPORTS:
            raise TerminalSourceFailure(
                "client transports differ from frozen transports"
            )

        validate_chain_ids(clients)
        boundary_audit = validate_boundaries(clients)
        finalized = validate_common_finalized_head(clients)

        retained_fees: list[int] = []
        retained_ratios: list[Decimal] = []
        epoch_rows: list[dict[str, Any]] = []
        previous_next_base_fee: int | None = None
        next_epoch_id = FIRST_EPOCH_ID
        discarded_padding = 0

        with working_paths[output_paths[0]].open(
            "xb" if production else "wb"
        ) as raw_handle:
            with gzip.GzipFile(
                fileobj=raw_handle,
                mode="wb",
                filename="",
                compresslevel=9,
                mtime=0,
            ) as raw_stream:
                for request_index in range(REQUEST_COUNT):
                    start = (
                        FIRST_REQUESTED_BLOCK
                        + request_index * REQUEST_CHUNK_BLOCKS
                    )
                    chunk = _fee_history_pair(
                        clients,
                        start=start,
                        block_count=REQUEST_CHUNK_BLOCKS,
                    )
                    previous_next_base_fee = validate_next_base_fee_overlap(
                        previous_next_base_fee, chunk
                    )
                    raw_stream.write(
                        _canonical_json_bytes(
                            {
                                "request_index": request_index,
                                "first_block": start,
                                "last_block": (
                                    start + REQUEST_CHUNK_BLOCKS - 1
                                ),
                                "oldestBlock": hex(chunk.oldest_block),
                                "baseFeePerGas": [
                                    hex(value) for value in chunk.base_fees
                                ],
                                "gasUsedRatio": list(
                                    chunk.gas_ratio_strings
                                ),
                            },
                            trailing_lf=True,
                        )
                    )
                    for offset, (base_fee, gas_ratio) in enumerate(
                        zip(chunk.base_fees[:-1], chunk.gas_ratios)
                    ):
                        block = start + offset
                        if block > LAST_RETAINED_BLOCK:
                            discarded_padding += 1
                            continue
                        retained_fees.append(base_fee)
                        retained_ratios.append(gas_ratio)
                        if len(retained_fees) != EPOCH_SIZE_BLOCKS:
                            continue
                        epoch_start, epoch_end, confirmation = (
                            prereg.epoch_blocks(next_epoch_id)
                        )
                        if epoch_start != block - EPOCH_SIZE_BLOCKS + 1:
                            raise TerminalSourceFailure(
                                "normalized epoch range is nonmonotone"
                            )
                        end_header = _header_pair(
                            clients,
                            hex(epoch_end),
                            expected_number=epoch_end,
                        )
                        confirmation_header = _header_pair(
                            clients,
                            hex(confirmation),
                            expected_number=confirmation,
                        )
                        epoch_rows.append(
                            build_epoch_row(
                                epoch_id=next_epoch_id,
                                base_fees=retained_fees,
                                gas_ratios=retained_ratios,
                                end_header=end_header,
                                confirmation_header=confirmation_header,
                            )
                        )
                        retained_fees = []
                        retained_ratios = []
                        next_epoch_id += 1
        if production:
            working_paths[output_paths[0]].chmod(0o444)
        _fsync_file(working_paths[output_paths[0]])
        if production:
            _fsync_directory(working_paths[output_paths[0]].parent)

        if retained_fees or retained_ratios:
            raise TerminalSourceFailure(
                "partial epoch remained after frozen replay"
            )
        if len(epoch_rows) != EPOCH_COUNT or next_epoch_id != LAST_EPOCH_ID + 1:
            raise TerminalSourceFailure("normalized epoch count differs")
        if discarded_padding != TERMINAL_PADDING_BLOCKS:
            raise TerminalSourceFailure("terminal padding discard count differs")
        if production:
            validate_protocol_seal(pre_replay_protocol_seal)

        epoch_bytes = _csv_gzip(epoch_rows)
        raw_temporary = working_paths[output_paths[0]]
        manifest = _manifest(
            cfg=cfg,
            raw_size=raw_temporary.stat().st_size,
            raw_sha256=_sha256_file(raw_temporary),
            epoch_bytes=epoch_bytes,
            finalized=finalized,
            boundary_audit=boundary_audit,
            builder_sha256=_sha256_file(BUILDER_PATH),
            claim_binding=claim_binding,
            pre_replay_protocol_seal=pre_replay_protocol_seal,
        )
        manifest_bytes = _canonical_json_bytes(manifest, trailing_lf=True)

        if production:
            _durable_write_once(working_paths[output_paths[1]], epoch_bytes)
            _durable_write_once(working_paths[output_paths[2]], manifest_bytes)
            return _publish_staged_generation(
                staged_paths=tuple(working_paths[path] for path in output_paths),
                claim_binding=claim_binding,
                pre_replay_protocol_seal=pre_replay_protocol_seal,
            )
        working_paths[output_paths[1]].write_bytes(epoch_bytes)
        working_paths[output_paths[2]].write_bytes(manifest_bytes)
        atomic_publish(working_paths)
        return manifest
    finally:
        if clean_working_paths:
            for temporary in working_paths.values():
                temporary.unlink(missing_ok=True)


def _build_source_synthetic(
    cfg: _SyntheticConfig, *, clients: Sequence[RpcLike]
) -> dict[str, Any]:
    """Synthetic-only custom-output entry point; never creates a replay claim."""

    _assert_synthetic_outputs(cfg)
    validate_preregistration()
    seal, claim = _synthetic_bindings(cfg)
    return _execute_source_replay(
        cfg,
        clients=clients,
        claim_binding=claim,
        pre_replay_protocol_seal=seal,
        production=False,
    )


def build_source() -> dict[str, Any]:
    """Claim and execute the sole canonical production source replay."""

    claim_path = _repository_candidate(REPLAY_CLAIM_PATH)
    output_paths = tuple(
        _repository_candidate(path)
        for path in (
            DEFAULT_RAW_OUTPUT,
            DEFAULT_EPOCH_OUTPUT,
            DEFAULT_MANIFEST_OUTPUT,
        )
    )
    if claim_path.exists() or claim_path.is_symlink():
        return _load_atomically_published_generation()
    if any(path.exists() or path.is_symlink() for path in output_paths):
        raise FileExistsError(
            "ESDI-288 canonical output exists without its one-shot claim; "
            "production replay is forbidden forever"
        )
    _assert_production_replay_available()
    validate_preregistration()
    protocol_seal = current_protocol_seal(require_committed=True)
    claim = _create_production_claim(protocol_seal)
    clients: tuple[RpcLike, RpcLike] = tuple(
        HttpJsonRpcClient(
            url,
            timeout_seconds=PRODUCTION_TIMEOUT_SECONDS,
        )
        for url in TRANSPORTS
    )
    cfg = _SyntheticConfig(
        raw_output=DEFAULT_RAW_OUTPUT.as_posix(),
        epoch_output=DEFAULT_EPOCH_OUTPUT.as_posix(),
        manifest_output=DEFAULT_MANIFEST_OUTPUT.as_posix(),
    )
    return _execute_source_replay(
        cfg,
        clients=clients,
        claim_binding=claim,
        pre_replay_protocol_seal=protocol_seal,
        production=True,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    parse_args(argv)
    manifest = build_source()
    print(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
