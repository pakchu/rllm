"""Freeze CSPG-288 before decoding new source states or market outcomes."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
import gzip
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np


POLICY_ID = "CSPG-288"
PROTOCOL_VERSION = "cboe_cross_surface_pressure_grammar_preregistration_v1"
DEFAULT_OUTPUT = (
    "results/cboe_cross_surface_pressure_grammar_"
    "preregistration_2026-07-24.json"
)

BOUNDARY_DOCUMENT = (
    "docs/cboe-cross-surface-pressure-grammar-boundary-2026-07-24.md"
)
BOUNDARY_DOCUMENT_SHA256 = (
    "0b6feb15d1e7b616b5b65bb266b15db7e3fdcf82765b5848c76d68e804cb39f2"
)
BOUNDARY_COMMIT = "e210951"
MECHANISM_DOCUMENT = (
    "docs/cboe-cross-surface-pressure-grammar-"
    "mechanism-decision-2026-07-24.md"
)
MECHANISM_DOCUMENT_SHA256 = (
    "3d8171b615580dd4c20d2b577265f52ac2646fb9697f73d3abd6ff8b82c3374f"
)
MECHANISM_COMMIT = "ad2907a"

TERM_SOURCE = (
    "data/cboe_volatility_term_structure_2018_2023/"
    "cboe_vix_term_structure_2018-01-01_2023-12-31.csv.gz"
)
TERM_SOURCE_SHA256 = (
    "6f1b2f7f3a5b1e4d5001d673e6ff54374791879c278248ce27b3d610e4f75dc7"
)
TERM_HEADER_SHA256 = (
    "b2fc60cae8d080d3b47a1a55c48438b63f91530cc345f1b6ef78cee05cc57e20"
)
TERM_MANIFEST = (
    "data/cboe_volatility_term_structure_2018_2023/build_manifest.json"
)
TERM_MANIFEST_SHA256 = (
    "42b2a35ad131bd63574d2adcf684e28766dc3060fa645fc749df10dd3fb27f27"
)
TERM_ALLOWLIST = (
    "observation_date",
    "VIX9D_close",
    "VIX_close",
    "VIX3M_close",
)

TAIL_SOURCE = (
    "data/cboe_tail_risk_2018_2023/"
    "cboe_tail_risk_2018-01-01_2023-12-31.csv.gz"
)
TAIL_SOURCE_SHA256 = (
    "cdde3f8d4bb1e23d00b192f5f9ef759aefba9087be5fd60653e9c02479dfa41a"
)
TAIL_HEADER_SHA256 = (
    "bdc2e42c1d356ebd815c491af9b20211d1bc8f2781c0917d92bbf04f1f0a5dc3"
)
TAIL_MANIFEST = "data/cboe_tail_risk_2018_2023/build_manifest.json"
TAIL_MANIFEST_SHA256 = (
    "9ef80ef3034c93d97c5b2a8160b2502527287d570d15f9d7166d631d9866c7bd"
)
TAIL_ALLOWLIST = (
    "observation_date",
    "SKEW_close",
    "VVIX_close",
    "VIX_close",
)

OPTION_SOURCE = (
    "data/cboe_option_flow_2020_2023/"
    "cboe_option_flow_2020-01-01_2023-12-31.csv.gz"
)
OPTION_SOURCE_SHA256 = (
    "35ef106ef01e3abadbcb4a6227187dd1d7cf2722191bd146bac06d08d1684a78"
)
OPTION_HEADER_SHA256 = (
    "a98314aa376428c5d237837121305c5cc4c4892e25ea3db3127d466b451281d7"
)
OPTION_MANIFEST = "data/cboe_option_flow_2020_2023/build_manifest.json"
OPTION_MANIFEST_SHA256 = (
    "0a513b146ad5857d9ab7311e978152c308de64db8ef29c4d463eb07ea503089e"
)
OPTION_ALLOWLIST = (
    "observation_date",
    "total_volume",
    "index_call_volume",
    "index_put_volume",
    "index_volume",
    "equity_call_volume",
    "equity_put_volume",
    "vix_call_volume",
    "vix_put_volume",
)

MARKET_SOURCE = (
    "data/binance_um_kline_reference_btc_2020_2023/"
    "BTCUSDT_5m_2020-01-01_2023-12-31.csv.gz"
)
MARKET_SOURCE_SHA256 = (
    "e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d"
)
MARKET_MANIFEST = (
    "data/binance_um_kline_reference_btc_2020_2023/build_manifest.json"
)
MARKET_MANIFEST_SHA256 = (
    "c04fbbd299cc748a6745c0ef030787da4d560833c744c81c98dd8840efc7913e"
)
FUNDING_SOURCE = "data/binance_um_btcusdt_funding_marks_2020_2023.csv.gz"
FUNDING_SOURCE_SHA256 = (
    "3284bbb6bb67946acb673c6b67459543e217f752589e1d47b6c7c3b659f733e6"
)
FUNDING_MANIFEST = (
    "results/binance_um_btcusdt_funding_marks_2020_2023_"
    "manifest_2026-07-17.json"
)
FUNDING_MANIFEST_SHA256 = (
    "a0b2d27e1aa8cf2d9ab8cb659b598ee0a6d7bd25401c9e10ae92d1a74415845b"
)

MODEL_ID = "google/gemma-2-2b-it"
MODEL_REVISION = "299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8"
MODEL_FILES = {
    "config.json": (
        "eacec6c5ca317a87ed2c46789d9705b9274db5027e7ba59da739bfae23addb55"
    ),
    "generation_config.json": (
        "a543a5d299bc2b20c52bd87ed174f561266510b57a392e12b5b5d758d798ce05"
    ),
    "tokenizer_config.json": (
        "cb32b7929c62608d46572e813112b3ad8a841fb98fdd6a4da8559e368a951c89"
    ),
    "tokenizer.json": (
        "3f289bc05132635a8bc7aca7aa21255efd5e18f3710f43e3cdb96bcd41be4922"
    ),
    "model.safetensors.index.json": (
        "ada0043f3e3b2e5ab2f445cad9c0fbbf9d91ad444675e6a82b822591c63abf5a"
    ),
    "model-00001-of-00002.safetensors": (
        "532d792c9178805064170a3ec485b7dedbfccc6fd297b92c31a6091b6c7e41bf"
    ),
    "model-00002-of-00002.safetensors": (
        "6d6d9ce84db398fb6e0191f91542e5da0a73da2cb695e172a24edc2146dc8d20"
    ),
    "special_tokens_map.json": (
        "baec30ea10906f16adb8c18af7a34023002c1746542612b8b41c9f09e1351351"
    ),
}

TOKEN_SCHEMA: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("term_level", ("LOW", "MID", "HIGH")),
    ("tail_level", ("LOW", "MID", "HIGH")),
    ("option_level", ("LOW", "MID", "HIGH")),
    ("term_change", ("DOWN", "SAME", "UP")),
    ("tail_change", ("DOWN", "SAME", "UP")),
    ("option_change", ("DOWN", "SAME", "UP")),
    ("stress_leader", ("TERM", "TAIL", "OPTION", "TIE")),
    ("relief_leader", ("TERM", "TAIL", "OPTION", "TIE")),
    ("dispersion", ("COMPRESSED", "SEPARATED", "FRACTURED")),
    ("agreement", ("UNISON", "ADJACENT", "POLARIZED")),
    ("topology_transition", ("STABLE", "ROTATING", "RESET")),
    ("pressure_breadth", ("FALLING", "BALANCED", "RISING")),
)
TOKEN_COLUMNS = tuple(name for name, _ in TOKEN_SCHEMA)
TOKEN_VOCABULARY = {name: values for name, values in TOKEN_SCHEMA}
SURFACES = ("TERM", "TAIL", "OPTION")
NEUTRAL_CODES = ("Q1", "Q2")
TASKS = ("ADMISSION", "DIRECTION")
ACTION_NAMES = ("ABSTAIN", "LONG", "SHORT")
FORBIDDEN_COMPARATOR_PATHS = (
    "data/premium_compression_breakout_relay_clocks_2020_2026.csv.gz",
    "data/premium_snapback_recenter_clocks_2020_2026.csv.gz",
)


@dataclass(frozen=True)
class Policy:
    policy_id: str = POLICY_ID
    rank_lookback_observations: int = 252
    rank_minimum_prior_observations: int = 126
    level_low_numerator: int = 1
    level_low_denominator: int = 3
    level_high_numerator: int = 2
    level_high_denominator: int = 3
    dispersion_compressed_numerator: int = 1
    dispersion_compressed_denominator: int = 6
    dispersion_fractured_numerator: int = 1
    dispersion_fractured_denominator: int = 3
    availability_local_hour: int = 9
    availability_local_minute: int = 30
    entry_local_hour: int = 9
    entry_local_minute: int = 35
    hold_bars: int = 288
    leverage: float = 0.50
    base_cost_notional_per_side: float = 0.0006
    stress_cost_notional_per_side: float = 0.0010
    held_path_drawdown_penalty: float = 1.0 / 3.0
    trade_utility_hurdle_account: float = 0.0010
    preference_margin: float = 0.0005
    random_seed: int = 20_260_724


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


def csv_header_bytes(path: str | Path) -> bytes:
    source = Path(path)
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rb") as handle:
        header = handle.readline()
    if not header.endswith(b"\n") or b"\n" in header[:-1]:
        raise RuntimeError(f"CSPG-288 CSV header is not one LF line: {path}")
    return header


def csv_header(path: str | Path) -> list[str]:
    return next(csv.reader([csv_header_bytes(path).decode("utf-8").rstrip("\n")]))


def sha256_csv_header(path: str | Path) -> str:
    return hashlib.sha256(csv_header_bytes(path)).hexdigest()


def strict_prior_midrank(
    current: float,
    prior_values: Iterable[float],
    *,
    minimum: int = Policy().rank_minimum_prior_observations,
    maximum: int = Policy().rank_lookback_observations,
) -> float:
    prior = np.asarray(list(prior_values), dtype=np.float64)
    if len(prior) < minimum:
        raise ValueError("CSPG strict-prior history is not ready")
    if len(prior) > maximum:
        prior = prior[-maximum:]
    value = np.float64(current)
    if not np.isfinite(value) or not np.isfinite(prior).all():
        raise ValueError("CSPG rank values must be finite")
    return float(
        (np.count_nonzero(prior < value) + 0.5 * np.count_nonzero(prior == value))
        / len(prior)
    )


def pressure_level(value: float) -> str:
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError("CSPG pressure must be finite and in [0,1]")
    if numeric < 1.0 / 3.0:
        return "LOW"
    if numeric > 2.0 / 3.0:
        return "HIGH"
    return "MID"


def change_token(current_level: str, previous_level: str) -> str:
    index = {"LOW": 0, "MID": 1, "HIGH": 2}
    if current_level not in index or previous_level not in index:
        raise ValueError("CSPG pressure level is invalid")
    delta = index[current_level] - index[previous_level]
    return "DOWN" if delta < 0 else "UP" if delta > 0 else "SAME"


def extreme_leader(pressures: Mapping[str, float], *, highest: bool) -> str:
    if tuple(pressures) != SURFACES:
        raise ValueError("CSPG pressure surface order changed")
    values = {name: float(pressures[name]) for name in SURFACES}
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("CSPG pressure is non-finite")
    extreme = (max if highest else min)(values.values())
    winners = [name for name in SURFACES if values[name] == extreme]
    return winners[0] if len(winners) == 1 else "TIE"


def dispersion_token(pressures: Mapping[str, float]) -> str:
    if tuple(pressures) != SURFACES:
        raise ValueError("CSPG pressure surface order changed")
    values = [float(pressures[name]) for name in SURFACES]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("CSPG pressure is non-finite")
    spread = max(values) - min(values)
    if spread < 1.0 / 6.0:
        return "COMPRESSED"
    if spread < 1.0 / 3.0:
        return "SEPARATED"
    return "FRACTURED"


def agreement_token(levels: Mapping[str, str]) -> str:
    if tuple(levels) != SURFACES:
        raise ValueError("CSPG level surface order changed")
    index = {"LOW": 0, "MID": 1, "HIGH": 2}
    try:
        values = [index[levels[name]] for name in SURFACES]
    except KeyError as exc:
        raise ValueError("CSPG pressure level is invalid") from exc
    spread = max(values) - min(values)
    return ("UNISON", "ADJACENT", "POLARIZED")[spread]


def topology_transition(
    current: Mapping[str, str],
    previous: Mapping[str, str],
) -> str:
    fields = (
        "term_level",
        "tail_level",
        "option_level",
        "stress_leader",
        "relief_leader",
    )
    if any(field not in current or field not in previous for field in fields):
        raise ValueError("CSPG topology transition fields are incomplete")
    changed = sum(current[field] != previous[field] for field in fields)
    if changed <= 1:
        return "STABLE"
    if changed <= 3:
        return "ROTATING"
    return "RESET"


def pressure_breadth(changes: Mapping[str, str]) -> str:
    fields = ("TERM", "TAIL", "OPTION")
    if tuple(changes) != fields:
        raise ValueError("CSPG change surface order changed")
    score_map = {"DOWN": -1, "SAME": 0, "UP": 1}
    try:
        score = sum(score_map[changes[name]] for name in fields)
    except KeyError as exc:
        raise ValueError("CSPG change token is invalid") from exc
    return "FALLING" if score < 0 else "RISING" if score > 0 else "BALANCED"


def validate_tokens(tokens: Mapping[str, str]) -> dict[str, str]:
    if tuple(tokens) != TOKEN_COLUMNS:
        raise ValueError("CSPG token order or schema changed")
    normalized = {name: str(tokens[name]) for name in TOKEN_COLUMNS}
    for name, value in normalized.items():
        if value not in TOKEN_VOCABULARY[name]:
            raise ValueError(f"CSPG token level is invalid: {name}={value}")
    return normalized


def neutral_code_orders() -> tuple[tuple[str, str], ...]:
    return tuple(itertools.permutations(NEUTRAL_CODES))


def opportunity_times(observation_date: date | str) -> dict[str, datetime]:
    source_day = (
        date.fromisoformat(observation_date)
        if isinstance(observation_date, str)
        else observation_date
    )
    if not isinstance(source_day, date):
        raise TypeError("CSPG source date must be a date or ISO date string")
    policy = Policy()
    next_day = source_day + timedelta(days=1)
    new_york = ZoneInfo("America/New_York")
    available_local = datetime.combine(
        next_day,
        time(policy.availability_local_hour, policy.availability_local_minute),
        tzinfo=new_york,
    )
    entry_local = datetime.combine(
        next_day,
        time(policy.entry_local_hour, policy.entry_local_minute),
        tzinfo=new_york,
    )
    available = available_local.astimezone(timezone.utc)
    entry = entry_local.astimezone(timezone.utc)
    return {
        "signal_available": available,
        "entry": entry,
        "exit": entry + timedelta(minutes=Policy().hold_bars * 5),
    }


def reserve_intervals(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: (row["entry"], row["exit"], str(row.get("id", ""))),
    )
    previous_exit: datetime | None = None
    result: list[dict[str, Any]] = []
    for row in ordered:
        entry = row["entry"]
        exit_time = row["exit"]
        if (
            not isinstance(entry, datetime)
            or not isinstance(exit_time, datetime)
            or entry.tzinfo is None
            or exit_time.tzinfo is None
            or exit_time <= entry
        ):
            raise ValueError("CSPG reservation interval is invalid")
        reserved = previous_exit is None or entry >= previous_exit
        row["reserved"] = reserved
        result.append(row)
        if reserved:
            previous_exit = exit_time
    return result


def comparator_contracts() -> list[dict[str, Any]]:
    contracts = [
        {
            "id": "VTR",
            "path": "results/cboe_volatility_term_rotation_clocks_2026-07-17.csv.gz",
            "sha256": "47f4ca447daa2b03a0827ad243ed1107eb34a37e5d7bab18ecd3c4331736959d",
            "header_sha256": "9628d5d9bb26e18964e87d96e33119d8eba8b11208ed516ce18a336b8e04041c",
            "group_column": "control",
        },
        {
            "id": "THD",
            "path": "results/cboe_tail_hedge_disagreement_clocks_2026-07-18.csv.gz",
            "sha256": "0e19455e2fb5ab2d36cc996c9adf514adc85c69dd1a325562344a8015464d546",
            "header_sha256": "ed0b1417ea6946fc8427f47f95b5b4dbcd6f377fad8da62484a2c95cbc85da92",
            "group_column": "control",
        },
        {
            "id": "IHM",
            "path": "results/cboe_institutional_hedge_migration_clocks_2026-07-18.csv.gz",
            "sha256": "5e04cffacb1754c3111fcc32b09d72f06b546a4803b40c77d655a9787b015c0b",
            "header_sha256": "6a763bf874f4cd5dc0ea16433d30868c3dee92a70e74f3dbcbfe6329a2d6d2ee",
            "group_column": "control",
        },
        {
            "id": "CXRT",
            "path": "data/cboe_cross_surface_risk_transfer_clocks_2020_2023.csv.gz",
            "sha256": "b3cc6f3d6a19cb39ef63ec0ba9908c983ce03c56a0c7dd8786e51c2ef1c0885f",
            "header_sha256": "d66a8a9e0593867005d8f47f026bd05556a9ff3c2c3a33e4b4dfc914d99c8591",
            "group_column": "control",
        },
        {
            "id": "OPRR",
            "path": "data/cboe_option_pressure_rank_rotation_clocks_2020_2023.csv.gz",
            "sha256": "a5c15e0d6444f79239276fb9c3da0555dea27a52eda254e7425d9b223d30d46c",
            "header_sha256": "4789240b9dc9d024165094f4c0e6f7d42d06776993891261715ad5dd58bf93bf",
            "group_column": "control",
        },
        {
            "id": "CCHR-live-pre2024",
            "path": "results/cchr_live_portfolio_pure_clocks_2026-07-21.csv.gz",
            "sha256": "73d6efbd35b3be64b0fa04fa9c8cb2db25866ef884f19b1ae673949e22a42b08",
            "header_sha256": "da21cfc42a55581971c304cc30122a72f9d062a7601db59a9702ec35504acb9a",
            "group_column": "candidate_id",
            "rows": 440,
        },
    ]
    for item in contracts:
        item.update(
            {
                "entry_column": "entry_time",
                "exit_column": "exit_time",
                "side_column": "side",
                "loader_allowlist": [
                    item["group_column"],
                    "entry_time",
                    "exit_time",
                    "side",
                ],
                "side_encoding": {"LONG": 1, "SHORT": -1},
                "group_selection": "every nonempty group over common coverage",
                "missing_common_coverage": "fail",
            }
        )
    return contracts


def frozen_dependencies() -> dict[str, str]:
    dependencies = {
        BOUNDARY_DOCUMENT: BOUNDARY_DOCUMENT_SHA256,
        MECHANISM_DOCUMENT: MECHANISM_DOCUMENT_SHA256,
        TERM_SOURCE: TERM_SOURCE_SHA256,
        TERM_MANIFEST: TERM_MANIFEST_SHA256,
        TAIL_SOURCE: TAIL_SOURCE_SHA256,
        TAIL_MANIFEST: TAIL_MANIFEST_SHA256,
        OPTION_SOURCE: OPTION_SOURCE_SHA256,
        OPTION_MANIFEST: OPTION_MANIFEST_SHA256,
        MARKET_SOURCE: MARKET_SOURCE_SHA256,
        MARKET_MANIFEST: MARKET_MANIFEST_SHA256,
        FUNDING_SOURCE: FUNDING_SOURCE_SHA256,
        FUNDING_MANIFEST: FUNDING_MANIFEST_SHA256,
    }
    dependencies.update(
        {item["path"]: item["sha256"] for item in comparator_contracts()}
    )
    return dependencies


def validate_frozen_dependencies() -> None:
    for path, expected in frozen_dependencies().items():
        if sha256_file(path) != expected:
            raise RuntimeError(f"CSPG-288 frozen dependency changed: {path}")
    for path, expected, allowlist in (
        (TERM_SOURCE, TERM_HEADER_SHA256, TERM_ALLOWLIST),
        (TAIL_SOURCE, TAIL_HEADER_SHA256, TAIL_ALLOWLIST),
        (OPTION_SOURCE, OPTION_HEADER_SHA256, OPTION_ALLOWLIST),
    ):
        if sha256_csv_header(path) != expected:
            raise RuntimeError(f"CSPG-288 source header changed: {path}")
        header = csv_header(path)
        if [column for column in header if column in allowlist] != list(allowlist):
            raise RuntimeError(f"CSPG-288 source allowlist/order changed: {path}")
    for item in comparator_contracts():
        if sha256_csv_header(item["path"]) != item["header_sha256"]:
            raise RuntimeError(f"CSPG-288 comparator header changed: {item['id']}")
    if any(path in frozen_dependencies() for path in FORBIDDEN_COMPARATOR_PATHS):
        raise RuntimeError("CSPG forbidden comparator entered dependency set")


def validate_model_snapshot() -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
            allow_patterns=sorted(MODEL_FILES),
        )
    )
    for filename, expected in MODEL_FILES.items():
        path = snapshot / filename
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"CSPG Gemma2 frozen file mismatch: {filename}")
    return {
        "validated": True,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": dict(MODEL_FILES),
    }


def _source_contract(
    path: str,
    sha256: str,
    header_sha256: str,
    manifest: str,
    manifest_sha256: str,
    allowlist: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "path": path,
        "sha256": sha256,
        "header_sha256": header_sha256,
        "manifest": manifest,
        "manifest_sha256": manifest_sha256,
        "allowlist": list(allowlist),
        "loader": "pandas.read_csv(usecols=allowlist); no load-and-drop",
    }


def _core_manifest() -> dict[str, Any]:
    policy = Policy()
    return {
        "protocol_version": PROTOCOL_VERSION,
        "policy": asdict(policy),
        "frozen_documents": {
            "boundary": {
                "path": BOUNDARY_DOCUMENT,
                "sha256": BOUNDARY_DOCUMENT_SHA256,
                "commit": BOUNDARY_COMMIT,
            },
            "mechanism": {
                "path": MECHANISM_DOCUMENT,
                "sha256": MECHANISM_DOCUMENT_SHA256,
                "commit": MECHANISM_COMMIT,
            },
        },
        "research_history_boundary": {
            "prior_cboe_source_rows_seen": True,
            "prior_cboe_family_outcomes_seen": True,
            "cxrt_oprr_source_support_seen": True,
            "cxrt_oprr_market_outcomes_opened": False,
            "new_cspg_pressure_or_token_seen": False,
            "new_cspg_incidence_seen": False,
            "new_cspg_market_outcome_seen": False,
            "global_pristine_holdout_claimed": False,
            "claim_scope": "candidate-level frozen 2023 outcome window",
        },
        "source_contracts": {
            "term": _source_contract(
                TERM_SOURCE,
                TERM_SOURCE_SHA256,
                TERM_HEADER_SHA256,
                TERM_MANIFEST,
                TERM_MANIFEST_SHA256,
                TERM_ALLOWLIST,
            ),
            "tail": _source_contract(
                TAIL_SOURCE,
                TAIL_SOURCE_SHA256,
                TAIL_HEADER_SHA256,
                TAIL_MANIFEST,
                TAIL_MANIFEST_SHA256,
                TAIL_ALLOWLIST,
            ),
            "option": _source_contract(
                OPTION_SOURCE,
                OPTION_SOURCE_SHA256,
                OPTION_HEADER_SHA256,
                OPTION_MANIFEST,
                OPTION_MANIFEST_SHA256,
                OPTION_ALLOWLIST,
            ),
            "intersection": "exact dates after independent causal histories",
            "missing_policy": "no fill, carry, interpolation, or zero replacement",
            "vix_cross_panel_equality": True,
            "pre_2024_only": True,
        },
        "rank_contract": {
            "lookback": policy.rank_lookback_observations,
            "minimum": policy.rank_minimum_prior_observations,
            "formula": "(count(prior<x)+0.5*count(prior==x))/len(prior)",
            "current_excluded": True,
            "source_histories_independent_before_join": True,
            "future_append_invariant": True,
        },
        "surface_algebra": {
            "term_pressure": "mean(rank(log(VIX9D/VIX)),rank(log(VIX/VIX3M)))",
            "tail_pressure": "mean(rank(log(SKEW/100)),rank(log(VVIX/VIX)))",
            "institutional_gap": (
                "log((index_put+0.5)/(index_call+0.5))"
                "-log((equity_put+0.5)/(equity_call+0.5))"
            ),
            "vix_call_pressure": "log((vix_call+0.5)/(vix_put+0.5))",
            "index_share": "log((index_volume+1)/(total_volume+1))",
            "option_pressure": (
                "mean(strict-prior ranks of each current-minus-prior option "
                "primitive delta)"
            ),
        },
        "token_contract": {
            "ordered_schema": [
                {"name": name, "values": list(values)}
                for name, values in TOKEN_SCHEMA
            ],
            "count": len(TOKEN_SCHEMA),
            "pressure_level": "<1/3 LOW; >2/3 HIGH; else MID",
            "change": "compare current/previous level indices",
            "extreme_ties": "TIE without epsilon",
            "dispersion": "<1/6 COMPRESSED; <1/3 SEPARATED; else FRACTURED",
            "agreement": "range of three level indices: 0/1/2",
            "transition": "0..1 STABLE; 2..3 ROTATING; 4..5 RESET",
            "breadth": "sign(count(UP)-count(DOWN))",
            "unknown_downstream_level": "ABSTAIN",
            "position": "deterministic guard, not a token",
            "forbidden": [
                "raw_values_or_ranks",
                "cxrt_votes_majority_side_or_run",
                "oprr_rotation_eligibility",
                "date_time_row_or_source_identity",
                "btc_price_return_funding_premium_oi_kimchi_dxy",
                "reward_action_history_pnl_cagr_mdd_split",
                "free_form_rationale_or_generated_feature",
            ],
        },
        "execution_contract": {
            "source_date": "D",
            "signal_available": "calendar D+1 09:30 America/New_York",
            "entry": "calendar D+1 09:35 America/New_York BTCUSDT 5m open",
            "exit": "entry+288*5m",
            "weekend_holiday_entries": True,
            "future_row_existence_affects_clock": False,
            "zone": "America/New_York via zoneinfo",
            "leverage": policy.leverage,
            "global_action_independent_reservation": True,
            "abstention_releases_reservation": False,
            "split_crossing_keeps_reservation": True,
            "scheduled_exit_only": True,
            "dynamic_exit_or_sizing": False,
            "one_bar_delay": "entry+5m, exit+5m, recompute reservation",
            "one_hour_delay": "entry+60m, exit+60m, recompute reservation",
        },
        "temporal_roles": {
            "initial_fit": ["2020-01-01T00:00:00Z", "2021-01-01T00:00:00Z"],
            "transfer_validation": [
                "2021-01-01T00:00:00Z",
                "2022-01-01T00:00:00Z",
            ],
            "final_fit": ["2020-01-01T00:00:00Z", "2022-01-01T00:00:00Z"],
            "selection": ["2022-01-01T00:00:00Z", "2023-01-01T00:00:00Z"],
            "untouched_eval": [
                "2023-01-01T00:00:00Z",
                "2024-01-01T00:00:00Z",
            ],
            "adaptation": None,
        },
        "source_support_gates": {
            "global_opportunities_min": 820,
            "train_2020_2021_min": 330,
            "year_2020_min": 100,
            "each_year_2021_2023_min": 230,
            "year_2020_active_months_min": 6,
            "each_year_2021_2023_active_months_min": 12,
            "each_half_2021_2023_min": 108,
            "each_quarter_2021_2023_min": 48,
            "year_2020_max_month_share": 0.20,
            "each_year_2021_2023_max_month_share": 0.12,
            "max_entry_gap_days": 10.0,
            "pressure_low_high_each_share_min": 0.08,
            "change_down_up_each_share_min": 0.08,
            "max_level_or_change_share": 0.80,
            "each_nontie_leader_must_occur": True,
            "max_nontie_leader_share": 0.80,
            "each_dispersion_agreement_share_min": 0.03,
            "each_transition_share_min": 0.02,
            "breadth_falling_rising_each_share_min": 0.12,
            "max_breadth_share": 0.75,
            "max_exact_signature_share": 0.05,
            "downstream_levels_must_exist_in_train": True,
            "counts_basis": "token-ready globally reserved split-contained",
            "required_tests": [
                "hash_header_allowlist",
                "date_order_uniqueness_positivity_pre2024",
                "cross_panel_vix_equality",
                "independent_history_before_intersection",
                "current_value_exclusion",
                "midrank_ties",
                "fixed_history",
                "option_delta_history",
                "missing_date_nonfill",
                "future_append",
                "prior_common_transition",
                "level_boundaries",
                "extreme_ties",
                "dst_fixed_calendar_clock",
                "neutral_code_order",
                "action_independent_reservation",
            ],
            "failure_action": "retire CSPG-288 unchanged before outcomes",
        },
        "economic_contract": {
            "market": {
                "path": MARKET_SOURCE,
                "sha256": MARKET_SOURCE_SHA256,
                "manifest": MARKET_MANIFEST,
                "manifest_sha256": MARKET_MANIFEST_SHA256,
            },
            "funding": {
                "path": FUNDING_SOURCE,
                "sha256": FUNDING_SOURCE_SHA256,
                "manifest": FUNDING_MANIFEST,
                "manifest_sha256": FUNDING_MANIFEST_SHA256,
                "interval": "entry<=settlement<=exit",
                "cash": "-side*fixed_quantity*settlement_mark*funding_rate",
                "boundary": "retain debits; drop credits exactly at entry/exit",
            },
            "quantity": "entry_equity*0.5/entry_open; fixed",
            "base_cost_notional_per_side": policy.base_cost_notional_per_side,
            "stress_cost_notional_per_side": policy.stress_cost_notional_per_side,
            "stress_replaces_base": True,
            "strict_mdd": (
                "global/pre-entry HWM; favorable then adverse held OHLC; "
                "funding and virtual adverse exit cost; scheduled exit"
            ),
            "cagr": "full half-open calendar including idle and abstention",
            "weekly_signflip_draws": 100_000,
            "weekly_signflip_seed": policy.random_seed,
        },
        "utility_contract": {
            "abstain": 0.0,
            "trade": (
                "log(max(account_multiplier,1e-12))"
                "-(1/3)*local_strict_drawdown-0.0010"
            ),
            "oracle_tie_priority": list(ACTION_NAMES),
            "admission": "ABSTAIN versus max(LONG,SHORT)",
            "direction": "LONG versus SHORT only when best trade beats ABSTAIN",
            "preference_margin": policy.preference_margin,
            "balancing": None,
            "synthetic_source_symmetry": None,
        },
        "baseline_contract": {
            "fit_transfer": "fit 2020, evaluate unchanged 2021",
            "refit_selection": "refit 2020-2021, evaluate 2022",
            "features": "main token one-hot plus 66 pair conjunctions; min count 3",
            "policies": [
                "always_abstain",
                "always_long",
                "always_short",
                "exact_signature_memory",
                "categorical_naive_bayes_alpha_1",
                "ridge_contextual_value_alpha_100",
                "extra_trees_value_512_depth5_leaf10",
                "32_shuffled_label_nb",
                "32_shuffled_utility_ridge",
                "12_single_token_ridge",
                "12_leave_one_out_ridge",
            ],
            "transfer_2021": {
                "return_positive": True,
                "ratio_min": 0.5,
                "strict_mdd_pct_max": 15.0,
                "both_halves_positive": True,
                "trades_min": 50,
                "each_side_min": 15,
                "each_side_contribution_positive": True,
                "max_action_share": 0.90,
                "stress_and_one_bar_delay_positive": True,
                "weekly_cluster_p_below": 0.25,
            },
            "selection_2022": {
                "return_positive": True,
                "ratio_min": 1.0,
                "strict_mdd_pct_max": 15.0,
                "both_halves_positive": True,
                "trades_min": 60,
                "each_side_min": 20,
                "each_side_contribution_positive": True,
                "max_action_share": 0.85,
                "stress_and_one_bar_delay_positive": True,
                "weekly_cluster_p_below": 0.15,
                "beat_shuffled_and_single_token": True,
            },
        },
        "rllm_contract": {
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "files": dict(MODEL_FILES),
            "loader": "AutoModelForCausalLM",
            "tokenizer": "AutoTokenizer",
            "trust_remote_code": False,
            "environment": {
                "torch": "2.9.0",
                "transformers_git_revision": (
                    "5d7ff4393ab99aa7cadf4cccd1f814dbb799f2bb"
                ),
                "trl": "0.29.0",
                "peft": "0.18.1",
                "bitsandbytes": "0.49.2",
                "numpy": "2.2.6",
                "pandas": "2.3.3",
                "scikit_learn": "1.7.2",
            },
            "quantization": {
                "load_in_4bit": True,
                "quant_type": "nf4",
                "double_quant": True,
                "compute_dtype": "float16",
            },
            "lora": {
                "r": 16,
                "alpha": 32,
                "dropout": 0.05,
                "bias": "none",
                "task_type": "CAUSAL_LM",
                "targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
            },
            "tasks": list(TASKS),
            "neutral_codes": list(NEUTRAL_CODES),
            "mapping": {
                "ADMISSION": {"Q1": "ABSTAIN", "Q2": "TRADE"},
                "DIRECTION": {"Q1": "LONG", "Q2": "SHORT"},
            },
            "code_orders": [list(order) for order in neutral_code_orders()],
            "generation": False,
            "score": "mean completion-token conditional log probability",
            "prior_correction": (
                "adapter_delta=adapted-base on same prompt; subtract task/code "
                "mean adapter_delta over original 2020-2021 fit states"
            ),
            "offset_reuse": "hash-frozen; never recompute downstream",
            "ties_or_errors": "ABSTAIN",
            "max_tokens": 384,
            "sft": {
                "optimizer": "AdamW",
                "learning_rate": 2.0e-4,
                "betas": [0.9, 0.999],
                "epsilon": 1.0e-8,
                "weight_decay": 0.01,
                "scheduler": "cosine",
                "warmup_steps": 8,
                "max_grad_norm": 1.0,
                "optimizer_steps": 64,
                "batch": 1,
                "gradient_accumulation": 8,
                "packing": False,
                "completion_only_loss": True,
                "fp16": True,
                "bf16": False,
                "seed": policy.random_seed,
            },
            "dpo": {
                "loss": "sigmoid",
                "beta": 0.1,
                "label_smoothing": 0.0,
                "reference": "final SFT with DPO updates disabled",
                "optimizer": "AdamW",
                "betas": [0.9, 0.999],
                "epsilon": 1.0e-8,
                "weight_decay": 0.01,
                "scheduler": "cosine",
                "warmup_steps": 8,
                "max_grad_norm": 1.0,
                "optimizer_steps": 96,
                "checkpoints": [24, 48, 72, 96],
                "learning_rate": 5.0e-6,
                "batch": 1,
                "gradient_accumulation": 8,
                "fp16": True,
                "bf16": False,
                "seed": policy.random_seed,
            },
            "memory_gib": {
                "inference_reserved_max": 6.5,
                "training_reserved_max": 16.0,
                "training_allocated_max": 14.0,
                "adapter_checkpoint_directory_max": 0.25,
                "retained_sft_plus_selected_dpo_max": 1.0,
            },
            "selection_2022": {
                "return_positive": True,
                "ratio_min": 2.0,
                "strict_mdd_pct_max": 15.0,
                "both_halves_positive": True,
                "trades_min": 70,
                "each_side_min": 25,
                "each_side_contribution_positive": True,
                "max_action_share": 0.85,
                "stress_and_one_bar_delay_positive": True,
                "weekly_cluster_p_below": 0.10,
                "beat_cheap_return": True,
                "ratio_margin_over_cheap": 0.25,
            },
        },
        "novelty_contract": {
            "comparators": comparator_contracts(),
            "cboe_timing_jaccard": "report_only",
            "cboe_same_side_reproduction_max": 0.70,
            "cboe_abs_signed_exposure_correlation_max": 0.60,
            "live_exact_jaccard_max": 0.10,
            "live_one_hour_jaccard_max": 0.25,
            "live_abs_signed_exposure_correlation_max": 0.40,
            "loader_semantics_allowlist": [
                "policy_or_group_id",
                "decision_or_admission",
                "entry",
                "exit",
                "action_or_side",
            ],
            "hash_drift": "fail",
            "undefined_correlation": "fail",
            "missing_required_common_coverage": "fail",
            "forbidden_paths": list(FORBIDDEN_COMPARATOR_PATHS),
            "parse_after": "2022 policy selection and pre-2024 action freeze",
        },
        "eval_2023_gate": {
            "return_positive": True,
            "ratio_min": 3.0,
            "strict_mdd_pct_max": 15.0,
            "both_halves_positive": True,
            "trades_min": 70,
            "each_side_min": 25,
            "each_side_contribution_positive": True,
            "active_months_min": 10,
            "max_month_share": 0.15,
            "max_action_share": 0.85,
            "weekly_clusters_min": 20,
            "weekly_cluster_p_below": 0.10,
            "mean_gross_underlying_move_bp_min": 20.0,
            "stress_and_one_bar_delay_positive": True,
            "one_hour_delay": "report_only",
            "every_neutral_code_order_return_positive": True,
            "beat_cheap_return": True,
            "ratio_margin_over_cheap": 0.50,
        },
        "strict_sequence": [
            "commit mechanism",
            "commit preregistration and synthetic tests",
            "commit source-only builder",
            "run source support once",
            "retire unchanged on source/support failure",
            "freeze evaluator baselines model controls",
            "open only 2020-2022 outcomes",
            "retire before GPU on transfer/cheap failure",
            "train one Gemma2 SFT and DPO checkpoints 24/48/72/96",
            "select on 2022 and freeze pre-2024 actions",
            "run novelty before 2023 outcomes",
            "evaluate 2023 once",
            "open sealed years sequentially after prior pass",
            "commit every completed unit with hashes and fresh tests",
        ],
        "outcome_boundary": {
            "source_artifact_bytes_hashed": True,
            "source_headers_read": True,
            "source_values_decoded": 0,
            "cspg_pressures_derived": 0,
            "cspg_tokens_derived": 0,
            "cspg_incidence_rows_derived": 0,
            "market_rows_loaded": 0,
            "funding_rows_loaded": 0,
            "comparator_rows_decoded": 0,
            "future_return_rows_loaded": 0,
            "return_or_pnl_fields": 0,
            "post_2023_rows_loaded": 0,
            "model_labels_created": 0,
            "model_training_runs": 0,
        },
    }


def build_manifest() -> dict[str, Any]:
    core = _core_manifest()
    return {**core, "manifest_hash": canonical_hash(core)}


def validate_manifest(payload: dict[str, Any]) -> None:
    expected = build_manifest()
    if payload != expected:
        raise RuntimeError("CSPG-288 manifest core differs from code")
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("CSPG-288 manifest hash mismatch")
    boundary = payload["outcome_boundary"]
    if any(
        boundary[field] != 0
        for field in (
            "source_values_decoded",
            "cspg_pressures_derived",
            "cspg_tokens_derived",
            "cspg_incidence_rows_derived",
            "market_rows_loaded",
            "funding_rows_loaded",
            "comparator_rows_decoded",
            "future_return_rows_loaded",
            "return_or_pnl_fields",
            "post_2023_rows_loaded",
            "model_labels_created",
            "model_training_runs",
        )
    ):
        raise RuntimeError("CSPG-288 evidence boundary opened")


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
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        actual = output.read_bytes()
        if actual != expected:
            raise RuntimeError("CSPG-288 existing manifest hash mismatch")
        return "verified_existing"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        dir=output.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(expected)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError:
            if output.read_bytes() != expected:
                raise RuntimeError("CSPG-288 manifest race drift")
            return "verified_existing"
        return "created"
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-model", action="store_true")
    args = parser.parse_args()
    if args.verify_model:
        validate_model_snapshot()
    payload = build_manifest()
    status = write_once(args.output, payload)
    print(
        json.dumps(
            {
                "status": status,
                "output": args.output,
                "manifest_hash": payload["manifest_hash"],
                "outcome_boundary": payload["outcome_boundary"],
                "model_verified": bool(args.verify_model),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
