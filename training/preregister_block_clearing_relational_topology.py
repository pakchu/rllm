"""Freeze BCRT-72 before decoding source values, incidence, or outcomes."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


POLICY_ID = "BCRT-72"
PROTOCOL_VERSION = "block_clearing_relational_topology_preregistration_v1"
DEFAULT_OUTPUT = (
    "results/block_clearing_relational_topology_"
    "preregistration_2026-07-24.json"
)

BOUNDARY_DOCUMENT = (
    "docs/block-clearing-relational-topology-boundary-2026-07-24.md"
)
BOUNDARY_DOCUMENT_SHA256 = (
    "ab3d71d5b7f52254a3d25f4eeada35acab16ec7a2460528e2727b56ae8039560"
)
BOUNDARY_COMMIT = "d4d2864"
MECHANISM_DOCUMENT = (
    "docs/block-clearing-relational-topology-"
    "mechanism-decision-2026-07-24.md"
)
MECHANISM_DOCUMENT_SHA256 = (
    "95d3ebffea956fbaee261d8da280d8ced8f12164823a9d86755771d0fb98b991"
)
MECHANISM_COMMIT = "c743851"

SOURCE = "data/bitcoin_utxo_fee_block_stats_2020_2023.csv.gz"
SOURCE_SHA256 = (
    "8d5160b59bd104819adb34752f2bd5b01e9c07d7aa108188abcd9e6d323d102f"
)
SOURCE_HEADER_SHA256 = (
    "66cefecff20d70a4229285fc5b93a7cd6126dfd53173acc9c8bffe805638c342"
)
SOURCE_MANIFEST = (
    "results/bitcoin_utxo_fee_block_stats_source_manifest_2026-07-20.json"
)
SOURCE_MANIFEST_SHA256 = (
    "ceb096b7bfeaf309729c4cab9f5cd4d2e0d526e261b1e460e9d03cd4f24e8084"
)
SOURCE_MANIFEST_HASH = (
    "98a84b0bd0338300f62eaa047b87498cc5a8d9505a03f6bd1912d1deb9564e8c"
)
SOURCE_BUILDER = "training/download_bitcoin_utxo_fee_stats.py"
SOURCE_BUILDER_SHA256 = (
    "099454feff009a5a4d44a96bd3790ff586d0365eba2e9b72e7b071d34e743633"
)
SOURCE_DECISION = (
    "docs/utxo-fee-clearing-polarity-mechanism-decision-2026-07-20.md"
)
SOURCE_DECISION_SHA256 = (
    "95bf889fd053987e1717b182dc5da4f19ef51d75a1cbda427913089368c4852e"
)
SOURCE_ALLOWLIST = (
    "height",
    "id",
    "previousblockhash",
    "timestamp",
    "mediantime",
    "tx_count",
    "size",
    "weight",
    "total_fees",
    "total_inputs",
    "total_outputs",
    "utxo_set_change",
)
REFERENCE = "data/bitcoin_block_summaries_2020_2023.csv.gz"
REFERENCE_SHA256 = (
    "1f8d1c0153717f2c18d8fa6f09428c780a850b2aecc8fb42bc497e16e68e1833"
)
REFERENCE_HEADER_SHA256 = (
    "afb4c6b31bc7909918b78f60ef1e14b9f59bd2e619f2c297fe1b3ce31f02d2fe"
)
REFERENCE_ALLOWLIST = SOURCE_ALLOWLIST[:8]

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

PRIMITIVES = (
    "CADENCE",
    "UTILIZATION",
    "PACKING",
    "FEE",
    "UTXO",
    "WITNESS",
    "LOAD_DISPERSION",
    "FEE_DISPERSION",
)
LEADER_VOCABULARY = (*PRIMITIVES, "TIE")
TOKEN_SCHEMA: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "cadence_utilization",
        ("CADENCE_LEADS", "BALANCED", "UTILIZATION_LEADS"),
    ),
    (
        "utilization_fee",
        ("UTILIZATION_LEADS", "BALANCED", "FEE_LEADS"),
    ),
    (
        "packing_witness",
        ("PACKING_LEADS", "BALANCED", "WITNESS_LEADS"),
    ),
    ("utxo_fee", ("UTXO_LEADS", "BALANCED", "FEE_LEADS")),
    (
        "load_fee_dispersion",
        ("LOAD_WIDER", "BALANCED", "FEE_WIDER"),
    ),
    ("high_leader", LEADER_VOCABULARY),
    ("low_leader", LEADER_VOCABULARY),
    ("rank_breadth", ("LOW_BROAD", "MIXED", "HIGH_BROAD")),
    ("extreme_occupancy", ("COMPACT", "FOCUSED", "FRACTURED")),
    ("relation_breadth", ("RIGHT_BROAD", "MIXED", "LEFT_BROAD")),
    ("order_transition", ("STABLE", "ROTATING", "RESET")),
    (
        "leader_transition",
        (
            "TIE_INVOLVED",
            "BOTH_STABLE",
            "HIGH_ROTATED",
            "LOW_ROTATED",
            "BOTH_ROTATED",
        ),
    ),
)
TOKEN_COLUMNS = tuple(name for name, _ in TOKEN_SCHEMA)
TOKEN_VOCABULARY = {name: values for name, values in TOKEN_SCHEMA}
NEUTRAL_CODES = ("Q1", "Q2")
TASKS = ("ADMISSION", "DIRECTION")
ACTION_NAMES = ("ABSTAIN", "LONG", "SHORT")


@dataclass(frozen=True)
class Policy:
    policy_id: str = POLICY_ID
    bucket_seconds: int = 43_200
    confirmation_blocks: int = 288
    minimum_embargo_seconds: int = 172_800
    bar_seconds: int = 300
    latency_bars: int = 1
    hold_bars: int = 72
    rank_lookback_buckets: int = 252
    rank_minimum_prior_buckets: int = 126
    pair_delta_numerator: int = 1
    pair_delta_denominator: int = 6
    rank_breadth_threshold: int = 2
    extreme_low_numerator: int = 1
    extreme_low_denominator: int = 6
    extreme_high_numerator: int = 5
    extreme_high_denominator: int = 6
    order_stable_max_changes: int = 6
    order_rotating_max_changes: int = 13
    leverage: float = 0.50
    base_cost_notional_per_side: float = 0.0006
    stress_cost_notional_per_side: float = 0.0010
    held_path_drawdown_penalty: float = 1.0 / 3.0
    trade_utility_hurdle_account: float = 0.0005
    preference_margin: float = 0.0003
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
        raise RuntimeError(f"BCRT CSV header is not one LF line: {path}")
    return header


def csv_header(path: str | Path) -> list[str]:
    return csv_header_bytes(path).decode("utf-8").rstrip("\n").split(",")


def sha256_csv_header(path: str | Path) -> str:
    return hashlib.sha256(csv_header_bytes(path)).hexdigest()


def strict_prior_midrank(
    current: float,
    prior_values: Iterable[float],
    *,
    minimum: int = Policy().rank_minimum_prior_buckets,
    maximum: int = Policy().rank_lookback_buckets,
) -> float:
    prior = np.asarray(list(prior_values), dtype=np.float64)
    if prior.ndim != 1 or minimum <= 0 or maximum < minimum:
        raise ValueError("BCRT rank window is invalid")
    if len(prior) < minimum:
        raise ValueError("BCRT strict-prior history is not ready")
    if len(prior) > maximum:
        prior = prior[-maximum:]
    value = np.float64(current)
    if not np.isfinite(value) or not np.isfinite(prior).all():
        raise ValueError("BCRT rank values must be finite")
    return float(
        (np.count_nonzero(prior < value) + 0.5 * np.count_nonzero(prior == value))
        / len(prior)
    )


def _validate_rank_mapping(ranks: Mapping[str, float]) -> dict[str, float]:
    if tuple(ranks) != PRIMITIVES:
        raise ValueError("BCRT primitive rank order changed")
    normalized = {name: float(ranks[name]) for name in PRIMITIVES}
    if any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0
        for value in normalized.values()
    ):
        raise ValueError("BCRT primitive rank must be finite and in [0,1]")
    return normalized


def pair_relation(
    left: float,
    right: float,
    *,
    left_token: str,
    right_token: str,
) -> str:
    left_value = float(left)
    right_value = float(right)
    if (
        not math.isfinite(left_value)
        or not math.isfinite(right_value)
        or not 0.0 <= left_value <= 1.0
        or not 0.0 <= right_value <= 1.0
    ):
        raise ValueError("BCRT pair ranks are invalid")
    delta = left_value - right_value
    boundary = 1.0 / 6.0
    if delta > boundary:
        return left_token
    if delta < -boundary:
        return right_token
    return "BALANCED"


def extreme_leader(ranks: Mapping[str, float], *, highest: bool) -> str:
    values = _validate_rank_mapping(ranks)
    extreme = (max if highest else min)(values.values())
    winners = [name for name in PRIMITIVES if values[name] == extreme]
    return winners[0] if len(winners) == 1 else "TIE"


def rank_breadth(ranks: Mapping[str, float]) -> str:
    values = _validate_rank_mapping(ranks)
    high = sum(value > 0.5 for value in values.values())
    low = sum(value < 0.5 for value in values.values())
    score = high - low
    if score >= 2:
        return "HIGH_BROAD"
    if score <= -2:
        return "LOW_BROAD"
    return "MIXED"


def extreme_occupancy(ranks: Mapping[str, float]) -> str:
    values = _validate_rank_mapping(ranks)
    count = sum(
        value < 1.0 / 6.0 or value > 5.0 / 6.0
        for value in values.values()
    )
    if count <= 2:
        return "COMPACT"
    if count <= 5:
        return "FOCUSED"
    return "FRACTURED"


def relation_breadth(scores: Sequence[int]) -> str:
    normalized = tuple(int(score) for score in scores)
    if len(normalized) != 5 or any(score not in (-1, 0, 1) for score in normalized):
        raise ValueError("BCRT relation-breadth scores are invalid")
    total = sum(normalized)
    if total >= 2:
        return "LEFT_BROAD"
    if total <= -2:
        return "RIGHT_BROAD"
    return "MIXED"


def _pair_order_states(ranks: Mapping[str, float]) -> tuple[int, ...]:
    values = _validate_rank_mapping(ranks)
    states: list[int] = []
    for left_index, left in enumerate(PRIMITIVES):
        for right in PRIMITIVES[left_index + 1 :]:
            delta = values[left] - values[right]
            states.append(1 if delta > 0.0 else -1 if delta < 0.0 else 0)
    if len(states) != 28:
        raise RuntimeError("BCRT pair-order state count drift")
    return tuple(states)


def order_transition(
    current: Mapping[str, float],
    previous: Mapping[str, float],
) -> str:
    changed = sum(
        left != right
        for left, right in zip(
            _pair_order_states(current),
            _pair_order_states(previous),
            strict=True,
        )
    )
    if changed <= 6:
        return "STABLE"
    if changed <= 13:
        return "ROTATING"
    return "RESET"


def leader_transition(
    current_high: str,
    current_low: str,
    previous_high: str,
    previous_low: str,
) -> str:
    leaders = (current_high, current_low, previous_high, previous_low)
    if any(leader not in LEADER_VOCABULARY for leader in leaders):
        raise ValueError("BCRT leader token is invalid")
    if "TIE" in leaders:
        return "TIE_INVOLVED"
    high_changed = current_high != previous_high
    low_changed = current_low != previous_low
    if not high_changed and not low_changed:
        return "BOTH_STABLE"
    if high_changed and not low_changed:
        return "HIGH_ROTATED"
    if not high_changed and low_changed:
        return "LOW_ROTATED"
    return "BOTH_ROTATED"


def validate_tokens(tokens: Mapping[str, str]) -> dict[str, str]:
    if tuple(tokens) != TOKEN_COLUMNS:
        raise ValueError("BCRT token order or schema changed")
    normalized = {name: str(tokens[name]) for name in TOKEN_COLUMNS}
    for name, value in normalized.items():
        if value not in TOKEN_VOCABULARY[name]:
            raise ValueError(f"BCRT token value is invalid: {name}={value}")
    return normalized


def ceil_5m(epoch_seconds: int) -> int:
    seconds = int(epoch_seconds)
    if seconds < 0 or seconds != epoch_seconds:
        raise ValueError("BCRT epoch seconds must be a nonnegative integer")
    width = Policy().bar_seconds
    return ((seconds + width - 1) // width) * width


def opportunity_times(
    *,
    bucket_end_seconds: int,
    prefix_max_timestamp: int,
    prefix_max_mediantime: int,
) -> dict[str, datetime]:
    policy = Policy()
    raw = (
        max(
            int(bucket_end_seconds),
            int(prefix_max_timestamp),
            int(prefix_max_mediantime),
        )
        + policy.minimum_embargo_seconds
    )
    signal_seconds = ceil_5m(raw)
    entry_seconds = signal_seconds + policy.latency_bars * policy.bar_seconds
    exit_seconds = entry_seconds + policy.hold_bars * policy.bar_seconds
    return {
        "signal_available": datetime.fromtimestamp(
            signal_seconds,
            tz=timezone.utc,
        ),
        "entry": datetime.fromtimestamp(entry_seconds, tz=timezone.utc),
        "exit": datetime.fromtimestamp(exit_seconds, tz=timezone.utc),
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
            raise ValueError("BCRT reservation interval is invalid")
        reserved = previous_exit is None or entry >= previous_exit
        row["reserved"] = reserved
        result.append(row)
        if reserved:
            previous_exit = exit_time
    return result


def neutral_code_orders() -> tuple[tuple[str, str], ...]:
    return tuple(itertools.permutations(NEUTRAL_CODES))


def comparator_contracts() -> list[dict[str, Any]]:
    contracts = [
        {
            "id": "BATE-288",
            "path": (
                "results/block_arrival_throughput_elasticity_"
                "clock_2026-07-20.csv"
            ),
            "sha256": (
                "cd4fbd01c104bd969ca1c12a53b8da82dd0e9376990e233c286ff009a5115c02"
            ),
            "header_sha256": (
                "562a0e666e50028f9118865f30d0e719830a7e0527e93fd870997b5a1138fe05"
            ),
            "group_column": "policy_id",
            "entry_column": "entry_time",
            "exit_column": "exit_time",
            "side_column": "side",
        },
        {
            "id": "UFCP-1",
            "path": (
                "results/utxo_fee_clearing_polarity_"
                "primary_clock_2026-07-20.csv"
            ),
            "sha256": (
                "8338c290d63b522531c8d55c8a79ba73cc13915c936733ec03ffcf6ab0e86c1b"
            ),
            "header_sha256": (
                "647b27c3660f5d06e621f31d9045c16917605b17b66a6ed62e139bd550a7a30e"
            ),
            "group_column": "policy_id",
            "entry_column": "entry_time",
            "exit_column": "exit_time",
            "side_column": "side",
        },
        {
            "id": "MCR-7",
            "path": "results/miner_cadence_recovery_clock_2026-07-17.csv",
            "sha256": (
                "2535244889b046ff00c369ee854973a91c23429dff82a6dd3c1a293a01352b0b"
            ),
            "header_sha256": (
                "113d5b413482187c07c71272b8edccfe14032d24fb5b5ca1d0363f63367dc72a"
            ),
            "group_column": "policy_id",
            "entry_column": "entry_date",
            "exit_column": "exit_date",
            "side_column": "side",
        },
        {
            "id": "NTB-7",
            "path": (
                "results/network_topology_broadening_clock_2026-07-17.csv"
            ),
            "sha256": (
                "6b1bd7c7458cffa062e40872c3ad1730007c01426790b1ba8e52c6eb853de42f"
            ),
            "header_sha256": (
                "ba82923c66192b28b0d240ffe29b2071043bb437c043fc1d35cfce77859ed13e"
            ),
            "group_column": "policy_id",
            "entry_column": "entry_date",
            "exit_column": "exit_date",
            "side_column": "side",
        },
        {
            "id": "BFC-3",
            "path": "results/blockspace_fee_confirmation_clock_2026-07-17.csv",
            "sha256": (
                "edda7bb8ae8a1de4e51a3b86e98d533748e73d203125a3ded1a487e9a0e93632"
            ),
            "header_sha256": (
                "cf484b9ccd564006f91a2a799c50ea84148f28ecd10322b06b5d34dd04b43562"
            ),
            "group_column": "policy_id",
            "entry_column": "entry_date",
            "exit_column": "exit_date",
            "side_column": "side",
        },
        {
            "id": "WCTR-288",
            "path": (
                "results/witness_composition_transport_"
                "primary_clock_2026-07-20.csv.gz"
            ),
            "sha256": (
                "7a6b56a3024d0d087322fad7b3229276c539b93374691cd2812af0630dc752b1"
            ),
            "header_sha256": (
                "cd90df97c515162cdd0d2fbca7f341ef65a61367168ff6bcc9d8d6e93eb3b6cb"
            ),
            "group_column": "clock",
            "entry_column": "entry_time_utc",
            "exit_column": "exit_time_utc",
            "side_column": "side",
        },
        {
            "id": "BFRT-288",
            "path": (
                "results/block_feerate_breadth_transport_"
                "primary_clock_2026-07-20.csv.gz"
            ),
            "sha256": (
                "33428d29c2ace9b23672b2dc9dc3e9ba0e3020fa1a6e3845d55fa5d75230d64a"
            ),
            "header_sha256": (
                "34f459439a8db95467cbbceca6ba14254bee97e8eca53a0840da2b68d1167c05"
            ),
            "group_column": "clock",
            "entry_column": "entry_time_utc",
            "exit_column": "exit_time_utc",
            "side_column": "side",
        },
        {
            "id": "EMFC-864",
            "path": (
                "results/exact_maturity_fee_cadence_"
                "polarity_clocks_2026-07-20.csv"
            ),
            "sha256": (
                "31af41f42ffe4dc73f0ff35ccf278e38c856d224184e802e46b370650d35951d"
            ),
            "header_sha256": (
                "9f57b86445928f284bc4039700f21cdbda2ccb7594ea72ce1426bce507aa1c4b"
            ),
            "group_column": "clock",
            "entry_column": "entry_time",
            "exit_column": "exit_time",
            "side_column": "side",
        },
        {
            "id": "CCHR-live-pre2024",
            "path": "results/cchr_live_portfolio_pure_clocks_2026-07-21.csv.gz",
            "sha256": (
                "73d6efbd35b3be64b0fa04fa9c8cb2db25866ef884f19b1ae673949e22a42b08"
            ),
            "header_sha256": (
                "da21cfc42a55581971c304cc30122a72f9d062a7601db59a9702ec35504acb9a"
            ),
            "group_column": "candidate_id",
            "entry_column": "entry_time",
            "exit_column": "exit_time",
            "side_column": "side",
            "rows": 440,
        },
    ]
    for item in contracts:
        item.update(
            {
                "loader_allowlist": [
                    item["group_column"],
                    item["entry_column"],
                    item["exit_column"],
                    item["side_column"],
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
        SOURCE: SOURCE_SHA256,
        SOURCE_MANIFEST: SOURCE_MANIFEST_SHA256,
        SOURCE_BUILDER: SOURCE_BUILDER_SHA256,
        SOURCE_DECISION: SOURCE_DECISION_SHA256,
        REFERENCE: REFERENCE_SHA256,
        MARKET_SOURCE: MARKET_SOURCE_SHA256,
        MARKET_MANIFEST: MARKET_MANIFEST_SHA256,
        FUNDING_SOURCE: FUNDING_SOURCE_SHA256,
        FUNDING_MANIFEST: FUNDING_MANIFEST_SHA256,
        "results/fee_endpoint_topology_disagreement_support_2026-07-20.json": (
            "03ba910a314ba6efb647f6588dff603261d414e5114680ca33bdc27d59aed035"
        ),
    }
    dependencies.update(
        {item["path"]: item["sha256"] for item in comparator_contracts()}
    )
    return dependencies


def validate_frozen_dependencies() -> None:
    for path, expected in frozen_dependencies().items():
        if sha256_file(path) != expected:
            raise RuntimeError(f"BCRT frozen dependency changed: {path}")
    if sha256_csv_header(SOURCE) != SOURCE_HEADER_SHA256:
        raise RuntimeError("BCRT source header hash changed")
    if csv_header(SOURCE) != list(SOURCE_ALLOWLIST):
        raise RuntimeError("BCRT source allowlist/order changed")
    if sha256_csv_header(REFERENCE) != REFERENCE_HEADER_SHA256:
        raise RuntimeError("BCRT reference header hash changed")
    if csv_header(REFERENCE) != list(REFERENCE_ALLOWLIST):
        raise RuntimeError("BCRT reference allowlist/order changed")
    for item in comparator_contracts():
        if sha256_csv_header(item["path"]) != item["header_sha256"]:
            raise RuntimeError(f"BCRT comparator header changed: {item['id']}")


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
            raise RuntimeError(f"BCRT Gemma2 frozen file mismatch: {filename}")
    return {
        "validated": True,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": dict(MODEL_FILES),
    }


def _core_manifest() -> dict[str, Any]:
    policy = Policy()
    pair_tokens = tuple(name for name, _ in TOKEN_SCHEMA[:5])
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
            "base_chain_source_values_seen": True,
            "base_chain_family_outcomes_seen": True,
            "bate_2021_2022_outcomes_seen": True,
            "bcrt_values_seen": False,
            "bcrt_tokens_or_incidence_seen": False,
            "bcrt_market_outcomes_seen": False,
            "global_pristine_holdout_claimed": False,
            "claim_scope": "candidate-level frozen 2023 outcome window",
        },
        "source_contract": {
            "path": SOURCE,
            "sha256": SOURCE_SHA256,
            "header_sha256": SOURCE_HEADER_SHA256,
            "manifest": SOURCE_MANIFEST,
            "manifest_sha256": SOURCE_MANIFEST_SHA256,
            "manifest_hash": SOURCE_MANIFEST_HASH,
            "source_builder": SOURCE_BUILDER,
            "source_builder_sha256": SOURCE_BUILDER_SHA256,
            "source_decision": SOURCE_DECISION,
            "source_decision_sha256": SOURCE_DECISION_SHA256,
            "allowlist": list(SOURCE_ALLOWLIST),
            "loader": "pandas.read_csv(usecols=allowlist); no load-and-drop",
            "reference": REFERENCE,
            "reference_sha256": REFERENCE_SHA256,
            "reference_header_sha256": REFERENCE_HEADER_SHA256,
            "reference_allowlist": list(REFERENCE_ALLOWLIST),
            "rows": 213_095,
            "height_start": 610_691,
            "height_end": 823_785,
            "end_timestamp_exclusive": 1_704_067_200,
            "identity": "utxo_set_change=total_outputs-total_inputs",
            "validation": {
                "integer_columns": [
                    "height",
                    "timestamp",
                    "mediantime",
                    "tx_count",
                    "size",
                    "weight",
                    "total_fees",
                    "total_inputs",
                    "total_outputs",
                    "utxo_set_change",
                ],
                "positive": ["tx_count", "size", "weight"],
                "nonnegative": ["total_fees"],
                "weight_bounds": "size <= weight <= 4*size",
                "block_ids": "lowercase 64-character hexadecimal",
                "height_and_id_unique": True,
                "timestamps_before_cutoff": True,
            },
            "production_contract": {
                "owned_bitcoin_core_required": True,
                "actual_first_seen_timestamps_required": True,
                "raw_response_hashing_required": True,
                "forward_field_parity_required": True,
                "live_mismatch_action": (
                    "block BCRT; do not rewrite historical values"
                ),
            },
            "pre_2024_only": True,
        },
        "bucket_contract": {
            "assignment": "floor(timestamp/43200)*43200 UTC",
            "interval": "[bucket_start,bucket_start+43200)",
            "minimum_blocks": 1,
            "previous_state": "immediately prior source-valid bucket",
            "previous_state_includes_reservation_or_split_suppressed": True,
            "anchor_height": (
                "first height whose mediantime is greater than or equal "
                "to bucket_end"
            ),
            "confirmation_height": "anchor_height+288",
            "membership": (
                "rows with height<=confirmation_height and "
                "bucket_start<=timestamp<bucket_end"
            ),
            "late_backdated_members": (
                "ignore for formed bucket and emit diagnostic"
            ),
            "anchor_and_confirmation_must_exist": True,
            "anchor_and_confirmation_heights_strictly_increasing": True,
            "availability_monotone": True,
            "prefix_replay": (
                "rebuild each formed bucket at confirmation_height and "
                "require byte-identical membership, primitives, ranks, "
                "tokens, availability, and reservation after every append"
            ),
            "latest_unconfirmed_buckets": "omit",
        },
        "primitive_contract": {
            "ordered": list(PRIMITIVES),
            "cadence": "log(block_count)",
            "utilization": "log((sum(weight)+1)/(4000000*n+1))",
            "packing": "log((sum(tx_count)+1)/(sum(weight)+1))",
            "fee_burden": "log((sum(total_fees)+1)/(sum(weight)+1))",
            "utxo_pressure": "sum(utxo_set_change)/(sum(tx_count)+1)",
            "witness_discount": "(4*sum(size)-sum(weight))/(4*sum(size))",
            "load_dispersion": "MAD(weight_i/4000000)",
            "fee_dispersion": (
                "MAD(log((total_fees_i+1)/(weight_i+1)))"
            ),
            "mad": "median(abs(x-median(x)))",
            "clipping_or_full_series_normalization": False,
        },
        "rank_contract": {
            "lookback": policy.rank_lookback_buckets,
            "minimum": policy.rank_minimum_prior_buckets,
            "formula": "(count(prior<x)+0.5*count(prior==x))/len(prior)",
            "current_excluded": True,
            "histories_independent_by_primitive": True,
            "first_complete_state_is_predecessor_only": True,
            "future_append_invariant": True,
        },
        "token_contract": {
            "ordered_schema": [
                {"name": name, "values": list(values)}
                for name, values in TOKEN_SCHEMA
            ],
            "count": len(TOKEN_SCHEMA),
            "pair_tokens": list(pair_tokens),
            "pair_relation": (
                "delta>1/6 left; delta<-1/6 right; else BALANCED"
            ),
            "pair_boundary_equality": "BALANCED",
            "leader_ties": "TIE without epsilon",
            "rank_breadth": (
                "high=count(rank>0.5),low=count(rank<0.5); "
                "difference >=2 HIGH_BROAD, <=-2 LOW_BROAD, else MIXED"
            ),
            "extreme_occupancy": (
                "count(rank<1/6 or rank>5/6); "
                "0..2 COMPACT,3..5 FOCUSED,6..8 FRACTURED"
            ),
            "relation_breadth": (
                "five semantic pair scores; sum>=2 LEFT_BROAD, "
                "<=-2 RIGHT_BROAD, else MIXED"
            ),
            "order_transition": (
                "changed 28 pair-order states; 0..6 STABLE, "
                "7..13 ROTATING, 14..28 RESET"
            ),
            "position": "deterministic guard, not a token",
            "unknown_downstream_value": "ABSTAIN",
            "forbidden": [
                "raw_source_values_or_numeric_ranks",
                "date_time_height_id_or_row_identity",
                "btc_price_return_funding_premium_oi_kimchi_dxy",
                "prior_action_reward_pnl_cagr_mdd_or_split",
                "prior_base_chain_policy_signal_side_state_or_outcome",
                "source_path_hash_or_transport_identity",
                "free_form_rationale_chain_of_thought_or_generated_feature",
            ],
        },
        "execution_contract": {
            "raw_available": (
                "max(bucket_end,prefix_max_timestamp_through_confirmation,"
                "prefix_max_mediantime_through_confirmation)+172800"
            ),
            "signal_available": "ceil raw availability to 300 seconds",
            "entry": "signal_available+300 seconds BTCUSDT 5m open",
            "exit": "entry+72*300 seconds",
            "leverage": policy.leverage,
            "global_action_independent_reservation": True,
            "abstention_releases_reservation": False,
            "split_crossing_keeps_reservation": True,
            "split_containment_fields": [
                "source_bucket",
                "anchor",
                "confirmation",
                "signal_available",
                "latency_bar",
                "entry",
                "held_bars",
                "exit",
            ],
            "split_interval": "half-open",
            "scheduled_exit_only": True,
            "dynamic_exit_or_sizing": False,
            "one_bar_delay": "entry+5m,exit+5m,recompute reservation",
            "one_hour_delay": "entry+60m,exit+60m,recompute reservation",
            "live_clock": (
                "max(frozen historical clock, actual node "
                "receipt/validation clock)"
            ),
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
            "boolean_scope": "[2020-01-01,2023-01-01) development only",
            "development_incidence": {
                "development_2020_2022_min": 2000,
                "train_2020_2021_min": 1250,
                "year_2020_min": 570,
                "each_year_2021_2022_min": 700,
                "year_2020_active_months_min": 9,
                "each_year_2021_2022_active_months_min": 12,
                "each_half_2021_2022_min": 340,
                "each_quarter_2021_2022_min": 165,
                "year_2020_max_month_share": 0.13,
                "each_year_2021_2022_max_month_share": 0.10,
                "max_entry_gap_days_2020_2022": 3,
            },
            "token_scope": ["train_2020_2021", "selection_2022"],
            "token_support": {
                "pair_each_value_share_min": 0.05,
                "pair_max_value_share": 0.80,
                "leader_nontie_distinct_min": 5,
                "leader_max_nontie_share": 0.40,
                "leader_tie_share_max": 0.20,
                "rank_breadth_each_share_min": 0.05,
                "rank_breadth_max_share": 0.80,
                "extreme_occupancy_each_share_min": 0.02,
                "extreme_occupancy_max_share": 0.90,
                "relation_breadth_each_share_min": 0.05,
                "relation_breadth_max_share": 0.80,
                "order_transition_each_share_min": 0.03,
                "order_transition_max_share": 0.85,
                "leader_transition_distinct_min": 4,
                "leader_transition_max_share": 0.75,
                "max_exact_signature_share": 0.05,
                "selection_values_must_exist_in_train": True,
            },
            "counts_basis": "token-ready globally reserved split-contained",
            "failure_action": "retire BCRT-72 unchanged before outcomes",
        },
        "eval_source_report_only": {
            "scope": "[2023-01-01,2024-01-01)",
            "reports": [
                "incidence",
                "calendar_coverage",
                "marginal_token_distribution",
                "exact_signature_distribution",
                "train_vocabulary_coverage",
            ],
            "boolean_gate": False,
            "may_authorize_continue_retire_repair_or_selection": False,
            "unseen_token_value": "frozen policy ABSTAIN",
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
            "cagr": "full half-open calendar including warmup idle abstention",
            "weekly_signflip_draws": 100_000,
            "weekly_signflip_seed": policy.random_seed,
        },
        "utility_contract": {
            "abstain": 0.0,
            "trade": (
                "log(max(account_multiplier,1e-12))"
                "-(1/3)*local_strict_drawdown-0.0005"
            ),
            "oracle_tie_priority": list(ACTION_NAMES),
            "admission": "ABSTAIN versus max(LONG,SHORT)",
            "direction": "LONG versus SHORT only when best trade beats ABSTAIN",
            "preference_margin": policy.preference_margin,
            "balancing": None,
            "synthetic_source_symmetry": None,
            "pre_gpu_label_gate": {
                "admission_max_target_share": 0.90,
                "direction_each_target_share_min": 0.20,
                "both_preference_sets_nonempty": True,
            },
        },
        "baseline_contract": {
            "fit_transfer": "fit 2020, evaluate unchanged 2021",
            "refit_selection": "refit 2020-2021, evaluate 2022",
            "features": "token one-hot plus 66 pair conjunctions; min count 5",
            "policies": [
                "always_abstain",
                "always_long",
                "always_short",
                "exact_signature_memory",
                "categorical_naive_bayes_alpha_1",
                "ridge_contextual_value_alpha_100",
                "extra_trees_value_512_depth6_leaf12_split24",
                "fit_majority_three_action_prior",
                "fit_admission_and_direction_prior",
                "utc_quarter_phase_prior",
                "32_shuffled_label_nb",
                "32_shuffled_utility_ridge",
                "16_circular_block_shift_controls",
                "12_single_token_ridge",
                "12_leave_one_token_out_ridge",
                "5_group_only_ridge",
                "5_leave_one_group_out_ridge",
            ],
            "shuffle_seeds": list(range(20_260_724, 20_260_756)),
            "circular_block_shift_offsets": [
                62,
                93,
                124,
                155,
                186,
                217,
                248,
                279,
                310,
                341,
                372,
                403,
                434,
                465,
                496,
                527,
            ],
            "token_groups": {
                "pair_relations_only": list(pair_tokens),
                "leaders_only": ["high_leader", "low_leader"],
                "breadth_occupancy_only": [
                    "rank_breadth",
                    "extreme_occupancy",
                    "relation_breadth",
                ],
                "transitions_only": [
                    "order_transition",
                    "leader_transition",
                ],
                "current_topology_without_transitions": list(TOKEN_COLUMNS[:10]),
            },
            "controls_never_qualify": [
                "prior_only",
                "quarter_phase",
                "shuffled",
                "circular_block_shift",
                "orientation_flipped_inference",
            ],
            "orientation_flipped_inference_controls": {
                "applies_to": (
                    "every learned primary and every later RLLM checkpoint"
                ),
                "refit": False,
                "pair_direction_flip": (
                    "swap both non-BALANCED values of all five pair tokens "
                    "only at downstream inference"
                ),
                "relation_breadth_flip": (
                    "swap LEFT_BROAD and RIGHT_BROAD only at downstream "
                    "inference"
                ),
                "unchanged": [
                    "BALANCED",
                    "MIXED",
                    "all_other_tokens",
                    "fitted_policy",
                    "threshold",
                ],
                "can_qualify": False,
            },
            "transfer_2021": {
                "return_positive": True,
                "ratio_min": 0.5,
                "strict_mdd_pct_max": 15.0,
                "both_halves_positive": True,
                "trades_min": 200,
                "each_side_min": 60,
                "each_side_contribution_positive": True,
                "max_action_share": 0.90,
                "stress_and_one_bar_delay_positive": True,
                "familywise_weekly_cluster_p_max_below": 0.25,
                "beat_always_abstain_return": True,
                "beat_always_long_short_exact_memory_return_and_ratio": True,
                "beat_every_prior_quarter_shuffled_circular": True,
                "beat_strongest_single_token_and_group_only": True,
                "beat_both_orientation_flipped_controls": True,
            },
            "selection_2022": {
                "return_positive": True,
                "ratio_min": 1.0,
                "strict_mdd_pct_max": 15.0,
                "both_halves_positive": True,
                "trades_min": 240,
                "each_side_min": 75,
                "each_side_contribution_positive": True,
                "max_action_share": 0.85,
                "stress_and_one_bar_delay_positive": True,
                "familywise_weekly_cluster_p_max_below": 0.10,
                "beat_always_abstain_return": True,
                "beat_always_long_short_exact_memory_return_and_ratio": True,
                "beat_every_prior_quarter_shuffled_circular": True,
                "beat_strongest_single_token_and_group_only": True,
                "beat_both_orientation_flipped_controls": True,
                "single_token_action_reproduction_max": 0.70,
            },
            "selection_order": [
                "higher_ratio",
                "higher_return",
                "lower_strict_mdd",
                "lexicographically_smaller_policy_id",
            ],
        },
        "familywise_signflip_contract": {
            "weeks": (
                "union of nonempty UTC weeks across the complete frozen family; "
                "flat policy-week return is zero"
            ),
            "statistic": (
                "mean(weekly_return)/(std(weekly_return,ddof=1)/sqrt("
                "union_week_count))"
            ),
            "zero_variance_statistic": "-infinity",
            "shared_null": (
                "same Rademacher sign for the same union week and every policy"
            ),
            "draws": 100_000,
            "seed": policy.random_seed,
            "null_statistic": "maximum t across complete frozen family",
            "adjusted_p": "(1+count(max_null_t>=selected_t))/100001",
            "family": [
                "every emitted learned primary",
                "exact_signature_memory",
                "prior_only",
                "quarter_phase_prior",
                "single_token",
                "group_only",
                "leave_one_token_out",
                "leave_one_group_out",
                "shuffled_label",
                "shuffled_utility",
                "circular_block_shift",
                "final_sft",
                "every_dpo_checkpoint",
                "masked_token_prior_adapter",
                "pair_direction_flipped_inference",
                "relation_breadth_flipped_inference",
            ],
            "post_result_omission": "forbidden",
            "transfer_p_max_below": 0.25,
            "cheap_selection_p_max_below": 0.10,
            "rllm_selection_p_max_below": 0.05,
            "eval_2023": "ordinary immutable one-policy p-value",
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
            "prompt_serialization": {
                "state": "KEY=VALUE one per line in canonical token order",
                "suffix_lines": [
                    "TASK=<ADMISSION or DIRECTION>",
                    "OPTIONS=<Q1,Q2 or Q2,Q1>",
                    "Return exactly CHOICE=<one option>.",
                ],
                "line_separator": "\\n",
                "trailing_newline": False,
                "completion_candidates": ["CHOICE=Q1", "CHOICE=Q2"],
                "text_only": True,
                "position_in_prompt": False,
            },
            "generation": False,
            "score": "mean completion-token conditional log probability",
            "prior_correction": (
                "adapter_delta=adapted-base on same prompt; subtract task/code "
                "mean adapter_delta over original 2020-2021 fit states"
            ),
            "masked_token_prior_adapter": {
                "recipe": "identical SFT and DPO recipe and labels",
                "input": "each token line replaced by key plus literal MASKED",
                "can_qualify": False,
                "frozen_pre_2024_predictions": True,
            },
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
                "warmup_steps": 16,
                "max_grad_norm": 1.0,
                "optimizer_steps": 128,
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
                "learning_rate": 5.0e-6,
                "betas": [0.9, 0.999],
                "epsilon": 1.0e-8,
                "weight_decay": 0.01,
                "scheduler": "cosine",
                "warmup_steps": 16,
                "max_grad_norm": 1.0,
                "optimizer_steps": 192,
                "checkpoints": [48, 96, 144, 192],
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
                "trades_min": 240,
                "each_side_min": 75,
                "each_side_contribution_positive": True,
                "max_action_share": 0.85,
                "stress_and_one_bar_delay_positive": True,
                "familywise_weekly_cluster_p_max_below": 0.05,
                "beat_every_frozen_non_rllm_prior_ablation_null": True,
                "beat_both_orientation_flipped_controls": True,
                "beat_masked_token_prior_adapter": True,
                "ratio_margin_over_strongest_non_rllm": 0.25,
                "max_token_value_non_abstain_action_share": 0.60,
                "single_token_action_reproduction_max": 0.70,
            },
            "selection_order": [
                "higher_ratio",
                "higher_return",
                "lower_strict_mdd",
                "earlier_optimizer_step",
            ],
        },
        "novelty_contract": {
            "comparators": comparator_contracts(),
            "exact_and_one_hour_entry_jaccard": "report",
            "unsigned_time_containment": "report",
            "absolute_signed_exposure_correlation_max": 0.35,
            "live_exact_jaccard_max": 0.10,
            "live_one_hour_jaccard_max": 0.25,
            "zero_variance_or_undefined": "fail",
            "hash_drift_or_missing_common_coverage": "fail",
            "parse_after": "2022 selection and immutable pre-2024 action freeze",
        },
        "eval_2023_gate": {
            "return_positive": True,
            "ratio_min": 3.0,
            "strict_mdd_pct_max": 15.0,
            "both_halves_positive": True,
            "trades_min": 240,
            "each_side_min": 75,
            "each_side_contribution_positive": True,
            "active_months_min": 10,
            "max_month_share": 0.15,
            "max_action_share": 0.85,
            "weekly_clusters_min": 30,
            "one_policy_weekly_cluster_p_below": 0.05,
            "mean_gross_underlying_move_bp_min": 20.0,
            "stress_and_one_bar_delay_positive": True,
            "one_hour_delay": "report_only",
            "every_neutral_code_order_return_positive": True,
            "beat_every_frozen_non_rllm_prior_ablation_null": True,
            "beat_both_orientation_flipped_controls": True,
            "beat_masked_token_prior_adapter": True,
            "ratio_margin_over_strongest_non_rllm": 0.50,
            "max_token_value_non_abstain_action_share": 0.60,
            "single_token_action_reproduction_max": 0.70,
        },
        "strict_sequence": [
            "commit mechanism",
            "commit preregistration and synthetic tests",
            "commit source-only builder",
            "run source support once",
            "retire unchanged on source/token failure",
            "freeze evaluator baselines model controls",
            "open only 2020-2022 outcomes",
            "retire before GPU on transfer/cheap failure",
            "train one Gemma2 SFT and DPO checkpoints 48/96/144/192",
            "select on 2022 and freeze pre-2024 actions",
            "run novelty before 2023 outcomes",
            "evaluate 2023 once",
            "open sealed years sequentially after prior pass",
            "commit every completed unit with hashes and fresh tests",
        ],
        "outcome_boundary": {
            "source_artifact_bytes_hashed": True,
            "source_manifest_aggregate_metadata_read": True,
            "source_header_read": True,
            "source_values_decoded": 0,
            "bcrt_buckets_derived": 0,
            "bcrt_primitive_or_rank_values_derived": 0,
            "bcrt_token_rows_derived": 0,
            "bcrt_opportunity_rows_derived": 0,
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
        raise RuntimeError("BCRT-72 manifest core differs from code")
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("BCRT-72 manifest hash mismatch")
    boundary = payload["outcome_boundary"]
    for field in (
        "source_values_decoded",
        "bcrt_buckets_derived",
        "bcrt_primitive_or_rank_values_derived",
        "bcrt_token_rows_derived",
        "bcrt_opportunity_rows_derived",
        "market_rows_loaded",
        "funding_rows_loaded",
        "comparator_rows_decoded",
        "future_return_rows_loaded",
        "return_or_pnl_fields",
        "post_2023_rows_loaded",
        "model_labels_created",
        "model_training_runs",
    ):
        if boundary[field] != 0:
            raise RuntimeError("BCRT-72 evidence boundary opened")


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
        if output.read_bytes() != expected:
            raise RuntimeError("BCRT-72 existing manifest hash mismatch")
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
                raise RuntimeError("BCRT-72 manifest race drift")
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
