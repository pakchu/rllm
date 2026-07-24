"""Freeze TPCT-120 before decoding source values, incidence, or outcomes."""
from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from fractions import Fraction
import gzip
import hashlib
from importlib import metadata as importlib_metadata
import itertools
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping, Sequence


UTC = timezone.utc
POLICY_ID = "TPCT-120"
PROTOCOL_VERSION = "tri_party_composition_topology_preregistration_v1"
DEFAULT_OUTPUT = (
    "results/tri_party_composition_topology_preregistration_2026-07-24.json"
)

BOUNDARY_DOCUMENT = (
    "docs/tri-party-composition-topology-boundary-2026-07-24.md"
)
BOUNDARY_DOCUMENT_SHA256 = (
    "a3e8d5c0dd2da652b4be93627a935f064c1d144560b124613f34f95287d28159"
)
BOUNDARY_COMMIT = "51cbcd0cff26ce7c25f9cf94ea78932d616b4af9"
MECHANISM_DOCUMENT = (
    "docs/tri-party-composition-topology-mechanism-decision-2026-07-24.md"
)
MECHANISM_DOCUMENT_SHA256 = (
    "b99883a3fe932a1083c791837d3fb3e91d8e5d598c198af0da4f6db652724137"
)
MECHANISM_COMMIT = "cc3eec15cde974e0a72b8c2cfc016d10c0f1bf2f"

SOURCE = (
    "data/ofr_repo_preliminary_2019_2023/"
    "ofr_repo_preliminary_observations_2019_2023.csv.gz"
)
SOURCE_SHA256 = (
    "6bb24b9a3d6ed6d266b0c885a9d9888bcccb3045df66d37d83031af66c6dc02a"
)
SOURCE_HEADER_SHA256 = (
    "743cb319f6fe1d2722cf8f249ff981c87b24def885a8776b0017f03cb060959c"
)
METADATA = (
    "data/ofr_repo_preliminary_2019_2023/"
    "ofr_repo_preliminary_metadata_2019_2023.json.gz"
)
METADATA_SHA256 = (
    "19a04e82eb5d8ddc6c3cb8dc64694438abd6b1987951470bb317659d9c53ef4f"
)
SELECTED_METADATA_HASH = (
    "e75d656e6ae322eeb0a44ef9e52450af21c54c3a17936c0791ec9fa4421c8edc"
)
SOURCE_MANIFEST = (
    "data/ofr_repo_preliminary_2019_2023/build_manifest.json"
)
SOURCE_MANIFEST_SHA256 = (
    "f937f567e1789ecb39a2b84d6288b2cbab931da4e9f1f4e51addea4b3423b705"
)
SOURCE_MANIFEST_HASH = (
    "802b83a9478711cd29d5b606d9e12eb1e90890e37f5908d4de64d7dd71f6d449"
)
SOURCE_AUDIT = "docs/ofr-repo-preliminary-source-audit-2026-07-23.md"
SOURCE_AUDIT_SHA256 = (
    "88e5ee4852acda41759b4c85731e3f6be170869a7485985b9c96507daa387ccb"
)

SOURCE_COLUMNS = (
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
SOURCE_ALLOWLIST = (
    "REPO-TRIV1_AR_OO-P",
    "REPO-TRIV1_TV_OO-P",
    "REPO-TRIV1_AR_B27-P",
    "REPO-TRIV1_TV_B27-P",
    "REPO-TRIV1_AR_B830-P",
    "REPO-TRIV1_TV_B830-P",
    "REPO-TRIV1_AR_G30-P",
    "REPO-TRIV1_TV_G30-P",
    "REPO-TRIV1_AR_T-P",
    "REPO-TRIV1_TV_T-P",
    "REPO-TRIV1_AR_AG-P",
    "REPO-TRIV1_TV_AG-P",
    "REPO-TRIV1_AR_CORD-P",
    "REPO-TRIV1_TV_CORD-P",
    "REPO-TRIV1_AR_O-P",
    "REPO-TRIV1_TV_O-P",
)
VALUE_KEYS = (
    "AR_OO",
    "TV_OO",
    "AR_B27",
    "TV_B27",
    "AR_B830",
    "TV_B830",
    "AR_G30",
    "TV_G30",
    "AR_T",
    "TV_T",
    "AR_AG",
    "TV_AG",
    "AR_CORD",
    "TV_CORD",
    "AR_O",
    "TV_O",
)
MNEMONIC_TO_VALUE_KEY = dict(zip(SOURCE_ALLOWLIST, VALUE_KEYS, strict=True))

PRIMITIVES = (
    "OVERNIGHT",
    "NEAR_TERM",
    "MEDIUM_TERM",
    "LONG_TERM",
    "TERM_PREMIUM",
    "GOVERNMENT_SHARE",
    "TREASURY_WITHIN_GOV",
    "CORPORATE_WITHIN_PRIVATE",
    "PRIVATE_PREMIUM",
    "CONCENTRATION_GAP",
)
LEADER_VOCABULARY = (*PRIMITIVES, "TIE")
TOKEN_SCHEMA: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "maturity_wings",
        ("OVERNIGHT_LEADS", "BALANCED", "LONG_TERM_LEADS"),
    ),
    (
        "term_belly",
        ("NEAR_TERM_LEADS", "BALANCED", "MEDIUM_TERM_LEADS"),
    ),
    (
        "term_volume_rate",
        ("LONG_TERM_VOLUME_LEADS", "BALANCED", "TERM_RATE_LEADS"),
    ),
    (
        "collateral_volume_rate",
        ("GOVERNMENT_VOLUME_LEADS", "BALANCED", "PRIVATE_RATE_LEADS"),
    ),
    (
        "safe_risky_composition",
        ("TREASURY_LEADS", "BALANCED", "CORPORATE_LEADS"),
    ),
    (
        "rate_surface",
        ("TERM_RATE_LEADS", "BALANCED", "PRIVATE_RATE_LEADS"),
    ),
    ("high_leader", LEADER_VOCABULARY),
    ("low_leader", LEADER_VOCABULARY),
    ("rank_breadth", ("LOW_BROAD", "MIXED", "HIGH_BROAD")),
    ("extreme_occupancy", ("COMPACT", "FOCUSED", "FRACTURED")),
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
ACTION_NAMES = ("ABSTAIN", "LONG", "SHORT")
NEUTRAL_CODES = ("Q1", "Q2", "Q3")
ACTION_BY_CODE = dict(zip(NEUTRAL_CODES, ACTION_NAMES, strict=True))
SOURCE_FAMILY_LEDGER = (
    "RVFC-72",
    "RMSR-72",
    "RCRE-72",
    "DMSH-168",
    "TPCT-120",
)
SERIALIZATION_SPECIMEN = OrderedDict(
    (
        ("maturity_wings", "OVERNIGHT_LEADS"),
        ("term_belly", "NEAR_TERM_LEADS"),
        ("term_volume_rate", "LONG_TERM_VOLUME_LEADS"),
        ("collateral_volume_rate", "GOVERNMENT_VOLUME_LEADS"),
        ("safe_risky_composition", "TREASURY_LEADS"),
        ("rate_surface", "TERM_RATE_LEADS"),
        ("high_leader", "OVERNIGHT"),
        ("low_leader", "OVERNIGHT"),
        ("rank_breadth", "LOW_BROAD"),
        ("extreme_occupancy", "COMPACT"),
        ("order_transition", "STABLE"),
        ("leader_transition", "TIE_INVOLVED"),
    )
)
SERIALIZATION_SELF_TEST_SHA256 = (
    "38bdb13f6fcfe23af8b2b47cdba412eaab66d411ededc4e6df1381356b9cea33"
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

COMPARATORS = (
    (
        "RVFC",
        "results/ofr_repo_venue_fragmentation_consensus_clocks_2026-07-23.csv.gz",
        "b2d251b147feecc98def94610834c7b11f078f35e580f6e7a950fc55bfe0724e",
    ),
    (
        "RMSR",
        "results/ofr_repo_mix_shock_resolution_race_clocks_2026-07-23.csv.gz",
        "bed2f522bd68fd389cb0af3d655980999324c0e6adb4f1dc027c6b6ea2cc5ae6",
    ),
    (
        "RCRE",
        "results/ofr_repo_collateral_routing_efficiency_clocks_2026-07-23.csv.gz",
        "cbe4e5f6fc52b66062abbf931e46ea4aa0d1f3c0157ffd365d0638aa573c2826",
    ),
    (
        "DMSH",
        "results/ofr_dvp_maturity_stock_flow_handoff_clocks_2026-07-23.csv.gz",
        "0cfb881b4e3a0123111eeab904eba7bee074767b9c1315f74e7bddf54e3371c3",
    ),
    (
        "FED_LIQUIDITY_COMPONENTS",
        "results/federal_liquidity_component_concordance_preregistered_clock_2026-07-17.csv.gz",
        "7ebb0450422d9265e46c596e0b6415b6a8816c66f5e0cbb9ccda14ca6cb4c67c",
    ),
    (
        "FROZEN_LIVE_SLEEVES",
        "results/cchr_live_portfolio_pure_clocks_2026-07-21.csv.gz",
        "73d6efbd35b3be64b0fa04fa9c8cb2db25866ef884f19b1ae673949e22a42b08",
    ),
)

MODEL_ID = "google/gemma-4-E2B-it"
MODEL_REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
MODEL_FILES = {
    "chat_template.jinja": (
        "0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5"
    ),
    "config.json": (
        "1b28f3d2c3100f6c594754b81107428bd7b822a7f48272ca681dae9d2ec38330"
    ),
    "generation_config.json": (
        "d4226bbe3117d2d253ba4609720ba82c6c4ce4627a9a6ae05387c78983ac03de"
    ),
    "model.safetensors": (
        "2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550"
    ),
    "processor_config.json": (
        "32bdf45d2ad4cc29a0822ddd157a182de76644f0419a6228d151495256e9813c"
    ),
    "tokenizer.json": (
        "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"
    ),
    "tokenizer_config.json": (
        "9f4fec4b1dc6ecddf8f4a92e9caea5971c0e67d81309f3f9066a2bee8c362633"
    ),
}
RUNTIME_DISTRIBUTIONS = OrderedDict(
    (
        ("torch", "2.9.0"),
        ("transformers", "5.7.0.dev0"),
        ("trl", "0.29.0"),
        ("peft", "0.18.1"),
        ("bitsandbytes", "0.49.2"),
        ("numpy", "2.2.6"),
        ("pandas", "2.3.3"),
        ("scikit-learn", "1.7.2"),
    )
)
TRANSFORMERS_GIT_COMMIT = "5d7ff4393ab99aa7cadf4cccd1f814dbb799f2bb"

DECIMAL_PATTERN = re.compile(r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$")
TRAIN_START = datetime(2020, 9, 10, tzinfo=UTC)
SELECTION_START = datetime(2022, 1, 1, tzinfo=UTC)
EVAL_START = datetime(2023, 1, 1, tzinfo=UTC)
SEALED_START = datetime(2024, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class Policy:
    policy_id: str = POLICY_ID
    bar_seconds: int = 300
    latency_bars: int = 1
    hold_hours: int = 120
    hold_bars: int = 1_440
    rank_lookback: int = 252
    pair_delta_numerator: int = 1
    pair_delta_denominator: int = 6
    rank_breadth_threshold: int = 3
    extreme_low_numerator: int = 1
    extreme_low_denominator: int = 6
    extreme_high_numerator: int = 5
    extreme_high_denominator: int = 6
    order_stable_max_changes: int = 5
    order_rotating_max_changes: int = 14
    leverage: float = 0.5
    base_cost_notional_per_side: float = 0.0006
    stress_cost_notional_per_side: float = 0.0010
    held_path_drawdown_penalty: float = 1.0 / 3.0
    utility_hurdle_account: float = 0.0005
    preference_margin: float = 0.0003
    random_seed: int = 20_260_724


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        raise ValueError("TPCT paths must be repository-relative")
    if not candidate.parts or any(part == ".." for part in candidate.parts):
        raise ValueError("TPCT path escaped repository")
    return repository_root() / candidate


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with repository_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_external_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_hash(payload: Any) -> str:
    return sha256_bytes(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    )


def source_header_bytes() -> bytes:
    with gzip.open(repository_path(SOURCE), "rb") as handle:
        raw = handle.readline()
    if not raw.endswith(b"\n"):
        raise RuntimeError("TPCT source header lost LF terminator")
    return raw.rstrip(b"\r\n")


def parse_exact_decimal(value: str) -> Fraction:
    text = str(value)
    if not DECIMAL_PATTERN.fullmatch(text):
        raise ValueError("TPCT value is not a canonical exact decimal")
    if text.startswith("-0") and Fraction(text) == 0:
        raise ValueError("TPCT negative zero is forbidden")
    parsed = Fraction(text)
    if parsed.denominator <= 0:
        raise ValueError("TPCT exact decimal denominator is invalid")
    return parsed


def ceil_5m(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("TPCT clock must be timezone-aware")
    normalized = value.astimezone(UTC)
    seconds = int(normalized.timestamp())
    if normalized.microsecond:
        seconds += 1
    width = Policy().bar_seconds
    rounded = ((seconds + width - 1) // width) * width
    return datetime.fromtimestamp(rounded, tz=UTC)


def opportunity_times(available_at_utc: datetime) -> dict[str, datetime]:
    policy = Policy()
    signal = ceil_5m(available_at_utc)
    entry = signal + timedelta(seconds=policy.bar_seconds * policy.latency_bars)
    exit_time = entry + timedelta(hours=policy.hold_hours)
    if exit_time - entry != timedelta(
        seconds=policy.hold_bars * policy.bar_seconds
    ):
        raise RuntimeError("TPCT hold hours/bars disagree")
    return {"signal_available": signal, "entry": entry, "exit": exit_time}


def source_value_may_decode(available_at_utc: datetime) -> bool:
    times = opportunity_times(available_at_utc)
    return times["entry"] >= TRAIN_START and times["exit"] < EVAL_START


def split_contains(
    entry: datetime,
    exit_time: datetime,
    split_start: datetime,
    split_end: datetime,
) -> bool:
    values = (entry, exit_time, split_start, split_end)
    if any(value.tzinfo is None for value in values):
        raise ValueError("TPCT split clock must be timezone-aware")
    if split_end <= split_start or exit_time <= entry:
        raise ValueError("TPCT split or opportunity interval is invalid")
    return entry >= split_start and exit_time < split_end


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
            raise ValueError("TPCT reservation interval is invalid")
        row["reserved"] = previous_exit is None or entry >= previous_exit
        result.append(row)
        if row["reserved"]:
            previous_exit = exit_time
    return result


def _exact_values(values: Mapping[str, Fraction]) -> OrderedDict[str, Fraction]:
    if tuple(values) != VALUE_KEYS:
        raise ValueError("TPCT source value order changed")
    normalized: OrderedDict[str, Fraction] = OrderedDict()
    for key in VALUE_KEYS:
        value = values[key]
        if not isinstance(value, Fraction):
            raise TypeError("TPCT primitive inputs must be exact Fractions")
        normalized[key] = value
    for key in VALUE_KEYS:
        if key.startswith("TV_") and normalized[key] <= 0:
            raise ValueError("TPCT transaction volume must be positive")
    return normalized


def build_primitives(
    values: Mapping[str, Fraction],
) -> OrderedDict[str, Fraction]:
    row = _exact_values(values)
    v_oo, v_b27, v_b830, v_g30 = (
        row["TV_OO"],
        row["TV_B27"],
        row["TV_B830"],
        row["TV_G30"],
    )
    v_t, v_ag, v_cord, v_o = (
        row["TV_T"],
        row["TV_AG"],
        row["TV_CORD"],
        row["TV_O"],
    )
    v_tenor = v_oo + v_b27 + v_b830 + v_g30
    v_term = v_b27 + v_b830 + v_g30
    v_collateral = v_t + v_ag + v_cord + v_o
    v_gov = v_t + v_ag
    v_private = v_cord + v_o
    tenor_shares = (
        v_oo / v_tenor,
        v_b27 / v_tenor,
        v_b830 / v_tenor,
        v_g30 / v_tenor,
    )
    collateral_shares = (
        v_t / v_collateral,
        v_ag / v_collateral,
        v_cord / v_collateral,
        v_o / v_collateral,
    )
    if sum(tenor_shares, Fraction()) != 1:
        raise RuntimeError("TPCT tenor simplex identity failed")
    if sum(collateral_shares, Fraction()) != 1:
        raise RuntimeError("TPCT collateral simplex identity failed")
    term_rate = (
        v_b27 * row["AR_B27"]
        + v_b830 * row["AR_B830"]
        + v_g30 * row["AR_G30"]
    ) / v_term
    government_rate = (
        v_t * row["AR_T"] + v_ag * row["AR_AG"]
    ) / v_gov
    private_rate = (
        v_cord * row["AR_CORD"] + v_o * row["AR_O"]
    ) / v_private
    return OrderedDict(
        (
            ("OVERNIGHT", tenor_shares[0]),
            ("NEAR_TERM", tenor_shares[1]),
            ("MEDIUM_TERM", tenor_shares[2]),
            ("LONG_TERM", tenor_shares[3]),
            ("TERM_PREMIUM", term_rate - row["AR_OO"]),
            ("GOVERNMENT_SHARE", v_gov / v_collateral),
            ("TREASURY_WITHIN_GOV", v_t / v_gov),
            ("CORPORATE_WITHIN_PRIVATE", v_cord / v_private),
            ("PRIVATE_PREMIUM", private_rate - government_rate),
            (
                "CONCENTRATION_GAP",
                sum((value * value for value in tenor_shares), Fraction())
                - sum(
                    (value * value for value in collateral_shares),
                    Fraction(),
                ),
            ),
        )
    )


def strict_prior_midrank(
    current: Fraction,
    prior_values: Iterable[Fraction],
) -> Fraction:
    if not isinstance(current, Fraction):
        raise TypeError("TPCT rank current value must be Fraction")
    prior = tuple(prior_values)
    if len(prior) != Policy().rank_lookback:
        raise ValueError("TPCT strict-prior history must contain exactly 252 rows")
    if any(not isinstance(value, Fraction) for value in prior):
        raise TypeError("TPCT rank prior values must be Fractions")
    less = sum(value < current for value in prior)
    equal = sum(value == current for value in prior)
    return Fraction(2 * less + equal, 2 * len(prior))


def _validate_ranks(
    ranks: Mapping[str, Fraction],
) -> OrderedDict[str, Fraction]:
    if tuple(ranks) != PRIMITIVES:
        raise ValueError("TPCT primitive rank order changed")
    normalized: OrderedDict[str, Fraction] = OrderedDict()
    for key in PRIMITIVES:
        value = ranks[key]
        if not isinstance(value, Fraction) or not 0 <= value <= 1:
            raise ValueError("TPCT primitive rank is invalid")
        normalized[key] = value
    return normalized


def pair_relation(
    left: Fraction,
    right: Fraction,
    *,
    left_token: str,
    right_token: str,
) -> str:
    if (
        not isinstance(left, Fraction)
        or not isinstance(right, Fraction)
        or not 0 <= left <= 1
        or not 0 <= right <= 1
    ):
        raise ValueError("TPCT pair ranks are invalid")
    delta = left - right
    boundary = Fraction(1, 6)
    if delta > boundary:
        return left_token
    if delta < -boundary:
        return right_token
    return "BALANCED"


def extreme_leader(
    ranks: Mapping[str, Fraction],
    *,
    highest: bool,
) -> str:
    values = _validate_ranks(ranks)
    extreme = (max if highest else min)(values.values())
    winners = [name for name in PRIMITIVES if values[name] == extreme]
    return winners[0] if len(winners) == 1 else "TIE"


def rank_breadth(ranks: Mapping[str, Fraction]) -> str:
    values = _validate_ranks(ranks)
    score = sum(value > Fraction(1, 2) for value in values.values()) - sum(
        value < Fraction(1, 2) for value in values.values()
    )
    if score >= Policy().rank_breadth_threshold:
        return "HIGH_BROAD"
    if score <= -Policy().rank_breadth_threshold:
        return "LOW_BROAD"
    return "MIXED"


def extreme_occupancy(ranks: Mapping[str, Fraction]) -> str:
    values = _validate_ranks(ranks)
    count = sum(
        value < Fraction(1, 6) or value > Fraction(5, 6)
        for value in values.values()
    )
    if count <= 3:
        return "COMPACT"
    if count <= 6:
        return "FOCUSED"
    return "FRACTURED"


def _pair_order_states(
    ranks: Mapping[str, Fraction],
) -> tuple[int, ...]:
    values = _validate_ranks(ranks)
    result: list[int] = []
    for left_index, left in enumerate(PRIMITIVES):
        for right in PRIMITIVES[left_index + 1 :]:
            delta = values[left] - values[right]
            result.append(1 if delta > 0 else -1 if delta < 0 else 0)
    if len(result) != 45:
        raise RuntimeError("TPCT pair-order state count changed")
    return tuple(result)


def order_transition(
    current: Mapping[str, Fraction],
    previous: Mapping[str, Fraction],
) -> str:
    changed = sum(
        left != right
        for left, right in zip(
            _pair_order_states(current),
            _pair_order_states(previous),
            strict=True,
        )
    )
    if changed <= Policy().order_stable_max_changes:
        return "STABLE"
    if changed <= Policy().order_rotating_max_changes:
        return "ROTATING"
    return "RESET"


def leader_transition(
    current_high: str,
    current_low: str,
    previous_high: str,
    previous_low: str,
) -> str:
    leaders = (current_high, current_low, previous_high, previous_low)
    if any(value not in LEADER_VOCABULARY for value in leaders):
        raise ValueError("TPCT leader token is invalid")
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


def build_tokens(
    current: Mapping[str, Fraction],
    previous: Mapping[str, Fraction],
) -> OrderedDict[str, str]:
    ranks = _validate_ranks(current)
    prior = _validate_ranks(previous)
    current_high = extreme_leader(ranks, highest=True)
    current_low = extreme_leader(ranks, highest=False)
    previous_high = extreme_leader(prior, highest=True)
    previous_low = extreme_leader(prior, highest=False)
    tokens: OrderedDict[str, str] = OrderedDict(
        (
            (
                "maturity_wings",
                pair_relation(
                    ranks["OVERNIGHT"],
                    ranks["LONG_TERM"],
                    left_token="OVERNIGHT_LEADS",
                    right_token="LONG_TERM_LEADS",
                ),
            ),
            (
                "term_belly",
                pair_relation(
                    ranks["NEAR_TERM"],
                    ranks["MEDIUM_TERM"],
                    left_token="NEAR_TERM_LEADS",
                    right_token="MEDIUM_TERM_LEADS",
                ),
            ),
            (
                "term_volume_rate",
                pair_relation(
                    ranks["LONG_TERM"],
                    ranks["TERM_PREMIUM"],
                    left_token="LONG_TERM_VOLUME_LEADS",
                    right_token="TERM_RATE_LEADS",
                ),
            ),
            (
                "collateral_volume_rate",
                pair_relation(
                    ranks["GOVERNMENT_SHARE"],
                    ranks["PRIVATE_PREMIUM"],
                    left_token="GOVERNMENT_VOLUME_LEADS",
                    right_token="PRIVATE_RATE_LEADS",
                ),
            ),
            (
                "safe_risky_composition",
                pair_relation(
                    ranks["TREASURY_WITHIN_GOV"],
                    ranks["CORPORATE_WITHIN_PRIVATE"],
                    left_token="TREASURY_LEADS",
                    right_token="CORPORATE_LEADS",
                ),
            ),
            (
                "rate_surface",
                pair_relation(
                    ranks["TERM_PREMIUM"],
                    ranks["PRIVATE_PREMIUM"],
                    left_token="TERM_RATE_LEADS",
                    right_token="PRIVATE_RATE_LEADS",
                ),
            ),
            ("high_leader", current_high),
            ("low_leader", current_low),
            ("rank_breadth", rank_breadth(ranks)),
            ("extreme_occupancy", extreme_occupancy(ranks)),
            ("order_transition", order_transition(ranks, prior)),
            (
                "leader_transition",
                leader_transition(
                    current_high,
                    current_low,
                    previous_high,
                    previous_low,
                ),
            ),
        )
    )
    validate_tokens(tokens)
    return tokens


def validate_tokens(tokens: Mapping[str, str]) -> dict[str, str]:
    if tuple(tokens) != TOKEN_COLUMNS:
        raise ValueError("TPCT token order or schema changed")
    normalized = {name: str(tokens[name]) for name in TOKEN_COLUMNS}
    for key, value in normalized.items():
        if value not in TOKEN_VOCABULARY[key]:
            raise ValueError(f"TPCT token is invalid: {key}={value}")
    return normalized


def action_option_orders() -> tuple[tuple[str, str, str], ...]:
    return tuple(itertools.permutations(NEUTRAL_CODES))


def build_user_text(
    tokens: Mapping[str, str],
    option_order: Sequence[str],
) -> str:
    normalized = validate_tokens(tokens)
    order = tuple(str(value) for value in option_order)
    if len(order) != len(NEUTRAL_CODES) or set(order) != set(NEUTRAL_CODES):
        raise ValueError("TPCT option order must be one Q1/Q2/Q3 permutation")
    lines = [
        "STATE:",
        *(f"{key}={normalized[key]}" for key in TOKEN_COLUMNS),
        "TASK=TPCT_ACTION",
        "OPTIONS:",
        *(f"{code}={ACTION_BY_CODE[code]}" for code in order),
        "Return exactly CHOICE=<one option>.",
    ]
    text = "\n".join(lines)
    if text.endswith("\n") or any(
        line != line.rstrip() for line in text.splitlines()
    ):
        raise RuntimeError("TPCT user text whitespace contract failed")
    return text


def completion_text(code: str) -> str:
    if code not in NEUTRAL_CODES:
        raise ValueError("TPCT completion code is invalid")
    return f"CHOICE={code}"


def chat_messages(
    user_text: str,
    *,
    assistant_completion: str | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(user_text, str) or not user_text:
        raise ValueError("TPCT user text is empty")
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [{"type": "text", "text": user_text}],
        }
    ]
    if assistant_completion is not None:
        if assistant_completion not in {
            completion_text(code) for code in NEUTRAL_CODES
        }:
            raise ValueError("TPCT assistant completion is invalid")
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": assistant_completion}
                ],
            }
        )
    return messages


def _render_chat_text(
    processor: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    add_generation_prompt: bool,
) -> str:
    rendered = processor.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
        preserve_thinking=False,
        tools=None,
    )
    if not isinstance(rendered, str):
        raise RuntimeError("TPCT chat template did not return text")
    return rendered


def _render_chat_ids(
    processor: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    rendered = processor.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
        preserve_thinking=False,
        tools=None,
        return_dict=True,
        return_tensors="pt",
    )
    forbidden = {
        "pixel_values",
        "image_sizes",
        "audio_values",
        "video_values",
    }
    if forbidden.intersection(rendered):
        raise RuntimeError("TPCT text-only chat created multimodal tensors")
    required = {"input_ids", "attention_mask", "mm_token_type_ids"}
    if not required.issubset(rendered):
        raise RuntimeError("TPCT chat tensor schema changed")
    input_ids = rendered["input_ids"]
    attention = rendered["attention_mask"]
    token_types = rendered["mm_token_type_ids"]
    if tuple(input_ids.shape[:1]) != (1,):
        raise RuntimeError("TPCT chat batch shape changed")
    ids = [int(value) for value in input_ids[0].tolist()]
    if [int(value) for value in attention[0].tolist()] != [1] * len(ids):
        raise RuntimeError("TPCT chat attention mask changed")
    if [int(value) for value in token_types[0].tolist()] != [0] * len(ids):
        raise RuntimeError("TPCT chat emitted non-text token types")
    return ids


def build_serialization_self_test(
    processor: Any,
    *,
    enforce_expected: bool = True,
) -> dict[str, Any]:
    specimen = validate_tokens(SERIALIZATION_SPECIMEN)
    prompt_prefixes: list[dict[str, Any]] = []
    prompt_state: dict[tuple[str, ...], tuple[str, list[int]]] = {}
    for option_order in action_option_orders():
        user_text = build_user_text(specimen, option_order)
        messages = chat_messages(user_text)
        rendered_text = _render_chat_text(
            processor,
            messages,
            add_generation_prompt=True,
        )
        rendered_ids = _render_chat_ids(
            processor,
            messages,
            add_generation_prompt=True,
        )
        retokenized = processor.tokenizer(
            rendered_text,
            add_special_tokens=False,
        )["input_ids"]
        if [int(value) for value in retokenized] != rendered_ids:
            raise RuntimeError("TPCT rendered prompt tokenization drifted")
        if rendered_text.count("<bos>") != 1 or not rendered_text.startswith(
            "<bos>"
        ):
            raise RuntimeError("TPCT chat template BOS contract changed")
        if "<eos>" in rendered_text:
            raise RuntimeError("TPCT scoring prefix unexpectedly contains EOS")
        order_key = tuple(option_order)
        prompt_state[order_key] = (rendered_text, rendered_ids)
        prompt_prefixes.append(
            {
                "option_order": list(option_order),
                "user_text_sha256": sha256_bytes(user_text.encode("utf-8")),
                "rendered_text_sha256": sha256_bytes(
                    rendered_text.encode("utf-8")
                ),
                "input_ids_sha256": canonical_hash(rendered_ids),
                "input_tokens": len(rendered_ids),
            }
        )

    special_ids = {int(value) for value in processor.tokenizer.all_special_ids}
    completions: list[dict[str, Any]] = []
    completion_ids: dict[str, list[int]] = {}
    for code in NEUTRAL_CODES:
        text = completion_text(code)
        ids = [
            int(value)
            for value in processor.tokenizer(
                text,
                add_special_tokens=False,
            )["input_ids"]
        ]
        if not ids or special_ids.intersection(ids):
            raise RuntimeError("TPCT completion tokenization is invalid")
        completion_ids[code] = ids
        completions.append(
            {
                "code": code,
                "action": ACTION_BY_CODE[code],
                "text": text,
                "input_ids": ids,
                "input_ids_sha256": canonical_hash(ids),
            }
        )

    full_examples: list[dict[str, Any]] = []
    for option_order in action_option_orders():
        order_key = tuple(option_order)
        prefix_text, prefix_ids = prompt_state[order_key]
        user_text = build_user_text(specimen, option_order)
        for code in NEUTRAL_CODES:
            completion = completion_text(code)
            messages = chat_messages(
                user_text,
                assistant_completion=completion,
            )
            rendered_text = _render_chat_text(
                processor,
                messages,
                add_generation_prompt=False,
            )
            rendered_ids = _render_chat_ids(
                processor,
                messages,
                add_generation_prompt=False,
            )
            retokenized = processor.tokenizer(
                rendered_text,
                add_special_tokens=False,
            )["input_ids"]
            if [int(value) for value in retokenized] != rendered_ids:
                raise RuntimeError("TPCT full-example tokenization drifted")
            ids = completion_ids[code]
            if rendered_ids[: len(prefix_ids)] != prefix_ids:
                raise RuntimeError("TPCT training/scoring prefixes disagree")
            if rendered_ids[
                len(prefix_ids) : len(prefix_ids) + len(ids)
            ] != ids:
                raise RuntimeError("TPCT completion mask location changed")
            if not rendered_text.startswith(prefix_text + completion):
                raise RuntimeError("TPCT rendered completion boundary changed")
            full_examples.append(
                {
                    "option_order": list(option_order),
                    "code": code,
                    "rendered_text_sha256": sha256_bytes(
                        rendered_text.encode("utf-8")
                    ),
                    "input_ids_sha256": canonical_hash(rendered_ids),
                    "input_tokens": len(rendered_ids),
                    "completion_start": len(prefix_ids),
                    "completion_stop": len(prefix_ids) + len(ids),
                }
            )

    body = {
        "version": "tpct_gemma4_serialization_self_test_v1",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "chat_template_sha256": MODEL_FILES["chat_template.jinja"],
        "tokenizer_sha256": MODEL_FILES["tokenizer.json"],
        "tokenizer_config_sha256": MODEL_FILES["tokenizer_config.json"],
        "specimen": specimen,
        "chat_arguments": {
            "tokenize": True,
            "add_generation_prompt_scoring": True,
            "add_generation_prompt_training": False,
            "enable_thinking": False,
            "preserve_thinking": False,
            "tools": None,
            "return_dict": True,
            "return_tensors": "pt",
            "caller_add_special_tokens": False,
            "completion_includes_eos_or_turn_close": False,
        },
        "prompt_prefixes": prompt_prefixes,
        "completions": completions,
        "full_examples": full_examples,
    }
    digest = canonical_hash(body)
    if enforce_expected and digest != SERIALIZATION_SELF_TEST_SHA256:
        raise RuntimeError("TPCT serialization self-test hash mismatch")
    return {**body, "self_test_sha256": digest}


def _read_json(path: str | Path) -> Any:
    with repository_path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_metadata() -> list[dict[str, Any]]:
    with gzip.open(repository_path(METADATA), "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise RuntimeError("TPCT metadata must be a JSON array")
    if any(not isinstance(row, dict) for row in payload):
        raise RuntimeError("TPCT metadata row is not an object")
    return payload


def validate_selected_metadata() -> dict[str, Any]:
    rows = _read_metadata()
    by_mnemonic = {str(row.get("mnemonic")): row for row in rows}
    if len(by_mnemonic) != len(rows):
        raise RuntimeError("TPCT metadata contains duplicate mnemonics")
    selected = [by_mnemonic[mnemonic] for mnemonic in SOURCE_ALLOWLIST]
    if canonical_hash(selected) != SELECTED_METADATA_HASH:
        raise RuntimeError("TPCT selected metadata hash mismatch")
    for mnemonic, row in zip(SOURCE_ALLOWLIST, selected, strict=True):
        value_key = MNEMONIC_TO_VALUE_KEY[mnemonic]
        measure, subset = value_key.split("_", 1)
        if row.get("segment") != "TRIV1":
            raise RuntimeError("TPCT metadata segment changed")
        if row.get("measure") != measure or row.get("subset") != subset:
            raise RuntimeError("TPCT metadata measure/subset changed")
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            raise RuntimeError("TPCT metadata body is missing")
        description = metadata.get("description")
        schedule = metadata.get("schedule")
        release = metadata.get("release")
        unit = metadata.get("unit")
        if not all(
            isinstance(value, dict)
            for value in (description, schedule, release, unit)
        ):
            raise RuntimeError("TPCT metadata nested schema changed")
        if description.get("vintage") != "Preliminary":
            raise RuntimeError("TPCT metadata vintage changed")
        if description.get("vintage_approach") != "Preliminary":
            raise RuntimeError("TPCT metadata vintage approach changed")
        if schedule.get("observation_frequency") != "Daily":
            raise RuntimeError("TPCT observation frequency changed")
        if release.get("frequency") != "Daily":
            raise RuntimeError("TPCT release frequency changed")
        expected_type = "Rate" if measure == "AR" else "Volume"
        expected_name = "Percent" if measure == "AR" else "USD"
        if unit.get("type") != expected_type or unit.get("name") != expected_name:
            raise RuntimeError("TPCT metadata unit changed")
        series_name = str(row.get("series_name", ""))
        if "excluding Federal Reserve transactions" not in series_name:
            raise RuntimeError("TPCT TRIV1 semantics changed")
    return {
        "rows_total": len(rows),
        "selected_rows": len(selected),
        "selected_hash": SELECTED_METADATA_HASH,
    }


def model_snapshot() -> Path:
    if "HF_HUB_CACHE" in os.environ:
        root = Path(os.environ["HF_HUB_CACHE"])
    elif "HF_HOME" in os.environ:
        root = Path(os.environ["HF_HOME"]) / "hub"
    else:
        root = Path.home() / ".cache" / "huggingface" / "hub"
    return (
        root
        / "models--google--gemma-4-E2B-it"
        / "snapshots"
        / MODEL_REVISION
    )


def load_model_processor() -> Any:
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(
        model_snapshot(),
        local_files_only=True,
        trust_remote_code=False,
    )


def validate_runtime() -> dict[str, Any]:
    versions: dict[str, str] = {}
    for distribution, expected in RUNTIME_DISTRIBUTIONS.items():
        actual = importlib_metadata.version(distribution)
        if actual != expected:
            raise RuntimeError(
                f"TPCT runtime version changed: {distribution}={actual}"
            )
        versions[distribution] = actual
    direct_url_raw = importlib_metadata.distribution(
        "transformers"
    ).read_text("direct_url.json")
    if direct_url_raw is None:
        raise RuntimeError("TPCT transformers direct_url binding is missing")
    try:
        direct_url = json.loads(direct_url_raw)
        commit = direct_url["vcs_info"]["commit_id"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "TPCT transformers direct_url binding is invalid"
        ) from exc
    if commit != TRANSFORMERS_GIT_COMMIT:
        raise RuntimeError("TPCT transformers git commit changed")
    return {
        "distributions": versions,
        "transformers_git_commit": commit,
    }


def frozen_binding_hashes() -> dict[str, str]:
    bindings = {
        BOUNDARY_DOCUMENT: BOUNDARY_DOCUMENT_SHA256,
        MECHANISM_DOCUMENT: MECHANISM_DOCUMENT_SHA256,
        SOURCE: SOURCE_SHA256,
        METADATA: METADATA_SHA256,
        SOURCE_MANIFEST: SOURCE_MANIFEST_SHA256,
        SOURCE_AUDIT: SOURCE_AUDIT_SHA256,
        MARKET_SOURCE: MARKET_SOURCE_SHA256,
        MARKET_MANIFEST: MARKET_MANIFEST_SHA256,
        FUNDING_SOURCE: FUNDING_SOURCE_SHA256,
        FUNDING_MANIFEST: FUNDING_MANIFEST_SHA256,
    }
    for _, path, expected in COMPARATORS:
        bindings[path] = expected
    return bindings


def validate_bindings() -> dict[str, Any]:
    bindings = frozen_binding_hashes()
    for path, expected in bindings.items():
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"TPCT binding hash mismatch: {path}")
    if sha256_bytes(source_header_bytes()) != SOURCE_HEADER_SHA256:
        raise RuntimeError("TPCT source header hash mismatch")
    if tuple(source_header_bytes().decode("utf-8").split(",")) != SOURCE_COLUMNS:
        raise RuntimeError("TPCT source header changed")
    source_manifest = _read_json(SOURCE_MANIFEST)
    if source_manifest.get("manifest_hash") != SOURCE_MANIFEST_HASH:
        raise RuntimeError("TPCT source manifest hash field changed")
    core = dict(source_manifest)
    recorded = core.pop("manifest_hash")
    if canonical_hash(core) != recorded:
        raise RuntimeError("TPCT source manifest is not canonical")
    checks = source_manifest.get("source_checks")
    if not isinstance(checks, dict) or not all(checks.values()):
        raise RuntimeError("TPCT source manifest contains a failed check")
    metadata_report = validate_selected_metadata()
    snapshot = model_snapshot()
    model_report: dict[str, Any] = {}
    for filename, expected in MODEL_FILES.items():
        path = snapshot / filename
        if not path.is_file():
            raise RuntimeError(f"TPCT model file is missing: {filename}")
        actual = sha256_external_file(path)
        if actual != expected:
            raise RuntimeError(f"TPCT model file hash mismatch: {filename}")
        model_report[filename] = {
            "sha256": actual,
            "bytes": path.stat().st_size,
        }
    serialization_report = build_serialization_self_test(
        load_model_processor()
    )
    runtime_report = validate_runtime()
    return {
        "repository_bindings": {
            path: {"sha256": expected}
            for path, expected in sorted(bindings.items())
        },
        "source_header_sha256": SOURCE_HEADER_SHA256,
        "metadata": metadata_report,
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "files": model_report,
            "runtime": runtime_report,
            "serialization_self_test": serialization_report,
        },
    }


def validate_anchor_contract(
    anchors: Mapping[str, Any],
    *,
    revalidate_files: bool,
) -> None:
    expected_repository_bindings = {
        path: {"sha256": expected}
        for path, expected in sorted(frozen_binding_hashes().items())
    }
    if anchors.get("repository_bindings") != expected_repository_bindings:
        raise RuntimeError("TPCT repository anchor bindings changed")
    if anchors.get("source_header_sha256") != SOURCE_HEADER_SHA256:
        raise RuntimeError("TPCT source header anchor changed")
    metadata = anchors.get("metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError("TPCT metadata anchor is missing")
    if metadata.get("selected_rows") != len(SOURCE_ALLOWLIST):
        raise RuntimeError("TPCT selected metadata anchor count changed")
    if metadata.get("selected_hash") != SELECTED_METADATA_HASH:
        raise RuntimeError("TPCT selected metadata anchor hash changed")
    model = anchors.get("model")
    if not isinstance(model, dict):
        raise RuntimeError("TPCT model anchor is missing")
    if model.get("id") != MODEL_ID or model.get("revision") != MODEL_REVISION:
        raise RuntimeError("TPCT model anchor identity changed")
    files = model.get("files")
    if not isinstance(files, dict) or set(files) != set(MODEL_FILES):
        raise RuntimeError("TPCT model file anchor schema changed")
    for filename, expected in MODEL_FILES.items():
        entry = files.get(filename)
        if (
            not isinstance(entry, dict)
            or entry.get("sha256") != expected
            or not isinstance(entry.get("bytes"), int)
            or entry["bytes"] <= 0
        ):
            raise RuntimeError(f"TPCT model file anchor changed: {filename}")
    runtime = model.get("runtime")
    if runtime != {
        "distributions": dict(RUNTIME_DISTRIBUTIONS),
        "transformers_git_commit": TRANSFORMERS_GIT_COMMIT,
    }:
        raise RuntimeError("TPCT runtime anchor changed")
    serialization = model.get("serialization_self_test")
    if not isinstance(serialization, dict):
        raise RuntimeError("TPCT serialization anchor is missing")
    serialization_body = dict(serialization)
    recorded_serialization_hash = serialization_body.pop(
        "self_test_sha256",
        None,
    )
    if (
        recorded_serialization_hash != SERIALIZATION_SELF_TEST_SHA256
        or canonical_hash(serialization_body)
        != SERIALIZATION_SELF_TEST_SHA256
    ):
        raise RuntimeError("TPCT serialization anchor changed")
    if revalidate_files and dict(anchors) != validate_bindings():
        raise RuntimeError("TPCT persisted anchors do not match frozen files")


def build_contract() -> dict[str, Any]:
    policy = Policy()
    return {
        "policy": asdict(policy),
        "research_boundary": {
            "source_support_seen": True,
            "ofr_market_outcomes_seen": False,
            "tpct_values_seen": False,
            "tpct_incidence_seen": False,
            "tpct_market_outcomes_seen": False,
            "global_pristine_claimed": False,
            "source_family_ledger": list(SOURCE_FAMILY_LEDGER),
            "source_family_concepts": len(SOURCE_FAMILY_LEDGER),
            "2023_p_value_scope": "conditional_descriptive_quality_gate",
            "first_confirmatory_claim": (
                "unchanged combined 2023_2024 one-policy p < 0.002"
            ),
            "new_ofr_candidate_before_confirmation_invalidates_budget": True,
            "2024_confirmation_required": True,
        },
        "source": {
            "path": SOURCE,
            "sha256": SOURCE_SHA256,
            "header_sha256": SOURCE_HEADER_SHA256,
            "columns": list(SOURCE_COLUMNS),
            "mnemonic_allowlist": list(SOURCE_ALLOWLIST),
            "metadata_path": METADATA,
            "metadata_sha256": METADATA_SHA256,
            "selected_metadata_hash": SELECTED_METADATA_HASH,
            "manifest_path": SOURCE_MANIFEST,
            "manifest_sha256": SOURCE_MANIFEST_SHA256,
            "manifest_hash": SOURCE_MANIFEST_HASH,
            "final_asof_tot_le30_dvp_gcf_tri_forbidden": True,
            "load_and_drop_forbidden": True,
        },
        "sealed_parser": {
            "clock_fields_before_value": [
                "mnemonic",
                "observation_date",
                "available_at_utc",
            ],
            "value_decode_condition": (
                "unreserved_entry>=2020-09-10T00:00:00Z and "
                "unreserved_exit<2023-01-01T00:00:00Z"
            ),
            "sealed_late_2022_and_2023_values_decoded": False,
            "candidate_statistics_for_sealed_rows": False,
        },
        "vector": {
            "exact_rows": 16,
            "disclosure_edit": "0",
            "value_parser": "fractions.Fraction over canonical decimal",
            "all_transaction_volumes_strictly_positive": True,
            "fill_or_repair": False,
            "availability": (
                "max(observation_date+8 elapsed days,"
                "2020-09-10T00:00:00Z)"
            ),
        },
        "batch": {
            "same_availability_is_one_batch": True,
            "current_batch_excluded_from_all_current_ranks": True,
            "current_batch_excluded_from_transition_predecessor": True,
            "decision_row": "greatest complete observation date",
            "batch_invalid_date_prevents_decision_and_breaks_continuity": True,
            "append_to_future_history_after_tokens_and_reservation": True,
        },
        "primitives": {
            "ordered": list(PRIMITIVES),
            "exact_rational": True,
            "formulas": {
                "OVERNIGHT": "TV_OO/V_TENOR",
                "NEAR_TERM": "TV_B27/V_TENOR",
                "MEDIUM_TERM": "TV_B830/V_TENOR",
                "LONG_TERM": "TV_G30/V_TENOR",
                "TERM_PREMIUM": "weighted_term_rate-AR_OO",
                "GOVERNMENT_SHARE": "V_GOV/V_COLLATERAL",
                "TREASURY_WITHIN_GOV": "TV_T/V_GOV",
                "CORPORATE_WITHIN_PRIVATE": "TV_CORD/V_PRIVATE",
                "PRIVATE_PREMIUM": "weighted_private_rate-weighted_gov_rate",
                "CONCENTRATION_GAP": "tenor_HHI-collateral_HHI",
            },
            "tenor_and_collateral_simplex_exact": True,
        },
        "rank": {
            "lookback": policy.rank_lookback,
            "minimum": policy.rank_lookback,
            "formula": "(count(prior<x)+0.5*count(prior==x))/252",
            "strictly_earlier_availability_only": True,
            "fallback": None,
        },
        "tokens": {
            "ordered_schema": [
                {"name": name, "values": list(values)}
                for name, values in TOKEN_SCHEMA
            ],
            "count": len(TOKEN_SCHEMA),
            "pair_threshold": "strict +/-1/6; equality BALANCED",
            "rank_breadth": ">=3 HIGH_BROAD; <=-3 LOW_BROAD; else MIXED",
            "extreme_occupancy": "0..3 COMPACT; 4..6 FOCUSED; 7..10 FRACTURED",
            "order_transition": "0..5 STABLE; 6..14 ROTATING; 15..45 RESET",
            "unknown_downstream_value": "ABSTAIN",
            "position_or_executability_token": False,
        },
        "execution": {
            "signal": "ceil_5m(max required availability)",
            "entry": "signal+5m",
            "exit": "entry+120h",
            "held_bars": policy.hold_bars,
            "leverage": policy.leverage,
            "global_action_independent_reservation": True,
            "abstention_releases_reservation": False,
            "reserve_before_split_containment": True,
            "split_interval": "[start,end)",
            "containment": "entry>=start and exit<end",
            "scheduled_exit_only": True,
        },
        "temporal_roles": {
            "train": [
                TRAIN_START.isoformat(),
                SELECTION_START.isoformat(),
            ],
            "selection": [
                SELECTION_START.isoformat(),
                EVAL_START.isoformat(),
            ],
            "untouched_eval": [
                EVAL_START.isoformat(),
                SEALED_START.isoformat(),
            ],
            "adaptation": None,
        },
        "source_support": {
            "train_total_min": 75,
            "train_2020_min": 15,
            "train_2021_min": 50,
            "selection_2022_min": 55,
            "active_months_2020_min": 3,
            "active_months_2021_2022_min": 11,
            "each_half_2021_2022_min": 23,
            "each_quarter_2021_2022_min": 10,
            "max_month_share": 0.20,
            "same_split_max_gap_days": 10,
            "train_start_delay_days_max": 21,
            "selection_start_delay_days_max": 15,
            "split_end_lead_days_max": 15,
            "cross_boundary_blackout_days_max": 20,
            "sealed_value_count": 0,
            "token_support": {
                "pair_each_value_count_min": 3,
                "pair_each_value_share_min": 0.03,
                "pair_max_value_share": 0.85,
                "train_each_leader_nontie_distinct_min": 5,
                "selection_each_leader_nontie_distinct_min": 4,
                "leader_max_nontie_share": 0.50,
                "leader_tie_share_max": 0.20,
                "rank_breadth_each_count_min": 3,
                "rank_breadth_max_share": 0.90,
                "extreme_occupancy_each_count_min": 2,
                "extreme_occupancy_max_share": 0.92,
                "order_transition_each_count_min": 2,
                "order_transition_max_share": 0.92,
                "train_leader_transition_distinct_min": 4,
                "selection_leader_transition_distinct_min": 3,
                "leader_transition_max_share": 0.85,
                "max_exact_signature_share": 0.15,
                "selection_values_must_exist_in_train": True,
            },
        },
        "source_controls": [
            "one_decision_stale",
            "five_decision_stale",
            "year_primitive_permutation",
            "joint_year_state_permutation",
            "pair_orientation_flip",
            "leader_role_flip",
        ],
        "novelty": {
            "comparators": [
                {"id": identifier, "path": path, "sha256": sha256}
                for identifier, path, sha256 in COMPARATORS
            ],
            "full_comparator_cohort_required": True,
            "exact_entry_jaccard_max": 0.20,
            "tolerant_24h_jaccard_max": 0.50,
            "unsigned_occupancy_abs_correlation_max": 0.75,
            "directional_signed_exposure_abs_correlation_max": 0.50,
            "live_exact_entry_jaccard_max": 0.10,
            "live_tolerant_24h_jaccard_max_preoutcome": 0.35,
            "live_tolerant_24h_jaccard_max_eval": 0.30,
            "live_signed_exposure_abs_correlation_max_eval": 0.35,
            "live_unsigned_occupancy_abs_correlation_max": 0.60,
            "missing_required_directional_side_fails": True,
        },
        "accounting": {
            "market": {"path": MARKET_SOURCE, "sha256": MARKET_SOURCE_SHA256},
            "funding": {
                "path": FUNDING_SOURCE,
                "sha256": FUNDING_SOURCE_SHA256,
            },
            "base_cost_notional_per_side": policy.base_cost_notional_per_side,
            "stress_cost_notional_per_side": (
                policy.stress_cost_notional_per_side
            ),
            "strict_mdd": "global pre-entry high-water with held OHLC path",
            "cagr": "full half-open calendar interval",
            "weekly_cluster_draws": 100_000,
            "familywise_shared_max_stat": True,
            "familywise_statistic": "weekly_return_t_policy",
            "development_primary_selector": "highest_observed_t_policy",
            "family_includes_failed_secondary_gate_policies": True,
            "secondary_selection_tie_break_only_after_exact_t_tie": True,
        },
        "utility": {
            "abstain": 0.0,
            "trade": (
                "log(max(account_multiplier,1e-12))"
                "-(1/3)*local_strict_drawdown-0.0005"
            ),
            "tie_priority": list(ACTION_NAMES),
            "preference_margin": policy.preference_margin,
            "outcome_dependent_sampling": False,
        },
        "cheap_gate_2022": {
            "primary_algorithms": [
                "categorical_naive_bayes",
                "ridge_contextual_utility",
                "extra_trees_contextual_utility",
            ],
            "ratio_min": 1.0,
            "strict_mdd_pct_max": 15.0,
            "trades_min": 30,
            "each_half_trades_min": 10,
            "each_side_trades_min": 8,
            "familywise_p_strictly_below": 0.20,
            "primary_selector": "highest_observed_t_policy",
            "tie_break": [
                "higher_ratio",
                "higher_return",
                "lower_mdd",
                "lexicographically_smaller_policy_id",
            ],
            "stress_and_one_hour_delay_positive": True,
            "beat_all_frozen_controls": True,
        },
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "loader": "transformers.AutoModelForMultimodalLM",
            "processor": "transformers.AutoProcessor",
            "text_only": True,
            "thinking": False,
            "file_hashes": dict(MODEL_FILES),
            "runtime": {
                "distributions": dict(RUNTIME_DISTRIBUTIONS),
                "transformers_git_commit": TRANSFORMERS_GIT_COMMIT,
            },
            "serialization": {
                "self_test_sha256": SERIALIZATION_SELF_TEST_SHA256,
                "specimen": dict(SERIALIZATION_SPECIMEN),
                "token_order": list(TOKEN_COLUMNS),
                "option_orders": [
                    list(order) for order in action_option_orders()
                ],
                "completions": [
                    completion_text(code) for code in NEUTRAL_CODES
                ],
                "message_roles": ["user"],
                "system_or_developer_message": False,
                "multimodal_content": False,
                "scoring_add_generation_prompt": True,
                "training_add_generation_prompt": False,
                "enable_thinking": False,
                "preserve_thinking": False,
                "tools": None,
                "caller_add_special_tokens": False,
                "completion_special_tokens_scored": False,
                "full_example_count": 18,
                "must_pass_before_economic_evaluator_or_training": True,
            },
            "quantization": {
                "load_in_4bit": True,
                "type": "nf4",
                "double_quant": True,
                "compute_dtype": "bfloat16",
            },
            "lora": {
                "r": 8,
                "alpha": 16,
                "dropout": 0.05,
                "bias": "none",
                "task_type": "CAUSAL_LM",
                "target_regex": (
                    r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj)$"
                ),
                "trainable_parameters": 2_678_784,
            },
            "resource_gates": {
                "visible_cuda_devices": 1,
                "bf16_required": True,
                "inference_peak_allocated_gib_max": 7.0,
                "inference_peak_reserved_gib_max": 7.25,
                "training_peak_allocated_gib_max": 20.0,
                "training_peak_reserved_gib_max": 24.0,
                "adapter_checkpoint_mib_max": 256,
                "retained_adapters_gib_max": 1.0,
            },
            "option_orders": [list(order) for order in action_option_orders()],
            "score": (
                "mean length-normalized adapter-minus-base completion logprob "
                "over six option orders, subtract train action offsets"
            ),
            "sft": {
                "optimizer_steps": 64,
                "gradient_accumulation": 8,
                "learning_rate": 1.0e-4,
                "warmup_steps": 8,
                "seed": policy.random_seed,
            },
            "dpo": {
                "optimizer_steps": 96,
                "gradient_accumulation": 8,
                "learning_rate": 5.0e-6,
                "warmup_steps": 8,
                "beta": 0.1,
                "checkpoints": [24, 48, 72, 96],
                "seed": policy.random_seed,
            },
            "selection_2022": {
                "ratio_min": 2.0,
                "strict_mdd_pct_max": 15.0,
                "trades_min": 30,
                "each_half_trades_min": 10,
                "each_side_trades_min": 8,
                "familywise_p_strictly_below": 0.10,
                "primary_selector": "highest_observed_t_policy",
                "tie_break": [
                    "higher_ratio",
                    "higher_return",
                    "lower_mdd",
                    "earlier_optimizer_step",
                ],
                "ratio_margin_over_cheap_min": 0.25,
            },
        },
        "eval_2023": {
            "atomic_source_novelty_outcome_stage": True,
            "novelty_uses_complete_frozen_comparator_cohort": True,
            "source_opportunities_min": 55,
            "ratio_min": 3.0,
            "strict_mdd_pct_max": 15.0,
            "trades_min": 30,
            "each_half_trades_min": 10,
            "each_side_trades_min": 8,
            "weekly_clusters_min": 20,
            "conditional_descriptive_one_policy_p_strictly_below": 0.05,
            "confirmatory_type_i_claim": False,
            "mean_signed_gross_move_bp_min": 40.0,
            "ratio_margin_over_cheap_min": 0.50,
            "stress_and_one_hour_delay_positive": True,
            "one_day_delay": "report_only",
        },
        "confirmation_2024": {
            "required_before_production": True,
            "unchanged_eval_gates": True,
            "source_family_concepts": len(SOURCE_FAMILY_LEDGER),
            "source_family_alpha": 0.01,
            "bonferroni_combined_2023_2024_p_strictly_below": 0.002,
            "first_confirmatory_statistical_claim": True,
            "additional_candidate_invalidates_budget": True,
            "retraining_or_repair": False,
        },
        "failure_action": "retire TPCT-120 unchanged at first failed stage",
    }


def build_manifest(*, validate_files: bool = True) -> dict[str, Any]:
    anchors = validate_bindings() if validate_files else {
        "validation_skipped_for_unit_test": True
    }
    contract = build_contract()
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "as_of_date": "2026-07-24",
        "candidate": POLICY_ID,
        "contract": contract,
        "contract_hash": canonical_hash(contract),
        "anchors": anchors,
        "decision": {
            "source_support_authorized": True,
            "comparator_novelty_authorized_after_support": True,
            "market_outcomes_authorized": False,
            "model_training_authorized": False,
            "sealed_eval_authorized": False,
            "next_action": (
                "commit and run source-only TPCT support builder without "
                "decoding sealed-boundary values"
            ),
        },
        "outcome_boundary": {
            "source_artifact_bytes_hashed": validate_files,
            "source_manifest_aggregate_metadata_read": validate_files,
            "source_header_read": validate_files,
            "selected_metadata_objects_read": (
                len(SOURCE_ALLOWLIST) if validate_files else 0
            ),
            "source_values_decoded": 0,
            "primitives_or_ranks_derived": 0,
            "token_rows_derived": 0,
            "opportunity_rows_derived": 0,
            "sealed_boundary_values_decoded": 0,
            "comparator_rows_parsed": 0,
            "market_rows_read": 0,
            "funding_rows_read": 0,
            "future_return_rows_read": 0,
            "return_or_pnl_fields_read": 0,
            "model_labels_created": 0,
            "model_training_runs": 0,
            "network_calls": 0,
        },
    }
    payload["manifest_hash"] = canonical_hash(payload)
    return payload


def validate_manifest(
    payload: Mapping[str, Any],
    *,
    allow_unvalidated_anchors: bool = False,
    revalidate_files: bool = True,
) -> None:
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("TPCT protocol version changed")
    if payload.get("candidate") != POLICY_ID:
        raise RuntimeError("TPCT candidate identity changed")
    manifest = dict(payload)
    recorded_manifest = manifest.pop("manifest_hash", None)
    if recorded_manifest != canonical_hash(manifest):
        raise RuntimeError("TPCT manifest hash mismatch")
    contract = payload.get("contract")
    if not isinstance(contract, dict):
        raise RuntimeError("TPCT contract is missing")
    if payload.get("contract_hash") != canonical_hash(contract):
        raise RuntimeError("TPCT contract hash mismatch")
    if contract != build_contract():
        raise RuntimeError("TPCT frozen contract drifted")
    anchors = payload.get("anchors")
    unvalidated_anchors = {"validation_skipped_for_unit_test": True}
    if anchors == unvalidated_anchors:
        if not allow_unvalidated_anchors:
            raise RuntimeError("TPCT persisted manifest has unvalidated anchors")
    elif not isinstance(anchors, dict):
        raise RuntimeError("TPCT manifest anchors are missing")
    else:
        validate_anchor_contract(
            anchors,
            revalidate_files=revalidate_files,
        )
    decision = payload.get("decision")
    if not isinstance(decision, dict):
        raise RuntimeError("TPCT decision is missing")
    if not decision.get("source_support_authorized"):
        raise RuntimeError("TPCT source support is not authorized")
    forbidden_authorizations = (
        "market_outcomes_authorized",
        "model_training_authorized",
        "sealed_eval_authorized",
    )
    if any(decision.get(key) for key in forbidden_authorizations):
        raise RuntimeError("TPCT preregistration opened a later stage")
    boundary = payload.get("outcome_boundary")
    if not isinstance(boundary, dict):
        raise RuntimeError("TPCT outcome boundary is missing")
    allowed_nonzero = {
        "source_artifact_bytes_hashed",
        "source_manifest_aggregate_metadata_read",
        "source_header_read",
        "selected_metadata_objects_read",
    }
    for key, value in boundary.items():
        if key in allowed_nonzero:
            continue
        if value != 0:
            raise RuntimeError(f"TPCT outcome boundary opened: {key}")
    expected_boundary_reads = {
        "source_artifact_bytes_hashed": anchors != unvalidated_anchors,
        "source_manifest_aggregate_metadata_read": (
            anchors != unvalidated_anchors
        ),
        "source_header_read": anchors != unvalidated_anchors,
        "selected_metadata_objects_read": (
            len(SOURCE_ALLOWLIST) if anchors != unvalidated_anchors else 0
        ),
    }
    if any(
        boundary.get(key) != expected
        for key, expected in expected_boundary_reads.items()
    ):
        raise RuntimeError("TPCT preregistration read boundary changed")


def write_manifest(
    payload: Mapping[str, Any],
    output: str | Path = DEFAULT_OUTPUT,
) -> Path:
    validate_manifest(payload, revalidate_files=True)
    path = repository_path(output)
    encoded = canonical_json_bytes(payload)
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(
                "existing TPCT preregistration differs; refusing overwrite"
            )
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        if path.exists():
            if path.read_bytes() != encoded:
                raise RuntimeError("concurrent TPCT preregistration differs")
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_manifest(validate_files=True)
    path = write_manifest(payload, args.output)
    print(
        json.dumps(
            {
                "candidate": POLICY_ID,
                "output": str(path.relative_to(repository_root())),
                "contract_hash": payload["contract_hash"],
                "manifest_hash": payload["manifest_hash"],
                "outcome_boundary": payload["outcome_boundary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
