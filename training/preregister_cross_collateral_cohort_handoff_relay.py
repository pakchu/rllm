"""Freeze the outcome-blind CCHR-288 singleton preregistration.

This stage reads the frozen source manifest and hashes source/provenance bytes.
It never parses source rows, comparator rows, market data, funding, returns, or
PnL, and it never derives real CCHR feature or event incidence.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence


POLICY_ID = "CCHR-288"
PROTOCOL_VERSION = "cross_collateral_cohort_handoff_relay_preregistration_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

SOURCE_PATH = Path(
    "data/binance_cross_collateral_metrics_btc_2021_2023/"
    "BTC_cross_collateral_metrics_5m_2021-07-08_2023-12-31.csv.gz"
)
SOURCE_SHA256 = "ab9f18ba7745f21b17ac1124c45bb755245d404d66100c595bb77631f4bc1757"
SOURCE_MANIFEST = Path(
    "results/binance_cross_collateral_metrics_btc_2021_2023_manifest.json"
)
SOURCE_MANIFEST_SHA256 = (
    "c0732ca47451209a9bb519545b0e349550994d870d476ee66ecbae81588fb159"
)
SOURCE_BUILDER = Path("training/build_binance_cross_collateral_metrics.py")
SOURCE_BUILDER_SHA256 = (
    "a86b15d404064d137f36b341a6a0df1a12ec865e9344cd16737ca4a1b31a3db6"
)
SOURCE_AUDIT = Path(
    "docs/binance-cross-collateral-positioning-metrics-source-audit-2026-07-17.md"
)
SOURCE_AUDIT_SHA256 = "2e72881dac5aae71b8a8a078ea0748fcce015e3c52c8bfb985dbcbe04a8e13a2"
MECHANISM_DECISION = Path(
    "docs/cross-collateral-cohort-handoff-relay-mechanism-decision-2026-07-21.md"
)
MECHANISM_DECISION_SHA256 = (
    "1a86a7179b1f38471e5074bce5832acbb64ee3b078385883647bc11b7e3eaea7"
)
PREREGISTRATION_SOURCE = Path(
    "training/preregister_cross_collateral_cohort_handoff_relay.py"
)
DEFAULT_OUTPUT = Path(
    "results/cross_collateral_cohort_handoff_relay_preregistration_2026-07-21.json"
)
COMPARATOR_FREEZE = Path(
    "results/cross_collateral_cohort_handoff_relay_comparator_freeze_2026-07-21.json"
)

SOURCE_COLUMNS = (
    "date",
    "um_count_long_short_ratio",
    "um_sum_taker_long_short_vol_ratio",
    "cm_sum_taker_long_short_vol_ratio",
)
FORBIDDEN_SOURCE_COLUMNS = (
    "source_complete",
    "um_sum_open_interest",
    "um_sum_open_interest_value",
    "cm_sum_open_interest",
    "cm_sum_open_interest_value",
    "um_count_toptrader_long_short_ratio",
    "um_sum_toptrader_long_short_ratio",
    "cm_count_long_short_ratio",
    "cm_count_toptrader_long_short_ratio",
    "cm_sum_toptrader_long_short_ratio",
    "open",
    "high",
    "low",
    "close",
    "funding_rate",
    "premium",
    "return",
    "pnl",
)

COMPARATOR_PROVENANCE_BINDINGS: dict[str, dict[str, str]] = {
    "ccpr_preregistration": {
        "path": "training/preregister_cross_collateral_positioning_recoil.py",
        "sha256": "6d1224c3a0d24686bf3b997f424b10f65d004c211269916f6007de8b8464a0a5",
        "role": "semantic_contract",
    },
    "ccpr_source_clock": {
        "path": "results/cross_collateral_positioning_recoil_clocks_2026-07-17.csv",
        "sha256": "2a864ec2b616a3118bf9ffa44f99f96fbe19e79d82870f21a0d7d9010d5c993a",
        "role": "raw_onset_clock",
    },
    "pdlh_source_implementation": {
        "path": "training/search_positioning_lifecycle_hazard_alpha.py",
        "sha256": "86d3f8dee6f3ce72ba2bc7f75daae559cd1e19d9260bc0ad535a53f6652e73f3",
        "role": "semantic_provenance_only",
    },
    "pdlh_result_commitment": {
        "path": "results/positioning_lifecycle_hazard_alpha_scan_2026-07-13.json",
        "sha256": "f72029be60dc63e2de78d30565acb6ca4d4879478e79167fc4f70168efcec0af",
        "role": "hash_only_outcome_bearing_provenance",
    },
    "dtv_source_implementation": {
        "path": "training/search_debt_transfer_velocity_alpha.py",
        "sha256": "babe8479d55853e2b4ab9263b44d835ec058455c17615c39275c3e85f87d1880",
        "role": "semantic_provenance_only",
    },
    "dtv_result_commitment": {
        "path": "results/debt_transfer_velocity_alpha_scan_2026-07-13.json",
        "sha256": "81a89f1c77d7d238e03faa842410378a19ebf3928443fc35e2b358c917589d55",
        "role": "hash_only_outcome_bearing_provenance",
    },
    "far_source_implementation": {
        "path": "training/search_funding_age_rollover_transfer_alpha.py",
        "sha256": "5e28f645f6368bf13879443e1f866feacc3e8296bd2591ef4e8e31d3d6d5062d",
        "role": "semantic_provenance_only",
    },
    "far_result_commitment": {
        "path": "results/funding_age_rollover_transfer_alpha_scan_2026-07-13.json",
        "sha256": "fef5a6c761e5393a6795ff5e91c8777ebc62faed204848a597b1f780b5ea0c79",
        "role": "hash_only_outcome_bearing_provenance",
    },
    "dlpd_source_clock": {
        "path": (
            "data/btcdom_leverage_polarity_decomposition_evaluation_clocks_"
            "2022_2023.csv.gz"
        ),
        "sha256": "38ccc18df700d24462d0cae91e34733856ed053dc400c584a3eedaf3f9ed60f1",
        "role": "accepted_clock_projection_source",
    },
    "live_portfolio_config": {
        "path": "configs/live/portfolio_gross385_trainmdd40_2026-07-12.json",
        "sha256": "86f255ca3967245b8b0676b00025b955d7f33668ab1ef9d813623191b4ecd1e7",
        "role": "live_component_registry",
    },
    "live_oi_upbit_config": {
        "path": "configs/live/oi_upbit_ratio288_low_candidate.json",
        "sha256": "659239373e1f51fc2df9615f5387686fd9252a56e1c366b45421bf39d3d6223f",
        "role": "live_component_contract",
    },
    "live_funding_premium_config": {
        "path": "configs/live/new_long_minimal_funding_premium_candidate.json",
        "sha256": "f0848c5fea1fcc7823ed15b6e4b865a8dc2731c2d2bfd2ba21b0f92c534f0f03",
        "role": "live_component_contract",
    },
    "live_rex_config": {
        "path": "configs/live/rex_veto_7_candidate.json",
        "sha256": "36df47c4737eb99f4ca5e2b257d9bd2fbf130df9d731b9ac02fcfe5192acd4db",
        "role": "live_component_contract",
    },
}

PURE_CLOCK_REQUIREMENTS: dict[str, dict[str, Any]] = {
    "pdlh": {
        "path": "results/cchr_pdlh_pure_clocks_2026-07-21.csv.gz",
        "export_manifest": "results/cchr_pdlh_pure_clock_manifest_2026-07-21.json",
        "required_member_count": 16,
    },
    "dtv": {
        "path": "results/cchr_dtv_pure_clocks_2026-07-21.csv.gz",
        "export_manifest": "results/cchr_dtv_pure_clock_manifest_2026-07-21.json",
        "required_member_count": 24,
    },
    "far": {
        "path": "results/cchr_far_pure_clocks_2026-07-21.csv.gz",
        "export_manifest": "results/cchr_far_pure_clock_manifest_2026-07-21.json",
        "required_member_count": 12,
    },
    "live": {
        "path": "results/cchr_live_portfolio_pure_clocks_2026-07-21.csv.gz",
        "export_manifest": (
            "results/cchr_live_portfolio_pure_clock_manifest_2026-07-21.json"
        ),
        "required_member_count": 3,
    },
}

CLOCK_SCHEMA = (
    "candidate_id",
    "split",
    "decision_time",
    "entry_time",
    "exit_time",
    "side",
)

OUTCOME_BOUNDARY = {
    "source_manifest_json_read": True,
    "source_artifact_bytes_hashed": True,
    "source_csv_values_read": 0,
    "comparator_artifact_bytes_hashed": True,
    "comparator_rows_read": 0,
    "outcome_bearing_provenance_json_parsed": 0,
    "cchr_feature_rows_derived": 0,
    "signal_incidence_rows_derived": 0,
    "market_rows_loaded": 0,
    "funding_rows_loaded": 0,
    "return_or_pnl_fields": 0,
    "post_2023_rows_loaded": 0,
    "network_calls": 0,
    "subprocess_calls": 0,
}

ARTIFACT_TOP_LEVEL_KEYS = frozenset(
    {
        "protocol_version",
        "policy_id",
        "config",
        "source_binding",
        "mechanism_decision",
        "comparator_provenance_bindings",
        "comparator_candidate_map",
        "comparator_candidate_map_hash",
        "pure_clock_requirements",
        "comparator_freeze_requirement",
        "policy",
        "policy_hash",
        "outcomes_opened",
        "outcome_boundary",
        "authorization",
        "preregistration_source",
        "manifest_hash",
    }
)


@dataclass(frozen=True)
class Config:
    source_manifest: str = str(SOURCE_MANIFEST)
    preregistration_output: str = str(DEFAULT_OUTPUT)


@dataclass(frozen=True)
class Split:
    name: str
    start: datetime
    end: datetime


@dataclass(frozen=True)
class AnchorObservation:
    time: datetime
    combined_valid: bool
    rank_ready: bool
    crowd_rank: float | None = None
    handoff_rank: float | None = None
    handoff_value: float | None = None


@dataclass(frozen=True)
class SourceRow:
    time: datetime
    um_global_ratio: float | None
    um_taker_ratio: float | None
    cm_taker_ratio: float | None


@dataclass(frozen=True)
class Candidate:
    episode_start: datetime
    handoff_anchor: datetime
    decision_time: datetime
    entry_time: datetime
    exit_time: datetime
    side: int


def _repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPOSITORY_ROOT / candidate
    return candidate.resolve()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _repository_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def _unique_object(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError(f"CCHR JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(
        _repository_path(path).read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
    )
    if not isinstance(payload, dict):
        raise RuntimeError("CCHR JSON must be an object")
    return payload


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("CCHR timestamps must be timezone-aware UTC")
    normalized = value.astimezone(timezone.utc)
    if normalized.second or normalized.microsecond:
        raise ValueError("CCHR timestamps must be minute-aligned")
    return normalized


def exact_sign(value: float) -> int:
    if not math.isfinite(value):
        raise ValueError("CCHR sign input must be finite")
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def empirical_midrank(current: float, prior: Sequence[float]) -> float:
    if len(prior) != 168:
        raise ValueError("CCHR midrank requires exactly 168 strict-prior values")
    values = [float(value) for value in prior]
    if not math.isfinite(current) or not all(math.isfinite(value) for value in values):
        raise ValueError("CCHR midrank values must be finite")
    below = sum(value < current for value in values)
    equal = sum(value == current for value in values)
    return (below + 0.5 * equal) / 168.0


def combined_anchor_valid(rows: Sequence[SourceRow], anchor: datetime) -> bool:
    anchor = _utc(anchor)
    if anchor.minute != 55 or len(rows) != 12:
        return False
    expected = tuple(
        anchor - timedelta(minutes=5 * offset) for offset in range(11, -1, -1)
    )
    try:
        observed = tuple(_utc(row.time) for row in rows)
    except ValueError:
        return False
    if observed != expected:
        return False

    current_global = rows[-1].um_global_ratio
    if (
        current_global is None
        or not math.isfinite(current_global)
        or current_global <= 0
    ):
        return False
    for row in rows:
        values = (row.um_taker_ratio, row.cm_taker_ratio)
        if any(
            value is None or not math.isfinite(value) or value <= 0 for value in values
        ):
            return False
    return True


def random_side(entry_time: datetime) -> int:
    stamp = _utc(entry_time).strftime("%Y-%m-%dT%H:%M:%SZ")
    first = hashlib.sha256(f"CCHR-288|20260721|{stamp}".encode("ascii")).digest()[0]
    return 1 if first % 2 == 0 else -1


def research_splits() -> tuple[Split, Split]:
    return (
        Split(
            "train",
            datetime(2021, 8, 8, tzinfo=timezone.utc),
            datetime(2023, 1, 1, tzinfo=timezone.utc),
        ),
        Split(
            "selection",
            datetime(2023, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 1, 1, tzinfo=timezone.utc),
        ),
    )


def validate_primary_candidate(candidate: Candidate) -> None:
    episode_start = _utc(candidate.episode_start)
    handoff = _utc(candidate.handoff_anchor)
    decision = _utc(candidate.decision_time)
    entry = _utc(candidate.entry_time)
    exit_time = _utc(candidate.exit_time)
    if any(
        value.minute % 5 != 0
        for value in (episode_start, handoff, decision, entry, exit_time)
    ):
        raise ValueError("CCHR candidate times must be five-minute aligned")
    if episode_start.minute != 55 or handoff.minute != 55:
        raise ValueError("CCHR episode and handoff anchors must have minute 55")
    if not episode_start < handoff:
        raise ValueError("CCHR handoff must follow episode start")
    age = handoff - episode_start
    if (
        age < timedelta(hours=12)
        or age > timedelta(hours=72)
        or (age.total_seconds() % 3600 != 0)
    ):
        raise ValueError("CCHR handoff age must be an integer 12 through 72 hours")
    if decision != handoff + timedelta(minutes=5):
        raise ValueError("CCHR decision must equal handoff plus five minutes")
    if entry != handoff + timedelta(minutes=10):
        raise ValueError("CCHR entry must equal handoff plus ten minutes")
    if exit_time != entry + timedelta(hours=24):
        raise ValueError("CCHR exit must equal entry plus 24 hours")
    if candidate.side not in (-1, 1):
        raise ValueError("CCHR candidate side must be -1 or +1")


def candidate_split(
    candidate: Candidate,
    splits: Sequence[Split] | None = None,
) -> str | None:
    validate_primary_candidate(candidate)
    declared = research_splits() if splits is None else tuple(splits)
    times = (
        candidate.episode_start,
        candidate.handoff_anchor,
        candidate.decision_time,
        candidate.entry_time,
    )
    for split in declared:
        start = _utc(split.start)
        end = _utc(split.end)
        if not start < end:
            raise ValueError("CCHR split must be non-empty")
        if all(start <= _utc(value) < end for value in times) and (
            start < _utc(candidate.exit_time) < end
        ):
            return split.name
    return None


def schedule_nonoverlap(candidates: Sequence[Candidate]) -> list[Candidate]:
    accepted: list[Candidate] = []
    prior_exit: datetime | None = None
    for candidate in sorted(candidates, key=lambda item: item.entry_time):
        validate_primary_candidate(candidate)
        entry = _utc(candidate.entry_time)
        exit_time = _utc(candidate.exit_time)
        if prior_exit is None or entry >= prior_exit:
            accepted.append(candidate)
            prior_exit = exit_time
    return accepted


class CCHRStateMachine:
    """Synthetic primary state machine over already-ranked hourly anchors."""

    def __init__(self) -> None:
        self.armed = False
        self.armed_at: datetime | None = None
        self.active = False
        self.episode_side = 0
        self.episode_start: datetime | None = None
        self.peak_extremity = 0.0
        self.previous_handoff_sign = 0
        self.previous_anchor: datetime | None = None

    def _reset(self, *, anchor: datetime) -> None:
        self.armed = False
        self.armed_at = None
        self.active = False
        self.episode_side = 0
        self.episode_start = None
        self.peak_extremity = 0.0
        self.previous_handoff_sign = 0
        self.previous_anchor = anchor

    @staticmethod
    def _validated(observation: AnchorObservation) -> tuple[float, float, float]:
        if not observation.rank_ready:
            raise ValueError("CCHR ranked observation must be rank-ready")
        values = (
            observation.crowd_rank,
            observation.handoff_rank,
            observation.handoff_value,
        )
        if any(value is None or not math.isfinite(value) for value in values):
            raise ValueError("CCHR ranked observation has a non-finite value")
        crowd_rank, handoff_rank, handoff_value = values
        assert crowd_rank is not None
        assert handoff_rank is not None
        assert handoff_value is not None
        if not 0.0 <= crowd_rank <= 1.0 or not 0.0 <= handoff_rank <= 1.0:
            raise ValueError("CCHR ranks must lie in [0,1]")
        return crowd_rank, handoff_rank, handoff_value

    def process(self, observation: AnchorObservation) -> Candidate | None:
        anchor = _utc(observation.time)
        if anchor.minute != 55:
            raise ValueError("CCHR hourly anchor minute must be exactly 55")
        contiguous = self.previous_anchor is None or (
            anchor - self.previous_anchor == timedelta(hours=1)
        )
        if (
            not observation.combined_valid
            or not observation.rank_ready
            or not contiguous
        ):
            self._reset(anchor=anchor)
            return None

        crowd_rank, handoff_rank, handoff_value = self._validated(observation)
        crowd_tail = 1 if crowd_rank >= 0.90 else -1 if crowd_rank <= 0.10 else 0
        crowd_side = 1 if crowd_rank > 0.50 else -1 if crowd_rank < 0.50 else 0
        extremity = abs(2.0 * crowd_rank - 1.0)
        handoff_sign = exact_sign(handoff_value)
        terminated = False

        if self.active:
            assert self.episode_start is not None
            if crowd_side == self.episode_side:
                self.peak_extremity = max(self.peak_extremity, extremity)
            else:
                self.active = False
                self.episode_side = 0
                self.episode_start = None
                self.peak_extremity = 0.0
                self.armed = False
                self.armed_at = None
                terminated = True

            if self.active:
                assert self.episode_start is not None
                elapsed = anchor - self.episode_start
                age_hours = elapsed.total_seconds() / 3600.0
                if age_hours < 0.0 or not age_hours.is_integer():
                    raise RuntimeError("CCHR episode age must be an integer hour")
                if 12.0 <= age_hours <= 72.0:
                    is_handoff = (
                        extremity <= self.peak_extremity - 0.20
                        and handoff_sign == -self.episode_side
                        and handoff_rank >= 0.75
                        and self.previous_handoff_sign != -self.episode_side
                    )
                    if is_handoff:
                        side = -self.episode_side
                        entry = anchor + timedelta(minutes=10)
                        candidate = Candidate(
                            episode_start=self.episode_start,
                            handoff_anchor=anchor,
                            decision_time=anchor + timedelta(minutes=5),
                            entry_time=entry,
                            exit_time=entry + timedelta(hours=24),
                            side=side,
                        )
                        self.active = False
                        self.episode_side = 0
                        self.episode_start = None
                        self.peak_extremity = 0.0
                        self.armed = False
                        self.armed_at = None
                        terminated = True
                        self.previous_handoff_sign = handoff_sign
                        self.previous_anchor = anchor
                        return candidate
                if age_hours >= 72.0:
                    self.active = False
                    self.episode_side = 0
                    self.episode_start = None
                    self.peak_extremity = 0.0
                    self.armed = False
                    self.armed_at = None
                    terminated = True

        if not self.active and not terminated:
            if not self.armed:
                if 0.25 < crowd_rank < 0.75:
                    self.armed = True
                    self.armed_at = anchor
            elif self.armed_at is not None and anchor > self.armed_at:
                setup = (
                    crowd_tail != 0
                    and handoff_sign == crowd_tail
                    and handoff_rank >= 0.60
                )
                if setup:
                    self.active = True
                    self.episode_side = crowd_tail
                    self.episode_start = anchor
                    self.peak_extremity = extremity
                    self.armed = False
                    self.armed_at = None

        self.previous_handoff_sign = handoff_sign
        self.previous_anchor = anchor
        return None


def comparator_candidate_map() -> dict[str, dict[str, Any]]:
    members: dict[str, dict[str, Any]] = {}
    for q in (0.80, 0.85, 0.90):
        for hold in (48, 96):
            candidate_id = f"ccpr:q={q:.2f}:hold={hold}"
            members[candidate_id] = {
                "family": "ccpr",
                "parameters": {"control": "primary", "q": q},
                "hold_bars": hold,
                "component_weight": None,
            }
    for disagreement in (
        "top_position_minus_global",
        "top_account_minus_global",
    ):
        for min_age in (144, 432):
            for trigger in ("contraction", "zero_cross"):
                for hold in (72, 216):
                    candidate_id = (
                        f"pdlh:{disagreement}:age={min_age}:"
                        f"trigger={trigger}:hold={hold}"
                    )
                    members[candidate_id] = {
                        "family": "pdlh",
                        "parameters": {
                            "disagreement": disagreement,
                            "min_age": min_age,
                            "trigger": trigger,
                        },
                        "hold_bars": hold,
                        "component_weight": None,
                    }
    for memory in (72, 288):
        for acceptance in (72, 288):
            for q in (0.90, 0.95):
                for hold in (72, 144, 288):
                    candidate_id = (
                        f"dtv:memory={memory}:accept={acceptance}:q={q:.2f}:hold={hold}"
                    )
                    members[candidate_id] = {
                        "family": "dtv",
                        "parameters": {
                            "memory": memory,
                            "acceptance_horizon": acceptance,
                            "q": q,
                        },
                        "hold_bars": hold,
                        "component_weight": None,
                    }
    for age in (1, 3, 6):
        for half_life in (288, 864):
            for hold in (72, 144):
                candidate_id = f"far:age={age}:half_life={half_life}:q=0.90:hold={hold}"
                members[candidate_id] = {
                    "family": "far",
                    "parameters": {
                        "min_age_settlements": age,
                        "half_life_bars": half_life,
                        "q": 0.90,
                    },
                    "hold_bars": hold,
                    "component_weight": None,
                }
    members["dlpd:DLPD-12:primary"] = {
        "family": "dlpd",
        "parameters": {"candidate": "DLPD-12", "control": "primary"},
        "hold_bars": 144,
        "component_weight": None,
    }
    for component, weight, hold in (
        ("oi_upbit_ratio288_low", 0.65, 30),
        ("new_long_minimal_funding_premium", 1.75, 576),
        ("cand_rex_veto_7", 1.45, 144),
    ):
        members[f"live:{component}"] = {
            "family": "live",
            "parameters": {"component": component},
            "hold_bars": hold,
            "component_weight": weight,
        }
    return dict(sorted(members.items()))


def expected_source_binding() -> dict[str, Any]:
    return {
        "path": str(SOURCE_PATH),
        "sha256": SOURCE_SHA256,
        "manifest": str(SOURCE_MANIFEST),
        "manifest_sha256": SOURCE_MANIFEST_SHA256,
        "builder": str(SOURCE_BUILDER),
        "builder_sha256": SOURCE_BUILDER_SHA256,
        "audit": str(SOURCE_AUDIT),
        "audit_sha256": SOURCE_AUDIT_SHA256,
        "rows": 261_216,
        "start_inclusive": "2021-07-08T00:00:00Z",
        "end_exclusive": "2024-01-01T00:00:00Z",
        "columns": list(SOURCE_COLUMNS),
    }


def comparator_freeze_requirement() -> dict[str, Any]:
    return {
        "path": str(COMPARATOR_FREEZE),
        "required_before_real_incidence": True,
        "must_bind": [
            "exporter_sha256",
            "raw_input_path_sha256_and_column_allowlist",
            "export_manifest_sha256",
            "pure_clock_sha256",
            "coverage",
            "member_count",
            "candidate_map_sha256",
        ],
    }


CONTROL_DEFINITIONS: dict[str, dict[str, Any]] = {
    "crowd_resolution_only": {
        "setup": "first armed C!=0; no H setup predicate",
        "handoff": "first age12..72 E<=peak-0.20 while S==episode_side",
        "side": "-episode_side",
    },
    "handoff_only": {
        "setup": "none",
        "handoff": "nonzero H sign flip from immediately prior combined-valid hour and R_H>=0.75",
        "side": "sign(H_current)",
    },
    "no_age": {"difference": "primary minimum_age_hours=0"},
    "um_taker_only": {"difference": "replace H/R_H with U/R_U everywhere"},
    "cm_stale_1h": {"difference": "H_stale[t]=U[t]-M[t-1h] with own 168-prior rank"},
    "one_hour_execution_delay": {
        "difference": "primary decision/entry/exit +12 five-minute bars",
        "containment": "original origins and all shifted execution times in one split",
    },
    "direction_flip": {"difference": "exact primary accepted clock; side *= -1"},
    "deterministic_random_side": {
        "difference": (
            "exact primary accepted clock; digest first-byte parity of ASCII "
            "CCHR-288|20260721|%Y-%m-%dT%H:%M:%SZ"
        )
    },
}


def policy() -> dict[str, Any]:
    members = comparator_candidate_map()
    return {
        "singleton": True,
        "source": {
            "columns": list(SOURCE_COLUMNS),
            "forbidden_columns": list(FORBIDDEN_SOURCE_COLUMNS),
            "existing_source_complete_excluded": True,
            "local_complete": (
                "exact unique 5m rows t-55m..t; current UM global ratio and all "
                "12 UM/CM taker ratios positive finite"
            ),
            "gap_action": (
                "reset both 168-anchor histories, cancel active episode, disarm; "
                "accepted trade reservation persists"
            ),
            "post_gap_quarantine": "full 168 combined-valid hourly history rebuild",
            "fill_policy": "no interpolation, forward fill, skip, or partial salvage",
        },
        "features": {
            "anchor_minute_utc": 55,
            "taker_median_bars": 12,
            "G": "log(um_count_long_short_ratio[t])",
            "U": "median(log(um_sum_taker_long_short_vol_ratio[t-11:t]))",
            "M": "median(log(cm_sum_taker_long_short_vol_ratio[t-11:t]))",
            "H": "U-M",
            "rank_lookback_hourly_anchors": 168,
            "rank_current_excluded": True,
            "rank_requires_contiguous_combined_valid": True,
            "midrank": "(count(prior<current)+0.5*count(prior==current))/168",
            "tie_rule": "exact parsed IEEE-754 binary64 equality",
            "parameter_grid": [],
        },
        "state_machine": {
            "processing_order": [
                "validate_and_rank",
                "invalid_reset_only",
                "active_peak_then_cancel_then_handoff_then_expire",
                "mark_termination",
                "inactive_nonterminated_neutral_rearm",
                "setup_only_if_armed_on_strictly_prior_anchor",
            ],
            "initial_armed": False,
            "neutral_rearm": "0.25<R_G<0.75",
            "same_anchor_rearm": False,
            "same_anchor_arm_and_setup": False,
            "crowd_tail": "C=+1 if R_G>=0.90; -1 if R_G<=0.10; else 0",
            "crowd_orientation": "S=sign(R_G-0.50), exact zero at 0.50",
            "extremity": "E=abs(2*R_G-1)",
            "setup": "C!=0 and sign(H)==C and R_H>=0.60",
            "minimum_age_hours": 12,
            "maximum_age_hours_inclusive": 72,
            "contraction": "E<=peak_extremity-0.20",
            "handoff": (
                "S==episode_side and sign(H)==-episode_side and R_H>=0.75 "
                "and prior combined-valid sign(H)!=-episode_side"
            ),
            "side": "-episode_side",
            "first_handoff_only": True,
            "no_retry": True,
        },
        "execution": {
            "decision_delay_bars": 1,
            "entry_delay_bars": 2,
            "hold_bars": 288,
            "hold_hours": 24,
            "notional_leverage": 0.5,
            "interval": "[entry_time,exit_time)",
            "stops_or_targets": False,
        },
        "scheduling": {
            "construct_once": "whole frozen pre-2024 panel",
            "split_order": ["train", "selection"],
            "split_containment_fields": [
                "episode_start",
                "handoff_anchor",
                "decision_time",
                "entry_time",
                "every_held_5m_bar",
                "exit_time",
            ],
            "boundary_action": "drop before scheduling; no reservation",
            "candidate_order": "entry_time ascending",
            "acceptance": "entry_time>=prior_accepted_exit",
            "overlap_action": "suppress; never queue or replace",
            "gap_does_not_clear_accepted_reservation": True,
        },
        "calendar": {
            "source_warmup": "[2021-07-08,2021-08-08)",
            "train": "[2021-08-08,2023-01-01)",
            "selection": "[2023-01-01,2024-01-01)",
            "sealed": "2024+",
        },
        "controls": deepcopy(CONTROL_DEFINITIONS),
        "support_gates": {
            "count_basis": "accepted primary after containment and global nonoverlap",
            "train_total_minimum": 60,
            "train_2021_partial_minimum": 18,
            "train_2022_minimum": 36,
            "train_eligible_half_minimum": 12,
            "eligible_half_source_days_minimum": 90,
            "selection_total_minimum": 40,
            "selection_each_half_minimum": 18,
            "selection_each_quarter_minimum": 7,
            "train_each_side_minimum": 15,
            "train_each_side_fraction_minimum": 0.25,
            "selection_each_side_minimum": 12,
            "maximum_month_share_each_split": 0.20,
            "maximum_weekday_share_each_split": 0.25,
            "failure_action": "retire CCHR-288 before outcomes; no repair",
        },
        "comparator_contract": {
            "clock_schema": list(CLOCK_SCHEMA),
            "candidate_member_count": len(members),
            "candidate_map_hash": canonical_hash(members),
            "generated_family_member_counts": {
                "pdlh": 16,
                "dtv": 24,
                "far": 12,
                "live": 3,
            },
            "legacy_search_execution_forbidden": True,
            "raw_market_or_funding_read_forbidden": True,
            "ccpr_reader_columns": [
                "signal_time",
                "entry_time",
                "q",
                "control",
                "side",
            ],
            "dlpd_reader_columns": [
                "candidate",
                "control",
                "split",
                "source_hour_start",
                "decision_time",
                "entry_time",
                "exit_time",
                "side",
            ],
            "exact_entry_jaccard_maximum": 0.10,
            "cchr_entry_within_six_hours_fraction_maximum": 0.35,
            "absolute_signed_exposure_pearson_maximum": 0.40,
            "zero_variance_action": "fail closed",
            "live_entry_aggregation": "set union of component entry times",
            "live_exposure_aggregation": "sum(weight_i*side_i); no clipping or renormalization",
            "internal_controls_source_threshold": None,
        },
        "performance_gates": {
            "required_sequence": ["train", "selection", "2024", "2025", "2026"],
            "absolute_return_positive_each": True,
            "cagr_to_strict_mdd_minimum_each": 3.0,
            "strict_mdd_maximum_each": 0.15,
            "each_declared_half_positive": True,
            "stress_cost_bp_per_notional_per_side": 10,
            "one_extra_5m_bar_positive": True,
            "mean_gross_underlying_bp_minimum": 35.0,
            "weekly_cluster_one_sided_p_maximum": 0.10,
            "mechanism_control_ratio_margin_minimum": 0.25,
            "post_outcome_repair": "forbidden",
        },
        "rllm_boundary": {
            "authorized_before_deterministic_train_selection_pass": False,
            "allowed_actions_after_pass": ["TRADE", "ABSTAIN", "REDUCE_SIZE"],
            "side_change": False,
            "hold_extension": False,
        },
    }


def _validate_source_manifest(path: str | Path) -> dict[str, Any]:
    if sha256_file(path) != SOURCE_MANIFEST_SHA256:
        raise RuntimeError("CCHR source manifest SHA drift")
    manifest = _read_json(path)
    protocol = manifest.get("protocol")
    file_binding = manifest.get("file")
    if not isinstance(protocol, dict) or not isinstance(file_binding, dict):
        raise RuntimeError("CCHR source manifest schema drift")
    expected_protocol = {
        "outcomes_opened": False,
        "start_inclusive": "2021-07-08 00:00:00",
        "end_exclusive": "2024-01-01 00:00:00",
        "post_2023_rows_requested": False,
        "source_only": True,
    }
    for key, expected in expected_protocol.items():
        if protocol.get(key) != expected:
            raise RuntimeError(f"CCHR source protocol drift: {key}")
    if file_binding.get("path") != str(SOURCE_PATH):
        raise RuntimeError("CCHR source path drift")
    if file_binding.get("sha256") != SOURCE_SHA256:
        raise RuntimeError("CCHR source file binding drift")
    if file_binding.get("rows") != 261_216:
        raise RuntimeError("CCHR source row-count drift")
    if file_binding.get("duplicate_rows_removed") != 0:
        raise RuntimeError("CCHR source duplicate-row drift")
    if sha256_file(SOURCE_PATH) != SOURCE_SHA256:
        raise RuntimeError("CCHR source file SHA drift")
    return expected_source_binding()


def _validate_static_bindings() -> dict[str, dict[str, str]]:
    fixed = {
        SOURCE_BUILDER: SOURCE_BUILDER_SHA256,
        SOURCE_AUDIT: SOURCE_AUDIT_SHA256,
        MECHANISM_DECISION: MECHANISM_DECISION_SHA256,
    }
    for path, expected in fixed.items():
        if sha256_file(path) != expected:
            raise RuntimeError(f"CCHR frozen binding SHA drift: {path}")
    validated: dict[str, dict[str, str]] = {}
    for name, binding in COMPARATOR_PROVENANCE_BINDINGS.items():
        path = _repository_path(binding["path"])
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"CCHR comparator provenance missing: {name}")
        if sha256_file(path) != binding["sha256"]:
            raise RuntimeError(f"CCHR comparator provenance SHA drift: {name}")
        validated[name] = dict(binding)
    return validated


def _validate_config(cfg: Config, *, require_new_output: bool) -> None:
    manifest = _repository_path(cfg.source_manifest)
    raw_output = Path(cfg.preregistration_output)
    if not raw_output.is_absolute():
        raw_output = REPOSITORY_ROOT / raw_output
    if raw_output.is_symlink():
        raise ValueError("CCHR preregistration output must not be a symlink")
    output = _repository_path(cfg.preregistration_output)
    if manifest != _repository_path(SOURCE_MANIFEST):
        raise RuntimeError("CCHR source manifest path differs from frozen source")
    if manifest.suffix != ".json" or output.suffix != ".json":
        raise ValueError("CCHR source manifest and preregistration must be JSON")
    protected = {
        manifest,
        _repository_path(SOURCE_PATH),
        _repository_path(SOURCE_BUILDER),
        _repository_path(SOURCE_AUDIT),
        _repository_path(MECHANISM_DECISION),
        _repository_path(PREREGISTRATION_SOURCE),
        *(
            _repository_path(binding["path"])
            for binding in COMPARATOR_PROVENANCE_BINDINGS.values()
        ),
    }
    if output in protected:
        raise ValueError("CCHR preregistration output aliases a protected input")
    if require_new_output and output.exists():
        raise FileExistsError("CCHR preregistration is immutable")


def build_manifest(cfg: Config | None = None) -> dict[str, Any]:
    cfg = Config() if cfg is None else cfg
    _validate_config(cfg, require_new_output=False)
    source_binding = _validate_source_manifest(cfg.source_manifest)
    comparators = _validate_static_bindings()
    members = comparator_candidate_map()
    frozen_policy = policy()
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "config": asdict(cfg),
        "source_binding": source_binding,
        "mechanism_decision": {
            "path": str(MECHANISM_DECISION),
            "sha256": MECHANISM_DECISION_SHA256,
        },
        "comparator_provenance_bindings": comparators,
        "comparator_candidate_map": members,
        "comparator_candidate_map_hash": canonical_hash(members),
        "pure_clock_requirements": deepcopy(PURE_CLOCK_REQUIREMENTS),
        "comparator_freeze_requirement": comparator_freeze_requirement(),
        "policy": frozen_policy,
        "policy_hash": canonical_hash(frozen_policy),
        "outcomes_opened": False,
        "outcome_boundary": dict(OUTCOME_BOUNDARY),
        "authorization": {
            "real_incidence_authorized_by_this_artifact": False,
            "reason": "mandatory pure-clock comparator freeze is a separate prerequisite",
            "outcome_evaluator_authorized": False,
        },
        "preregistration_source": {
            "path": str(PREREGISTRATION_SOURCE),
            "sha256": sha256_file(PREREGISTRATION_SOURCE),
        },
    }
    payload["manifest_hash"] = canonical_hash(payload)
    return payload


def validate_manifest(
    payload: Mapping[str, Any],
    *,
    verify_sources: bool = True,
    expected_output: str | Path | None = None,
) -> None:
    if frozenset(payload) != ARTIFACT_TOP_LEVEL_KEYS:
        raise RuntimeError("CCHR top-level schema drift")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("CCHR protocol version drift")
    if payload.get("policy_id") != POLICY_ID:
        raise RuntimeError("CCHR policy ID drift")
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("CCHR manifest hash mismatch")
    if payload.get("outcomes_opened") is not False:
        raise RuntimeError("CCHR preregistration opened outcomes")
    if payload.get("outcome_boundary") != OUTCOME_BOUNDARY:
        raise RuntimeError("CCHR outcome boundary drift")
    if payload.get("source_binding") != expected_source_binding():
        raise RuntimeError("CCHR source binding drift")
    if payload.get("mechanism_decision") != {
        "path": str(MECHANISM_DECISION),
        "sha256": MECHANISM_DECISION_SHA256,
    }:
        raise RuntimeError("CCHR mechanism decision binding drift")
    if payload.get("comparator_provenance_bindings") != (
        COMPARATOR_PROVENANCE_BINDINGS
    ):
        raise RuntimeError("CCHR comparator provenance binding drift")
    if payload.get("policy") != policy():
        raise RuntimeError("CCHR policy drift")
    if payload.get("policy_hash") != canonical_hash(policy()):
        raise RuntimeError("CCHR policy hash drift")
    members = comparator_candidate_map()
    if payload.get("comparator_candidate_map") != members:
        raise RuntimeError("CCHR comparator candidate map drift")
    if payload.get("comparator_candidate_map_hash") != canonical_hash(members):
        raise RuntimeError("CCHR comparator candidate map hash drift")
    if payload.get("pure_clock_requirements") != PURE_CLOCK_REQUIREMENTS:
        raise RuntimeError("CCHR pure-clock requirement drift")
    if payload.get("comparator_freeze_requirement") != comparator_freeze_requirement():
        raise RuntimeError("CCHR comparator freeze requirement drift")
    authorization = payload.get("authorization")
    if not isinstance(authorization, dict) or any(
        authorization.get(key) is not False
        for key in (
            "real_incidence_authorized_by_this_artifact",
            "outcome_evaluator_authorized",
        )
    ):
        raise RuntimeError("CCHR authorization drift")
    config = payload.get("config")
    if not isinstance(config, dict) or set(config) != {
        "source_manifest",
        "preregistration_output",
    }:
        raise RuntimeError("CCHR config drift")
    if config["source_manifest"] != str(SOURCE_MANIFEST):
        raise RuntimeError("CCHR config source-manifest drift")
    preregistration_source = payload.get("preregistration_source")
    if not isinstance(preregistration_source, dict) or preregistration_source.get(
        "path"
    ) != str(PREREGISTRATION_SOURCE):
        raise RuntimeError("CCHR preregistration-source binding drift")
    if expected_output is not None and _repository_path(
        config["preregistration_output"]
    ) != _repository_path(expected_output):
        raise RuntimeError("CCHR output-path binding drift")
    if verify_sources:
        expected = build_manifest(Config(**config))
        if payload != expected:
            raise RuntimeError("CCHR frozen binding or source drift")


def write_preregistration(cfg: Config | None = None) -> dict[str, Any]:
    cfg = Config() if cfg is None else cfg
    _validate_config(cfg, require_new_output=True)
    payload = build_manifest(cfg)
    validate_manifest(payload, expected_output=cfg.preregistration_output)
    output = _repository_path(cfg.preregistration_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return payload


def load_preregistration(path: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    payload = _read_json(path)
    validate_manifest(payload, expected_output=path)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", default=str(SOURCE_MANIFEST))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_preregistration(
        Config(
            source_manifest=args.source_manifest,
            preregistration_output=args.output,
        )
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
