"""Freeze CDLTR-72A before opening source incidence or market outcomes.

This stage hashes bound files and reads source/comparator CSV headers only. It
never parses a source value row, comparator event row, comparator manifest,
BTC market row, funding value, return, or strategy outcome.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


POLICY_ID = "CDLTR-72A"
PROTOCOL_VERSION = "cross_domain_liquidity_transmission_relay_preregistration_v2"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION_SOURCE = Path(
    "training/preregister_cross_domain_liquidity_transmission_relay.py"
)
MECHANISM_DECISION = Path(
    "docs/cross-domain-liquidity-transmission-relay-mechanism-decision-2026-07-21.md"
)
MECHANISM_DECISION_SHA256 = (
    "970a114b7dab6b39bea8110264eb4ab05fd9794b5cb239bc643acb53619eebe5"
)
COMPARATOR_AMENDMENT = Path(
    "docs/cdltr72a-preincidence-comparator-amendment-2026-07-21.md"
)
COMPARATOR_AMENDMENT_SHA256 = (
    "fba002d78e0c29d5824d2bfd922d74c1d5477f2eb63f55959f14aafd88661064"
)
PREREGISTRATION_DOCUMENT = Path(
    "docs/cross-domain-liquidity-transmission-relay-preregistration-2026-07-21.md"
)
PREREGISTRATION_DOCUMENT_SHA256 = (
    "b768a63da6809230e4fbd87bc7106e19460817aa3f5bb645d20645e00b18582a"
)
DEFAULT_OUTPUT = Path(
    "results/cross_domain_liquidity_transmission_relay_preregistration_2026-07-21.json"
)

RRP_SOURCE = Path(
    "data/new_york_fed_overnight_rrp_2018_2023/"
    "new_york_fed_overnight_rrp_2018-01-01_2023-12-31.csv.gz"
)
RRP_MANIFEST = Path("data/new_york_fed_overnight_rrp_2018_2023/build_manifest.json")
RRP_BUILDER = Path("training/build_new_york_fed_overnight_rrp.py")
CBOE_SOURCE = Path(
    "data/cboe_volatility_term_structure_2018_2023/"
    "cboe_vix_term_structure_2018-01-01_2023-12-31.csv.gz"
)
CBOE_MANIFEST = Path(
    "data/cboe_volatility_term_structure_2018_2023/build_manifest.json"
)
CBOE_BUILDER = Path("training/build_cboe_volatility_term_structure.py")
NETWORK_SOURCE = Path("data/coinmetrics_btc_network_daily_2020_2023.csv.gz")
NETWORK_MANIFEST = Path(
    "results/coinmetrics_btc_network_daily_pre2024_manifest_2026-07-16.json"
)
NETWORK_BUILDER = Path("training/download_coinmetrics_btc_network_daily.py")

SOURCE_BINDINGS = {
    "rrp": {
        "source": RRP_SOURCE,
        "source_sha256": (
            "49f67ed44b7eb81fd35c17a8209cf14d6a8019d7e9f77fce8c343d1a7fb66b27"
        ),
        "manifest": RRP_MANIFEST,
        "manifest_sha256": (
            "4f87e2219da71c94832c8708086ba01387efc145e3488b62cd3b3d07c62d8fee"
        ),
        "builder": RRP_BUILDER,
        "builder_sha256": (
            "0567157dde18b1c6ccfb37b669ceead521360f23dd0b73033fccc08e37c0d42c"
        ),
        "header": (
            "operation_id",
            "operation_date",
            "settlement_date",
            "maturity_date",
            "close_time_et",
            "result_available_at_utc",
            "last_updated_et",
            "total_amount_submitted_usd",
            "total_amount_accepted_usd",
            "participating_counterparties",
            "accepted_counterparties",
            "source_complete",
            "quarantine_reason",
        ),
        "allowed_columns": (
            "operation_date",
            "result_available_at_utc",
            "total_amount_accepted_usd",
            "source_complete",
            "quarantine_reason",
        ),
    },
    "cboe": {
        "source": CBOE_SOURCE,
        "source_sha256": (
            "6f1b2f7f3a5b1e4d5001d673e6ff54374791879c278248ce27b3d610e4f75dc7"
        ),
        "manifest": CBOE_MANIFEST,
        "manifest_sha256": (
            "42b2a35ad131bd63574d2adcf684e28766dc3060fa645fc749df10dd3fb27f27"
        ),
        "builder": CBOE_BUILDER,
        "builder_sha256": (
            "0cd9fb50d6f665e9cc4f20539bacde328d1b7587b624ebc15f8a7b3489eeec2d"
        ),
        "header": (
            "observation_date",
            "VIX9D_close",
            "VIX_close",
            "VIX3M_close",
        ),
        "allowed_columns": (
            "observation_date",
            "VIX9D_close",
            "VIX3M_close",
        ),
    },
    "network": {
        "source": NETWORK_SOURCE,
        "source_sha256": (
            "97ab2ca9d0c347d85221b51734f98072763370072ca51f1c40e3214191159b42"
        ),
        "manifest": NETWORK_MANIFEST,
        "manifest_sha256": (
            "66b185769800c4732cf748b40ca9cb48c5eee239abf0425ff193c0688111c372"
        ),
        "builder": NETWORK_BUILDER,
        "builder_sha256": (
            "73929fe4f7b8463ee187d008657cdad5be45df0d5f2c74ded1d541f61e87b763"
        ),
        "header": (
            "observation_date",
            "available_at",
            "AdrActCnt",
            "TxCnt",
            "TxTfrCnt",
        ),
        "allowed_columns": (
            "observation_date",
            "available_at",
            "AdrActCnt",
            "TxCnt",
            "TxTfrCnt",
        ),
    },
}

COMPARATOR_CLOCK = Path("results/cdltr_prior_comparator_views_2026-07-21.csv.gz")
COMPARATOR_CLOCK_SHA256 = (
    "bffdcf158d7d4e38db5794fb4761de528fb73b0b772ae950f3a087a93ab63f1a"
)
COMPARATOR_MANIFEST = Path(
    "results/cdltr_prior_comparator_views_manifest_2026-07-21.json"
)
COMPARATOR_MANIFEST_SHA256 = (
    "a795f384287f24200e00d2cc5a5721610bb5282d1b044b3a653a053190c44261"
)
COMPARATOR_HEADER = (
    "comparator",
    "capability",
    "decision_time",
    "entry_time",
    "exit_time",
    "side",
    "source_clock",
)
DIRECTIONAL_COMPARATORS = (
    "CVTR-1",
    "DFFB-601",
    "FLCC-1:FLCC-H4-Q60",
    "FLCC-1:FLCC-H4-Q65",
    "FLCC-1:FLCC-H8-Q60",
    "FLCC-1:FLCC-H8-Q65",
    "NTB-7",
    "NWE-8",
    "ORFR-1",
    "chain_activity_impulse_momentum",
)
TIMESTAMP_ONLY_COMPARATORS = (
    "NWE-7",
    "live_anchor_2023",
    "prior_microstructure:cbfr72",
    "prior_microstructure:mfic_fast",
    "prior_microstructure:mfic_slow",
    "prior_microstructure:mfic_union",
    "prior_microstructure:netf_fast",
    "prior_microstructure:netf_slow",
    "prior_microstructure:netf_union",
    "prior_microstructure:terminal_absorption_wait72_h72",
    "prior_microstructure:wfrs_l288_q90_h144",
)

EXPECTED_OUTCOME_BOUNDARY = {
    "source_value_rows_read": 0,
    "source_feature_rows_derived": 0,
    "real_event_incidence_rows_derived": 0,
    "comparator_event_rows_read": 0,
    "comparator_manifest_values_parsed": 0,
    "btc_market_rows_loaded": 0,
    "funding_rows_loaded": 0,
    "return_rows_loaded": 0,
    "return_or_pnl_fields_read": 0,
    "post_2023_rows_read": 0,
    "network_calls": 0,
    "subprocess_calls": 0,
}


@dataclass(frozen=True)
class Config:
    output: str = str(DEFAULT_OUTPUT)


def _repository_path(path: str | Path) -> Path:
    raw = str(path)
    candidate = Path(path)
    if raw.startswith("~") or candidate.is_absolute() or ".." in candidate.parts:
        raise RuntimeError("CDLTR path must be repository-relative")
    root = REPOSITORY_ROOT.resolve(strict=True)
    target = REPOSITORY_ROOT / candidate
    current = REPOSITORY_ROOT
    for part in candidate.parts:
        current /= part
        if current.is_symlink():
            raise RuntimeError("CDLTR repository path contains a symlink")
        if not current.exists():
            break
    resolved = target.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise RuntimeError("CDLTR path escapes the repository") from error
    return target


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_file(path: str | Path) -> str:
    return _sha256_path(_repository_path(path))


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_regular_file(path: str | Path, expected_sha256: str, label: str) -> str:
    target = _repository_path(path)
    if target.is_symlink():
        raise RuntimeError(f"{label} is a symlink")
    if not target.is_file():
        raise RuntimeError(f"{label} is missing")
    observed = _sha256_path(target)
    if observed != expected_sha256:
        raise RuntimeError(f"{label} SHA drift")
    return observed


def _read_gzip_csv_header(path: str | Path) -> tuple[str, ...]:
    target = _repository_path(path)
    with gzip.open(target, "rt", encoding="utf-8", newline="") as handle:
        return tuple(next(csv.reader(handle)))


def _serialize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    return value


def _source_binding_payload() -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, contract in SOURCE_BINDINGS.items():
        source = contract["source"]
        _validate_regular_file(
            source,
            str(contract["source_sha256"]),
            f"CDLTR {name} source",
        )
        header = _read_gzip_csv_header(source)
        if header != contract["header"]:
            raise RuntimeError(f"CDLTR {name} source header drift")
        allowed = tuple(str(item) for item in contract["allowed_columns"])
        if any(column not in header for column in allowed):
            raise RuntimeError(f"CDLTR {name} allowlist drift")
        _validate_regular_file(
            contract["manifest"],
            str(contract["manifest_sha256"]),
            f"CDLTR {name} manifest",
        )
        _validate_regular_file(
            contract["builder"],
            str(contract["builder_sha256"]),
            f"CDLTR {name} builder",
        )
        output[name] = _serialize(contract)
    return output


def _comparator_binding_payload() -> dict[str, Any]:
    _validate_regular_file(
        COMPARATOR_CLOCK,
        COMPARATOR_CLOCK_SHA256,
        "CDLTR complete comparator clock",
    )
    if _read_gzip_csv_header(COMPARATOR_CLOCK) != COMPARATOR_HEADER:
        raise RuntimeError("CDLTR complete comparator clock header drift")
    _validate_regular_file(
        COMPARATOR_MANIFEST,
        COMPARATOR_MANIFEST_SHA256,
        "CDLTR complete comparator manifest",
    )
    return {
        "clock": str(COMPARATOR_CLOCK),
        "clock_sha256": COMPARATOR_CLOCK_SHA256,
        "manifest": str(COMPARATOR_MANIFEST),
        "manifest_sha256": COMPARATOR_MANIFEST_SHA256,
        "format": "csv_gzip",
        "header": list(COMPARATOR_HEADER),
        "rows": 9_985,
        "directional_rows": 1_788,
        "timestamp_only_rows": 8_197,
        "directional_comparators": list(DIRECTIONAL_COMPARATORS),
        "timestamp_only_comparators": list(TIMESTAMP_ONLY_COMPARATORS),
        "event_rows_read_during_preregistration": 0,
        "manifest_values_parsed_during_preregistration": 0,
    }


def policy_payload() -> dict[str, Any]:
    return {
        "candidate": POLICY_ID,
        "singleton": True,
        "predecessor_disposition": (
            "CDLTR-72 rejected before preregistration and incidence because "
            "not every comparator defined a directional interval"
        ),
        "non_pristine_boundary": (
            "candidate-level new interaction; component-family development results "
            "are research-seen and cannot validate CDLTR"
        ),
        "source_votes": {
            "rrp": {
                "lookback_normal_operation_slots": 5,
                "required_complete_slots_inclusive": 6,
                "quarantine_breaks_baseline": True,
                "long": "accepted_now - accepted_fifth_prior_slot < 0",
                "short": "accepted_now - accepted_fifth_prior_slot > 0",
                "neutral": "delta == 0",
                "causal_time": "result_available_at_utc",
            },
            "cboe": {
                "long": "VIX9D_close < VIX3M_close",
                "short": "VIX9D_close > VIX3M_close",
                "neutral": "VIX9D_close == VIX3M_close",
                "causal_time": (
                    "09:35 America/New_York on the next exact date in the "
                    "three-index intersection"
                ),
                "forward_fill": False,
            },
            "network": {
                "lookback_calendar_days": 7,
                "required_consecutive_dates": 8,
                "metrics": ["AdrActCnt", "TxCnt", "TxTfrCnt"],
                "metric_transform": "sign(log(current / exact_7d_prior))",
                "long": "at least two positive metric votes",
                "short": "at least two negative metric votes",
                "neutral": "otherwise",
                "causal_time": "available_at",
            },
        },
        "relay": {
            "state_age_clock": "actual UTC availability timestamp",
            "rrp_vote_expiry_hours": 36,
            "cboe_vote_expiry_hours": 36,
            "macro_episode_transition": "absent_or_opposite_to_same_nonzero_side",
            "network_confirmation": "first report strictly after macro onset",
            "network_deadline_hours_after_onset": 36,
            "failed_confirmation_retry": False,
            "reactivation_requires_macro_exit_and_reentry": True,
        },
        "execution": {
            "decision_time": "confirming network available_at",
            "entry": "ceil_to_5m(decision_time) + 5 minutes",
            "exact_boundary_still_adds_minutes": 5,
            "hold_hours": 72,
            "notional_exposure": 0.5,
            "stops_or_take_profit": False,
            "global_nonoverlap": True,
            "split_crossing_action": "skip",
        },
        "windows": {
            "warmup": ["2020-01-01T00:00:00Z", "2021-01-01T00:00:00Z"],
            "train": ["2021-01-01T00:00:00Z", "2023-01-01T00:00:00Z"],
            "selection": ["2023-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "sealed_from": "2024-01-01T00:00:00Z",
        },
        "support_gates": {
            "train_total_minimum": 60,
            "each_train_year_minimum": 25,
            "each_train_half_year_minimum": 12,
            "selection_total_minimum": 30,
            "each_selection_half_year_minimum": 12,
            "train_each_side_minimum": 18,
            "selection_each_side_minimum": 8,
            "maximum_month_share": 0.20,
            "maximum_weekday_share": 0.35,
            "all_controls_must_pass_calendar_and_containment": True,
        },
        "controls": {
            "macro_only": "macro-episode onset, same side",
            "network_only": "network-vote onset, same side",
            "reverse_order": "network onset followed by first macro update",
            "one_network_report_delay": "primary side on next valid network report",
            "direction_flip": "exact primary clock, opposite side",
            "deterministic_random_side": (
                "LONG iff first byte SHA256('CDLTR-72|20260721|' + "
                "entry_time_utc) < 128, otherwise SHORT; predecessor seed retained"
            ),
        },
        "novelty_gates": {
            "decision_date_jaccard_maximum": 0.30,
            "cdltr_dates_within_one_utc_day_fraction_maximum": 0.50,
            "signed_occupied_exposure_absolute_pearson_maximum": 0.40,
            "exposure_grid": (
                "complete 5m UTC common span; entry-inclusive/exit-exclusive; "
                "flat 0, long +1, short -1"
            ),
            "timestamp_gates_apply_to": [
                *DIRECTIONAL_COMPARATORS,
                *TIMESTAMP_ONLY_COMPARATORS,
            ],
            "signed_exposure_applies_to": list(DIRECTIONAL_COMPARATORS),
            "timestamp_only_comparators": list(TIMESTAMP_ONLY_COMPARATORS),
            "flcc_candidates_pass_independently": True,
            "missing_side_exit_or_union_may_not_be_invented": True,
            "directional_comparators_must_have_nonzero_variance": True,
            "failure_action": "reject CDLTR-72A without repair or outcomes",
        },
        "outcome_sequence": [
            "source support and novelty",
            "separately frozen strict train",
            "strict selection only after train pass",
            "future shadow only after final freeze",
        ],
        "llm_boundary": {
            "authorized_before_deterministic_train_and_selection_pass": False,
            "later_role": "TRADE/ABSTAIN veto only",
            "may_change_side_timing_hold_or_relay": False,
            "rl_reward_requirements": ["strict drawdown penalty", "turnover penalty"],
        },
        "stopping_rule": (
            "any source, support, control, novelty, or strict economic failure "
            "permanently rejects CDLTR-72A; repair requires a new candidate identity"
        ),
    }


def build_preregistration(cfg: Config | None = None) -> dict[str, Any]:
    frozen_cfg = Config() if cfg is None else cfg
    mechanism_sha = _validate_regular_file(
        MECHANISM_DECISION,
        MECHANISM_DECISION_SHA256,
        "CDLTR mechanism decision",
    )
    amendment_sha = _validate_regular_file(
        COMPARATOR_AMENDMENT,
        COMPARATOR_AMENDMENT_SHA256,
        "CDLTR comparator amendment",
    )
    document_sha = _validate_regular_file(
        PREREGISTRATION_DOCUMENT,
        PREREGISTRATION_DOCUMENT_SHA256,
        "CDLTR preregistration document",
    )
    source_sha = sha256_file(PREREGISTRATION_SOURCE)
    policy = policy_payload()
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "candidate": POLICY_ID,
        "config": asdict(frozen_cfg),
        "mechanism_decision": {
            "path": str(MECHANISM_DECISION),
            "sha256": mechanism_sha,
        },
        "comparator_amendment": {
            "path": str(COMPARATOR_AMENDMENT),
            "sha256": amendment_sha,
        },
        "preregistration_document": {
            "path": str(PREREGISTRATION_DOCUMENT),
            "sha256": document_sha,
        },
        "preregistration_source": {
            "path": str(PREREGISTRATION_SOURCE),
            "sha256": source_sha,
        },
        "source_bindings": _source_binding_payload(),
        "comparator_binding": _comparator_binding_payload(),
        "policy": policy,
        "policy_hash": canonical_hash(policy),
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "comparator_incidence_opened": False,
        "performance_values_opened": False,
        "outcome_boundary": dict(EXPECTED_OUTCOME_BOUNDARY),
        "next_action": "run exact source-only CDLTR-72A support and novelty evaluator",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def _protected_paths() -> set[Path]:
    protected = {
        _repository_path(PREREGISTRATION_SOURCE),
        _repository_path(MECHANISM_DECISION),
        _repository_path(COMPARATOR_AMENDMENT),
        _repository_path(PREREGISTRATION_DOCUMENT),
        _repository_path(COMPARATOR_CLOCK),
        _repository_path(COMPARATOR_MANIFEST),
    }
    for contract in SOURCE_BINDINGS.values():
        for key in ("source", "manifest", "builder"):
            protected.add(_repository_path(contract[key]))
    return protected


def write_preregistration(cfg: Config | None = None) -> dict[str, Any]:
    frozen_cfg = Config() if cfg is None else cfg
    output = _repository_path(frozen_cfg.output)
    if output.suffix != ".json":
        raise ValueError("CDLTR preregistration output must be JSON")
    if output in _protected_paths():
        raise ValueError("CDLTR preregistration output aliases a protected input")
    if output.exists() or output.is_symlink():
        raise FileExistsError("CDLTR preregistration is immutable")
    payload = build_preregistration(frozen_cfg)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=output.parent, prefix=f".{output.name}.", delete=False
    ) as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.replace(temporary, output)
        output.chmod(0o444)
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def load_preregistration(path: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    target = _repository_path(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("CDLTR preregistration must be a JSON object")
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("CDLTR preregistration canonical hash mismatch")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("CDLTR preregistration protocol drift")
    if payload.get("candidate") != POLICY_ID:
        raise RuntimeError("CDLTR preregistration candidate drift")
    if payload.get("policy") != policy_payload():
        raise RuntimeError("CDLTR preregistration policy drift")
    if payload.get("policy_hash") != canonical_hash(policy_payload()):
        raise RuntimeError("CDLTR preregistration policy hash drift")
    if payload.get("outcomes_opened") is not False:
        raise RuntimeError("CDLTR preregistration opened outcomes")
    if payload.get("outcome_boundary") != EXPECTED_OUTCOME_BOUNDARY:
        raise RuntimeError("CDLTR preregistration outcome boundary drift")
    config = payload.get("config")
    if not isinstance(config, dict):
        raise RuntimeError("CDLTR preregistration config missing")
    expected = build_preregistration(Config(**config))
    if payload != expected:
        raise RuntimeError("CDLTR preregistration binding drift")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_preregistration(Config(output=args.output))
    print(
        json.dumps(
            {
                "candidate": payload["candidate"],
                "manifest_hash": payload["manifest_hash"],
                "outcomes_opened": payload["outcomes_opened"],
                "output": args.output,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
