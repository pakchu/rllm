"""Compare the frozen BAFR clock with a hash-bound prior-clock bundle only."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


FIVE_MINUTES = pd.Timedelta(minutes=5)
TOLERANCE_BARS = 12
JACCARD_MAXIMUM = 0.20
BAFR_CONTAINMENT_MAXIMUM = 0.30

BAFR_SUPPORT_SHA256 = "cf6edad6a4eb46c6630dbb5008c88da1ddd39f9ac5c1606785be02f2b323fb62"
BAFR_CLOCK_SHA256 = "f3b816a76decce31136ed23d22f043eb8e80ef1b8697b869241b060062f01747"
# Pinned after the independent prior-clock freezer was run, committed, and reviewed.
PRIOR_BUNDLE_SHA256 = "c5584256140799b380973f9f376e5751ad754a81c9683473467b9d05af0bb9f0"

REQUIRED_COMPARATORS = (
    "cbfr72",
    "mfic_fast",
    "mfic_slow",
    "mfic_union",
    "netf_fast",
    "netf_slow",
    "netf_union",
    "wfrs_l288_q90_h144",
    "terminal_absorption_wait72_h72",
)


@dataclass(frozen=True)
class NoveltyConfig:
    bafr_support: str = "results/binance_aggressor_frustration_support_2026-07-20.json"
    bafr_clock: str = "results/binance_aggressor_frustration_clock_2026-07-20.csv"
    prior_bundle: str = "results/prior_microstructure_comparator_clock_bundle_2026-07-20.json"
    output: str = "results/binance_aggressor_frustration_novelty_2026-07-20.json"


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require_sha256(path: str | Path, expected: str, *, label: str) -> str:
    observed = _sha256(path)
    if observed != expected:
        raise ValueError(f"{label} hash differs from its frozen value")
    return observed


def normalize_clock(clock: pd.DataFrame) -> pd.DataFrame:
    required = {"signal_date", "side"}
    if not required.issubset(clock.columns):
        raise ValueError(f"clock is missing columns: {sorted(required - set(clock.columns))}")
    output = clock.loc[:, ["signal_date", "side"]].copy()
    output["signal_date"] = pd.to_datetime(
        output["signal_date"], utc=True, errors="raise"
    ).dt.tz_convert(None)
    output["side"] = pd.to_numeric(output["side"], errors="raise").astype(np.int8)
    if not output["side"].isin((-1, 1)).all():
        raise ValueError("clock side must be exactly -1 or +1")
    if output["signal_date"].duplicated().any():
        raise ValueError("clock contains duplicate signal timestamps")
    if not output["signal_date"].is_monotonic_increasing:
        raise ValueError("clock timestamps must be strictly increasing")
    return output.reset_index(drop=True)


def clock_hash(clock: pd.DataFrame) -> str:
    normalized = normalize_clock(clock)
    events = [
        {
            "signal_date": row.signal_date.strftime("%Y-%m-%d %H:%M:%S"),
            "side": int(row.side),
        }
        for row in normalized.itertuples(index=False)
    ]
    payload = json.dumps(events, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def one_to_one_pairs(
    left_ns: Iterable[int],
    right_ns: Iterable[int],
    *,
    tolerance_ns: int,
) -> list[tuple[int, int]]:
    if tolerance_ns < 0:
        raise ValueError("match tolerance cannot be negative")
    left = sorted(int(value) for value in left_ns)
    right = sorted(int(value) for value in right_ns)
    left_cursor = right_cursor = 0
    pairs: list[tuple[int, int]] = []
    while left_cursor < len(left) and right_cursor < len(right):
        first = left[left_cursor]
        second = right[right_cursor]
        if abs(first - second) <= tolerance_ns:
            pairs.append((first, second))
            left_cursor += 1
            right_cursor += 1
        elif first < second:
            left_cursor += 1
        else:
            right_cursor += 1
    return pairs


def compare_clock(
    bafr_clock: pd.DataFrame,
    prior_clock: pd.DataFrame,
    *,
    name: str,
    coverage_start: str | pd.Timestamp,
    coverage_end: str | pd.Timestamp,
    tolerance_bars: int = TOLERANCE_BARS,
) -> dict[str, Any]:
    start = pd.Timestamp(coverage_start)
    end = pd.Timestamp(coverage_end)
    if start >= end:
        raise ValueError("comparison coverage start must precede end")
    bafr = normalize_clock(bafr_clock)
    prior = normalize_clock(prior_clock)
    bafr = bafr.loc[bafr["signal_date"].ge(start) & bafr["signal_date"].lt(end)].copy()
    prior = prior.loc[prior["signal_date"].ge(start) & prior["signal_date"].lt(end)].copy()
    tolerance_ns = int(FIVE_MINUTES.value * tolerance_bars)
    pairs = one_to_one_pairs(
        bafr["signal_date"].astype("int64"),
        prior["signal_date"].astype("int64"),
        tolerance_ns=tolerance_ns,
    )
    matches = len(pairs)
    union = len(bafr) + len(prior) - matches

    same_side_matches = 0
    for side in (-1, 1):
        same_side_matches += len(
            one_to_one_pairs(
                bafr.loc[bafr["side"].eq(side), "signal_date"].astype("int64"),
                prior.loc[prior["side"].eq(side), "signal_date"].astype("int64"),
                tolerance_ns=tolerance_ns,
            )
        )

    jaccard = matches / union if union else 1.0
    containment = matches / len(bafr) if len(bafr) else 1.0
    prior_containment = matches / len(prior) if len(prior) else 1.0
    has_common_clock = bool(len(bafr) and len(prior))
    passes = bool(
        has_common_clock
        and jaccard <= JACCARD_MAXIMUM
        and containment <= BAFR_CONTAINMENT_MAXIMUM
    )
    return {
        "comparator": name,
        "coverage_start_inclusive": str(start),
        "coverage_end_exclusive": str(end),
        "tolerance_bars": int(tolerance_bars),
        "bafr_events": int(len(bafr)),
        "prior_events": int(len(prior)),
        "time_matches": int(matches),
        "same_side_matches": int(same_side_matches),
        "time_jaccard": float(jaccard),
        "bafr_time_containment": float(containment),
        "prior_time_containment_diagnostic": float(prior_containment),
        "jaccard_maximum": JACCARD_MAXIMUM,
        "bafr_containment_maximum": BAFR_CONTAINMENT_MAXIMUM,
        "passes": passes,
    }


def load_bafr(
    cfg: NoveltyConfig,
    *,
    expected_support_sha256: str = BAFR_SUPPORT_SHA256,
    expected_clock_sha256: str = BAFR_CLOCK_SHA256,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    support_path = Path(cfg.bafr_support)
    clock_path = Path(cfg.bafr_clock)
    support_hash = _require_sha256(
        support_path, expected_support_sha256, label="BAFR support"
    )
    clock_file_hash = _require_sha256(
        clock_path, expected_clock_sha256, label="BAFR clock"
    )
    support = json.loads(support_path.read_text())
    identity = {
        "candidate": "BAFR-24F",
        "stage": "outcome_blind_support",
        "outcomes_opened": False,
        "passed": True,
        "next_stage": "outcome_blind_novelty_gate",
    }
    for key, expected in identity.items():
        if support.get(key) != expected:
            raise ValueError(f"BAFR support identity mismatch: {key}")
    if support.get("source", {}).get("market_columns_loaded") != ["date"]:
        raise ValueError("BAFR support parsed non-timestamp market columns")
    if support.get("source", {}).get("price_or_outcome_columns_loaded") != []:
        raise ValueError("BAFR support parsed price or outcome columns")
    if support.get("clock", {}).get("sha256") != clock_file_hash:
        raise ValueError("BAFR clock hash differs from support artifact")
    clock = normalize_clock(
        pd.read_csv(clock_path, usecols=["signal_date", "side"])
    )
    if len(clock) != int(support.get("clock", {}).get("rows", -1)):
        raise ValueError("BAFR clock row count differs from support artifact")
    return clock, {
        "support_path": str(support_path),
        "support_sha256": support_hash,
        "clock_path": str(clock_path),
        "clock_file_sha256": clock_file_hash,
        "clock_canonical_sha256": clock_hash(clock),
        "clock_rows": int(len(clock)),
    }


def load_prior_bundle(
    path: str | Path,
    *,
    expected_sha256: str = PRIOR_BUNDLE_SHA256,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    bundle_hash = _require_sha256(path, expected_sha256, label="prior clock bundle")
    payload = json.loads(Path(path).read_text())
    if payload.get("stage") != "prior_comparator_clock_freeze":
        raise ValueError("prior bundle stage is invalid")
    protocol = payload.get("protocol", {})
    required_false = (
        "bafr_source_loaded",
        "bafr_clock_loaded",
        "bafr_support_loaded",
        "bafr_outcomes_opened",
        "post_entry_outcomes_computed",
    )
    if any(protocol.get(field) is not False for field in required_false):
        raise ValueError("prior bundle touched BAFR or post-entry outcomes")
    if protocol.get("output_fields") != ["signal_date", "side"]:
        raise ValueError("prior bundle exposes non-clock output fields")
    raw_comparators = payload.get("comparators")
    if not isinstance(raw_comparators, dict):
        raise ValueError("prior bundle comparators are missing")
    if tuple(raw_comparators) != REQUIRED_COMPARATORS:
        raise ValueError("prior bundle comparator set or order differs from freeze")

    comparators: dict[str, dict[str, Any]] = {}
    metadata: dict[str, Any] = {}
    for name, descriptor in raw_comparators.items():
        clock = normalize_clock(pd.DataFrame(descriptor.get("events", [])))
        if len(clock) != int(descriptor.get("clock_rows", -1)):
            raise ValueError(f"{name} clock row count differs from bundle")
        observed_hash = clock_hash(clock)
        if observed_hash != descriptor.get("clock_sha256"):
            raise ValueError(f"{name} canonical clock hash differs from bundle")
        start = pd.Timestamp(descriptor.get("coverage_start_inclusive"))
        end = pd.Timestamp(descriptor.get("coverage_end_exclusive"))
        if start >= end or end > pd.Timestamp("2024-01-01"):
            raise ValueError(f"{name} coverage is invalid")
        comparators[name] = {
            "clock": clock,
            "coverage_start": start,
            "coverage_end": end,
        }
        metadata[name] = {
            key: value for key, value in descriptor.items() if key != "events"
        }
    return comparators, {
        "path": str(path),
        "sha256": bundle_hash,
        "stage": payload["stage"],
        "protocol": protocol,
        "sources": payload.get("sources"),
        "comparators": metadata,
    }


def run_novelty(cfg: NoveltyConfig) -> dict[str, Any]:
    bafr_clock, bafr_source = load_bafr(cfg)
    comparators, prior_bundle = load_prior_bundle(cfg.prior_bundle)
    comparisons = [
        compare_clock(
            bafr_clock,
            descriptor["clock"],
            name=name,
            coverage_start=descriptor["coverage_start"],
            coverage_end=descriptor["coverage_end"],
        )
        for name, descriptor in comparators.items()
    ]
    passed = bool(comparisons and all(item["passes"] for item in comparisons))
    result = {
        "as_of": "2026-07-20",
        "candidate": "BAFR-24F",
        "stage": "outcome_blind_novelty_gate",
        "outcomes_opened": False,
        "protocol": {
            "comparison_fields_loaded": ["signal_date", "side"],
            "bafr_market_ohlc_or_funding_loaded": False,
            "prior_market_data_loaded": False,
            "common_coverage": "clip both clocks to each frozen comparator's declared source window",
            "primary_match": "deterministic one-to-one timestamp match; side-agnostic time matching is the conservative gate and same-side matching is diagnostic",
            "tolerance_bars": TOLERANCE_BARS,
            "jaccard_maximum": JACCARD_MAXIMUM,
            "bafr_containment_maximum": BAFR_CONTAINMENT_MAXIMUM,
            "family_policy": "gate each frozen variant and the timestamp-deduplicated union for multi-variant MFIC and NETF families",
        },
        "config": asdict(cfg),
        "bafr_source": bafr_source,
        "prior_bundle": prior_bundle,
        "comparisons": comparisons,
        "passed": passed,
        "next_stage": "freeze_outcome_evaluator" if passed else "reject_without_outcomes",
    }
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    for field in NoveltyConfig.__dataclass_fields__.values():
        parser.add_argument(
            "--" + field.name.replace("_", "-"),
            default=getattr(NoveltyConfig, field.name),
        )
    result = run_novelty(NoveltyConfig(**vars(parser.parse_args())))
    print(
        json.dumps(
            {
                "outcomes_opened": result["outcomes_opened"],
                "passed": result["passed"],
                "comparisons": result["comparisons"],
                "next_stage": result["next_stage"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
