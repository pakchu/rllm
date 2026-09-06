"""Split frozen train-only PPOSM lifecycle residual rows by candidate specialist.

The input is intentionally frozen: 204 pre-2024 lifecycle-anchor pair rows
(102 SKIP, 102 TP12) from the train-only lifecycle residual builder.  This
splitter fails closed if the input bytes, identity order, train/window markers,
reference-anchor markers, or candidate pair contract drift.  It writes no OOS
surface; outputs are deterministic train JSONLs for candidate-specific SFT/RLVR
specialists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import build_pposm_counterfactual_action_data as counterfactual
from training import build_pposm_lifecycle_residual_data as lifecycle

DEFAULT_INPUT = lifecycle.DEFAULT_TRAIN_OUTPUT
DEFAULT_SKIP_OUTPUT = Path(
    "data/pposm_lifecycle_residual_specialist_skip_train_pre2024_2026-09-02.jsonl"
)
DEFAULT_TP12_OUTPUT = Path(
    "data/pposm_lifecycle_residual_specialist_tp12_train_pre2024_2026-09-02.jsonl"
)
DEFAULT_SUMMARY_OUTPUT = Path(
    "results/pposm_lifecycle_specialist_split_summary_2026-09-02.json"
)
DEFAULT_EXPECTED_INPUT_SHA256 = "64d03c9baa68e8968ffddb66e531561a7724410e4324d9d3a04eeba58f4796f1"
DEFAULT_EXPECTED_IDENTITY_SHA256 = "d0d2578ee463b2282915933afdcbc168a4178efe584a98886dbabc7099cdf8c2"
DEFAULT_EXPECTED_ROWS = 204
DEFAULT_EXPECTED_ROWS_PER_CANDIDATE = 102
TRAIN_WINDOW = lifecycle.TRAIN_WINDOW
CANDIDATE_ACTIONS = lifecycle.CANDIDATE_ACTIONS
LABELS = lifecycle.LABELS


@dataclass(frozen=True)
class Config:
    input_jsonl: Path = DEFAULT_INPUT
    skip_output: Path = DEFAULT_SKIP_OUTPUT
    tp12_output: Path = DEFAULT_TP12_OUTPUT
    summary_output: Path = DEFAULT_SUMMARY_OUTPUT
    expected_input_sha256: str = DEFAULT_EXPECTED_INPUT_SHA256
    expected_identity_sha256: str = DEFAULT_EXPECTED_IDENTITY_SHA256
    expected_rows: int = DEFAULT_EXPECTED_ROWS
    expected_rows_per_candidate: int = DEFAULT_EXPECTED_ROWS_PER_CANDIDATE


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join((canonical_json(row) + "\n").encode("utf-8") for row in rows)


def _load_jsonl_with_sha(path: str | Path) -> tuple[list[dict[str, Any]], str]:
    source = Path(path)
    raw = source.read_bytes()
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"line {number} in {source} is not a JSON object")
        rows.append(value)
    return rows, _sha256_bytes(raw)


def _identity_sha256(rows: Sequence[dict[str, Any]]) -> str:
    identities = [str(_metadata(row)["identity"]) for row in rows]
    return _sha256_bytes("\n".join(identities).encode("utf-8"))


def _metadata(row: Mapping[str, Any]) -> Mapping[str, Any]:
    meta = row.get("metadata")
    if not isinstance(meta, Mapping):
        raise ValueError("lifecycle row is missing metadata object")
    return meta


def _signal_time_pre2024(meta: Mapping[str, Any]) -> bool:
    ts = pd.Timestamp(str(meta.get("signal_time"))).tz_localize(None)
    return ts < pd.Timestamp(TRAIN_WINDOW[2])


def _switch_utility(row: Mapping[str, Any]) -> float:
    meta = _metadata(row)
    utilities = meta.get("residual_utilities")
    if not isinstance(utilities, Mapping) or "SWITCH" not in utilities:
        raise ValueError("lifecycle row missing residual_utilities.SWITCH")
    value = float(utilities["SWITCH"])
    if not math.isfinite(value):
        raise ValueError("residual_utilities.SWITCH must be finite")
    return value


def _median_nonzero_abs_switch_utility(rows: Sequence[dict[str, Any]]) -> float | None:
    values = [abs(_switch_utility(row)) for row in rows if _switch_utility(row) != 0.0]
    if not values:
        return None
    return float(np.median(values))


def validate_frozen_lifecycle_rows(rows: Sequence[dict[str, Any]], cfg: Config, input_sha256: str) -> dict[str, Any]:
    """Validate the frozen 204-row train lifecycle pair contract."""

    if cfg.expected_input_sha256 and input_sha256 != cfg.expected_input_sha256:
        raise ValueError(
            f"frozen lifecycle input sha mismatch: expected {cfg.expected_input_sha256}, observed {input_sha256}"
        )
    if len(rows) != int(cfg.expected_rows):
        raise ValueError(f"expected {cfg.expected_rows} frozen lifecycle rows, observed {len(rows)}")

    identities: list[str] = []
    by_base: dict[str, list[dict[str, Any]]] = defaultdict(list)
    candidate_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()

    for row in rows:
        meta = _metadata(row)
        identity = str(meta.get("identity", ""))
        if not identity:
            raise ValueError("every lifecycle row must include metadata.identity")
        identities.append(identity)
        if row.get("split") != "train" or meta.get("window") != TRAIN_WINDOW[0]:
            raise ValueError("frozen lifecycle rows must be train/pre_2024 only")
        if not _signal_time_pre2024(meta):
            raise ValueError("frozen lifecycle signal_time must be before 2024-01-01")
        if meta.get("reference_anchor") is not True:
            raise ValueError("frozen lifecycle row must be a reference_anchor")
        if meta.get("offline_label_only") is not True:
            raise ValueError("frozen lifecycle row must be marked offline_label_only")
        if meta.get("future_outcome_present_in_prompt") is not False:
            raise ValueError("prompt/outcome leak guard changed")
        candidate = str(meta.get("candidate_action"))
        if candidate not in CANDIDATE_ACTIONS:
            raise ValueError("candidate_action must be SKIP or TP12")
        signal_position = int(meta.get("signal_position"))
        expected_identity = lifecycle.lifecycle_identity(TRAIN_WINDOW[0], signal_position, candidate)
        if identity != expected_identity:
            raise ValueError("metadata.identity does not match candidate/window/signal_position")
        base_identity = str(meta.get("base_identity", ""))
        if base_identity != counterfactual.signal_identity(TRAIN_WINDOW[0], signal_position):
            raise ValueError("metadata.base_identity does not match window/signal_position")
        target = str(row.get("target"))
        if target not in LABELS:
            raise ValueError("target must be KEEP or SWITCH")
        _switch_utility(row)
        by_base[base_identity].append(dict(row))
        candidate_counts[candidate] += 1
        target_counts[target] += 1

    if len(set(identities)) != len(identities):
        raise ValueError("duplicate frozen lifecycle identity")
    identity_sha256 = _sha256_bytes("\n".join(identities).encode("utf-8"))
    if cfg.expected_identity_sha256 and identity_sha256 != cfg.expected_identity_sha256:
        raise ValueError(
            f"frozen lifecycle identity sha mismatch: expected {cfg.expected_identity_sha256}, observed {identity_sha256}"
        )
    for candidate in CANDIDATE_ACTIONS:
        observed = candidate_counts.get(candidate, 0)
        if observed != int(cfg.expected_rows_per_candidate):
            raise ValueError(
                f"expected {cfg.expected_rows_per_candidate} {candidate} rows, observed {observed}"
            )
    for base_identity, pair in by_base.items():
        candidates = sorted(str(_metadata(row)["candidate_action"]) for row in pair)
        if candidates != sorted(CANDIDATE_ACTIONS):
            raise ValueError(f"base identity {base_identity} lacks exactly one SKIP and TP12 pair")
        signal_positions = {int(_metadata(row)["signal_position"]) for row in pair}
        signal_times = {str(_metadata(row)["signal_time"]) for row in pair}
        if len(signal_positions) != 1 or len(signal_times) != 1:
            raise ValueError(f"base identity {base_identity} pair metadata diverged")

    expected_pairs = int(cfg.expected_rows) // len(CANDIDATE_ACTIONS)
    if len(by_base) != expected_pairs:
        raise ValueError(f"expected {expected_pairs} candidate pairs, observed {len(by_base)}")

    return {
        "input_sha256": input_sha256,
        "identity_sha256": identity_sha256,
        "rows": len(rows),
        "reference_anchors": len(by_base),
        "reference_anchor_pairs": len(rows),
        "candidate_counts": dict(sorted(candidate_counts.items())),
        "target_counts": dict(sorted(target_counts.items())),
    }


def _write_jsonl(path: str | Path, rows: Sequence[dict[str, Any]]) -> str:
    data = _jsonl_bytes(rows)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return _sha256_bytes(data)


def _candidate_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "target_counts": dict(sorted(Counter(str(row["target"]) for row in rows).items())),
        "identity_sha256": _identity_sha256(rows),
        "median_nonzero_absolute_switch_utility": _median_nonzero_abs_switch_utility(rows),
    }


def split_lifecycle_specialists(cfg: Config) -> dict[str, Any]:
    rows, input_sha = _load_jsonl_with_sha(cfg.input_jsonl)
    validation = validate_frozen_lifecycle_rows(rows, cfg, input_sha)
    by_candidate = {
        candidate: [row for row in rows if _metadata(row)["candidate_action"] == candidate]
        for candidate in CANDIDATE_ACTIONS
    }
    output_paths = {"SKIP": cfg.skip_output, "TP12": cfg.tp12_output}
    output_hashes = {
        candidate: _write_jsonl(output_paths[candidate], candidate_rows)
        for candidate, candidate_rows in by_candidate.items()
    }
    summary = {
        "protocol": "pposm_train_only_lifecycle_specialist_split_v1",
        "config": {key: str(value) for key, value in asdict(cfg).items()},
        "input": {
            "path": str(cfg.input_jsonl),
            **validation,
        },
        "outputs": {
            candidate: {
                "path": str(output_paths[candidate]),
                "sha256": output_hashes[candidate],
                **_candidate_summary(candidate_rows),
            }
            for candidate, candidate_rows in by_candidate.items()
        },
        "candidate_actions": list(CANDIDATE_ACTIONS),
        "labels": list(LABELS),
        "split": "train",
        "window": {
            "name": TRAIN_WINDOW[0],
            "start": TRAIN_WINDOW[1],
            "end_exclusive": TRAIN_WINDOW[2],
        },
        "rows": {
            "input": len(rows),
            "SKIP": len(by_candidate["SKIP"]),
            "TP12": len(by_candidate["TP12"]),
            "oos": 0,
        },
        "target_counts": dict(sorted(Counter(str(row["target"]) for row in rows).items())),
        "candidate_target_counts": dict(
            sorted(
                Counter(
                    f"{_metadata(row)['candidate_action']}={row['target']}" for row in rows
                ).items()
            )
        ),
        "identity_sha256": validation["identity_sha256"],
        "median_nonzero_absolute_switch_utility": _median_nonzero_abs_switch_utility(rows),
        "leakage_guard": {
            "train_only": True,
            "oos_surface_written": False,
            "input_sha_fail_closed": True,
            "identity_order_fail_closed": True,
            "reference_anchor_pair_contract_fail_closed": True,
        },
    }
    cfg.summary_output.parent.mkdir(parents=True, exist_ok=True)
    cfg.summary_output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--skip-output", type=Path, default=DEFAULT_SKIP_OUTPUT)
    parser.add_argument("--tp12-output", type=Path, default=DEFAULT_TP12_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--expected-input-sha256", default=DEFAULT_EXPECTED_INPUT_SHA256)
    parser.add_argument("--expected-identity-sha256", default=DEFAULT_EXPECTED_IDENTITY_SHA256)
    parser.add_argument("--expected-rows", type=int, default=DEFAULT_EXPECTED_ROWS)
    parser.add_argument("--expected-rows-per-candidate", type=int, default=DEFAULT_EXPECTED_ROWS_PER_CANDIDATE)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(split_lifecycle_specialists(Config(**vars(parse_args()))), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
