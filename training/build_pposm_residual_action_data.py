"""Build pairwise residual PPOSM action data around an ALWAYS-TP4 default.

Each frozen PPOSM signal becomes two causal comparison rows: candidate SKIP vs
TP4 and candidate TP12 vs TP4.  Future execution utilities are label/reward
metadata only and never appear in the prompt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from training import build_pposm_counterfactual_action_data as counterfactual

DEFAULT_TRAIN_OUTPUT = Path(
    "data/pposm_residual_action_train_pre2024_2026-09-02.jsonl"
)
DEFAULT_OOS_OUTPUT = Path(
    "data/pposm_residual_action_oos_2024_2026_2026-09-02.jsonl"
)
DEFAULT_SUMMARY_OUTPUT = Path("results/pposm_residual_action_data_summary_2026-09-02.json")
DEFAULT_CANDIDATES = ("SKIP", "TP12")
CANDIDATE_ACTIONS = DEFAULT_CANDIDATES
DEFAULT_ACTION = "TP4"
LABELS = ("KEEP", "SWITCH")


@dataclass(frozen=True)
class Config:
    manifest: Path = counterfactual.DEFAULT_MANIFEST
    train_output: Path = DEFAULT_TRAIN_OUTPUT
    oos_output: Path = DEFAULT_OOS_OUTPUT
    summary_output: Path = DEFAULT_SUMMARY_OUTPUT


def canonical_json(value: Any) -> str:
    return counterfactual.canonical_json(value)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def residual_identity(base_identity: str, candidate: str) -> str:
    if candidate not in DEFAULT_CANDIDATES:
        raise ValueError(f"candidate must be one of {DEFAULT_CANDIDATES}")
    return f"pposm-residual-action|{candidate}|{base_identity}"


def residual_prompt(base_prompt: str, *, candidate: str) -> str:
    if candidate not in DEFAULT_CANDIDATES:
        raise ValueError(f"candidate must be one of {DEFAULT_CANDIDATES}")
    causal_lines = [
        line
        for line in str(base_prompt).splitlines()
        if line.startswith(("causal_predicates:", "signal_time_state:", "predicate_priority:"))
    ]
    if not causal_lines:
        raise ValueError("base prompt does not expose causal signal context")
    return "\n".join(
        (
            "Frozen PPOSM residual action router.",
            "Default action is TP4. Decide whether to keep TP4 or switch to the candidate.",
            "Return exactly one token: KEEP or SWITCH.",
            f"candidate_action: {candidate}",
            f"default_action: {DEFAULT_ACTION}",
            "causal_context_begin",
            *causal_lines,
            "causal_context_end",
        )
    )

def residual_label(values: dict[str, float]) -> str:
    if set(values) != set(LABELS):
        raise ValueError("residual utilities must contain KEEP and SWITCH")
    return "SWITCH" if float(values["SWITCH"]) > float(values["KEEP"]) else "KEEP"
def build_pair_rows(base_row: dict[str, Any] | None = None, **kwargs: Any) -> list[dict[str, Any]]:
    if base_row is None:
        base_row = counterfactual.build_row(**kwargs)
    metadata = base_row.get("metadata") if isinstance(base_row.get("metadata"), dict) else {}
    utilities = metadata.get("action_utilities")
    if not isinstance(utilities, dict) or DEFAULT_ACTION not in utilities:
        raise ValueError("counterfactual row lacks action_utilities with TP4")
    base_identity = metadata.get("identity")
    if not isinstance(base_identity, str) or not base_identity:
        raise ValueError("counterfactual row lacks metadata.identity")
    default_utility = float(utilities[DEFAULT_ACTION])
    rows: list[dict[str, Any]] = []
    for candidate in DEFAULT_CANDIDATES:
        residual = float(utilities[candidate]) - default_utility
        target = residual_label({"KEEP": 0.0, "SWITCH": residual})
        pair_meta = {
            "identity": residual_identity(base_identity, candidate),
            "base_identity": base_identity,
            "window": metadata.get("window"),
            "signal_position": int(metadata["signal_position"]),
            "signal_time": metadata.get("signal_time"),
            "default_action": DEFAULT_ACTION,
            "base_action": DEFAULT_ACTION,
            "candidate_action": candidate,
            "residual_advantage": residual,
            "residual_utilities": {"KEEP": 0.0, "SWITCH": residual},
            "action_utilities": utilities,
            "action_take_profit_bps": metadata.get("action_take_profit_bps"),
            "executable_positions": metadata.get("executable_positions"),
            "entry_rule": metadata.get("entry_rule", "next_5m_open"),
            "utility_source": "candidate_minus_tp4_from_offline_exact_base_cost_execution",
            "offline_label_only": bool(metadata.get("offline_label_only", False)),
            "future_outcome_present_in_prompt": False,
        }
        rows.append(
            {
                "task": "pposm_residual_action",
                "split": base_row.get("split"),
                "prompt": residual_prompt(str(base_row["prompt"]), candidate=candidate),
                "target": target,
                "metadata": pair_meta,
            }
        )
    return rows


def rows_from_counterfactual(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        out.extend(build_pair_rows(row))
    try:
        validate_pair_rows(out)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    return out

def validate_pair_rows(rows: Sequence[dict[str, Any]]) -> None:
    identities = [str(row["metadata"]["identity"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate residual pair identity")
    by_base: dict[str, list[str]] = {}
    for row in rows:
        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        by_base.setdefault(str(meta.get("base_identity")), []).append(str(meta.get("candidate_action")))
        if row.get("target") not in LABELS:
            raise ValueError("residual target must be KEEP or SWITCH")
    for base, candidates in by_base.items():
        if tuple(sorted(candidates)) != tuple(sorted(DEFAULT_CANDIDATES)):
            raise ValueError(f"base identity {base} does not have exactly SKIP and TP12 candidates")

def _jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join((canonical_json(row) + "\n").encode("utf-8") for row in rows)


def build(cfg: Config) -> dict[str, Any]:
    manifest, _ = counterfactual.frozen.load_frozen_manifest(cfg.manifest)
    cf_train, cf_oos = _build_counterfactual_rows(cfg.manifest)
    train = rows_from_counterfactual(cf_train)
    oos = rows_from_counterfactual(cf_oos)
    train_bytes, oos_bytes = _jsonl_bytes(train), _jsonl_bytes(oos)
    for path, payload in ((cfg.train_output, train_bytes), (cfg.oos_output, oos_bytes)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    all_rows = train + oos
    pair_rows_by_window = Counter(
        str(row["metadata"].get("window")) for row in all_rows
    )
    train_pre2024 = all(
        pd.Timestamp(row["metadata"]["signal_time"]).tz_localize(None)
        < pd.Timestamp("2024-01-01")
        for row in train
    )
    summary = {
        "protocol": "pposm_residual_action_pairwise_tp4_default_v1",
        "config": {key: str(value) for key, value in asdict(cfg).items()},
        "manifest_freeze_hash": manifest["freeze_hash"],
        "default_action": DEFAULT_ACTION,
        "candidate_actions": list(DEFAULT_CANDIDATES),
        "labels": list(LABELS),
        "rows": {"train": len(train), "oos": len(oos), "total": len(all_rows)},
        "signals": {"train": len(train)//2, "oos": len(oos)//2, "total": len(all_rows)//2},
        "pair_rows_by_window": dict(sorted(pair_rows_by_window.items())),
        "signals_by_window": {
            window: count // len(CANDIDATE_ACTIONS)
            for window, count in sorted(pair_rows_by_window.items())
        },
        "targets": dict(sorted(Counter(row["target"] for row in all_rows).items())),
        "targets_by_split": {
            "train": dict(sorted(Counter(row["target"] for row in train).items())),
            "oos": dict(sorted(Counter(row["target"] for row in oos).items())),
        },
        "candidate_targets": dict(
            sorted(
                Counter(
                    f"{row['metadata']['candidate_action']}={row['target']}"
                    for row in all_rows
                ).items()
            )
        ),
        "output_sha256": {"train": _sha256(train_bytes), "oos": _sha256(oos_bytes)},
        "identity_sha256": _sha256("\n".join(row["metadata"]["identity"] for row in all_rows).encode("utf-8")),
        "identity_sha256_by_split": {
            "train": _sha256(
                "\n".join(row["metadata"]["identity"] for row in train).encode(
                    "utf-8"
                )
            ),
            "oos": _sha256(
                "\n".join(row["metadata"]["identity"] for row in oos).encode(
                    "utf-8"
                )
            ),
        },
        "causality": {
            "train_signal_time_pre2024_only": train_pre2024,
            "default_action_fixed_before_oos": DEFAULT_ACTION,
            "future_outcome_in_prompt": False,
            "oos_utilities_are_offline_labels_only": True,
        },
    }
    cfg.summary_output.parent.mkdir(parents=True, exist_ok=True)
    cfg.summary_output.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def _build_counterfactual_rows(manifest: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    def load(path: Path) -> list[dict[str, Any]]:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    with tempfile.TemporaryDirectory(prefix="pposm-residual-") as directory:
        root = Path(directory)
        train = root / "counterfactual_train.jsonl"
        oos = root / "counterfactual_oos.jsonl"
        counterfactual.build(
            counterfactual.Config(
                manifest=manifest,
                train_output=train,
                oos_output=oos,
                summary_output=root / "counterfactual_summary.json",
            )
        )
        return load(train), load(oos)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=counterfactual.DEFAULT_MANIFEST)
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument("--oos-output", type=Path, default=DEFAULT_OOS_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(Config(**vars(parse_args()))), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
