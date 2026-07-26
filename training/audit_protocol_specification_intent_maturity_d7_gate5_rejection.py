#!/usr/bin/env python3
"""Read-only post-terminal cardinality audit for the PSIM-D7 rejection."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from training import (
    build_protocol_specification_intent_maturity_d7_source_support as runner,
)


PROTOCOL_VERSION = "psim_d7_gate5_post_terminal_forensic_v1"
TERMINAL_RESULT_HASH = (
    "45846070617398860a03f5a401047c95a37c7ba3526c37fbcea5a11687e8658b"
)
TERMINAL_SHA256 = (
    "36702b4737f1bb37e901241a96e04f30e77132bb6a18ade1fab277a83f15557e"
)
DEFAULT_OUTPUT = Path(
    "results/protocol_specification_intent_maturity_d7_gate5_forensic_"
    "2026-07-27.json"
)
EXPECTED_EXCEPTION = {
    "type": "ValueError",
    "message": "PSIM-D7 relation card exceeds frozen event bound",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_manifest(root: Path) -> str:
    rows: list[list[Any]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            rows.append(["symlink", relative, os.readlink(path)])
        elif path.is_file():
            rows.append(
                ["file", relative, path.stat().st_size, _sha256_file(path)]
            )
        elif path.is_dir():
            rows.append(["directory", relative])
        else:
            rows.append(["other", relative])
    return runner.canonical_hash(rows)


def relation_unit_count(
    ethereum_events: int,
    bitcoin_events: int,
) -> int:
    if ethereum_events < 0 or bitcoin_events < 0:
        raise ValueError("PSIM-D7 forensic event count is negative")
    if ethereum_events and bitcoin_events:
        return ethereum_events * bitcoin_events
    if ethereum_events or bitcoin_events:
        return ethereum_events + bitcoin_events
    return 1


def cardinality_census(
    events: Sequence[runner.ProposalEvent],
) -> dict[str, Any]:
    visible = runner._model_visible_events(events)
    rows: list[dict[str, Any]] = []
    for schedule_row in runner.core.prereg.ARCHIVE_SCHEDULES:
        schedule = schedule_row.name
        by_day: dict[Any, list[runner.ProposalEvent]] = defaultdict(list)
        for event in visible:
            by_day[event.available_at[schedule].date()].append(event)
        for decision_at in runner._decision_times():
            day_events = by_day.get(decision_at.date(), [])
            ethereum_events = sum(
                event.protocol == "ethereum" for event in day_events
            )
            bitcoin_events = sum(
                event.protocol == "bitcoin" for event in day_events
            )
            rows.append(
                {
                    "schedule": schedule,
                    "decision_day": decision_at.date().isoformat(),
                    "ethereum_events": ethereum_events,
                    "bitcoin_events": bitcoin_events,
                    "event_count": len(day_events),
                    "relation_units": relation_unit_count(
                        ethereum_events,
                        bitcoin_events,
                    ),
                }
            )

    limit = runner.core.prereg.MAX_MODEL_EVENTS_PER_CARD
    overflow = [row for row in rows if row["relation_units"] > limit]
    top = sorted(
        rows,
        key=lambda row: (
            -row["relation_units"],
            row["schedule"],
            row["decision_day"],
        ),
    )
    by_schedule = Counter(row["schedule"] for row in overflow)
    return {
        "maximum_model_events_per_card": limit,
        "card_cells_total": len(rows),
        "overflow_card_cells": len(overflow),
        "overflow_card_cells_by_schedule": dict(sorted(by_schedule.items())),
        "first_overflow": overflow[0] if overflow else None,
        "maximum_cardinality": top[0],
        "top_cardinalities": top[:20],
    }


def _load_events(
    source_root: Path,
    ledger: runner.AccessLedger,
) -> tuple[list[runner.ProposalEvent], dict[str, Any]]:
    events: list[runner.ProposalEvent] = []
    summaries: dict[str, Any] = {}
    for protocol in ("ethereum", "bitcoin"):
        repository = source_root / f"{protocol}-a.git"
        records = runner.collect_commit_chain(repository, protocol, ledger)
        groups, issues = runner.collect_proposal_groups(
            repository,
            records,
            ledger,
        )
        if issues:
            raise RuntimeError(
                f"PSIM-D7 forensic proposal-group issues: {protocol}"
            )
        object_ids = sorted(
            {
                oid
                for group in groups
                for oid in (group.old_blob_oid, group.new_blob_oid)
                if oid is not None
            }
        )
        raw_by_oid = dict(
            runner._cat_file_batch_local(
                repository,
                object_ids,
                expected_type="blob",
                ledger=ledger,
            )
        )
        protocol_events = runner._materialize_events_from_raw(
            groups,
            raw_by_oid,
            ledger,
        )
        events.extend(protocol_events)
        summaries[protocol] = {
            "commit_records": len(records),
            "proposal_groups": len(groups),
            "blob_objects": len(object_ids),
            "events": len(protocol_events),
            "model_visible_events": sum(
                not event.administrative_quarantined
                and event.model_visibility == "MODEL_VISIBLE"
                for event in protocol_events
            ),
            "administrative_events": sum(
                event.administrative_quarantined
                for event in protocol_events
            ),
            "semantic_outcomes": dict(
                sorted(
                    Counter(
                        event.semantic_outcome_id
                        for event in protocol_events
                    ).items()
                )
            ),
        }
    return events, summaries


def run_audit(
    *,
    source_root: Path = runner.DEFAULT_SOURCE_ROOT,
    output_path: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    if source_root != runner.DEFAULT_SOURCE_ROOT.resolve():
        raise ValueError("PSIM-D7 forensic source root is frozen")
    terminal_path = runner.REPO_ROOT / runner.DEFAULT_REJECTION_PATH
    terminal_payload = json.loads(terminal_path.read_text(encoding="utf-8"))
    if (
        terminal_payload.get("result_hash") != TERMINAL_RESULT_HASH
        or _sha256_file(terminal_path) != TERMINAL_SHA256
    ):
        raise RuntimeError("PSIM-D7 terminal rejection authority changed")

    source_before = _tree_manifest(source_root)
    terminal_before = _sha256_file(terminal_path)
    ledger = runner.AccessLedger()
    events, protocol_summaries = _load_events(source_root, ledger)
    census = cardinality_census(events)

    card_ledger = runner.AccessLedger()
    failure: dict[str, str] | None = None
    try:
        runner.build_daily_cards(events, ledger=card_ledger)
    except Exception as error:  # Exact exception is part of the audit result.
        failure = {
            "type": type(error).__name__,
            "message": str(error),
        }
    if failure != EXPECTED_EXCEPTION:
        raise RuntimeError("PSIM-D7 Gate-5 forensic exception changed")

    source_after = _tree_manifest(source_root)
    terminal_after = _sha256_file(terminal_path)
    if source_before != source_after or terminal_before != terminal_after:
        raise RuntimeError("PSIM-D7 forensic audit mutated terminal evidence")
    if ledger.network_commands != 0:
        raise RuntimeError("PSIM-D7 forensic audit used the network")

    visible_events = runner._model_visible_events(events)
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "authority": {
            "terminal_path": runner.DEFAULT_REJECTION_PATH.as_posix(),
            "terminal_result_hash": TERMINAL_RESULT_HASH,
            "terminal_sha256": TERMINAL_SHA256,
            "source_root": str(source_root),
        },
        "boundary": {
            "official_run_reexecuted": False,
            "market_model_or_outcomes_accessed": False,
            "source_root_repaired_or_reused_for_candidate": False,
            "network_commands": ledger.network_commands,
        },
        "counts": {
            "events": len(events),
            "model_visible_events": len(visible_events),
            "administrative_events": len(events) - len(visible_events),
            "daily_cards_completed": card_ledger.daily_cards_built,
        },
        "protocol_summaries": protocol_summaries,
        "cardinality": census,
        "failure": failure,
        "source_replay": {
            "git_commands": ledger.git_commands,
            "network_commands": ledger.network_commands,
        },
        "integrity": {
            "source_tree_manifest_before": source_before,
            "source_tree_manifest_after": source_after,
            "source_tree_unchanged": source_before == source_after,
            "terminal_sha256_before": terminal_before,
            "terminal_sha256_after": terminal_after,
            "terminal_artifact_unchanged": terminal_before == terminal_after,
        },
    }
    result = {**core, "result_hash": runner.canonical_hash(core)}
    runner._write_once_bytes(output_path, runner.canonical_json_bytes(result))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=runner.DEFAULT_SOURCE_ROOT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    result = run_audit(
        source_root=arguments.source_root,
        output_path=arguments.output,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
