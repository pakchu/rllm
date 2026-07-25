from __future__ import annotations

import gzip
import json
from collections import OrderedDict
from pathlib import Path

import pandas as pd
import pytest

from training import bctp_schedule_seal as seal
from training import bctp_stage_sources as stages
from training import freeze_block_clearing_target_position_evaluator as freeze


def _rows(policy_id: str, stage: str, *, minute: int = 0, target: str = "TARGET_FLAT") -> list[dict[str, object]]:
    start = pd.Timestamp(f"{stage}-01-01T00:00:00Z") + pd.Timedelta(minutes=minute)
    return [
        {
            "policy_id": policy_id,
            "sequence_id": f"{policy_id}-{minute}",
            "entry_time": start,
            "target": target,
        },
        {
            "policy_id": policy_id,
            "sequence_id": f"{policy_id}-{minute + 5}",
            "entry_time": start + pd.Timedelta(minutes=5),
            "target": "TARGET_LONG",
        },
    ]


def _schedules(stage: str = "2021") -> tuple[OrderedDict[str, object], OrderedDict[str, object]]:
    base: OrderedDict[str, object] = OrderedDict(
        (policy_id, pd.DataFrame(_rows(policy_id, stage), columns=stages.SCHEDULE_COLUMNS))
        for policy_id in freeze.FAMILY_IDS
    )
    delayed: OrderedDict[str, object] = OrderedDict()
    for policy_id in stages.PROMOTABLE_PRIMARY_IDS:
        frame = base[policy_id].copy()
        frame["sequence_id"] = (
            frame["sequence_id"].astype(str) + ":delay_5m"
        )
        frame["entry_time"] = pd.to_datetime(
            frame["entry_time"],
            utc=True,
        ) + pd.Timedelta(minutes=5)
        delayed[policy_id] = frame
    return base, delayed


def _gunzip_bytes(path: str | Path) -> bytes:
    with gzip.open(path, "rb") as handle:
        return handle.read()


def test_deterministic_write_once_manifest_and_gzip(tmp_path: Path) -> None:
    base, delayed = _schedules("2021")
    first = seal.seal_transfer_year_schedule("2021", base, delayed, output_root=tmp_path / "a")
    second = seal.seal_transfer_year_schedule("2021", base, delayed, output_root=tmp_path / "b")
    again = seal.seal_transfer_year_schedule("2021", base, delayed, output_root=tmp_path / "a")

    assert Path(first["base_schedules"]["path"]).read_bytes() == Path(second["base_schedules"]["path"]).read_bytes()
    assert Path(first["delayed_primary_schedules"]["path"]).read_bytes() == Path(second["delayed_primary_schedules"]["path"]).read_bytes()
    assert _gunzip_bytes(first["base_schedules"]["path"]).startswith(b"policy_id,sequence_id,entry_time,target\n")
    assert first["manifest_hash"] == again["manifest_hash"]
    assert first["base_schedules"]["frame_hash"] == second["base_schedules"]["frame_hash"]
    assert first["strategy_outcomes_calculated"] is False
    assert first["outcome_payload_opened"] is False
    assert first["market_or_funding_payload_bytes_hashed"] is False
    assert first["stress_reuses_base_target_sequences"] is True


def test_drift_is_rejected_for_existing_schedule_bytes(tmp_path: Path) -> None:
    base, delayed = _schedules("2021")
    seal.seal_transfer_year_schedule("2021", base, delayed, output_root=tmp_path)
    bad = base.copy()
    first_policy = next(iter(bad))
    changed = bad[first_policy].copy()
    changed.loc[0, "target"] = "TARGET_SHORT"
    bad[first_policy] = changed

    with pytest.raises(RuntimeError, match="write-once.*drift"):
        seal.seal_transfer_year_schedule("2021", bad, delayed, output_root=tmp_path)


def test_malformed_and_out_of_stage_schedules_are_rejected(tmp_path: Path) -> None:
    base, delayed = _schedules("2021")
    first_policy = next(iter(base))
    malformed = base.copy()
    malformed[first_policy] = malformed[first_policy].drop(columns=["target"])
    with pytest.raises(ValueError, match="schema"):
        seal.seal_transfer_year_schedule("2021", malformed, delayed, output_root=tmp_path / "bad_schema")

    out_of_stage = base.copy()
    changed = out_of_stage[first_policy].copy()
    changed.loc[0, "entry_time"] = pd.Timestamp("2022-01-01T00:00:00Z")
    out_of_stage[first_policy] = changed
    with pytest.raises(ValueError, match="out-of-stage|order"):
        seal.seal_transfer_year_schedule("2021", out_of_stage, delayed, output_root=tmp_path / "bad_stage")

    naive = base.copy()
    changed = naive[first_policy].copy()
    naive_rows = changed.to_dict(orient="records")
    naive_rows[0]["entry_time"] = "2021-01-01 00:00:00"
    naive[first_policy] = naive_rows
    with pytest.raises(ValueError, match="timezone aware"):
        seal.seal_transfer_year_schedule("2021", naive, delayed, output_root=tmp_path / "bad_tz")

    with pytest.raises(ValueError, match="stage"):
        seal.seal_transfer_year_schedule("2020", base, delayed, output_root=tmp_path / "fit")  # type: ignore[arg-type]


def test_policy_order_and_delayed_primary_contract_are_enforced(tmp_path: Path) -> None:
    base, delayed = _schedules("2021")
    reversed_base = OrderedDict(reversed(list(base.items())))
    with pytest.raises(ValueError, match="policy order"):
        seal.seal_transfer_year_schedule("2021", reversed_base, delayed, output_root=tmp_path / "order")

    bad_delayed = OrderedDict(delayed)
    first_primary = next(iter(bad_delayed))
    changed = pd.DataFrame(bad_delayed[first_primary], columns=stages.SCHEDULE_COLUMNS)
    changed.loc[0, "entry_time"] = pd.Timestamp("2021-01-01T00:00:00Z")
    bad_delayed[first_primary] = changed
    with pytest.raises(ValueError, match=r"\+5m"):
        seal.seal_transfer_year_schedule("2021", base, bad_delayed, output_root=tmp_path / "delay")


def test_reload_validation_catches_manifest_and_artifact_drift(tmp_path: Path) -> None:
    base, delayed = _schedules("2022")
    manifest = seal.seal_transfer_year_schedule("2022", base, delayed, output_root=tmp_path)
    reloaded = seal.load_transfer_year_schedule_seal("2022", manifest_path=manifest["path"])
    stage_gate = stages._validate_required_schedule("2022", manifest["path"])
    assert stage_gate is not None
    assert stage_gate["manifest_hash"] == manifest["manifest_hash"]
    assert reloaded["manifest_hash"] == manifest["manifest_hash"]
    assert reloaded["evaluator_manifest_hash"] == stages.EXPECTED_EVALUATOR_MANIFEST_HASH
    assert reloaded["family_ids"] == list(freeze.FAMILY_IDS)
    assert reloaded["promotable_primary_ids"] == list(stages.PROMOTABLE_PRIMARY_IDS)

    path = Path(manifest["path"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["outcome_payload_opened"] = True
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="outcome payload|manifest hash"):
        seal.load_transfer_year_schedule_seal("2022", manifest_path=path)

    # Restore and corrupt the sealed CSV; reload must check gzip SHA/frame binding.
    seal.seal_transfer_year_schedule("2022", base, delayed, output_root=tmp_path / "clean")
    clean_manifest = tmp_path / "clean" / "2022" / seal.MANIFEST_FILENAME
    clean_payload = json.loads(clean_manifest.read_text(encoding="utf-8"))
    with gzip.open(clean_payload["base_schedules"]["path"], "at", encoding="utf-8") as handle:
        handle.write("always_flat,extra,2022-01-01T00:10:00Z,TARGET_FLAT\n")
    with pytest.raises(ValueError, match="hash mismatch|row count|timestamps"):
        seal.load_transfer_year_schedule_seal("2022", manifest_path=clean_manifest)
