from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from training import (
    probe_protocol_specification_intent_maturity_d3_transport as probe,
)

DECISION_PATH = Path(
    "docs/post-psim-d2-alpha-mechanism-audit-2026-07-25.md"
)
RESULT_PATH = Path(
    "results/protocol_specification_intent_maturity_d3_transport_probe_"
    "2026-07-25.json"
)


@pytest.fixture(scope="module")
def payload(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    root = tmp_path_factory.mktemp("psim-d3-probe") / "root"
    return probe.build_probe(root)


def test_git_binary_and_transport_are_exactly_bound() -> None:
    assert probe.GIT_VERSION == "git version 2.43.0"
    assert probe.GIT_BINARY == Path("/usr/bin/git")
    assert probe.sha256_file(probe.GIT_BINARY) == probe.GIT_BINARY_SHA256
    assert probe.BULK_FETCH_ARGUMENTS == (
        "-c",
        "fetch.negotiationAlgorithm=noop",
        "fetch",
        "origin",
        "--no-tags",
        "--no-write-fetch-head",
        "--recurse-submodules=no",
        "--filter=blob:none",
        "--no-auto-maintenance",
        "--stdin",
    )


def test_single_bulk_fetch_hydrates_only_requested_blobs(
    payload: dict[str, object],
) -> None:
    bulk = payload["bulk_fetch_probe"]
    assert isinstance(bulk, dict)
    assert bulk["fetch_invocations"] == 1
    assert bulk["command"][0] == "/usr/bin/git"
    assert bulk["requested_blob_count"] == 6
    assert bulk["pack_delta"] == 1
    assert bulk["promisor_pack_delta"] == 1
    assert bulk["new_pack_object_count"] == 6
    assert bulk["new_loose_object_count"] == 0
    assert bulk["new_pack_exact_requested_blobs"] is True
    assert bulk["new_total_object_store_exact_requested_blobs"] is True
    assert bulk["nonrequested_blob_present"] is False
    assert bulk["fetch_head_absent"] is True
    assert bulk["refs_unchanged"] is True
    assert bulk["maintenance_child_processes"] == 0


def test_post_hydration_read_is_network_and_pack_invariant(
    payload: dict[str, object],
) -> None:
    bulk = payload["bulk_fetch_probe"]
    no_lazy = payload["no_lazy_fetch_probe"]
    assert isinstance(bulk, dict)
    assert isinstance(no_lazy, dict)
    assert bulk["post_hydration_fetch_child_processes"] == 0
    assert bulk["post_hydration_pack_delta"] == 0
    assert bulk["post_hydration_object_store_unchanged"] is True
    assert no_lazy == {
        "environment": "GIT_NO_LAZY_FETCH=1",
        "fetch_child_processes": 0,
        "missing_object_exit_nonzero": True,
        "pack_delta": 0,
        "semantic_probe_passed": True,
        "upstream_version_contract": (
            "behaviorally_bound_local_backport_not_upstream_2.43_cli"
        ),
    }


def test_cat_file_buffering_is_not_network_batching(
    payload: dict[str, object],
) -> None:
    control = payload["buffered_cat_file_control"]
    assert isinstance(control, dict)
    assert control == {
        "command": "git cat-file --batch-command --buffer",
        "interpretation": (
            "stdout_buffering_only_not_network_want_batching"
        ),
        "promisor_pack_delta": 6,
        "requested_blob_count": 6,
    }


def test_transport_delta_accepts_multiple_promisor_packs() -> None:
    result = probe._validate_hydration_delta(
        before_packs=("pack-base.pack",),
        after_packs=("pack-a.pack", "pack-b.pack", "pack-base.pack"),
        before_promisors=("pack-base.promisor",),
        after_promisors=(
            "pack-a.promisor",
            "pack-b.promisor",
            "pack-base.promisor",
        ),
        before_loose=(),
        after_loose=(),
        before_objects={"0" * 40: "commit"},
        after_objects={
            "0" * 40: "commit",
            "1" * 40: "blob",
            "2" * 40: "blob",
        },
        new_pack_objects={
            "pack-a.pack": {"1" * 40: "blob"},
            "pack-b.pack": {"2" * 40: "blob"},
        },
        requested=("1" * 40, "2" * 40),
    )
    assert result["new_pack_count"] == 2
    assert result["new_promisor_count"] == 2
    assert result["new_total_object_count"] == 2


@pytest.mark.parametrize(
    ("after_objects", "new_pack_objects", "after_promisors", "after_loose"),
    [
        (
            {"0" * 40: "commit", "1" * 40: "blob", "3" * 40: "blob"},
            {"pack-a.pack": {"1" * 40: "blob", "3" * 40: "blob"}},
            ("pack-a.promisor", "pack-base.promisor"),
            (),
        ),
        (
            {"0" * 40: "commit"},
            {"pack-a.pack": {}},
            ("pack-a.promisor", "pack-base.promisor"),
            (),
        ),
        (
            {"0" * 40: "commit", "1" * 40: "blob"},
            {"pack-a.pack": {"1" * 40: "blob"}},
            ("pack-base.promisor",),
            (),
        ),
        (
            {"0" * 40: "commit", "1" * 40: "blob"},
            {"pack-a.pack": {"1" * 40: "blob"}},
            ("pack-a.promisor", "pack-base.promisor"),
            ("11/" + "1" * 38,),
        ),
    ],
)
def test_transport_delta_fails_closed_on_boundary_changes(
    after_objects: dict[str, str],
    new_pack_objects: dict[str, dict[str, str]],
    after_promisors: tuple[str, ...],
    after_loose: tuple[str, ...],
) -> None:
    with pytest.raises(RuntimeError):
        probe._validate_hydration_delta(
            before_packs=("pack-base.pack",),
            after_packs=("pack-a.pack", "pack-base.pack"),
            before_promisors=("pack-base.promisor",),
            after_promisors=after_promisors,
            before_loose=(),
            after_loose=after_loose,
            before_objects={"0" * 40: "commit"},
            after_objects=after_objects,
            new_pack_objects=new_pack_objects,
            requested=("1" * 40,),
        )


def test_trace_child_ambiguity_fails_closed(tmp_path: Path) -> None:
    trace = tmp_path / "trace.json"
    trace.write_text(
        json.dumps({"event": "child_start", "child_id": 1}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="trace child argv is ambiguous"):
        probe._trace_child_arguments(trace)


def test_child_classification_detects_maintenance_and_fetch() -> None:
    assert probe._is_maintenance_child(
        ("git", "maintenance", "run", "--auto")
    )
    assert probe._is_maintenance_child(("git-gc", "--auto"))
    assert probe._is_fetch_child(("git", "fetch", "origin"))
    assert probe._is_fetch_child(("git-remote-https", "origin", "url"))
    assert not probe._is_fetch_child(("git", "cat-file", "--batch"))


def test_post_read_snapshot_mutation_fails_closed() -> None:
    before = {
        "fetch_head_absent": True,
        "loose_objects": [],
        "objects": {"1" * 40: "blob"},
        "packs": ["pack-a.pack"],
        "promisors": ["pack-a.promisor"],
        "refs": ["refs/heads/master " + "0" * 40],
    }
    probe._assert_post_read_invariant(before, dict(before))
    for field, changed in (
        ("fetch_head_absent", False),
        ("loose_objects", ["11/" + "1" * 38]),
        ("packs", ["pack-a.pack", "pack-b.pack"]),
        ("promisors", []),
        ("refs", ["refs/heads/master " + "2" * 40]),
    ):
        after = dict(before)
        after[field] = changed
        with pytest.raises(
            RuntimeError,
            match="post-hydration object store changed",
        ):
            probe._assert_post_read_invariant(before, after)


def test_subprocess_run_drains_stderr_without_deadlock() -> None:
    completed = probe._run(
        [
            sys.executable,
            "-c",
            (
                "import sys;"
                "sys.stderr.buffer.write(b'x' * 2_000_000);"
                "sys.stdout.write('ok')"
            ),
        ]
    )
    assert completed.stdout == b"ok"
    assert len(completed.stderr) == 2_000_000


def test_probe_is_outcome_blind_and_hash_complete(
    payload: dict[str, object],
) -> None:
    assert payload["synthetic_only"] is True
    assert payload["access_boundary"] == {
        "official_eip_bip_source_accessed": False,
        "market_data_accessed": False,
        "model_accessed": False,
        "outcomes_accessed": False,
    }
    assert payload["synthetic_remote_contract"] == {
        "transport": "file",
        "uploadpack_allow_any_sha1_in_want": True,
        "uploadpack_allow_filter": True,
    }
    unhashed = dict(payload)
    result_hash = unhashed.pop("result_hash")
    assert result_hash == probe.canonical_hash(unhashed)


def test_written_probe_is_canonical_and_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, object],
) -> None:
    target = tmp_path / "probe.json"
    monkeypatch.setattr(probe, "run_probe", lambda: payload)
    first = probe.write_probe(target)
    second = probe.write_probe(target)
    assert first == second == payload
    assert target.read_bytes() == probe.canonical_json_bytes(payload)
    assert json.loads(target.read_text(encoding="utf-8")) == payload


def test_existing_probe_mismatch_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, object],
) -> None:
    target = tmp_path / "probe.json"
    target.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(probe, "run_probe", lambda: payload)
    with pytest.raises(
        RuntimeError,
        match="existing PSIM-D3 transport probe differs",
    ):
        probe.write_probe(target)


def test_decision_document_preserves_terminal_and_outcome_boundaries() -> None:
    text = DECISION_PATH.read_text(encoding="utf-8")
    assert "PSIM-D3" in text
    assert "targeted batch-hydration bare replay" in text
    assert "REJECT_PSIM_D2_UNCHANGED_BEFORE_MARKET_MODEL_OR_OUTCOMES" in text
    assert (
        "b8134ab47a1c69916593d1092b9125e0a8a78da11cf3080660064b12a2e6387c"
        in text
    )
    assert "/tmp/psim-d3-source/ethereum-a.git" in text
    assert "One explicit batch fetch is allowed per replica." in text
    assert "one-pack observation is synthetic evidence" in text
    assert "Full clone or `--refetch` is" in text
    assert "also forbidden because either could hydrate" in text
    assert "https://git-scm.com/docs/git-fetch/2.43.0.html" in text
    assert (
        "https://github.com/git/git/blob/v2.43.0/promisor-remote.c#L17-L45"
        in text
    )
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    assert result["result_hash"] in text
    assert probe.sha256_file(RESULT_PATH) in text
