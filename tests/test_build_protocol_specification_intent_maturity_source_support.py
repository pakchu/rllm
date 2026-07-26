from __future__ import annotations

import ast
import copy
import dataclasses
import json
import subprocess
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import pytest

import training.build_protocol_specification_intent_maturity_source_support as runner
import training.preregister_protocol_specification_intent_maturity as prereg

UTC = timezone.utc


def _eip_blob(number: int = 123, *, requires: str = "1, 2") -> bytes:
    return (
        f"---\neip: {number}\ntitle: Synthetic EIP\nrequires: {requires}\n---\n"
        "# Abstract\nSynthetic abstract.\n"
        "# Motivation\nSynthetic motivation.\n"
        "# Specification\nSynthetic specification.\n"
    ).encode("utf-8")


def _bip_blob(number: int = 123, *, requires: str = "1, 2") -> bytes:
    return (
        f"<pre>\n  BIP: {number}\n  Title: Synthetic BIP\n"
        f"  Requires: {requires}\n</pre>\n"
        "== Abstract ==\nSynthetic abstract.\n"
        "== Motivation ==\nSynthetic motivation.\n"
        "== Specification ==\nSynthetic specification.\n"
    ).encode("utf-8")


def _synthetic_cards_events():
    events = runner.synthetic_events()
    cards = runner.build_daily_cards(events)
    return events, cards


def _run(cmd: list[str], cwd: Path) -> str:
    completed = subprocess.run(
        cmd,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def _git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(["git", "init"], repo)
    _run(["git", "config", "user.email", "synthetic@example.test"], repo)
    _run(["git", "config", "user.name", "Synthetic"], repo)
    return repo


def _commit_all(repo: Path, message: str = "synthetic") -> str:
    _run(["git", "add", "."], repo)
    env_cmd = [
        "bash",
        "-lc",
        "GIT_AUTHOR_DATE='2020-01-10T00:00:00+0000' "
        "GIT_COMMITTER_DATE='2020-01-10T00:00:00+0000' "
        f"git commit -m {message!r} >/dev/null && git rev-parse HEAD",
    ]
    return _run(env_cmd, repo)


def _commit_all_at(repo: Path, timestamp: str, message: str) -> str:
    _run(["git", "add", "."], repo)
    return _run(
        [
            "bash",
            "-lc",
            f"GIT_AUTHOR_DATE={timestamp!r} "
            f"GIT_COMMITTER_DATE={timestamp!r} "
            f"git commit -m {message!r} >/dev/null && git rev-parse HEAD",
        ],
        repo,
    )


def test_self_check_is_synthetic_only_and_opens_no_git_network_or_source(monkeypatch):
    def fail_git(*args, **kwargs):  # pragma: no cover - only runs on regression
        raise AssertionError("self-check must not invoke git/source transport")

    monkeypatch.setattr(runner, "_run_git", fail_git)
    monkeypatch.setattr(runner, "_git_text", fail_git)
    monkeypatch.setattr(runner, "_cat_file_batch", fail_git)

    payload = runner.build_self_check_manifest()

    assert payload["failed"] == []
    assert payload["network_calls"] == 0
    assert payload["source_event_rows_opened"] == 0
    assert payload["outcomes_opened"] is False
    assert payload["forbidden_access"] == runner.AccessLedger.zero().snapshot()


def test_runner_has_no_model_market_or_network_imports():
    source = runner.RUNNER_PATH.read_text()
    tree = ast.parse(source)
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])

    forbidden_import_roots = {
        "ccxt",
        "downloader",
        "models",
        "pandas_datareader",
        "requests",
        "sklearn",
        "torch",
        "yfinance",
    }
    assert imported_roots.isdisjoint(forbidden_import_roots)


def test_authority_and_preregistration_bindings_match_frozen_constants():
    assert runner.POLICY_ID == "PSIM-D1"
    assert runner.RUNNER_PROTOCOL == "psim_d1_source_support_runner_v1"
    assert runner.SEAL_PROTOCOL == "psim_d1_source_support_execution_seal_v1"
    assert runner.RESULT_PROTOCOL == "psim_d1_source_support_result_v1"
    assert runner.PREREGISTRATION_SCRIPT_PATH == prereg.SCRIPT_PATH
    assert runner.PREREGISTRATION_PATH == prereg.DEFAULT_OUTPUT
    assert runner.DECISION_COMMIT == prereg.SELECTION_COMMIT
    assert runner.DECISION_SHA256 == prereg.DECISION_SHA256
    assert len(runner.PREREGISTRATION_COMMIT) == 40
    assert len(runner.PREREGISTRATION_SHA256) == 64
    assert len(runner.PREREGISTRATION_MANIFEST_HASH) == 64
    assert len(runner.PREREGISTRATION_SCRIPT_SHA256) == 64
    assert len(runner.PREREGISTRATION_DOC_SHA256) == 64


def test_commit_object_parser_accepts_monotone_commit_and_rejects_hash_mismatch():
    raw = runner._synthetic_commit_raw(
        tree_oid="1" * 40,
        parent_oid="2" * 40,
        epoch=1_577_923_200,
    )
    oid = runner.git_object_sha1("commit", raw)

    record = runner.parse_commit_object(
        "ethereum", oid, raw, 7, date(2020, 1, 3)
    )

    assert record.tree_oid == "1" * 40
    assert record.parent_oid == "2" * 40
    assert record.committer_day == date(2020, 1, 2)
    assert record.effective_day == date(2020, 1, 3)
    with pytest.raises(ValueError, match="commit object SHA-1 mismatch"):
        runner.parse_commit_object("ethereum", "0" * 40, raw, 0, None)


@pytest.mark.parametrize(
    "raw, message",
    [
        (b"tree " + b"1" * 40 + b"\n\nmissing committer\n", "no committer"),
        (
            b"tree " + b"1" * 39 + b"z\ncommitter A <a@b> 1 +0000\n\nx\n",
            "tree OID is malformed",
        ),
        (
            b"tree " + b"1" * 40 + b"\ncommitter A <a@b> nope +0000\n\nx\n",
            "committer identity time is malformed",
        ),
    ],
)
def test_commit_object_parser_rejects_malformed_commit_headers(raw, message):
    oid = runner.git_object_sha1("commit", raw)
    with pytest.raises(ValueError, match=message):
        runner.parse_commit_object("bitcoin", oid, raw, 0, None)


def test_raw_delta_parser_accepts_git_raw_z_and_rejects_duplicates_and_bad_grammar():
    row = (
        b":000000 100644 "
        + runner.ZERO_OID.encode("ascii")
        + b" "
        + b"2" * 40
        + b" A\x00EIPS/eip-123.md\x00"
    )
    parsed = runner.parse_raw_path_delta(row)
    assert parsed == [
        runner.PathChange(
            path="EIPS/eip-123.md",
            old_mode="000000",
            new_mode="100644",
            old_oid=runner.ZERO_OID,
            new_oid="2" * 40,
            status="A",
        )
    ]

    with pytest.raises(ValueError, match="not NUL terminated"):
        runner.parse_raw_path_delta(row.rstrip(b"\x00"))
    with pytest.raises(ValueError, match="repeats path"):
        runner.parse_raw_path_delta(row + row)
    bad_status = row.replace(b" A\x00", b" R\x00")
    with pytest.raises(ValueError, match="status is unsupported"):
        runner.parse_raw_path_delta(bad_status)


@pytest.mark.parametrize(
    "protocol,path,expected",
    [
        ("ethereum", "EIPS/eip-123.md", (123, "md")),
        ("bitcoin", "bip-0123.mediawiki", (123, "mediawiki")),
        ("bitcoin", "bip-0123.md", (123, "md")),
        ("ethereum", "eips/eip-123.md", None),
        ("bitcoin", "bip-123.mediawiki", None),
    ],
)
def test_path_identity_enforces_exact_eip_bip_path_grammar(protocol, path, expected):
    assert runner._path_identity(protocol, path) == expected


def test_blob_feature_parser_strictly_parses_eip_and_bip_and_rejects_mismatches():
    eip_raw = _eip_blob(123)
    eip_oid = runner.git_object_sha1("blob", eip_raw)
    eip = runner.parse_blob_features("ethereum", 123, eip_oid, eip_raw)
    assert eip.header["eip"] == "123"
    assert eip.dependency_edges["requires"] == (1, 2)
    assert {"ABSTRACT", "MOTIVATION", "SPECIFICATION"}.issubset(
        set(eip.section_presence)
    )

    bip_raw = _bip_blob(123)
    bip_oid = runner.git_object_sha1("blob", bip_raw)
    bip = runner.parse_blob_features("bitcoin", 123, bip_oid, bip_raw)
    assert bip.header["bip"] == "123"
    assert bip.dependency_edges["requires"] == (1, 2)
    assert {"ABSTRACT", "MOTIVATION", "SPECIFICATION"}.issubset(
        set(bip.section_presence)
    )

    with pytest.raises(ValueError, match="blob object SHA-1 mismatch"):
        runner.parse_blob_features("ethereum", 123, "0" * 40, eip_raw)
    with pytest.raises(ValueError, match="path number differs"):
        runner.parse_blob_features("bitcoin", 999, bip_oid, bip_raw)
    malformed_bip = b"\n\n\n\n<pre>\n  BIP: 123\n</pre>\n"
    with pytest.raises(ValueError, match="leading blank prefix is invalid"):
        runner.parse_blob_features(
            "bitcoin",
            123,
            runner.git_object_sha1("blob", malformed_bip),
            malformed_bip,
        )


def test_proposal_grouping_uses_fake_local_git_and_reports_duplicate_bip_paths(tmp_path):
    repo = _git_repo(tmp_path)
    (repo / "bip-0001.mediawiki").write_text(_bip_blob(1).decode(), encoding="utf-8")
    (repo / "bip-0001.md").write_text(_bip_blob(1).decode(), encoding="utf-8")
    oid = _commit_all(repo)
    record = runner.CommitRecord(
        protocol="bitcoin",
        oid=oid,
        tree_oid=_run(["git", "rev-parse", f"{oid}^{{tree}}"], repo),
        parent_oid=None,
        first_parent_index=0,
        committer_epoch=1_578_614_400,
        committer_day=date(2020, 1, 10),
        effective_day=date(2020, 1, 10),
    )

    groups, issues = runner.proposal_groups_for_commit(
        repo, record, runner.AccessLedger.zero()
    )

    assert groups == []
    assert any("duplicate_new_tree_paths" in issue for issue in issues)
    assert any("ambiguous_old_or_new_blob" in issue for issue in issues)


def test_proposal_grouping_materializes_single_local_eip_event(tmp_path):
    repo = _git_repo(tmp_path)
    (repo / "EIPS").mkdir()
    (repo / "EIPS" / "eip-123.md").write_text(
        _eip_blob(123).decode(), encoding="utf-8"
    )
    oid = _commit_all(repo)
    record = runner.CommitRecord(
        protocol="ethereum",
        oid=oid,
        tree_oid=_run(["git", "rev-parse", f"{oid}^{{tree}}"], repo),
        parent_oid=None,
        first_parent_index=0,
        committer_epoch=1_578_614_400,
        committer_day=date(2020, 1, 10),
        effective_day=date(2020, 1, 10),
    )
    ledger = runner.AccessLedger.zero()

    groups, issues = runner.proposal_groups_for_commit(repo, record, ledger)
    events = runner.materialize_events(repo, groups, ledger)

    assert issues == []
    assert len(groups) == 1
    assert groups[0].event_type == "CREATE"
    assert events[0].proposal_number == 123
    assert events[0].dependency_delta_state == "NO_PRIOR"
    assert events[0].changed_sections
    assert ledger.git_commands >= 3
    # materialize_events marks cat-file as network-capable because official
    # partial clones may hydrate blobs; self-check separately proves no network.


def test_first_in_window_update_parses_old_side_only_as_pre_window_baseline(
    tmp_path,
):
    repo = _git_repo(tmp_path)
    (repo / "EIPS").mkdir()
    path = repo / "EIPS" / "eip-123.md"
    path.write_text(
        _eip_blob(123).decode().replace(
            "Synthetic abstract.",
            "PREWINDOW BASELINE",
        ),
        encoding="utf-8",
    )
    parent = _commit_all_at(
        repo,
        "2019-12-31T00:00:00+0000",
        "pre-window",
    )
    path.write_text(
        _eip_blob(123).decode().replace(
            "Synthetic abstract.",
            "INWINDOW CHANGE",
        ),
        encoding="utf-8",
    )
    oid = _commit_all_at(
        repo,
        "2020-01-10T00:00:00+0000",
        "in-window",
    )
    raw = subprocess.run(
        ["git", "cat-file", "commit", oid],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout
    record = runner.parse_commit_object(
        "ethereum",
        oid,
        raw,
        1,
        date(2019, 12, 31),
    )
    assert record.parent_oid == parent
    ledger = runner.AccessLedger.zero()

    groups, issues = runner.proposal_groups_for_commit(repo, record, ledger)
    events = runner.materialize_events(repo, groups, ledger)

    assert issues == []
    assert len(groups) == len(events) == 1
    event = events[0]
    assert event.event_type == "UPDATE"
    assert event.window_revision_count == 0
    assert event.window_age_days == 0
    assert event.update_gap_days is None
    assert event.old_blob_role == "PRE_WINDOW_BASELINE"
    assert event.prior_dependency_state == "PRE_WINDOW_UNKNOWN"
    assert "ABSTRACT|REMOVE|PREWINDOW BASELINE" in event.intent_text
    assert "ABSTRACT|ADD|INWINDOW CHANGE" in event.intent_text
    assert ledger.proposal_blobs_opened == 2
    assert ledger.pre_2020_proposal_blobs_opened == 0


@pytest.mark.parametrize(
    "effective_day,counter_name,error_text",
    [
        (
            date(2019, 12, 31),
            "pre_2020_proposal_blobs_opened",
            "pre-2020",
        ),
        (
            date(2024, 1, 1),
            "post_2023_proposal_blobs_opened",
            "post-2023",
        ),
    ],
)
def test_materialize_events_rejects_out_of_window_groups_before_blob_access(
    monkeypatch,
    tmp_path,
    effective_day,
    counter_name,
    error_text,
):
    event = runner.synthetic_events()[0]
    group = runner.ProposalGroup(
        protocol=event.protocol,
        proposal_number=event.proposal_number,
        commit_oid=event.commit_oid,
        first_parent_index=event.first_parent_index,
        committer_day=effective_day,
        effective_day=effective_day,
        old_path=event.old_path,
        new_path=event.new_path,
        old_blob_oid=event.old_blob_oid,
        new_blob_oid=event.new_blob_oid,
        event_type=event.event_type,
        event_id=event.event_id,
    )
    monkeypatch.setattr(
        runner,
        "_cat_file_batch",
        lambda *args, **kwargs: pytest.fail(
            "out-of-window groups must fail before blob access"
        ),
    )
    ledger = runner.AccessLedger.zero()

    with pytest.raises(RuntimeError, match=error_text):
        runner.materialize_events(tmp_path, [group], ledger)

    assert getattr(ledger, counter_name) == 1
    assert ledger.proposal_blobs_opened == 0


def test_daily_cards_cover_all_four_schedules_pairing_states_reset_and_quarantine():
    events, cards = _synthetic_cards_events()
    gate = runner.gate_pairing_reset_quarantine(events, cards)

    assert gate.passed, gate.failure
    assert {card.schedule for card in cards} == {row.name for row in prereg.ARCHIVE_SCHEDULES}
    assert {row.name: row.delay_calendar_days for row in prereg.ARCHIVE_SCHEDULES} == {
        "ARCHIVE_D2": 2,
        "ARCHIVE_D7": 7,
        "ARCHIVE_D30": 30,
        "ARCHIVE_D90": 90,
    }
    assert {unit["counterpart_state"] for card in cards for unit in card.local_payload["relation_units"]} >= {
        "SAME_DAY_CARTESIAN",
        "NO_ANCHOR",
    }
    first_events = {}
    for event in events:
        first_events.setdefault((event.protocol, event.proposal_number), event)
    assert all(event.window_revision_count == 0 for event in first_events.values())
    assert all(event.window_age_days == 0 for event in first_events.values())
    assert all(event.update_gap_days is None for event in first_events.values())

    broken = dataclasses.replace(
        events[0],
        available_at={**events[0].available_at, "ARCHIVE_D2": datetime(2020, 1, 12, 13, tzinfo=UTC)},
    )
    broken_gate = runner.gate_pairing_reset_quarantine([broken, *events[1:]], cards)
    assert not broken_gate.passed
    assert "four_schedule_delays" in broken_gate.failure


def test_daily_cards_reject_complete_cartesian_relation_unit_overflow():
    events = []
    for protocol, count, offset in (
        ("ethereum", 9, 0),
        ("bitcoin", 8, 100),
    ):
        for index in range(count):
            events.append(
                runner._synthetic_event(
                    protocol=protocol,
                    proposal_number=6_000 + offset + index,
                    effective_day=date(2020, 1, 10),
                    event_type="CREATE",
                    revision=0,
                    first_parent_index=offset + index,
                )
            )

    with pytest.raises(
        ValueError,
        match="relation card exceeds frozen event bound",
    ):
        runner.build_daily_cards(events)


def test_control_metrics_cover_all_seven_controls_and_exact_cell_threshold_logic():
    events, cards = _synthetic_cards_events()
    metrics = runner.build_control_metrics(events, cards)

    assert tuple(metrics) == prereg.RELATION_CONTROLS
    for control, control_row in metrics.items():
        assert set(control_row["cells"]) == {
            f"{schedule.name}:{split}"
            for schedule in prereg.ARCHIVE_SCHEDULES
            for split in ("train", "test", "eval")
        }
        for cell, row in control_row["cells"].items():
            assert row["eligible"] >= 4, (control, cell, row)
            assert row["changed"] * 10 >= row["eligible"], (control, cell, row)
            assert row["changed_fraction"] == (
                f"{row['changed']}/{row['eligible']}"
            )
            assert row["passed"] is True
    assert runner.gate_control_sensitivity(metrics).passed

    # Exact boundary: 1/10 passes (>= 0.10), 0/10 fails, and 3/3 fails (<4 eligible).
    controlled = copy.deepcopy(metrics)
    first_control = prereg.RELATION_CONTROLS[0]
    cells = controlled[first_control]["cells"]
    cells["ARCHIVE_D2:train"] = {
        "eligible": 10,
        "changed": 1,
        "changed_fraction": "1/10",
        "passed": True,
    }
    cells["ARCHIVE_D2:test"] = {
        "eligible": 10,
        "changed": 0,
        "changed_fraction": "0/10",
        "passed": False,
    }
    cells["ARCHIVE_D2:eval"] = {
        "eligible": 3,
        "changed": 3,
        "changed_fraction": "3/3",
        "passed": False,
    }
    controlled[first_control]["passed"] = False
    gate = runner.gate_control_sensitivity(controlled)
    assert not gate.passed
    assert f"{first_control}:ARCHIVE_D2:test" in gate.failure
    assert f"{first_control}:ARCHIVE_D2:eval" in gate.failure
    assert f"{first_control}:ARCHIVE_D2:train" not in gate.failure


def test_split_vocabulary_daily_and_future_append_gates_pass_for_synthetic_events():
    events, cards = _synthetic_cards_events()
    split_metrics = runner.split_support_metrics(events, cards)

    assert runner.gate_split_support(split_metrics).passed
    assert runner.gate_vocabulary(events).passed
    assert runner.gate_daily_cards(cards).passed
    assert runner.gate_future_append(events, cards).passed


def test_split_gate_reports_each_support_floor_when_metrics_are_below_thresholds():
    metrics = {}
    for split in prereg.SPLITS:
        name = split["name"]
        metrics[name] = {
            "events_total": 0,
            "events_per_protocol": {"ethereum": 0, "bitcoin": 0},
            "events_per_protocol_source_year": {
                f"{protocol}:{year}": 0
                for protocol in ("ethereum", "bitcoin")
                for year in range(
                    runner.parse_time(str(split["decision_start"])).year,
                    runner.parse_time(str(split["decision_end_exclusive"])).year,
                )
            },
            "unique_proposals_total": 0,
            "unique_proposals_per_protocol": {"ethereum": 0, "bitcoin": 0},
            "unique_event_days_per_protocol": {"ethereum": 0, "bitcoin": 0},
            "active_months_per_protocol": {"ethereum": 0, "bitcoin": 0},
            "active_quarters_per_protocol": {"ethereum": 0, "bitcoin": 0},
            "top_proposal_event_count": 1,
            "top_event_day_count": 1,
            "relation_units": 0,
            "relation_units_nonexcluded": 0,
            "relation_units_with_counterpart_nonexcluded": 0,
            "d90_daily_cards": 0,
        }

    gate = runner.gate_split_support(metrics)

    assert not gate.passed
    for split in ("train", "test", "eval"):
        assert f"{split}:events_total" in gate.failure
        assert f"{split}:counterpart_fraction" in gate.failure
        assert f"{split}:top_proposal_event_share" in gate.failure


def test_forbidden_access_gate_lists_every_forbidden_source_market_model_result_field():
    assert runner.gate_forbidden_access(runner.AccessLedger.zero()).passed
    ledger = runner.AccessLedger(
        pre_2020_proposal_blobs_opened=1,
        post_2023_proposal_blobs_opened=1,
        btc_market_rows_read=1,
        funding_rows_read=1,
        future_return_rows_read=1,
        reward_rows_built=1,
        models_loaded=1,
        model_outputs_built=1,
        trade_rows_built=1,
        pnl_rows_built=1,
        cagr_values_built=1,
        strict_mdd_values_built=1,
    )
    gate = runner.gate_forbidden_access(ledger)

    assert not gate.passed
    assert set(gate.failure.split(";")) == set(gate.metrics["forbidden_fields"])


def test_safe_write_allows_idempotent_identical_artifact_and_rejects_conflict_escape_and_symlink(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.setattr(runner, "REPO_ROOT", repo_root)

    artifact = runner._write_once_bytes("results/synthetic.json", b"{}\n")
    assert artifact == repo_root / "results" / "synthetic.json"
    assert runner._write_once_bytes("results/synthetic.json", b"{}\n") == artifact
    with pytest.raises(RuntimeError, match="existing PSIM artifact differs"):
        runner._write_once_bytes("results/synthetic.json", b"{\"changed\":true}\n")
    with pytest.raises(RuntimeError, match="escapes repository"):
        runner._safe_destination(tmp_path / "outside.json")

    symlink_parent = repo_root / "link-parent"
    symlink_parent.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlinked"):
        runner._safe_destination("link-parent/out.json")

    broken = repo_root / "results" / "broken.json"
    broken.symlink_to(repo_root / "missing-target.json")
    with pytest.raises(RuntimeError, match="existing PSIM artifact differs"):
        runner._write_once_bytes("results/broken.json", b"{}\n")


def test_terminal_gate_reports_existing_conflicting_outputs_and_safe_paths(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.setattr(runner, "REPO_ROOT", repo_root)
    config = runner.Config(
        source_root=tmp_path / "source",
        result_path=Path("results/result.json"),
        rejection_path=Path("results/rejection.json"),
        events_path=Path("data/events.jsonl.gz"),
        cards_path=Path("data/cards.jsonl.gz"),
        controls_path=Path("results/controls.json"),
    )

    assert runner.gate_terminal_publication_ready(config).passed
    (repo_root / "results").mkdir(exist_ok=True)
    (repo_root / "results" / "result.json").write_text("already here", encoding="utf-8")
    gate = runner.gate_terminal_publication_ready(config)
    assert not gate.passed
    assert "exists:results/result.json" in gate.failure

    (repo_root / "results" / "result.json").unlink()
    (repo_root / "results" / "result.json").symlink_to(
        repo_root / "missing-result.json"
    )
    gate = runner.gate_terminal_publication_ready(config)
    assert not gate.passed
    assert "exists:results/result.json" in gate.failure


def test_replay_result_helpers_detect_matching_and_mismatched_replicas():
    events, cards = _synthetic_cards_events()
    event_map = {
        (protocol, replica): [event for event in events if event.protocol == protocol]
        for protocol in ("ethereum", "bitcoin")
        for replica in ("a", "b")
    }
    assert runner.gate_event_parser_replay(event_map).passed
    assert runner.gate_independent_replay(event_map, cards, cards).passed

    mismatched_events = dict(event_map)
    mismatched_events[("ethereum", "b")] = mismatched_events[("ethereum", "b")][1:]
    event_gate = runner.gate_event_parser_replay(mismatched_events)
    assert not event_gate.passed
    assert "ethereum" in event_gate.failure

    mismatched_cards = list(cards)
    mismatched_cards[0] = dataclasses.replace(
        mismatched_cards[0], local_payload_sha256="0" * 64
    )
    replay_gate = runner.gate_independent_replay(event_map, cards, mismatched_cards)
    assert not replay_gate.passed
    assert replay_gate.failure == "cards"


def test_preregistration_loader_replays_exact_frozen_manifest():
    payload = runner._load_preregistration()

    assert payload == prereg.build_preregistration()
    assert payload["manifest_hash"] == runner.PREREGISTRATION_MANIFEST_HASH
    assert runner.sha256_file(runner.PREREGISTRATION_PATH) == (
        runner.PREREGISTRATION_SHA256
    )


def test_daily_card_gate_recomputes_payload_and_chain_hashes():
    _events, cards = _synthetic_cards_events()
    tampered_payload = copy.deepcopy(cards[0].local_payload)
    tampered_payload["protocol_state"]["ethereum"]["new_event_state"] = (
        "TAMPERED"
    )
    tampered = dataclasses.replace(
        cards[0],
        local_payload=tampered_payload,
    )

    gate = runner.gate_daily_cards([tampered, *cards[1:]])

    assert not gate.passed
    assert "chain" in gate.failure


def test_future_append_gate_uses_valid_four_digit_bip_sentinel_when_bitcoin_is_first():
    events = sorted(
        runner.synthetic_events(),
        key=lambda row: (
            row.effective_day,
            row.protocol,
            row.first_parent_index,
        ),
    )
    assert events[0].protocol == "bitcoin"
    cards = runner.build_daily_cards(events)

    gate = runner.gate_future_append(events, cards)

    assert gate.passed, gate.failure
    assert gate.metrics["checks"]["sentinel_path_grammar"] is True


def _mock_official_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    valid_receipts: bool,
) -> runner.Config:
    repo_root = tmp_path / "repo-root"
    repo_root.mkdir()
    source_root = tmp_path / "source-root"
    monkeypatch.setattr(runner, "REPO_ROOT", repo_root)
    monkeypatch.setattr(
        runner,
        "validate_execution_seal",
        lambda: {
            "seal_hash": "a" * 64,
            "shared_commit": "b" * 40,
            "runner": {},
            "tests": {},
        },
    )
    monkeypatch.setattr(
        runner,
        "static_authority",
        lambda: {"source_authority_hash": "c" * 64},
    )
    monkeypatch.setattr(
        runner,
        "_authority_report",
        lambda seal, authority: {
            "seal_hash": seal["seal_hash"],
            **authority,
        },
    )
    monkeypatch.setattr(runner, "_worktree_clean", lambda: True)
    monkeypatch.setattr(runner, "enforce_disk_guard", lambda path: 1)

    def prepare(config, protocol, replica, ledger):
        spec = runner._repository_spec(protocol)
        return {
            "protocol": protocol,
            "replica": replica,
            "remote": spec.remote if valid_receipts else "invalid",
            "remote_head_symref": f"refs/heads/{spec.branch}",
            "remote_head_oid": "f" * 40,
            "local_tracking_symref": spec.remote_head_symref,
            "sealed_tip": spec.sealed_tip,
            "object_format": spec.object_format,
            "git_fsck_no_dangling": True,
            "shared_object_alternates": False,
            "worktree_porcelain_empty": True,
            "disk_used_gib": 1,
        }

    monkeypatch.setattr(runner, "prepare_source_repository", prepare)
    if not valid_receipts:
        monkeypatch.setattr(
            runner,
            "collect_commit_chain",
            lambda *args, **kwargs: pytest.fail(
                "first failed gate must stop without repair"
            ),
        )
        return runner.Config(source_root=source_root)

    synthetic = runner.synthetic_events()

    def chain(repo, protocol, ledger):
        spec = runner._repository_spec(protocol)
        return [
            runner.CommitRecord(
                protocol=protocol,
                oid=spec.sealed_tip,
                tree_oid="1" * 40,
                parent_oid=None,
                first_parent_index=0,
                committer_epoch=1_578_614_400,
                committer_day=date(2020, 1, 10),
                effective_day=date(2020, 1, 10),
            )
        ]

    def proposal_groups(repo, records, ledger):
        protocol = records[0].protocol
        event = next(row for row in synthetic if row.protocol == protocol)
        return (
            [
                runner.ProposalGroup(
                    protocol=protocol,
                    proposal_number=event.proposal_number,
                    commit_oid=event.commit_oid,
                    first_parent_index=event.first_parent_index,
                    committer_day=event.committer_day,
                    effective_day=event.effective_day,
                    old_path=event.old_path,
                    new_path=event.new_path,
                    old_blob_oid=event.old_blob_oid,
                    new_blob_oid=event.new_blob_oid,
                    event_type=event.event_type,
                    event_id=event.event_id,
                )
            ],
            [],
        )

    def materialize(repo, groups, ledger):
        protocol = groups[0].protocol
        return [row for row in synthetic if row.protocol == protocol]

    monkeypatch.setattr(runner, "collect_commit_chain", chain)
    monkeypatch.setattr(runner, "collect_proposal_groups", proposal_groups)
    monkeypatch.setattr(runner, "materialize_events", materialize)
    return runner.Config(source_root=source_root)


def test_official_run_stops_at_first_failed_gate_and_publishes_only_rejection(
    monkeypatch,
    tmp_path,
):
    config = _mock_official_source(
        monkeypatch,
        tmp_path,
        valid_receipts=False,
    )

    report = runner.run_official(config)

    assert report["decision"] == "reject"
    assert report["first_failure"] == {
        "gate_id": 1,
        "name": runner.GATE_NAMES[0],
    }
    assert report["source_incidence_opened"] is False
    assert (runner.REPO_ROOT / config.rejection_path).is_file()
    assert not any(
        (runner.REPO_ROOT / path).exists()
        for path in (
            config.result_path,
            config.events_path,
            config.cards_path,
            config.controls_path,
        )
    )
    assert not (runner.REPO_ROOT / runner.RUN_LOCK_PATH).exists()


def test_official_run_replays_all_thirteen_gates_and_atomically_publishes_pass(
    monkeypatch,
    tmp_path,
):
    config = _mock_official_source(
        monkeypatch,
        tmp_path,
        valid_receipts=True,
    )

    report = runner.run_official(config)
    replay = runner.run_official(config)

    assert report["decision"] == "pass"
    assert replay == report
    assert [row["name"] for row in report["gates"]] == list(
        runner.GATE_NAMES
    )
    assert all(row["passed"] for row in report["gates"])
    assert report["profitability_result"] is False
    assert report["outcomes_opened"] is False
    assert not any(
        report["access_ledger"][name]
        for name in runner.FORBIDDEN_ACCESS_FIELDS
    )
    for entry in report["artifacts"].values():
        artifact = runner.REPO_ROOT / entry["path"]
        assert artifact.is_file()
        assert runner.sha256_file(artifact) == entry["sha256"]
    stored = json.loads(
        (runner.REPO_ROOT / config.result_path).read_text(encoding="utf-8")
    )
    assert stored == report
    assert not (runner.REPO_ROOT / runner.RUN_LOCK_PATH).exists()


def test_source_configuration_rejects_symlinked_ancestor(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "source-link"
    link.symlink_to(outside, target_is_directory=True)
    config = runner.Config(source_root=link / "nested")

    with pytest.raises(ValueError, match="symlink ancestor"):
        runner._validate_source_configuration(config)


def test_publication_failure_becomes_terminal_gate_thirteen_rejection(
    monkeypatch,
    tmp_path,
):
    config = _mock_official_source(
        monkeypatch,
        tmp_path,
        valid_receipts=True,
    )
    monkeypatch.setattr(
        runner,
        "_publish_pass_group",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            FileExistsError("synthetic publication race")
        ),
    )

    report = runner.run_official(config)

    assert report["decision"] == "reject"
    assert report["first_failure"] == {
        "gate_id": 13,
        "name": runner.GATE_NAMES[12],
    }
    assert report["gates"][-1]["passed"] is False
    assert report["error"] == {"type": "FileExistsError"}
    assert (runner.REPO_ROOT / config.rejection_path).is_file()
    assert not (runner.REPO_ROOT / runner.RUN_LOCK_PATH).exists()
