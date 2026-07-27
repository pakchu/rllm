from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from training import preregister_psim_d8_rllm1 as rllm1
from training import preregister_psim_d8_rllm2_source_feature_seal as prereg


def test_predecessor_pass_is_exact_and_market_remains_closed() -> None:
    predecessor = prereg.validate_predecessor()
    result = predecessor["result"]
    assert result["result_hash"] == prereg.RLLM2_RESULT_HASH
    assert result["challenge"]["decision"] == "pass"
    assert (
        result["challenge"]["source_feature_construction_authorized"]
        is True
    )
    assert result["challenge"]["market_access_authorized"] is False
    assert result["access_boundary"]["market_or_funding_paths_read"] == []
    assert (
        result["access_boundary"][
            "market_or_funding_payload_bytes_hashed"
        ]
        is False
    )


def test_source_row_roster_and_prompts_are_deterministic() -> None:
    rows = prereg.build_source_rows()
    repeated = prereg.build_source_rows()
    assert [row["row_hash"] for row in rows] == [
        row["row_hash"] for row in repeated
    ]
    roster = prereg.source_roster_contract(rows)
    assert roster["row_count"] == 1_461
    assert roster["split_counts"] == {
        "eval": 365,
        "test": 365,
        "train": 731,
    }
    assert roster["year_counts"] == {
        "2020": 366,
        "2021": 365,
        "2022": 365,
        "2023": 365,
    }
    assert roster["forced_no_eligible_rows"] == 117
    assert roster["embedding_forward_count"] == 1_461
    assert roster["relation_teacher_forward_count"] == 1_344
    assert [row["row_index"] for row in rows] == list(range(1_461))
    assert [row["decision_at"] for row in rows] == sorted(
        row["decision_at"] for row in rows
    )


def test_source_payload_and_prompt_boundaries_are_exact() -> None:
    rows = prereg.build_source_rows()
    digit_pattern = re.compile(r"\d")
    for row in rows:
        source_text = json.dumps(
            row["source_payload"],
            ensure_ascii=False,
            sort_keys=True,
        )
        assert digit_pattern.search(source_text) is None
        assert row["policy_prompt"].endswith("POLICY_STATE=")
        assert row["relation_teacher_prompt"].endswith("RELATION_CODE=")
        assert (
            tuple(row["relation_teacher_code_to_label"])
            == prereg.RELATION_CODE_ORDER
        )
        assert set(row["relation_teacher_code_to_label"].values()) == set(
            rllm1.RELATION_LABELS
        )
        assert row["row_hash"] == prereg.canonical_hash(
            {key: value for key, value in row.items() if key != "row_hash"}
        )
        if row["forced_no_eligible"]:
            assert row["source_payload"] == {
                "selected_subcard_relation": "NO_MODEL_ELIGIBLE_RELATION",
                "events": [],
                "relation_edges": [],
                "forced_relation": "INSUFFICIENT_EVIDENCE",
                "forced_target_rule": (
                    "KEEP_CURRENT_OR_FLAT_AT_SPLIT_START"
                ),
            }
            assert row["relation_teacher_forward_required"] is False
            assert (
                row["forced_relation_when_teacher_skipped"]
                == "INSUFFICIENT_EVIDENCE"
            )
        else:
            assert row["relation_teacher_forward_required"] is True
            assert row["forced_relation_when_teacher_skipped"] is None


def test_preregistration_is_canonical_and_closes_all_outcomes(
    tmp_path: Path,
) -> None:
    payload = prereg.build_preregistration()
    core = {
        key: value
        for key, value in payload.items()
        if key != "manifest_hash"
    }
    assert payload["manifest_hash"] == prereg.canonical_hash(core)
    assert payload["candidate"]["id"] == "PSIM-D8-RLLM2-S1"
    assert payload["candidate"]["profitability_claim"] is False
    assert payload["embedding_contract"]["shape"] == [1_461, 2_560]
    assert payload["relation_teacher_contract"]["forward_count"] == 1_344
    assert payload["execution_contract"]["checkpoint_shard_size"] == 1
    assert payload["execution_contract"][
        "inflight_sentinel_before_each_row_forward"
    ] is True
    assert payload["access_boundary"]["market_or_funding_paths_read"] == []
    assert (
        payload["access_boundary"][
            "market_or_funding_payload_bytes_hashed"
        ]
        is False
    )
    assert payload["access_boundary"]["rewards_created"] == 0
    assert payload["access_boundary"]["economic_metrics_computed"] == 0
    assert payload["access_boundary"]["test_outcomes_opened"] is False
    assert payload["access_boundary"]["eval_outcomes_opened"] is False

    output = tmp_path / "preregistration.json"
    first = prereg.write_preregistration(output)
    second = prereg.write_preregistration(output)
    assert first == second == payload
    assert output.read_bytes() == prereg.canonical_json_bytes(payload)


def test_direct_script_entrypoint_imports_without_pythonpath() -> None:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, str(Path(prereg.__file__)), "--help"],
        cwd=prereg.REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--output" in completed.stdout


def test_preregistration_has_no_market_or_economic_import_surface() -> None:
    source_path = Path(prereg.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    assert not imports.intersection(
        {
            "backtest",
            "database",
            "numpy",
            "pandas",
            "psycopg",
            "sqlalchemy",
        }
    )
