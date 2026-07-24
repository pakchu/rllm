from __future__ import annotations

import builtins
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

import training.preregister_edgar_claim_relation_language as ecrl


@pytest.fixture(scope="module")
def splits() -> dict[str, list[dict[str, object]]]:
    return ecrl.generate_splits()


def _numbered() -> tuple[str, str]:
    return (
        "P1: Prior context.\nP2: Prior evidence.",
        "C1: Current context.\nC2: Current evidence.",
    )


def _config(root: Path) -> ecrl.Config:
    return ecrl.Config(
        output=str(root / "report.json"),
        prompt_output=str(root / "prompt.txt"),
        inventory_output=str(root / "inventory.json"),
        train_output=str(root / "train.jsonl"),
        calibration_output=str(root / "calibration.jsonl"),
        adversarial_output=str(root / "adversarial.jsonl"),
        swap_output=str(root / "swap.jsonl"),
    )


def test_parser_accepts_frozen_targets_and_one_optional_lf() -> None:
    prior, current = _numbered()
    for target in ecrl.SCENARIO_TARGETS.values():
        parsed = ecrl.parse_model_output(target, prior=prior, current=current)
        assert parsed["valid"], (target, parsed)
        parsed_lf = ecrl.parse_model_output(
            target + "\n",
            prior=prior,
            current=current,
        )
        assert parsed_lf["valid"], (target, parsed_lf)


@pytest.mark.parametrize(
    "output",
    [
        " A|U|N|C2|NONE",
        "A|U|N|C2|NONE ",
        "A|U|N|C2|NONE\r\n",
        "A|U|N|C2|NONE\n\n",
        "```A|U|N|C2|NONE```",
        '{"output":"A|U|N|C2|NONE"}',
        "A｜U｜N｜C2｜NONE",
        "Ａ|U|N|C2|NONE",
        "a|U|N|C2|NONE",
        "A|U|N|C0|NONE",
        "A|U|N|C9|NONE",
        "A|U|N|C2|P９",
    ],
)
def test_parser_rejects_non_exact_output(output: str) -> None:
    prior, current = _numbered()
    assert ecrl.parse_model_output(
        output,
        prior=prior,
        current=current,
    ) == {"valid": False, "error": "malformed"}


@pytest.mark.parametrize(
    ("output", "error"),
    [
        ("G|U|F|C8|P8", "01_missing_evidence_id"),
        ("G|U|F|C2|P2", "02_mixed_contract"),
        ("D|U|F|C2|P2", "03_risk_or_third_contract"),
        ("F|U|F|C2|P2", "04_no_claim_contract"),
        ("A|U|N|C2|P2", "05_new_evidence_contract"),
        ("A|U|F|C2|NONE", "06_relational_evidence_contract"),
        ("A|U|I|NONE|NONE", "07_directional_evidence_contract"),
        ("A|U|P|C2|P2", "08_repeat_delta_contract"),
        ("A|W|F|C2|P2", "09_unchanged_relation_contract"),
        ("A|X|I|C2|NONE", "10_supported_incomparable_contract"),
    ],
)
def test_consistency_failures_follow_frozen_order(
    output: str,
    error: str,
) -> None:
    prior, current = _numbered()
    assert ecrl.parse_model_output(
        output,
        prior=prior,
        current=current,
    ) == {"valid": False, "error": error}


@pytest.mark.parametrize(
    "output",
    [
        "A|U|F|C1|P1",
        "A|V|R|C2|P2",
        "A|U|N|C2|NONE",
        "A|W|P|C2|P2",
        "B|W|P|C2|P2",
        "C|W|P|C2|P2",
        "D|X|I|C2|NONE",
        "E|X|I|C2|NONE",
        "F|X|I|NONE|NONE",
        "G|X|I|NONE|NONE",
    ],
)
def test_consistency_accepts_boundary_combinations(output: str) -> None:
    prior, current = _numbered()
    assert ecrl.parse_model_output(
        output,
        prior=prior,
        current=current,
    )["valid"]


def test_redactor_uses_authorized_alias_variants_and_stable_placeholders() -> None:
    text = (
        "Atlas Digital Holdings Inc. and Atlas Digital filed on "
        "January 3, 2022 at 12:30 UTC for $1.2 million; "
        "NASDAQ: ATLS and BTC appeared with CIK 1234567890."
    )
    observed = ecrl.redact_line(
        text,
        aliases=("Atlas Digital Holdings Inc.",),
    )
    assert observed.count("[ENTITY]") == 2
    assert "[DATE]" in observed
    assert "[TIME]" in observed
    assert "[NUM]" in observed
    assert observed.count("[SYMBOL]") >= 2
    assert "[ID]" in observed
    assert "[[SYMBOL]]" not in observed
    assert "[ENTITY]" in observed


def test_redactor_nfkc_boundaries_and_no_unauthorized_nickname() -> None:
    fullwidth = "Ａｔｌａｓ Ｄｉｇｉｔａｌ Ｈｏｌｄｉｎｇｓ Ｉｎｃ．"
    assert ecrl.redact_line(
        fullwidth,
        aliases=("Atlas Digital Holdings Inc.",),
    ) == "[ENTITY]"
    observed = ecrl.redact_line(
        "XAtlas DigitalX and Atlas remained.",
        aliases=("Atlas Digital Holdings Inc.",),
    )
    assert "XAtlas DigitalX" in observed
    assert "Atlas remained" in observed


def test_redactor_rejects_source_private_use_code_points() -> None:
    with pytest.raises(ValueError, match="private-use"):
        ecrl.redact_line("Issuer \ue000 text.", aliases=())


@pytest.mark.parametrize("literal", ecrl.PREFILTER_LITERALS)
def test_prefilter_is_case_insensitive_and_fails_closed(literal: str) -> None:
    assert (
        ecrl.prefilter_reason(
            f"P1: {literal.swapcase()}",
            "C1: ordinary text",
        )
        == literal
    )


def test_generator_counts_balance_keys_and_sort_order(
    splits: dict[str, list[dict[str, object]]],
) -> None:
    assert {name: len(rows) for name, rows in splits.items()} == {
        "train": 4096,
        "calibration": 512,
        "adversarial": 768,
        "swap": 512,
    }
    for split, rows in splits.items():
        assert Counter(str(row["scenario_id"]) for row in rows) == Counter(
            {
                scenario: ecrl.SPLIT_ROWS_PER_SCENARIO[split]
                for scenario in ecrl.SCENARIO_TARGETS
            }
        )
        assert all(tuple(row) == ecrl.ROW_KEYS for row in rows)
        row_ids = [str(row["row_id"]) for row in rows]
        assert row_ids == sorted(
            row_ids,
            key=lambda value: tuple(value.split(":")),
        )


def test_hash_choice_formula_is_exact() -> None:
    pool = ("zero", "one", "two", "three")
    expected_index = (
        int.from_bytes(
            hashlib.sha256(
                b"ECRL-1|20260725|train|FULFILL_UP|7|issuer"
            ).digest(),
            "big",
        )
        % len(pool)
    )
    assert ecrl._choice(
        pool,
        split="train",
        scenario_id="FULFILL_UP",
        ordinal=7,
        field="issuer",
    ) == pool[expected_index]


def test_regeneration_is_byte_identical() -> None:
    first = {
        split: ecrl._jsonl_bytes(rows)
        for split, rows in ecrl.generate_splits().items()
    }
    second = {
        split: ecrl._jsonl_bytes(rows)
        for split, rows in ecrl.generate_splits().items()
    }
    assert first == second


def test_template_ids_are_disjoint_across_splits(
    splits: dict[str, list[dict[str, object]]],
) -> None:
    values = {
        split: {str(row["template_id"]) for row in rows}
        for split, rows in splits.items()
    }
    for left in values:
        for right in values:
            if left < right:
                assert values[left].isdisjoint(values[right])


def test_guard_rows_are_exact_and_model_denominator_is_736(
    splits: dict[str, list[dict[str, object]]],
) -> None:
    guarded = [
        row
        for row in splits["adversarial"]
        if ecrl.render_prompt(row) is None
    ]
    assert len(guarded) == 32
    assert Counter(str(row["scenario_id"]) for row in guarded) == Counter(
        {scenario: 2 for scenario in ecrl.SCENARIO_TARGETS}
    )
    assert len(splits["adversarial"]) - len(guarded) == 736


def test_relation_contrast_groups_are_current_identical_and_complete(
    splits: dict[str, list[dict[str, object]]],
) -> None:
    groups = ecrl.relation_contrast_groups(splits["adversarial"])
    assert len(groups) == 16
    lookup = {
        str(row["row_id"]): row for row in splits["adversarial"]
    }
    for row_ids in groups.values():
        rows = [lookup[row_id] for row_id in row_ids]
        assert len({str(row["current"]) for row in rows}) == 1
        assert {
            str(row["target"]).split("|")[2] for row in rows
        } == {"F", "R", "N", "P"}
        assert len({str(row["target"]) for row in rows}) == 4
        assert max(
            Counter(str(row["current"]) for row in rows).values()
        ) / len(rows) == 1.0
        assert max(
            Counter(str(row["target"]) for row in rows).values()
        ) / len(rows) == 0.25


def test_swap_pairs_change_only_surfaces_and_render_identically(
    splits: dict[str, list[dict[str, object]]],
) -> None:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in splits["swap"]:
        grouped[str(row["pair_id"])].append(row)
    assert len(grouped) == 256
    for rows in grouped.values():
        assert len(rows) == 2
        assert rows[0]["target"] == rows[1]["target"]
        assert rows[0]["prior"] != rows[1]["prior"]
        assert ecrl._surface_skeleton(str(rows[0]["prior"])) == (
            ecrl._surface_skeleton(str(rows[1]["prior"]))
        )
        assert ecrl._surface_skeleton(str(rows[0]["current"])) == (
            ecrl._surface_skeleton(str(rows[1]["current"]))
        )
        assert ecrl.render_prompt(rows[0]) == ecrl.render_prompt(rows[1])


def _lexicon_pair(
    prior_decision: str,
    current_decision: str,
) -> tuple[str, str]:
    return (
        f"P1: Context.\nP2: {prior_decision}\nP3: Context.",
        f"C1: Context.\nC2: {current_decision}\nC3: Context.",
    )


@pytest.mark.parametrize(
    ("prior_decision", "current_decision", "expected"),
    [
        (
            "The issuer planned to increase Bitcoin inventory.",
            "The issuer completed an increase in Bitcoin inventory.",
            "A|U|F|C2|P2",
        ),
        (
            "The issuer planned to reduce Bitcoin inventory.",
            "The issuer completed an increase in Bitcoin inventory.",
            "A|U|R|C2|P2",
        ),
        (
            "The issuer completed an increase in Bitcoin inventory.",
            "The issuer completed an increase in Bitcoin inventory.",
            "A|W|P|C2|P2",
        ),
        (
            "The issuer planned to increase Bitcoin inventory.",
            "The issuer planned to increase Bitcoin inventory.",
            "B|W|P|C2|P2",
        ),
        (
            "Routine Bitcoin accounting language.",
            "The issuer may increase Bitcoin inventory if financing closes.",
            "C|U|N|C2|NONE",
        ),
        (
            "Routine Bitcoin accounting language.",
            "The issuer faces adverse Bitcoin risk.",
            "D|X|I|C2|NONE",
        ),
        (
            "Routine Bitcoin accounting language.",
            "Customers may purchase Bitcoin through a third party.",
            "E|X|I|C2|NONE",
        ),
        (
            "Routine Bitcoin accounting language.",
            "The issuer did not increase Bitcoin inventory.",
            "F|X|I|NONE|NONE",
        ),
    ],
)
def test_lexicon_outputs_are_frozen(
    prior_decision: str,
    current_decision: str,
    expected: str,
) -> None:
    prior, current = _lexicon_pair(prior_decision, current_decision)
    assert ecrl.lexicon_output(prior, current) == expected


def test_lexicon_mixed_direction_is_fail_closed() -> None:
    prior = "P1: Context.\nP2: Context."
    current = (
        "C1: The issuer completed an increase in Bitcoin inventory.\n"
        "C2: The issuer completed a step and reduced Bitcoin inventory."
    )
    assert ecrl.lexicon_output(prior, current) == "G|X|I|NONE|NONE"


def test_inventory_pins_prompt_scenarios_lexicon_and_redaction() -> None:
    inventory = ecrl.template_inventory()
    assert inventory["scenario_targets"] == dict(ecrl.SCENARIO_TARGETS)
    assert inventory["lexicon"] == {
        key: list(values) for key, values in ecrl.LEXICON.items()
    }
    assert inventory["redaction"]["sentinels"] == dict(ecrl.SENTINELS)
    assert inventory["relation_contrast"]["ordinals"] == list(range(2, 18))


def test_build_outputs_is_deterministic_self_hashed_and_zero_boundary(
    tmp_path: Path,
) -> None:
    cfg = _config(tmp_path)
    first = ecrl.build_outputs(cfg)
    second = ecrl.build_outputs(cfg)
    assert first == second
    report = json.loads(first[cfg.output])
    self_hash = report.pop("self_hash")
    assert self_hash == ecrl.canonical_hash(report)
    assert set(report["m0_counters"].values()) == {0}
    assert report["decision"]["status"] == "PASS"
    datasets = report["contract"]["datasets"]
    assert datasets["adversarial"]["guard_rows"] == 32
    assert datasets["adversarial"]["rendered_prompt_rows"] == 736
    assert report["contract"]["relation_contrast"]["group_count"] == 16
    assert report["contract"]["relation_contrast"]["row_count"] == 64


def test_build_outputs_reads_no_source_history_or_market(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[Path] = []
    original_open = Path.open

    def guarded_open(path: Path, *args: object, **kwargs: object):
        resolved = path.resolve()
        opened.append(resolved)
        forbidden = (
            "sec_edgar_bitcoin_8k_6k_source",
            "market",
            "funding",
            "premium",
            "reward",
        )
        assert not any(token in resolved.name for token in forbidden)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    ecrl.build_outputs(_config(tmp_path))
    assert set(opened) == {
        (ecrl.REPOSITORY_ROOT / ecrl.MECHANISM_DOCUMENT).resolve(),
        Path(ecrl.__file__).resolve(),
    }


def test_build_outputs_does_not_import_model_or_network_modules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__
    forbidden = {
        "transformers",
        "torch",
        "peft",
        "trl",
        "bitsandbytes",
        "accelerate",
        "requests",
        "httpx",
        "urllib",
        "socket",
    }

    def guarded_import(name: str, *args: object, **kwargs: object):
        assert name.split(".", 1)[0] not in forbidden
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    ecrl.build_outputs(_config(tmp_path))


def test_write_once_refuses_overwrite_without_modifying_bytes(
    tmp_path: Path,
) -> None:
    cfg = _config(tmp_path)
    report = ecrl.write_outputs(cfg)
    assert report["decision"]["status"] == "PASS"
    before = {
        path: Path(path).read_bytes()
        for path in (
            cfg.output,
            cfg.prompt_output,
            cfg.inventory_output,
            cfg.train_output,
            cfg.calibration_output,
            cfg.adversarial_output,
            cfg.swap_output,
        )
    }
    with pytest.raises(FileExistsError, match="write-once"):
        ecrl.write_outputs(cfg)
    assert {path: Path(path).read_bytes() for path in before} == before


def test_required_count_uses_ceiling() -> None:
    assert ecrl.required_count(0.98, 736) == 722
    assert ecrl.required_count(0.95, 32) == 31
    assert ecrl.required_count(1.0, 16) == 16
    with pytest.raises(ValueError):
        ecrl.required_count(1.01, 10)
