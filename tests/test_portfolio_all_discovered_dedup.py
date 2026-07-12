import numpy as np

import training.portfolio_opt_all_discovered_alpha_gross10 as subject


def test_exact_duplicate_map_collapses_identical_return_and_adverse_paths(monkeypatch):
    monkeypatch.setattr(subject.base, "SLEEVES", ["a", "b", "c"])
    by = {}
    for split in ("train", "test2024", "eval2025", "ytd2026"):
        by[split] = {
            "R": np.array([[0.0, 0.1], [0.0, 0.1], [0.0, -0.1]]),
            "A": np.array([[0.0, -0.2], [0.0, -0.2], [0.0, -0.2]]),
        }

    canonical, groups = subject.exact_duplicate_map(by)

    assert groups == [["a", "b"]]
    assert canonical == {"a": "a", "b": "a"}


def test_sleeve_family_groups_generated_candidates():
    assert subject.sleeve_family("cand_calendar_1") == "calendar"
    assert subject.sleeve_family("cand_rex_veto_7") == "rex_veto"
    assert subject.sleeve_family("new_long_minimal_funding_premium") == "new"
    assert subject.sleeve_family("oi_upbit_ratio288_low") == "legacy"
