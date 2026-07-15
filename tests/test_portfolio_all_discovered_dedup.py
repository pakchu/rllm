import numpy as np
import pandas as pd

import training.portfolio_opt_all_discovered_alpha_gross10 as subject
import training.portfolio_opt_combined_rex_new_alpha as combined


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


def test_new_alpha_event_builder_includes_train_split(monkeypatch):
    rows = 180
    market = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=rows, freq="5min"),
            "open": np.ones(rows),
            "high": np.ones(rows),
            "low": np.ones(rows),
        }
    )
    masks = {
        "train": np.ones(rows, dtype=bool),
        "test2024": np.ones(rows, dtype=bool),
    }
    monkeypatch.setattr(
        combined,
        "build_market_feature_frame",
        lambda market, window_size: pd.DataFrame({"base": np.ones(len(market))}),
    )
    monkeypatch.setattr(
        combined,
        "build_interest_features",
        lambda market, base: pd.DataFrame({"interest": np.ones(len(market))}),
    )
    monkeypatch.setattr(
        combined.na,
        "ALPHAS",
        {"probe": {"side": "long", "hold": 2}},
    )
    monkeypatch.setattr(combined.na, "LONG_COMPONENTS", {})
    monkeypatch.setattr(
        combined.na,
        "_alpha_active",
        lambda features, name: np.ones(len(features), dtype=bool),
    )

    def event_path(market, signal_pos, **kwargs):
        ret = np.zeros(len(market))
        adverse = np.zeros(len(market))
        ret[signal_pos + 1] = 0.01
        return ret, adverse, 0.01

    monkeypatch.setattr(combined.na, "_event_path", event_path)
    events = []

    combined._append_new_alpha_events(
        events,
        market,
        masks,
        combined.CombinedOptConfig(random_samples=0),
    )

    assert {event["split"] for event in events} == {"train", "test2024"}
