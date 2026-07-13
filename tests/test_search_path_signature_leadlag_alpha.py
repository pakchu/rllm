from __future__ import annotations

import json

import numpy as np
import pandas as pd

from training.search_funding_premium_external_state_gate_alpha import _manifest_core_hash
from training.search_path_signature_leadlag_alpha import PathSignatureConfig, _signals, rolling_signed_area, run


def test_signed_area_distinguishes_price_then_flow_from_flow_then_price() -> None:
    price_then_flow = rolling_signed_area(pd.Series([1.0, 0.0]), pd.Series([0.0, 1.0]), 2)
    flow_then_price = rolling_signed_area(pd.Series([0.0, 1.0]), pd.Series([1.0, 0.0]), 2)
    assert price_then_flow.iloc[-1] > 0.0
    assert flow_then_price.iloc[-1] < 0.0
    assert np.isclose(price_then_flow.iloc[-1], -flow_then_price.iloc[-1])


def test_signed_area_prefix_is_unchanged_by_future_increments() -> None:
    dx = pd.Series([1.0, 0.0, -0.5, 0.25])
    dy = pd.Series([0.0, 1.0, 0.25, -0.5])
    baseline = rolling_signed_area(dx, dy, 3)
    extended = rolling_signed_area(pd.concat([dx, pd.Series([99.0])], ignore_index=True), pd.concat([dy, pd.Series([-99.0])], ignore_index=True), 3)
    np.testing.assert_allclose(baseline.to_numpy(), extended.iloc[: len(baseline)].to_numpy(), equal_nan=True)


def test_flow_led_continuation_uses_area_and_hourly_stride() -> None:
    features = pd.DataFrame({"ps_area_24": np.zeros(13), "ps_flow_direction_24": np.zeros(13)})
    features.loc[1, ["ps_area_24", "ps_flow_direction_24"]] = [-0.8, 0.8]
    features.loc[12, ["ps_area_24", "ps_flow_direction_24"]] = [-0.8, 0.8]
    spec = {
        "path_window": 24,
        "family": "flow_led_continuation",
        "hold": 24,
        "stride": 12,
        "area_lower": -0.5,
        "area_upper": 0.5,
        "flow_lower": -0.5,
        "flow_upper": 0.5,
    }
    active, side = _signals(features, spec)
    assert np.flatnonzero(active).tolist() == [12]
    assert side[12] == 1


def test_price_led_fade_and_flip_are_exact_opposites() -> None:
    features = pd.DataFrame({"ps_area_24": [0.8], "ps_flow_direction_24": [0.8]})
    spec = {
        "path_window": 24,
        "family": "price_led_crowding_fade",
        "hold": 24,
        "stride": 12,
        "area_lower": -0.5,
        "area_upper": 0.5,
        "flow_lower": -0.5,
        "flow_upper": 0.5,
    }
    active, side = _signals(features, spec)
    _, flipped = _signals(features, spec, flip=True)
    assert active.tolist() == [True]
    assert side.tolist() == [-1]
    assert flipped.tolist() == [1]


def test_flow_only_ablation_removes_only_area_gate() -> None:
    features = pd.DataFrame({"ps_area_24": [0.0], "ps_flow_direction_24": [0.8]})
    spec = {
        "path_window": 24,
        "family": "flow_led_continuation",
        "hold": 24,
        "stride": 12,
        "area_lower": -0.5,
        "area_upper": 0.5,
        "flow_lower": -0.5,
        "flow_upper": 0.5,
    }
    active, _ = _signals(features, spec)
    ablated, side = _signals(features, spec, flow_only=True)
    assert active.tolist() == [False]
    assert ablated.tolist() == [True]
    assert side.tolist() == [1]


def test_persistent_state_does_not_reenter_at_next_hour() -> None:
    features = pd.DataFrame({"ps_area_24": np.zeros(25), "ps_flow_direction_24": np.zeros(25)})
    features.loc[12:, "ps_area_24"] = -0.8
    features.loc[12:, "ps_flow_direction_24"] = 0.8
    spec = {
        "path_window": 24,
        "family": "flow_led_continuation",
        "hold": 24,
        "stride": 12,
        "area_lower": -0.5,
        "area_upper": 0.5,
        "flow_lower": -0.5,
        "flow_upper": 0.5,
    }
    active, side = _signals(features, spec)
    assert np.flatnonzero(active).tolist() == [12]
    assert side[12] == 1


def test_empty_preflight_manifest_never_opens_oos(tmp_path, monkeypatch) -> None:
    core = {
        "protocol": {},
        "source_prefix_hashes": {},
        "feature_hash": "unused",
        "search_space": {"raw_specs": 6, "eligible_unique_paths": 0, "top_n": 6},
        "preflight_diagnostics": [],
        "selected": [],
    }
    manifest = {"as_of": "test", "sha256": _manifest_core_hash(core), **core}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    def fail_replay(*_args, **_kwargs):
        raise AssertionError("OOS replay must remain unopened")

    monkeypatch.setattr("training.search_path_signature_leadlag_alpha._replay", fail_replay)
    report = run(
        PathSignatureConfig(
            input_csv="unused",
            funding_csv="unused",
            premium_csv="unused",
            output=str(tmp_path / "report.json"),
            manifest_output=str(manifest_path),
            docs_output=str(tmp_path / "report.md"),
        )
    )
    assert report["preflight_only"] is True
    assert report["oos_opened"] is False
