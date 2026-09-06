import numpy as np
import pandas as pd
import pytest

from training import build_high_volatility_seismic_stress_rotation_relay_support as support


def event_xml(magnitudes: str) -> bytes:
    return f'''<q:quakeml xmlns:q="http://quakeml.org/xmlns/quakeml/1.2" xmlns="http://quakeml.org/xmlns/bed/1.2"><eventParameters><event publicID="event/a"><origin publicID="origin/1"><time><value>2023-07-01T00:00:00Z</value></time><creationInfo><creationTime>2023-07-01T00:01:00Z</creationTime></creationInfo></origin>{magnitudes}</event></eventParameters></q:quakeml>'''.encode()


def magnitude(value: float, creation: str = "2023-07-01T00:02:00Z", origin: str = "origin/1") -> str:
    return f"<magnitude><mag><value>{value}</value></mag><originID>{origin}</originID><creationInfo><creationTime>{creation}</creationTime></creationInfo></magnitude>"


def test_parse_event_retains_only_ever_eligible_histories():
    low = ET_from(event_xml(magnitude(4.9)))
    high = ET_from(event_xml(magnitude(4.9) + magnitude(5.1)))
    assert support.parse_event(low) is None
    parsed = support.parse_event(high)
    assert parsed is not None and len(parsed["magnitudes"]) == 2
    assert len(parsed["event_id_sha256"]) == 64


def test_event_without_magnitude_is_non_candidate():
    raw = b'''<q:quakeml xmlns:q="http://quakeml.org/xmlns/quakeml/1.2" xmlns="http://quakeml.org/xmlns/bed/1.2"><eventParameters><event publicID="event/a"/></eventParameters></q:quakeml>'''
    assert support.parse_event(ET_from(raw)) is None


def ET_from(raw: bytes):
    import xml.etree.ElementTree as ET
    return ET.fromstring(raw).find(".//b:event", support.NS)


def test_causal_panel_uses_latest_version_available_by_decision():
    parsed = support.parse_event(ET_from(event_xml(magnitude(5.0) + magnitude(5.2, "2023-07-04T13:00:00Z"))))
    panel = support.causal_stress_panel([parsed])
    row = panel[panel.source_day.eq(pd.Timestamp("2023-07-01T00:00:00Z"))].iloc[0]
    assert row.causal_event_count == 1
    assert row.daily_seismic_stress == pytest.approx(10 ** 7.5)


def test_strict_prior_midrank_excludes_current():
    values = pd.Series(list(range(60)) + [100.0]); ranks = support.strict_prior_midrank(values)
    assert np.isnan(ranks.iloc[59]); assert ranks.iloc[60] == 1.0


def test_primary_clock_uses_both_frozen_gates():
    features = pd.DataFrame({"source_day": pd.to_datetime(["2023-07-01", "2023-07-03"], utc=True), "decision_time": pd.to_datetime(["2023-07-03T12:00Z", "2023-07-05T12:00Z"]), "daily_seismic_stress": [10.0, 5.0], "stress_change": [2.0, -5.0], "stress_change_rank": [0.7, 0.8], "seismic_side": [-1, 1], "causal_event_count": [2, 1], "btc_realized_variation": [0.1, 0.2], "btc_variation_rank": [0.7, 0.8]})
    clock = support.build_clock(features)
    assert clock.side.tolist() == [-1, 1]
    assert (pd.to_datetime(clock.entry_time, utc=True) - pd.to_datetime(clock.decision_time, utc=True)).eq(pd.Timedelta(minutes=5)).all()
