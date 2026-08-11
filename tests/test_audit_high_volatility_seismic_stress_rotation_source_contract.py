import xml.etree.ElementTree as ET

import pytest

from training import audit_high_volatility_seismic_stress_rotation_source_contract as audit


def test_quakeml_contract_accepts_linked_revision_history():
    raw = b'''<q:quakeml xmlns:q="http://quakeml.org/xmlns/quakeml/1.2" xmlns="http://quakeml.org/xmlns/bed/1.2"><eventParameters><event publicID="event/a"><origin publicID="origin/1"><time><value>2023-07-01T00:00:00Z</value></time><creationInfo><creationTime>2023-07-01T00:01:00Z</creationTime></creationInfo></origin><origin publicID="origin/2"><time><value>2023-07-01T00:00:01Z</value></time><creationInfo><creationTime>2023-07-01T00:02:00Z</creationTime></creationInfo></origin><magnitude><mag><value>5.0</value></mag><originID>origin/1</originID><creationInfo><creationTime>2023-07-01T00:01:00Z</creationTime></creationInfo></magnitude><magnitude><mag><value>5.1</value></mag><originID>origin/2</originID><creationInfo><creationTime>2023-07-01T00:02:00Z</creationTime></creationInfo></magnitude></event></eventParameters></q:quakeml>'''
    result = audit.inspect_quakeml(raw)
    assert result["events"] == 1
    assert result["events_with_multiple_origins"] == 1
    assert result["events_with_multiple_magnitudes"] == 1


def test_quakeml_contract_rejects_unlinked_magnitude():
    raw = b'''<q:quakeml xmlns:q="http://quakeml.org/xmlns/quakeml/1.2" xmlns="http://quakeml.org/xmlns/bed/1.2"><eventParameters><event publicID="event/a"><origin publicID="origin/1"><time><value>x</value></time><creationInfo><creationTime>y</creationTime></creationInfo></origin><magnitude><mag><value>5</value></mag><originID>origin/missing</originID><creationInfo><creationTime>z</creationTime></creationInfo></magnitude></event></eventParameters></q:quakeml>'''
    with pytest.raises(RuntimeError, match="linkage"):
        audit.inspect_quakeml(raw)


def test_geojson_contract_rejects_duplicate_ids():
    raw = b'{"features":[{"id":"a","properties":{"ids":",a,"}},{"id":"a","properties":{"ids":",a,"}}]}'
    with pytest.raises(RuntimeError, match="identity"):
        audit.inspect_geojson(raw)


def test_quakeml_id_normalization():
    assert audit.normalize_quakeml_ids({"quakeml:earthquake.usgs.gov/fdsnws/event/1/query?eventid=us6000x&format=quakeml"}) == {"us6000x"}
