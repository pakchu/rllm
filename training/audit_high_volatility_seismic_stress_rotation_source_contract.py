"""Audit HVSSR-24 USGS causal-version transport without opening incidence or outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import preregister_high_volatility_seismic_stress_rotation_relay as prereg


PREREG_SHA = "92dc3d2086e7197311e386f4c62da3b2cdf247256d6b5ac02482415dcf0b079d"
SCRIPT = Path("training/audit_high_volatility_seismic_stress_rotation_source_contract.py")
RESULT = Path("results/high_volatility_seismic_stress_rotation_relay_source_contract_2026-08-12.json")
PROBE_START, PROBE_END = "2023-07-01", "2023-07-02"
NS = {"b": "http://quakeml.org/xmlns/bed/1.2"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def fetch(parameters: dict[str, Any]) -> tuple[str, bytes]:
    url = prereg.USGS_EVENT_API + "?" + urlencode(parameters)
    request = Request(url, headers={"User-Agent": "rllm-hvssr-source-contract/1.0"})
    with urlopen(request, timeout=120) as response:
        return url, response.read()


def inspect_quakeml(raw: bytes) -> dict[str, Any]:
    root = ET.fromstring(raw)
    events = root.findall(".//b:event", NS)
    if not events:
        raise RuntimeError("HVSSR empty QuakeML probe")
    identifiers: set[str] = set()
    events_with_multiple_origins = 0
    events_with_multiple_magnitudes = 0
    for event in events:
        event_id = event.attrib.get("publicID")
        if not event_id or event_id in identifiers:
            raise RuntimeError("HVSSR duplicate or missing QuakeML event id")
        identifiers.add(event_id)
        origins = event.findall("b:origin", NS)
        magnitudes = event.findall("b:magnitude", NS)
        if not origins or not magnitudes:
            raise RuntimeError("HVSSR event lacks causal origin or magnitude history")
        events_with_multiple_origins += len(origins) > 1
        events_with_multiple_magnitudes += len(magnitudes) > 1
        origin_ids = {item.attrib.get("publicID") for item in origins}
        for origin in origins:
            if origin.findtext("b:time/b:value", namespaces=NS) is None or origin.findtext("b:creationInfo/b:creationTime", namespaces=NS) is None:
                raise RuntimeError("HVSSR origin lacks event or publication time")
        for magnitude in magnitudes:
            if magnitude.findtext("b:mag/b:value", namespaces=NS) is None or magnitude.findtext("b:creationInfo/b:creationTime", namespaces=NS) is None:
                raise RuntimeError("HVSSR magnitude lacks value or publication time")
            if magnitude.findtext("b:originID", namespaces=NS) not in origin_ids:
                raise RuntimeError("HVSSR magnitude-to-origin linkage drift")
    if events_with_multiple_origins < 1 or events_with_multiple_magnitudes < 1:
        raise RuntimeError("HVSSR probe does not preserve revision histories")
    return {
        "events": len(events),
        "events_with_multiple_origins": events_with_multiple_origins,
        "events_with_multiple_magnitudes": events_with_multiple_magnitudes,
        "event_ids": identifiers,
    }


def inspect_geojson(raw: bytes) -> dict[str, Any]:
    document = json.loads(raw)
    features = document.get("features")
    if not isinstance(features, list) or not features:
        raise RuntimeError("HVSSR empty GeoJSON probe")
    identifiers = {str(item.get("properties", {}).get("ids", "")) for item in features}
    public_ids = {str(item.get("id")) for item in features}
    if "None" in public_ids or len(public_ids) != len(features):
        raise RuntimeError("HVSSR GeoJSON event identity drift")
    return {"events": len(features), "deleted_events": sum(item.get("properties", {}).get("status") == "deleted" for item in features), "event_ids": public_ids, "source_ids_present": all(identifiers)}


def normalize_quakeml_ids(ids: set[str]) -> set[str]:
    normalized: set[str] = set()
    for item in ids:
        values = parse_qs(urlparse(item).query).get("eventid", [])
        if len(values) != 1 or not values[0]:
            raise RuntimeError("HVSSR QuakeML public event id drift")
        normalized.add(values[0])
    return normalized


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVSSR preregistration hash drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text()); prereg.validate(registration)
    common = {"starttime": PROBE_START, "endtime": PROBE_END, "eventtype": "earthquake"}
    geo_url, geo_raw = fetch({**common, "format": "geojson", "includedeleted": "true"})
    xml_url, xml_raw = fetch({**common, "format": "quakeml", "includeallorigins": "true", "includeallmagnitudes": "true"})
    geo = inspect_geojson(geo_raw); quake = inspect_quakeml(xml_raw)
    if geo["event_ids"] != normalize_quakeml_ids(quake["event_ids"]):
        raise RuntimeError("HVSSR GeoJSON and QuakeML probe event universes differ")
    core = {
        "protocol_version": "hvssr_24_source_contract_v1", "policy_id": "HVSSR-24",
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]},
        "source_evaluator": {"path": str(SCRIPT), "sha256": sha(SCRIPT)},
        "probe": {
            "window": [PROBE_START, PROBE_END], "geojson_url": geo_url, "quakeml_url": xml_url,
            "geojson_events": geo["events"], "quakeml_events": quake["events"], "deleted_events": geo["deleted_events"],
            "events_with_multiple_origins": quake["events_with_multiple_origins"],
            "events_with_multiple_magnitudes": quake["events_with_multiple_magnitudes"],
            "event_universe_identity_matches": True, "origin_event_and_creation_times_present": True,
            "magnitude_value_creation_time_and_origin_link_present": True,
        },
        "numeric_magnitudes_published": False, "event_identifiers_published": False,
        "candidate_incidence_opened": False, "btc_rows_opened": False,
        "postentry_return_pnl_execution_price_opened": False, "gross9_rows_opened": False,
        "source_contract_passed": True, "advance_to_source_incidence": True,
        "advance_to_gross9_novelty": False, "advance_to_economic_outcomes": False,
        "decision": "pass_to_source_incidence", "repair_authorized": False,
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args(); report = run()
    print(json.dumps({"decision": report["decision"], "probe": report["probe"]}, indent=2))
