from __future__ import annotations

import base64
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from training import preregister_tor_consensus_relay_reconfiguration as tcrr


RESULT_PATH = Path(
    "results/tor_consensus_relay_reconfiguration_source_protocol_2026-07-23.json"
)


def _relay_identity(seed: int) -> str:
    return base64.b64encode(bytes([seed]) * 20).decode("ascii").rstrip("=")


def _signature_payload(seed: int) -> str:
    return base64.b64encode(bytes([seed]) * 128).decode("ascii")


def _synthetic_consensus(label: datetime) -> bytes:
    authority_ids = [f"{seed:040X}" for seed in range(1, 6)]
    lines = [
        "@type network-status-consensus-3 1.0",
        "network-status-version 3",
        "vote-status consensus",
        "consensus-method 32",
        f"valid-after {label:%Y-%m-%d %H:%M:%S}",
        f"fresh-until {(label + timedelta(hours=1)):%Y-%m-%d %H:%M:%S}",
        f"valid-until {(label + timedelta(hours=3)):%Y-%m-%d %H:%M:%S}",
        "voting-delay 300 300",
        "known-flags Exit Fast Guard Running Stable V2Dir Valid",
    ]
    for seed, identity in enumerate(authority_ids, start=1):
        lines.extend(
            [
                f"dir-source auth{seed} {identity} auth{seed}.example 192.0.2.{seed} 80 443",
                f"contact authority {seed}",
                f"vote-digest {seed:040X}",
            ]
        )
    lines.extend(
        [
            f"r relay1 {_relay_identity(11)} ignored 2023-01-01 00:00:00 192.0.2.20 9001 0",
            "s Fast Guard Running Stable Valid",
            "w Bandwidth=1000",
            f"r relay2 {_relay_identity(12)} ignored 2023-01-01 00:00:00 192.0.2.21 9001 0",
            "s Exit Fast Running Stable Valid",
            "w Bandwidth=500 Unmeasured=1",
            "directory-footer",
        ]
    )
    for seed, identity in enumerate(authority_ids, start=1):
        signing_key = f"{seed + 100:040X}"
        lines.extend(
            [
                f"directory-signature {identity} {signing_key}",
                "-----BEGIN SIGNATURE-----",
                _signature_payload(seed),
                "-----END SIGNATURE-----",
            ]
        )
    return ("\n".join(lines) + "\n").encode("ascii")


def _replace(raw: bytes, old: str, new: str) -> bytes:
    text = raw.decode("ascii")
    assert old in text
    return text.replace(old, new, 1).encode("ascii")


def _remove_signature(raw: bytes, seed: int) -> bytes:
    identity = f"{seed:040X}"
    signing_key = f"{seed + 100:040X}"
    block = "\n".join(
        [
            f"directory-signature {identity} {signing_key}",
            "-----BEGIN SIGNATURE-----",
            _signature_payload(seed),
            "-----END SIGNATURE-----",
            "",
        ]
    )
    return _replace(raw, block, "")


def test_archive_manifest_and_target_envelope_are_exact() -> None:
    manifest = tcrr.archive_manifest()
    assert len(manifest) == tcrr.EXPECTED_ARCHIVES == 48
    assert manifest[0] == {
        "month": "2020-01",
        "filename": "consensuses-2020-01.tar.xz",
        "url": (
            "https://collector.torproject.org/archive/relay-descriptors/"
            "consensuses/consensuses-2020-01.tar.xz"
        ),
        "compressed_bytes": 23463012,
        "compressed_sha256": (
            "16ff174aefea61518243120b2c3ada54d0b3bdb0ccca3e051f6079e46d23ff8e"
        ),
    }
    assert manifest[-1]["month"] == "2023-12"
    assert sum(row["compressed_bytes"] for row in manifest) == 1068849328
    assert len({row["compressed_sha256"] for row in manifest}) == 48

    labels = tcrr.anchor_labels()
    assert len(labels) == tcrr.EXPECTED_TARGET_DOCUMENTS == 5844
    assert labels[0] == datetime(2020, 1, 1, tzinfo=timezone.utc)
    assert labels[-1] == datetime(2023, 12, 31, 18, tzinfo=timezone.utc)
    assert {label.hour for label in labels} == set(tcrr.ANCHOR_HOURS)


def test_member_name_round_trip_and_tar_safety_are_fail_closed() -> None:
    label = datetime(2023, 1, 13, 18, tzinfo=timezone.utc)
    name = "consensuses-2023-01/13/2023-01-13-18-00-00-consensus"
    assert tcrr.expected_member_name(label) == name
    assert tcrr.label_from_member_name(name) == label
    assert tcrr.is_target_member(name) is True
    tcrr.validate_tar_member(
        name=name,
        size=2_000_000,
        is_file=True,
        is_symlink=False,
        is_hardlink=False,
    )

    for bad in (
        "/consensuses-2023-01/13/2023-01-13-18-00-00-consensus",
        "../consensuses-2023-01/13/2023-01-13-18-00-00-consensus",
        "consensuses-2023-01/12/2023-01-13-18-00-00-consensus",
    ):
        with pytest.raises(ValueError):
            tcrr.validate_tar_member(
                name=bad,
                size=1,
                is_file=True,
                is_symlink=False,
                is_hardlink=False,
            )

    non_target = "consensuses-2023-01/13/2023-01-13-17-00-00-consensus"
    assert tcrr.is_target_member(non_target) is False
    tcrr.validate_tar_member(
        name=non_target,
        size=2_000_000,
        is_file=True,
        is_symlink=False,
        is_hardlink=False,
    )
    with pytest.raises(ValueError, match="target tar member"):
        tcrr.label_from_member_name(non_target)

    with pytest.raises(ValueError, match="links"):
        tcrr.validate_tar_member(
            name=name,
            size=1,
            is_file=False,
            is_symlink=True,
            is_hardlink=False,
        )
    with pytest.raises(ValueError, match="size"):
        tcrr.validate_tar_member(
            name=name,
            size=True,  # type: ignore[arg-type]
            is_file=True,
            is_symlink=False,
            is_hardlink=False,
        )


def test_availability_is_label_plus_fifteen_minutes() -> None:
    label = datetime(2022, 6, 1, 12, tzinfo=timezone.utc)
    assert tcrr.public_availability_time(label) == label + timedelta(minutes=15)
    with pytest.raises(ValueError, match="anchor hours"):
        tcrr.public_availability_time(label.replace(hour=11))
    with pytest.raises(ValueError, match="UTC-aware"):
        tcrr.public_availability_time(label.replace(tzinfo=None))


def test_consensus_parser_retains_only_frozen_topology_fields() -> None:
    label = datetime(2023, 1, 1, tzinfo=timezone.utc)
    raw = _synthetic_consensus(label)
    parsed = tcrr.parse_consensus_document(raw, expected_label=label)
    assert parsed["member_bytes"] == len(raw)
    assert parsed["member_sha256"] == tcrr.sha256_bytes(raw)
    assert parsed["valid_after"] == "2023-01-01T00:00:00Z"
    assert parsed["availability_time"] == "2023-01-01T00:15:00Z"
    assert parsed["relay_count"] == 2
    assert parsed["authority_count"] == parsed["signature_count"] == 5
    assert parsed["relays"] == sorted(
        parsed["relays"], key=lambda relay: relay["relay_identity"]
    )
    assert set(parsed["relays"][0]) == {
        "relay_identity",
        "flags",
        "consensus_bandwidth",
    }
    forbidden = json.dumps(parsed)
    assert "192.0.2" not in forbidden
    assert "relay1" not in forbidden
    assert "auth1.example" not in forbidden
    assert tcrr.parse_consensus_document(raw, expected_label=label) == parsed
    assert tcrr.canonical_member_identity(label=label, parsed=parsed) == (
        tcrr.canonical_member_identity(label=label, parsed=dict(parsed))
    )


@pytest.mark.parametrize(
    ("raw_mutation", "match"),
    [
        (
            lambda raw: _replace(
                raw,
                "@type network-status-consensus-3 1.0",
                "@type network-status-microdesc-consensus-3 1.0",
            ),
            "annotation",
        ),
        (
            lambda raw: _replace(
                raw,
                "fresh-until 2023-01-01 01:00:00",
                "fresh-until 2023-01-01 00:10:00",
            ),
            "validity interval",
        ),
        (
            lambda raw: _replace(raw, "s Fast Guard", "s Mystery Fast Guard"),
            "flags",
        ),
        (
            lambda raw: _replace(raw, "w Bandwidth=1000", "v Tor 0.4.7.13"),
            "lacks one s or w",
        ),
    ],
)
def test_consensus_parser_rejects_schema_and_membership_drift(
    raw_mutation, match: str
) -> None:
    label = datetime(2023, 1, 1, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match=match):
        tcrr.parse_consensus_document(
            raw_mutation(_synthetic_consensus(label)), expected_label=label
        )


def test_consensus_parser_rejects_label_and_signature_membership_mismatch() -> None:
    label = datetime(2023, 1, 1, tzinfo=timezone.utc)
    raw = _synthetic_consensus(label)
    with pytest.raises(ValueError, match="valid-after"):
        tcrr.parse_consensus_document(
            raw, expected_label=label + timedelta(hours=6)
        )

    first_identity = f"{1:040X}"
    signature_header = f"directory-signature {first_identity} {101:040X}"
    damaged = _replace(raw, signature_header, f"directory-signature {9:040X} {101:040X}")
    with pytest.raises(ValueError, match="membership"):
        tcrr.parse_consensus_document(damaged, expected_label=label)


def test_consensus_parser_accepts_majority_subset_and_rejects_below_majority() -> None:
    label = datetime(2023, 1, 1, tzinfo=timezone.utc)
    raw = _synthetic_consensus(label)
    four_signatures = _remove_signature(raw, 5)
    parsed = tcrr.parse_consensus_document(
        four_signatures, expected_label=label
    )
    assert parsed["authority_count"] == 5
    assert parsed["signature_count"] == 4

    two_signatures = four_signatures
    for seed in (2, 3):
        two_signatures = _remove_signature(two_signatures, seed)
    with pytest.raises(ValueError, match="strict authority majority"):
        tcrr.parse_consensus_document(two_signatures, expected_label=label)


def test_canonical_member_identity_rejects_unfrozen_fields() -> None:
    label = datetime(2023, 1, 1, tzinfo=timezone.utc)
    parsed = tcrr.parse_consensus_document(
        _synthetic_consensus(label), expected_label=label
    )
    parsed["relay_ip"] = "future-leaking-field"
    with pytest.raises(ValueError, match="fields differ"):
        tcrr.canonical_member_identity(label=label, parsed=parsed)


def test_protocol_artifact_is_reproducible_and_keeps_outcomes_closed() -> None:
    payload = tcrr.build_protocol()
    tcrr.validate_protocol(payload)
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == tcrr.canonical_hash(core)
    assert payload["outcomes_opened"] is False
    assert payload["market_clocks_opened"] is False
    assert payload["full_source_incidence_opened"] is False
    assert payload["mechanism_features_opened"] is False
    assert payload["bounded_probe"]["valid_consensus_signatures"] == 8
    assert payload["bounded_probe"]["trusted_authority_match_count"] == 8
    assert payload["source_contract"]["archive_total_compressed_bytes"] == 1068849328

    committed = json.loads((tcrr.REPO_ROOT / RESULT_PATH).read_text(encoding="utf-8"))
    assert committed == payload
