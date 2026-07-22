"""Freeze TCRR source support before relay incidence or BTC outcomes."""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import operator
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path(
    "results/tor_consensus_relay_reconfiguration_source_protocol_2026-07-23.json"
)
DECISION_PATH = Path(
    "docs/tor-consensus-relay-reconfiguration-source-axis-decision-2026-07-23.md"
)
DECISION_SHA256 = "3e88b71bcef2ec4dfec87f82fccfb68d66cf76111bde9adb5140f5ce50a0d799"
SCRIPT_PATH = Path("training/preregister_tor_consensus_relay_reconfiguration.py")

SOURCE_START = datetime(2020, 1, 1, tzinfo=timezone.utc)
SOURCE_END_EXCLUSIVE = datetime(2024, 1, 1, tzinfo=timezone.utc)
ANCHOR_HOURS = (0, 6, 12, 18)
EXPECTED_ARCHIVES = 48
EXPECTED_TARGET_DOCUMENTS = 5844
PUBLICATION_DELAY_MINUTES = 15
DISK_LIMIT_GIB = 300
MAXIMUM_ARCHIVE_BYTES = 64 * 1024 * 1024
MAXIMUM_MEMBER_BYTES = 8 * 1024 * 1024
ARCHIVE_BASE = (
    "https://collector.torproject.org/archive/relay-descriptors/consensuses"
)
INDEX_URL = "https://collector.torproject.org/index/index.json"
TYPE_ANNOTATION = b"@type network-status-consensus-3 1.0\n"

ARCHIVE_ROWS: tuple[tuple[str, int, str], ...] = (
    ("2020-01", 23463012, "16ff174aefea61518243120b2c3ada54d0b3bdb0ccca3e051f6079e46d23ff8e"),
    ("2020-02", 21571612, "3036026566164b812448b5515f0d24e88527b9450a5bf2f4911f46d01fa9c372"),
    ("2020-03", 23770476, "494669194d2a5aca88aca26f46027a018f8efc2fdc57aea5b8237ad6c70ef730"),
    ("2020-04", 24783940, "edb2a296f2d4e2481def1287051c417dc545fc9c40fca528789311fd612c406c"),
    ("2020-05", 23189976, "cafa06305427035568acd7761a33a85d70c3a889e237a52d626e8a2d83506f46"),
    ("2020-06", 21027892, "dcc9b4df18ca781e04618d84278c00c252dd626aa404a69fbfb4feb65c7ff52b"),
    ("2020-07", 21942088, "63e9aa1805232a39053a5874db9dd7cd2be504c915d8700568df1c314a307456"),
    ("2020-08", 22893996, "0082fea4dd585a3063063861f525f71fc3f064f5ac7508fa694f0f7715d098ea"),
    ("2020-09", 22780968, "d0b7a060b5108c3aec8bbbc5a4748deaa1e46e91935abeea9c9b00219f6d6333"),
    ("2020-10", 23935792, "41d56b4dd901485babff8910b51bf0accd8a1caa823d22b36f0399773152cd13"),
    ("2020-11", 23035688, "5e38b5cd0aedc89be0ccd981c626bbdd067bcd6eee369b5e5f2bdcb29819ce46"),
    ("2020-12", 24008312, "79771dbf9595adb89239fdce8c9b99d709cd5a1886b32f0b465719a6671fed69"),
    ("2021-01", 24924132, "ccbecafbee99541f23bc43d29ff5690070caa5af8c79b652012db0b6a2481c1d"),
    ("2021-02", 21257040, "2c145e00edf4dd6dba202a512640699cf5c494dd8a31a92854622ca888b52a27"),
    ("2021-03", 23154724, "be214f5db235382cd946327d34169bb675f28e8a6c913131872694a062d89d12"),
    ("2021-04", 22384232, "80bc6ff4e5e52b1a6d5c85522a300753ea43b6fdba56b6dbcdd26165aa0120b1"),
    ("2021-05", 22195308, "122dcdc41286156a8922cf32e11c1b0bb085b76f8bb0d50d24debbbb74ba9bc6"),
    ("2021-06", 21904692, "35a424b9cac53dc553acd8bd6147f250754978434734b0f46e7c351bbf02b457"),
    ("2021-07", 22293300, "3a4ef11808fae3b1a3a61d2ba715e378a34680b172949b8ffd1b45ec53d1f971"),
    ("2021-08", 22685408, "bc44f830cdb79bbd54ca1bd29d832c60ae2fb4e0c866fbf91cab1f8347f00e29"),
    ("2021-09", 21191904, "4a4fcc1d8c2a3b4201d5736fa6bc0f4cf92f9ddf40dcb0d00be1fdf222e4d8ca"),
    ("2021-10", 21316040, "07067f36051ddec840a714363ead3995eb1931fb914a5420e0c11c43bbad474d"),
    ("2021-11", 19403724, "156422f9a6ebac18df7a896b4842140f76437ffb5e0e7e5a32975d6b4313770b"),
    ("2021-12", 19986552, "50874f66cf19fffc104d8b92a9d8854d26eb398dd45851c60b7d03fd52e9d61f"),
    ("2022-01", 20778508, "7c7d7c45388e3e0c69df2ff94ddc4eae7185baa2331c65e257b0a5a2f384e854"),
    ("2022-02", 18503640, "3bcd2f32de2ad7ad24c95ad470b5daf18e779f846047947bffb1e2f01400adea"),
    ("2022-03", 22975008, "1369c719caaf6eba834fc7745edd823501081741cc30255b0c5828930d48def2"),
    ("2022-04", 21021924, "d73159b556235fde95474b4056ac50bd923ab026fbf32360600fd74121ecf946"),
    ("2022-05", 22162332, "0ae63b4f721bbaf59b3ae56a31e2691f53044878dd5c2d07c854d3b6e83d40a9"),
    ("2022-06", 22003828, "70fe4954d2b6fe4aa3d2fbcd739c181dcdd2f4f2d11830596569254b56f03e01"),
    ("2022-07", 23809376, "1a57eab5e9c64282e95ca661adc7e54abe1096e742af1ed65a9e5515432805e5"),
    ("2022-08", 23976992, "b6382f4b8690bdf274f0672f35b0ef6ad6b447f842849f9ea294ddb2f7902698"),
    ("2022-09", 22149816, "adc56e9eb77c800c3e39d66b514eac59dc85e0608df0a211da3cea72bcef693a"),
    ("2022-10", 23315884, "75d600c4e58e888342832ded0ea82bfea22e6223a66a7eec335bee44e91edad0"),
    ("2022-11", 21576156, "c71652ef4ed20fb3327e1ed0c0f164a68eb3a420c88ac0b31196b59e00951d75"),
    ("2022-12", 21039044, "98bebf067dd68dc34824962d34298467db1dde638dfd5c5b44c9d9b631f4fdef"),
    ("2023-01", 20753132, "1e9bf983be549fa8ec74c3d40f296921d3b2fe914d91ee246364c0897abe3e44"),
    ("2023-02", 19139020, "91b16759ddce2a68f5afaf9f7915dc6a83035549237de5da7b290806992a028a"),
    ("2023-03", 20312860, "78903ab208072e249793d38926dcf932abd6dfb98e9984c4e4f3f8fb515ad995"),
    ("2023-04", 19564096, "03654a91a8df92f83f4f1f99c083fddb5e9bcebe41fb9bce4ee9f3cb8dbe10c3"),
    ("2023-05", 20774184, "75582da87625e867ed28222da93a5a4b9b9be3d6936648a0ebd268fedd15bc1e"),
    ("2023-06", 20390484, "e66027fd903817dee61ed741b5acd02acb49f15f2c6bc6cd0fd83fa6f19eaff9"),
    ("2023-07", 21652788, "c1b2df834670ad74a030904f7c2b6dfefa2d75f6c425120319ba45b60c146dc0"),
    ("2023-08", 23244812, "fac7be9929b7a23bea9157882d8bf46c4e43ffb24a1c1eb1ae86e59df22d12a0"),
    ("2023-09", 24070276, "7167c9c4a7231e0fbfa225960b9e0cba222c7585fc4f98123a1f4cc756c3473e"),
    ("2023-10", 25387136, "44845a22c287e395ec4f52a46886d6f5ccaea60becba2b3d6bdbdfa9f919ebf4"),
    ("2023-11", 25309404, "e1392b4532177d05dee2728fc6ba6a41f16c4ab2bc5d17ecf220ddd6fcf3879d"),
    ("2023-12", 25837820, "888fad2c9e4d3aa4ce6f8232fbb1137db0de423e43bb22dfe4c608469be5bbdd"),
)

FINGERPRINT_PATTERN = re.compile(r"[A-F0-9]{40}", re.ASCII)
MONTH_PATTERN = re.compile(r"20(?:20|21|22|23)-(?:0[1-9]|1[0-2])", re.ASCII)
TARGET_MEMBER_PATTERN = re.compile(
    r"consensuses-(?P<month>20(?:20|21|22|23)-(?:0[1-9]|1[0-2]))/"
    r"(?P<day>0[1-9]|[12][0-9]|3[01])/"
    r"(?P<date>20(?:20|21|22|23)-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01]))-"
    r"(?P<hour>00|06|12|18)-00-00-consensus",
    re.ASCII,
)
ALL_CONSENSUS_MEMBER_PATTERN = re.compile(
    r"consensuses-(?P<month>20(?:20|21|22|23)-(?:0[1-9]|1[0-2]))/"
    r"(?P<day>0[1-9]|[12][0-9]|3[01])/"
    r"(?P<date>20(?:20|21|22|23)-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01]))-"
    r"(?P<hour>[01][0-9]|2[0-3])-00-00-consensus",
    re.ASCII,
)

PROBE = {
    "archive_url": f"{ARCHIVE_BASE}/consensuses-2023-01.tar.xz",
    "member_name": (
        "consensuses-2023-01/13/2023-01-13-09-00-00-consensus"
    ),
    "member_bytes": 2477964,
    "member_sha256": (
        "b43dbe5c0297e8d6f2c4344c6c8461eb864aaaf5f117151493072caf3311eba7"
    ),
    "consensus_method": 32,
    "valid_after": "2023-01-13T09:00:00Z",
    "fresh_until": "2023-01-13T10:00:00Z",
    "valid_until": "2023-01-13T12:00:00Z",
    "voting_delay_seconds": [300, 300],
    "relay_count": 6307,
    "authority_count": 8,
    "signature_count": 8,
    "valid_certificate_self_signatures": 8,
    "valid_consensus_signatures": 8,
    "trusted_authority_match_count": 8,
    "tor_source_release": "0.4.7.13",
    "tor_source_sha256": (
        "2079172cce034556f110048e26083ce9bea751f3154b0ad2809751815b11ea9d"
    ),
    "market_or_outcomes_opened": False,
}


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: str | Path) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return sha256_bytes(candidate.read_bytes())


def canonical_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256_bytes(raw)


def archive_manifest() -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    expected = datetime(2020, 1, 1, tzinfo=timezone.utc)
    for month, size, digest in ARCHIVE_ROWS:
        expected_month = expected.strftime("%Y-%m")
        if month != expected_month:
            raise RuntimeError("TCRR archive month sequence is not contiguous")
        if MONTH_PATTERN.fullmatch(month) is None:
            raise RuntimeError("TCRR archive month is malformed")
        if isinstance(size, bool) or size <= 0 or size > MAXIMUM_ARCHIVE_BYTES:
            raise RuntimeError("TCRR archive byte count is outside the frozen guard")
        if re.fullmatch(r"[0-9a-f]{64}", digest, re.ASCII) is None:
            raise RuntimeError("TCRR archive SHA-256 is malformed")
        filename = f"consensuses-{month}.tar.xz"
        manifest.append(
            {
                "month": month,
                "filename": filename,
                "url": f"{ARCHIVE_BASE}/{filename}",
                "compressed_bytes": size,
                "compressed_sha256": digest,
            }
        )
        year = expected.year + (expected.month // 12)
        month_number = expected.month % 12 + 1
        expected = expected.replace(year=year, month=month_number)
    if len(manifest) != EXPECTED_ARCHIVES or expected != SOURCE_END_EXCLUSIVE:
        raise RuntimeError("TCRR archive envelope is not exactly 48 months")
    return manifest


def anchor_labels() -> list[datetime]:
    labels: list[datetime] = []
    current = SOURCE_START
    while current < SOURCE_END_EXCLUSIVE:
        if current.hour in ANCHOR_HOURS:
            labels.append(current)
        current += timedelta(hours=1)
    if len(labels) != EXPECTED_TARGET_DOCUMENTS:
        raise RuntimeError("TCRR target envelope is not exactly 5,844 documents")
    return labels


def _validate_label(label: datetime) -> None:
    if label.tzinfo is None or label.utcoffset() != timedelta(0):
        raise ValueError("TCRR label must be UTC-aware")
    if label.minute or label.second or label.microsecond:
        raise ValueError("TCRR label must be an exact hour")
    if label.hour not in ANCHOR_HOURS:
        raise ValueError("TCRR label is outside the frozen anchor hours")
    if not SOURCE_START <= label < SOURCE_END_EXCLUSIVE:
        raise ValueError("TCRR label is outside the frozen interval")


def expected_member_name(label: datetime) -> str:
    _validate_label(label)
    return (
        f"consensuses-{label:%Y-%m}/{label:%d}/"
        f"{label:%Y-%m-%d-%H}-00-00-consensus"
    )


def label_from_member_name(name: str) -> datetime:
    if not isinstance(name, str):
        raise ValueError("TCRR tar member name must be text")
    match = TARGET_MEMBER_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError("TCRR target tar member path is malformed")
    if match.group("month") != match.group("date")[:7]:
        raise ValueError("TCRR target tar member month disagrees with date")
    if match.group("day") != match.group("date")[-2:]:
        raise ValueError("TCRR target tar member day disagrees with date")
    try:
        label = datetime.strptime(
            f"{match.group('date')} {match.group('hour')}", "%Y-%m-%d %H"
        ).replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ValueError("TCRR target tar member calendar date is invalid") from exc
    _validate_label(label)
    if expected_member_name(label) != name:
        raise ValueError("TCRR target tar member is not canonical")
    return label


def is_target_member(name: str) -> bool:
    try:
        label_from_member_name(name)
    except ValueError:
        return False
    return True


def _validate_consensus_member_calendar(name: str) -> None:
    match = ALL_CONSENSUS_MEMBER_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError("TCRR consensus tar member path is malformed")
    if match.group("month") != match.group("date")[:7]:
        raise ValueError("TCRR consensus tar member month disagrees with date")
    if match.group("day") != match.group("date")[-2:]:
        raise ValueError("TCRR consensus tar member day disagrees with date")
    try:
        datetime.strptime(
            f"{match.group('date')} {match.group('hour')}", "%Y-%m-%d %H"
        )
    except ValueError as exc:
        raise ValueError("TCRR consensus tar member calendar date is invalid") from exc


def validate_tar_member(
    *, name: str, size: int, is_file: bool, is_symlink: bool, is_hardlink: bool
) -> None:
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise ValueError("TCRR tar member path is unsafe")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("TCRR tar member path is unsafe")
    if is_symlink or is_hardlink:
        raise ValueError("TCRR tar links are forbidden")
    if isinstance(size, bool):
        raise ValueError("TCRR tar member size must be an integer")
    try:
        member_size = operator.index(size)
    except TypeError as exc:
        raise ValueError("TCRR tar member size must be an integer") from exc
    if member_size < 0 or member_size > MAXIMUM_MEMBER_BYTES:
        raise ValueError("TCRR tar member size exceeds the frozen guard")
    if is_file and name.endswith("-consensus"):
        _validate_consensus_member_calendar(name)


def public_availability_time(label: datetime) -> datetime:
    _validate_label(label)
    return label + timedelta(minutes=PUBLICATION_DELAY_MINUTES)


def _single_line(document: str, keyword: str) -> str:
    prefix = f"{keyword} "
    values = [line[len(prefix) :] for line in document.splitlines() if line.startswith(prefix)]
    if len(values) != 1:
        raise ValueError(f"TCRR requires exactly one {keyword} line")
    return values[0]


def _parse_consensus_time(document: str, keyword: str) -> datetime:
    value = _single_line(document, keyword)
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise ValueError(f"TCRR {keyword} time is malformed") from exc


def _decode_relay_identity(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9+/]{27}", value, re.ASCII) is None:
        raise ValueError("TCRR relay identity is not canonical unpadded base64")
    try:
        decoded = base64.b64decode(value + "=", validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("TCRR relay identity is malformed base64") from exc
    if len(decoded) != 20:
        raise ValueError("TCRR relay identity is not a 20-byte RSA digest")
    return decoded.hex().upper()


def _parse_authorities(lines: Sequence[str]) -> list[str]:
    identities: list[str] = []
    for line in lines:
        if not line.startswith("dir-source "):
            continue
        parts = line.split()
        if len(parts) < 7 or FINGERPRINT_PATTERN.fullmatch(parts[2]) is None:
            raise ValueError("TCRR dir-source line is malformed")
        if not parts[1].endswith("-legacy"):
            identities.append(parts[2])
    if len(identities) < 5 or len(set(identities)) != len(identities):
        raise ValueError("TCRR authority identities are insufficient or duplicated")
    return sorted(identities)


def _parse_signatures(raw: bytes, authority_ids: Sequence[str]) -> dict[str, Any]:
    start = raw.find(b"network-status-version")
    first_signature = raw.find(b"directory-signature ", start)
    if start < 0 or first_signature < 0:
        raise ValueError("TCRR signed consensus boundary is missing")
    signed = raw[start : first_signature + len(b"directory-signature ")]
    pattern = re.compile(
        rb"^directory-signature ([^\n]+)\n"
        rb"-----BEGIN SIGNATURE-----\n"
        rb"([A-Za-z0-9+/=\n]+)"
        rb"-----END SIGNATURE-----$",
        re.MULTILINE,
    )
    signatures: list[dict[str, Any]] = []
    for match in pattern.finditer(raw):
        fields = match.group(1).decode("ascii").split()
        if len(fields) == 2:
            algorithm, identity, signing_key = "sha1", fields[0], fields[1]
        elif len(fields) == 3:
            algorithm, identity, signing_key = fields
        else:
            raise ValueError("TCRR directory-signature header is malformed")
        if algorithm != "sha1":
            raise ValueError("TCRR ns consensus signature must use SHA-1")
        if (
            FINGERPRINT_PATTERN.fullmatch(identity) is None
            or FINGERPRINT_PATTERN.fullmatch(signing_key) is None
        ):
            raise ValueError("TCRR signature fingerprint is malformed")
        try:
            signature = base64.b64decode(
                re.sub(rb"\s+", b"", match.group(2)), validate=True
            )
        except (binascii.Error, ValueError) as exc:
            raise ValueError("TCRR signature payload is malformed") from exc
        if len(signature) < 128 or len(signature) > 512:
            raise ValueError("TCRR signature payload length is outside the guard")
        signatures.append(
            {
                "identity": identity,
                "signing_key": signing_key,
                "signature_bytes": len(signature),
            }
        )
    identities = [row["identity"] for row in signatures]
    signing_keys = [row["signing_key"] for row in signatures]
    if len(identities) != len(set(identities)):
        raise ValueError("TCRR has duplicate authority signatures")
    if len(signing_keys) != len(set(signing_keys)):
        raise ValueError("TCRR has duplicate signing-key digests")
    if not set(identities).issubset(authority_ids):
        raise ValueError("TCRR signature membership is outside dir-source authorities")
    majority = len(authority_ids) // 2 + 1
    if len(signatures) < majority:
        raise ValueError("TCRR does not carry a strict authority majority")
    return {
        "signed_portion_sha1": hashlib.sha1(signed).hexdigest(),
        "signatures": sorted(signatures, key=lambda row: row["identity"]),
    }


def _parse_relays(lines: Sequence[str], known_flags: set[str]) -> list[dict[str, Any]]:
    relays: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    def finish() -> None:
        if current is None:
            return
        if "flags" not in current or "consensus_bandwidth" not in current:
            raise ValueError("TCRR relay entry lacks one s or w line")
        relays.append(current)

    for line in lines:
        if line.startswith("r "):
            finish()
            parts = line.split()
            if len(parts) != 9:
                raise ValueError("TCRR relay r line is malformed")
            current = {"relay_identity": _decode_relay_identity(parts[2])}
        elif line.startswith("s ") or line == "s":
            if current is None or "flags" in current:
                raise ValueError("TCRR relay s line is misplaced or duplicated")
            flags = line.split()[1:]
            if len(flags) != len(set(flags)) or not set(flags).issubset(known_flags):
                raise ValueError("TCRR relay flags are duplicated or unknown")
            current["flags"] = sorted(flags)
        elif line.startswith("w "):
            if current is None or "consensus_bandwidth" in current:
                raise ValueError("TCRR relay w line is misplaced or duplicated")
            values: dict[str, str] = {}
            for token in line.split()[1:]:
                if "=" not in token:
                    raise ValueError("TCRR relay w token is malformed")
                key, value = token.split("=", 1)
                if key in values:
                    raise ValueError("TCRR relay w key is duplicated")
                values[key] = value
            bandwidth = values.get("Bandwidth")
            if bandwidth is None or re.fullmatch(r"0|[1-9][0-9]*", bandwidth) is None:
                raise ValueError("TCRR relay consensus bandwidth is malformed")
            current["consensus_bandwidth"] = int(bandwidth)
        elif line == "directory-footer":
            finish()
            current = None
            break
    if current is not None:
        finish()
    identities = [relay["relay_identity"] for relay in relays]
    if not relays or len(identities) != len(set(identities)):
        raise ValueError("TCRR relay identities are empty or duplicated")
    return sorted(relays, key=lambda relay: relay["relay_identity"])


def parse_consensus_document(raw: bytes, *, expected_label: datetime) -> dict[str, Any]:
    _validate_label(expected_label)
    if not isinstance(raw, bytes) or not raw:
        raise ValueError("TCRR consensus must be non-empty bytes")
    if len(raw) > MAXIMUM_MEMBER_BYTES:
        raise ValueError("TCRR consensus exceeds the frozen member guard")
    if not raw.startswith(TYPE_ANNOTATION):
        raise ValueError("TCRR consensus type annotation differs from the contract")
    if b"\r" in raw or b"\x00" in raw or not raw.endswith(b"\n"):
        raise ValueError("TCRR consensus bytes are not canonical LF text")
    try:
        document = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("TCRR consensus contains non-ASCII bytes") from exc
    lines = document.splitlines()
    if lines[1] != "network-status-version 3":
        raise ValueError("TCRR consensus network-status version differs")
    if _single_line(document, "vote-status") != "consensus":
        raise ValueError("TCRR document is not a consensus")
    method_text = _single_line(document, "consensus-method")
    if re.fullmatch(r"[1-9][0-9]*", method_text) is None:
        raise ValueError("TCRR consensus method is malformed")
    valid_after = _parse_consensus_time(document, "valid-after")
    fresh_until = _parse_consensus_time(document, "fresh-until")
    valid_until = _parse_consensus_time(document, "valid-until")
    if valid_after != expected_label:
        raise ValueError("TCRR valid-after differs from the target label")
    availability = public_availability_time(expected_label)
    if not valid_after < availability < fresh_until < valid_until:
        raise ValueError("TCRR validity interval does not cover frozen availability")
    delay = _single_line(document, "voting-delay").split()
    if len(delay) != 2 or any(re.fullmatch(r"[1-9][0-9]*", item) is None for item in delay):
        raise ValueError("TCRR voting-delay is malformed")
    known_flags_list = _single_line(document, "known-flags").split()
    if not known_flags_list or known_flags_list != sorted(set(known_flags_list)):
        raise ValueError("TCRR known-flags are empty, duplicated, or unsorted")
    authority_ids = _parse_authorities(lines)
    signature_data = _parse_signatures(raw, authority_ids)
    relays = _parse_relays(lines, set(known_flags_list))
    summary = {
        "valid_after": valid_after.isoformat().replace("+00:00", "Z"),
        "availability_time": availability.isoformat().replace("+00:00", "Z"),
        "fresh_until": fresh_until.isoformat().replace("+00:00", "Z"),
        "valid_until": valid_until.isoformat().replace("+00:00", "Z"),
        "consensus_method": int(method_text),
        "voting_delay_seconds": [int(value) for value in delay],
        "known_flags": known_flags_list,
        "authority_identities": authority_ids,
        "signature_headers": signature_data["signatures"],
        "signed_portion_sha1": signature_data["signed_portion_sha1"],
        "relays": relays,
    }
    return {
        "member_bytes": len(raw),
        "member_sha256": sha256_bytes(raw),
        "relay_count": len(relays),
        "authority_count": len(authority_ids),
        "signature_count": len(signature_data["signatures"]),
        "summary_identity_sha256": canonical_hash(summary),
        **summary,
    }


def canonical_member_identity(*, label: datetime, parsed: Mapping[str, Any]) -> str:
    _validate_label(label)
    allowed = {
        "member_bytes",
        "member_sha256",
        "relay_count",
        "authority_count",
        "signature_count",
        "summary_identity_sha256",
        "valid_after",
        "availability_time",
        "fresh_until",
        "valid_until",
        "consensus_method",
        "voting_delay_seconds",
        "known_flags",
        "authority_identities",
        "signature_headers",
        "signed_portion_sha1",
        "relays",
    }
    if set(parsed) != allowed:
        raise ValueError("TCRR parsed consensus fields differ from the frozen set")
    return canonical_hash(
        {
            "label": label.isoformat().replace("+00:00", "Z"),
            "member_name": expected_member_name(label),
            "parsed": dict(parsed),
        }
    )


def build_protocol() -> dict[str, Any]:
    manifest = archive_manifest()
    labels = anchor_labels()
    core: dict[str, Any] = {
        "protocol_version": "tor_consensus_relay_reconfiguration_source_v1",
        "source_id": "TCRR",
        "outcomes_opened": False,
        "market_clocks_opened": False,
        "full_source_incidence_opened": False,
        "mechanism_features_opened": False,
        "source_only_probe_opened": True,
        "decision_binding": {
            "path": str(DECISION_PATH),
            "sha256": DECISION_SHA256,
        },
        "implementation_binding": {
            "path": str(SCRIPT_PATH),
            "sha256": sha256_file(SCRIPT_PATH),
        },
        "source_contract": {
            "operator": "The Tor Project CollecTor",
            "index_url": INDEX_URL,
            "archive_base": ARCHIVE_BASE,
            "transport": "public HTTPS XZ-compressed POSIX tar",
            "type_annotation": TYPE_ANNOTATION.decode("ascii").rstrip("\n"),
            "source_interval": {
                "start_inclusive": SOURCE_START.isoformat().replace("+00:00", "Z"),
                "end_exclusive": SOURCE_END_EXCLUSIVE.isoformat().replace("+00:00", "Z"),
                "anchor_hours_utc": list(ANCHOR_HOURS),
                "expected_archives": EXPECTED_ARCHIVES,
                "expected_target_documents": EXPECTED_TARGET_DOCUMENTS,
                "first_target": labels[0].isoformat().replace("+00:00", "Z"),
                "last_target": labels[-1].isoformat().replace("+00:00", "Z"),
            },
            "archive_manifest": manifest,
            "archive_manifest_sha256": canonical_hash({"archives": manifest}),
            "archive_total_compressed_bytes": sum(
                row["compressed_bytes"] for row in manifest
            ),
            "member_path_grammar": TARGET_MEMBER_PATTERN.pattern,
            "retained_fields": [
                "consensus validity times and method",
                "voting-delay",
                "known relay flags",
                "authority identity and signing-key digests",
                "consensus signature byte counts and signed-portion SHA-1",
                "relay RSA identity",
                "relay flags",
                "consensus bandwidth",
                "archive/member byte counts and SHA-256 identities",
            ],
            "forbidden_fields": [
                "relay IP address, nickname, contact, software version, exit policy, geolocation, or free text",
                "Onionoo, third-party mirrors, microdescriptor consensus, server descriptors, or current-state fallback",
                "BTC bars/returns/funding/PnL",
                "prior alpha clocks or portfolio outcomes",
            ],
        },
        "availability_contract": {
            "source_field": "valid-after",
            "delay_minutes": PUBLICATION_DELAY_MINUTES,
            "effective_time": "valid-after + 15 minutes",
            "must_be_strictly_before": "fresh-until",
            "stale_consensus_grace_allowed": False,
            "archive HTTP timestamp_used_as_availability": False,
            "market_execution_rule": "strictly later market bar only",
        },
        "parser_contract": {
            "maximum_archive_bytes": MAXIMUM_ARCHIVE_BYTES,
            "maximum_member_bytes": MAXIMUM_MEMBER_BYTES,
            "disk_limit_gib": DISK_LIMIT_GIB,
            "tar_order_defines_time_order": False,
            "links_allowed": False,
            "absolute_or_parent_paths_allowed": False,
            "line_encoding": "ASCII with LF only and final LF required",
            "signature_algorithm": "SHA-1 for plain ns consensus",
            "authority_signature_membership": "distinct subset with strict majority",
            "strict_majority_required": True,
            "parse_twice_identity_required": True,
            "raw_archive_persistence": False,
        },
        "bounded_probe": dict(PROBE),
        "gates": {
            "all_48_archive_hashes_and_sizes_match": True,
            "all_5844_target_members_present_once": True,
            "no_unsafe_tar_members_or_links": True,
            "every_target_valid_after_matches_label": True,
            "every_availability_time_is_fresh": True,
            "every_signature_identity_is_a_distinct_dir_source_authority": True,
            "every_target_has_nonempty_unique_relays_with_s_and_w": True,
            "parse_twice_and_monthly_replay_identities_match": True,
            "coverage_denominator": EXPECTED_TARGET_DOCUMENTS,
            "any_failure_effect": "REJECT_NO_REPAIR before mechanism or BTC data",
        },
    }
    core["manifest_hash"] = canonical_hash(core)
    return core


def validate_protocol(payload: Mapping[str, Any]) -> None:
    if payload.get("source_id") != "TCRR":
        raise ValueError("TCRR source_id differs")
    for key in (
        "outcomes_opened",
        "market_clocks_opened",
        "full_source_incidence_opened",
        "mechanism_features_opened",
    ):
        if payload.get(key) is not False:
            raise ValueError(f"TCRR {key} must remain false")
    if sha256_file(DECISION_PATH) != DECISION_SHA256:
        raise ValueError("TCRR decision document hash differs")
    manifest = payload.get("source_contract", {}).get("archive_manifest")
    if manifest != archive_manifest():
        raise ValueError("TCRR archive manifest differs from the frozen rows")
    if payload.get("source_contract", {}).get("archive_total_compressed_bytes") != 1068849328:
        raise ValueError("TCRR archive byte total differs")
    stored_hash = payload.get("manifest_hash")
    if not isinstance(stored_hash, str):
        raise ValueError("TCRR manifest hash is missing")
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if stored_hash != canonical_hash(core):
        raise ValueError("TCRR manifest hash differs")


def write_protocol(output: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    payload = build_protocol()
    validate_protocol(payload)
    candidate = Path(output)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_protocol(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "source_id": payload["source_id"],
                "manifest_hash": payload["manifest_hash"],
                "expected_archives": EXPECTED_ARCHIVES,
                "expected_target_documents": EXPECTED_TARGET_DOCUMENTS,
                "market_or_outcomes_opened": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
