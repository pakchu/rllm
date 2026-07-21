"""Export the frozen DEHR-72 selection on a DEWH-comparable execution grid."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Mapping, cast

import pandas as pd

from training import preregister_deribit_expiry_hedge_release_support as dehr


POLICY_ID = "DEWH-144"
COMPARATOR_ID = "dehr:dehr_72_normalized"
PROTOCOL_VERSION = "dewh_dehr_comparator_clock_export_v1"
IMPLEMENTATION = Path("training/export_dewh_dehr_comparator_clocks.py")
DEWH_DECISION = Path(
    "docs/deribit-expiry-wall-handoff-mechanism-decision-2026-07-21.md"
)
DEWH_DECISION_SHA256 = (
    "f5b378b75e3d32c18e32b245f62674a7a7b25f90ec7761d865ddb6c627a93ce8"
)
DEHR_IMPLEMENTATION = Path(
    "training/preregister_deribit_expiry_hedge_release_support.py"
)
DEHR_IMPLEMENTATION_SHA256 = (
    "7600d3d4f7f72eb5f7462c872b1a3e5d1f864a36a79f1f1e5eb31eeb232cf1fc"
)
DEHR_PREREGISTRATION = Path(
    "results/deribit_expiry_hedge_release_support_preregistration_2026-07-20.json"
)
DEHR_PREREGISTRATION_SHA256 = (
    "21ef708fdbacca52323fc3d61086f8ed61af38ab143bab82fa5c16092fa4a36c"
)
DEHR_SUPPORT = Path("results/deribit_expiry_hedge_release_support_2026-07-20.json")
DEHR_SUPPORT_SHA256 = "fcc33c324263bc10709041504b0b78fc055d83e18224a6c6622fa3c9f47c9231"
DEHR_SOURCE = Path("data/deribit_btc_option_delivery_release_2019_2022.csv.gz")
DEHR_SOURCE_SHA256 = "a59953eb0efddbab7a28af9fdd0f61f204fa98d2de330cf1a4090293378b0fda"
DEHR_SOURCE_MANIFEST = Path(
    "results/deribit_btc_option_delivery_source_manifest_2026-07-20.json"
)
DEHR_SOURCE_MANIFEST_SHA256 = (
    "b1a2ed3a39b8e71adc0a46a5411d4f568eda3bdaa910cef64d9746fa6f5ea3e5"
)
DEHR_REJECTION = Path(
    "docs/deribit-expiry-hedge-release-support-rejection-2026-07-20.md"
)
DEHR_REJECTION_SHA256 = (
    "c8b0c18743057e9b217c932e742784f2f810241a4723869cbe9c912db88a25c2"
)
EXPECTED_PREREGISTRATION_HASH = (
    "d1797ea2eae04a85f9d917e27412bc8456878bd4b1d350b461b4bd64208e4c1e"
)
EXPECTED_SOURCE_MANIFEST_HASH = (
    "44b54dcd895a127dc89dc9c45f40f65c845814badeaf6c35bdabbc37e4e1b852"
)
EXPECTED_SUPPORT_RESULT_HASH = (
    "b118f24ac6e5796477865d3d9b95a3c0448b057f9b1fd30179588113e522a0f7"
)
EXPECTED_FROZEN_EVENT_CLOCK_HASH = (
    "319b548995f20f7db8065f1ff979762c39c39f2a5d143cb2575ef36a21f32310"
)
DEFAULT_CLOCK = Path("results/dewh_dehr_comparator_clocks_2026-07-21.csv.gz")
DEFAULT_MANIFEST = Path("results/dewh_dehr_comparator_clock_manifest_2026-07-21.json")
CLOCK_FIELDS = (
    "candidate_id",
    "causal_origin",
    "decision_time",
    "original_entry_time",
    "entry_time",
    "exit_time",
    "side",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle, object_pairs_hook=_reject_duplicate_pairs)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def normalized_entry(observation: pd.Timestamp) -> pd.Timestamp:
    if observation.tzinfo is None:
        raise ValueError("DEHR comparator observation lacks timezone")
    boundary = cast(pd.Timestamp, observation.ceil("5min"))
    return cast(pd.Timestamp, boundary + pd.Timedelta(minutes=5))


def _load_frozen_inputs() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    bindings = {
        DEWH_DECISION: DEWH_DECISION_SHA256,
        DEHR_IMPLEMENTATION: DEHR_IMPLEMENTATION_SHA256,
        DEHR_PREREGISTRATION: DEHR_PREREGISTRATION_SHA256,
        DEHR_SUPPORT: DEHR_SUPPORT_SHA256,
        DEHR_SOURCE: DEHR_SOURCE_SHA256,
        DEHR_SOURCE_MANIFEST: DEHR_SOURCE_MANIFEST_SHA256,
        DEHR_REJECTION: DEHR_REJECTION_SHA256,
    }
    for path, expected in bindings.items():
        if sha256_file(path) != expected:
            raise ValueError(f"DEWH DEHR comparator binding changed: {path}")

    preregistration = load_json(DEHR_PREREGISTRATION)
    support = load_json(DEHR_SUPPORT)
    source_manifest = load_json(DEHR_SOURCE_MANIFEST)
    if preregistration.get("artifact_hash") != EXPECTED_PREREGISTRATION_HASH:
        raise ValueError("DEHR preregistration hash changed")
    if support.get("result_hash") != EXPECTED_SUPPORT_RESULT_HASH:
        raise ValueError("DEHR support result hash changed")
    if support.get("event_clock_hash") != EXPECTED_FROZEN_EVENT_CLOCK_HASH:
        raise ValueError("DEHR frozen event clock hash changed")
    if support.get("outcomes_opened") is not False:
        raise ValueError("DEHR support artifact opened outcomes")
    if source_manifest.get("manifest_hash") != EXPECTED_SOURCE_MANIFEST_HASH:
        raise ValueError("DEHR source manifest hash changed")
    if source_manifest.get("aggregate", {}).get("sha256") != DEHR_SOURCE_SHA256:
        raise ValueError("DEHR source manifest binding changed")
    if (
        source_manifest.get("outcome_boundary", {}).get(
            "post_delivery_return_or_pnl_loaded"
        )
        is not False
    ):
        raise ValueError("DEHR source manifest opened outcomes")

    source = pd.read_csv(DEHR_SOURCE, compression="gzip")
    if list(source.columns) != dehr.SOURCE_COLUMNS:
        raise ValueError("DEHR source schema changed")
    for column in (
        "expiry_time",
        "delivery_event_time",
        "source_observation_earliest",
    ):
        source[column] = pd.to_datetime(source[column], utc=True, errors="raise")
    if len(source) != 1119 or source["expiry_time"].duplicated().any():
        raise ValueError("DEHR source row identity changed")
    if bool(source["expiry_time"].ge(pd.Timestamp("2023-01-01", tz="UTC")).any()):
        raise ValueError("DEHR comparator opened post-2022 source")
    return source, preregistration, support


def build_outputs() -> tuple[dict[str, Any], bytes]:
    source, preregistration, support = _load_frozen_inputs()
    cfg = dehr.Config()
    panel = dehr.build_signal_panel(source, cfg)
    schedule = panel.loc[panel["candidate"]].reset_index(drop=True)
    events = dehr.event_records(schedule)
    frozen_clock_hash = dehr.event_clock_hash(
        events,
        cfg=cfg,
        preregistration_hash=str(preregistration["artifact_hash"]),
        source_manifest_hash=EXPECTED_SOURCE_MANIFEST_HASH,
        source_sha256=DEHR_SOURCE_SHA256,
    )
    if frozen_clock_hash != EXPECTED_FROZEN_EVENT_CLOCK_HASH:
        raise ValueError("DEHR exact candidate selection changed")
    if len(schedule) != 159:
        raise ValueError("DEHR exact candidate incidence changed")

    rows: list[dict[str, Any]] = []
    for raw in schedule.to_dict(orient="records"):
        observation = cast(pd.Timestamp, raw["source_observation_earliest"])
        original_entry = cast(pd.Timestamp, raw["entry_time"])
        entry = normalized_entry(observation)
        exit_time = cast(pd.Timestamp, entry + pd.Timedelta(minutes=5 * 72))
        rows.append(
            {
                "candidate_id": COMPARATOR_ID,
                "causal_origin": cast(pd.Timestamp, raw["expiry_time"]),
                "decision_time": observation,
                "original_entry_time": original_entry,
                "entry_time": entry,
                "exit_time": exit_time,
                "side": int(raw["side"]),
            }
        )
    if any(row["side"] not in {-1, 1} for row in rows):
        raise ValueError("DEHR comparator side changed")
    if any(
        row["entry_time"].second
        or row["entry_time"].microsecond
        or row["entry_time"].minute % 5
        for row in rows
    ):
        raise ValueError("normalized DEHR entry left five-minute grid")
    if any(
        current["entry_time"] < previous["exit_time"]
        for previous, current in zip(rows, rows[1:])
    ):
        raise ValueError("normalized DEHR comparator overlaps")

    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=CLOCK_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            cast(
                Any,
                {
                    key: value.isoformat() if isinstance(value, pd.Timestamp) else value
                    for key, value in row.items()
                },
            )
        )
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0) as handle:
        handle.write(text.getvalue().encode("utf-8"))
    clock_bytes = output.getvalue()

    deltas = [
        (row["entry_time"] - row["original_entry_time"]).total_seconds() for row in rows
    ]
    core: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "comparator_id": COMPARATOR_ID,
        "as_of_date": "2026-07-21",
        "implementation_binding": {
            "path": str(IMPLEMENTATION),
            "sha256": sha256_file(IMPLEMENTATION),
        },
        "dewh_decision_binding": {
            "path": str(DEWH_DECISION),
            "sha256": DEWH_DECISION_SHA256,
        },
        "frozen_dehr_bindings": {
            "implementation_sha256": DEHR_IMPLEMENTATION_SHA256,
            "preregistration_sha256": DEHR_PREREGISTRATION_SHA256,
            "preregistration_hash": EXPECTED_PREREGISTRATION_HASH,
            "support_sha256": DEHR_SUPPORT_SHA256,
            "support_result_hash": EXPECTED_SUPPORT_RESULT_HASH,
            "source_sha256": DEHR_SOURCE_SHA256,
            "source_manifest_sha256": DEHR_SOURCE_MANIFEST_SHA256,
            "source_manifest_hash": EXPECTED_SOURCE_MANIFEST_HASH,
            "rejection_document_sha256": DEHR_REJECTION_SHA256,
            "frozen_event_clock_hash": frozen_clock_hash,
        },
        "normalization": {
            "selection_changed": False,
            "side_changed": False,
            "hold_bars": 72,
            "rule": (
                "ceil source_observation_earliest to five-minute boundary, "
                "then wait one complete five-minute bucket"
            ),
            "rows": len(rows),
            "original_off_grid_rows": sum(
                bool(
                    row["original_entry_time"].second
                    or row["original_entry_time"].microsecond
                    or row["original_entry_time"].minute % 5
                )
                for row in rows
            ),
            "normalized_off_grid_rows": 0,
            "minimum_delay_seconds": min(deltas),
            "maximum_delay_seconds": max(deltas),
        },
        "clock": {
            "path": str(DEFAULT_CLOCK),
            "sha256": hashlib.sha256(clock_bytes).hexdigest(),
            "gzip_mtime": 0,
            "schema": list(CLOCK_FIELDS),
            "rows": len(rows),
            "first_entry": rows[0]["entry_time"].isoformat(),
            "last_exit": rows[-1]["exit_time"].isoformat(),
            "longs": sum(row["side"] == 1 for row in rows),
            "shorts": sum(row["side"] == -1 for row in rows),
        },
        "outcome_boundary": {
            "dehr_source_rows_read": len(source),
            "dehr_candidate_rows_reconstructed": len(rows),
            "market_rows_loaded": 0,
            "funding_rows_loaded": 0,
            "performance_artifacts_parsed": 0,
            "return_or_pnl_fields_read": 0,
            "post_2022_source_rows_loaded": 0,
            "network_calls": 0,
            "economic_outcomes_computed": False,
        },
        "dehr_reopened_or_repaired": False,
        "authorized_use": "DEWH source-only related-family novelty comparison only",
    }
    manifest = {**core, "manifest_hash": canonical_hash(core)}
    return manifest, clock_bytes


def publish(
    manifest_path: Path,
    clock_path: Path,
    manifest: Mapping[str, Any],
    clock_bytes: bytes,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    clock_path.parent.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    try:
        clock_fd = os.open(clock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        created.append(clock_path)
        with os.fdopen(clock_fd, "wb") as handle:
            handle.write(clock_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        manifest_fd = os.open(
            manifest_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
        created.append(manifest_path)
        payload = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        with os.fdopen(manifest_fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    manifest, clock_bytes = build_outputs()
    publish(DEFAULT_MANIFEST, DEFAULT_CLOCK, manifest, clock_bytes)
    print(
        json.dumps(
            {
                "manifest": str(DEFAULT_MANIFEST),
                "manifest_hash": manifest["manifest_hash"],
                "clock": str(DEFAULT_CLOCK),
                "clock_sha256": manifest["clock"]["sha256"],
                "rows": manifest["clock"]["rows"],
                "original_off_grid_rows": manifest["normalization"][
                    "original_off_grid_rows"
                ],
                "economic_outcomes_computed": manifest["outcome_boundary"][
                    "economic_outcomes_computed"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
