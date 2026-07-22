"""Hash-seal the built SMCC source without parsing any feature row."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training import evaluate_same_millisecond_cascade_support as evaluator
from training import preregister_same_millisecond_cascade as prereg


EXPECTED_ROWS = 420_768


@dataclass(frozen=True)
class SealConfig:
    source: str = str(evaluator.DEFAULT_SOURCE)
    source_manifest: str = str(evaluator.DEFAULT_SOURCE_MANIFEST)
    output: str = str(evaluator.SOURCE_ACCESS_SEAL_PATH)


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_seal(cfg: SealConfig, *, expected_rows: int = EXPECTED_ROWS) -> dict[str, Any]:
    payload = evaluator.load_preregistration()
    source = Path(cfg.source)
    manifest_path = Path(cfg.source_manifest)
    source_hash = sha256_file(source)
    manifest_hash = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol = manifest.get("protocol", {})
    if protocol.get("outcomes_opened") is not False:
        raise ValueError("SMCC source manifest opened outcomes")
    if manifest.get("combined_output") != cfg.source:
        raise ValueError("SMCC source manifest path mismatch")
    if manifest.get("combined_sha256") != source_hash:
        raise ValueError("SMCC source manifest hash mismatch")
    if int(manifest.get("rows", -1)) != expected_rows:
        raise ValueError("SMCC source manifest row count mismatch")
    if protocol.get("source_archive_manifest_sha256") != prereg.ARCHIVE_MANIFEST_SHA256:
        raise ValueError("SMCC archive source contract mismatch")
    if protocol.get("source_audit_sha256") != prereg.SOURCE_AUDIT_SHA256:
        raise ValueError("SMCC source audit contract mismatch")
    return {
        "protocol_version": "same_millisecond_cascade_source_access_seal_v1",
        "preregistration_hash": payload["manifest_hash"],
        "preregistration_file_sha256": evaluator.PREREGISTRATION_FILE_SHA256,
        "source_path": cfg.source,
        "source_sha256": source_hash,
        "source_manifest_path": cfg.source_manifest,
        "source_manifest_sha256": manifest_hash,
        "source_manifest_rows": expected_rows,
        "source_manifest_outcomes_opened": False,
        "source_rows_parsed": 0,
        "outcomes_opened": False,
    }


def write_once(path: str | Path, payload: dict[str, Any]) -> str:
    output = Path(path)
    content = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_bytes() != content:
            raise RuntimeError("refusing to overwrite frozen SMCC source access seal")
        return "verified_existing"
    with output.open("xb") as handle:
        handle.write(content)
    return "created"


def parse_args() -> SealConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(evaluator.DEFAULT_SOURCE))
    parser.add_argument("--source-manifest", default=str(evaluator.DEFAULT_SOURCE_MANIFEST))
    parser.add_argument("--output", default=str(evaluator.SOURCE_ACCESS_SEAL_PATH))
    return SealConfig(**vars(parser.parse_args()))


def main() -> None:
    cfg = parse_args()
    payload = build_seal(cfg)
    status = write_once(cfg.output, payload)
    print(
        json.dumps(
            {
                "status": status,
                "source_sha256": payload["source_sha256"],
                "source_manifest_sha256": payload["source_manifest_sha256"],
                "source_rows_parsed": 0,
                "outcomes_opened": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
