import csv
import hashlib
import json
from pathlib import Path


def test_hash_bound_gpr_snapshot_and_exact_derived_coverage():
    manifest = json.loads(Path("data/global_daily_gpr_recent_1985_2026_aug_manifest.json").read_text())
    raw = Path(manifest["raw_path"])
    derived = Path(manifest["derived_csv_path"])
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == manifest["raw_sha256"]
    assert hashlib.sha256(derived.read_bytes()).hexdigest() == manifest["derived_csv_sha256"]
    core = {k: v for k, v in manifest.items() if k != "manifest_hash"}
    payload = json.dumps(core, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    assert hashlib.sha256(payload).hexdigest() == manifest["manifest_hash"]
    with derived.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == manifest["rows"] == 15190
    assert rows[0]["DAY"] == manifest["first_day"] == "19850101"
    assert rows[-1]["DAY"] == manifest["last_day"] == "20260803"
    assert list(rows[0]) == manifest["columns"] == ["DAY", "GPRD", "GPRD_ACT", "GPRD_THREAT"]
