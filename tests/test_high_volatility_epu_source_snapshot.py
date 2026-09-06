import csv,hashlib,json
from pathlib import Path
def test_epu_snapshot_is_complete_and_was_downloaded_after_preregistration():
 p=Path("data/us_daily_epu_1985_2026_aug.csv");m=json.loads(Path("data/us_daily_epu_1985_2026_aug_manifest.json").read_text());assert hashlib.sha256(p.read_bytes()).hexdigest()==m["sha256"];rows=list(csv.DictReader(p.open()));assert len(rows)==15195;assert m["first_observation"]=="1985-01-01";assert m["last_observation"]=="2026-08-08";assert m["downloaded_after_preregistration_commit"]=="fd581aa3";h=m.pop("manifest_hash");encoded=json.dumps(m,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode();assert hashlib.sha256(encoded).hexdigest()==h
