import csv,hashlib,json
from pathlib import Path
def test_hash_bound_news_sentiment_snapshot():
 m=json.loads(Path('data/frbsf_daily_news_sentiment_1980_2026_aug_manifest.json').read_text());raw=Path(m['raw_path']);derived=Path(m['derived_csv_path']);assert hashlib.sha256(raw.read_bytes()).hexdigest()==m['raw_sha256'];assert hashlib.sha256(derived.read_bytes()).hexdigest()==m['derived_csv_sha256'];core={k:v for k,v in m.items() if k!='manifest_hash'};assert hashlib.sha256(json.dumps(core,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()==m['manifest_hash'];rows=list(csv.DictReader(derived.open()));assert len(rows)==m['rows'];assert rows[0]['date']==m['first_day']=='1980-01-01';assert rows[-1]['date']==m['last_day']=='2026-08-02'
