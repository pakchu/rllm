from __future__ import annotations
import gzip,hashlib,json
from pathlib import Path
import pandas as pd
from training import export_gross9_structural_clocks as e

EXPECTED={
'cand_rex_veto_7':'4a0bf038122d7e9d009a02faf28e0d3ea21408c76b7edef8b42d5515b6d9d9a9',
'fresh_kimchi_fx':'749d6ac664001510d404c308c3c5abdca4b1b045f8f13dd1f6ec73223cf7a059',
'frozen_annual_rank7':'9d0125530975a3a7a4699f68f3c05d093ea56912fbc067094260ce150876bc01',
'markov_transition_long':'d9bf01d624795c2654157ef03d3b343ee01357c8c13e94a5f0f8eaa41c104b33',
'rex_taker_low_range_position':'cb3e31a337d9950078e1071fb606e1936f0c8a54e09202f03c9cd0cf561cc3cd'}

def digest(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()

def test_manifest_binds_complete_authoritative_roster_and_counts()->None:
 p=e.DEFAULT_MANIFEST;assert digest(p)=='5433812da786a959cda1cfcf4825bc2e4a228ea8152a4b8cce1e867f29adf073'
 d=json.loads(p.read_text());core={k:v for k,v in d.items() if k!='manifest_hash'}
 assert d['manifest_hash']==e.canonical_hash(core)=='c1f7c2096cea035d053dd3d7b887b13f3220b6d96ddb99893b5be26cb44ae650'
 assert d['authority']['weights']==e.EXPECTED_WEIGHTS
 assert d['all_authoritative_counts_verified'] is True
 assert set(d['clocks'])==set(e.EXPECTED_WEIGHTS)
 for sleeve,record in d['clocks'].items():
  assert record['counts']==e.EXPECTED_COUNTS[sleeve]
  assert record['rows']==sum(e.EXPECTED_COUNTS[sleeve].values())
  assert record['sha256']==EXPECTED[sleeve]==digest(Path(record['path']))

def test_clock_bytes_are_deterministic_gzip_and_valid()->None:
 for sleeve,expected in EXPECTED.items():
  path=e.DEFAULT_OUTPUT_DIR/f'{sleeve}.csv.gz';raw=path.read_bytes()
  assert raw[:2]==b'\x1f\x8b' and raw[4:8]==b'\0\0\0\0'
  frame=pd.read_csv(path,compression='gzip')
  checked=e.validate_clock(frame,sleeve)
  assert len(checked)==sum(e.EXPECTED_COUNTS[sleeve].values())
  assert hashlib.sha256(gzip.compress(e.deterministic_csv_bytes(checked),mtime=0)).hexdigest()==expected

def test_anchor_and_all_input_hashes_validate()->None:
 assert e.validate_anchor()['weights']==e.EXPECTED_WEIGHTS
 assert set(e.validate_frozen_inputs())==set(e.INPUT_BINDINGS)
