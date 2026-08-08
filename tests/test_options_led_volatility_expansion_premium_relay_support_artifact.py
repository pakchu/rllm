from __future__ import annotations
import hashlib,json
from training import build_options_led_volatility_expansion_premium_relay_support as s

def test_support_artifact_hashes_and_manifest_are_frozen() -> None:
 assert hashlib.sha256(s.DEFAULT_CLOCK.read_bytes()).hexdigest()=='b79bd105784db59980a83d1e1e75e3334e954f76e0f06a6d44eca1dc017e6bf1'
 assert hashlib.sha256(s.DEFAULT_RESULT.read_bytes()).hexdigest()=='afbf8157c2c85aec0470563cdfba1b45afe18472a617633d2140b6ec6c1c15a7'
 report=json.loads(s.DEFAULT_RESULT.read_text())
 core={k:v for k,v in report.items() if k!='manifest_hash'}
 assert report['manifest_hash']==s.chash(core)
 assert report['clock']['sha256']==hashlib.sha256(s.DEFAULT_CLOCK.read_bytes()).hexdigest()
