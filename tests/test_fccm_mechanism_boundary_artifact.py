from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "results/fccm_mechanism_boundary_2026-07-23.json"
FILE_SHA256 = "08eced75e484d5e0cc18882fef2672d24928f81995cb75ad0191131034c05184"
MANIFEST_HASH = "571554f181747fb61dd02612d36eacd16af04f81c90643c7266e12b8a0753dec"


def test_fccm_mechanism_boundary_is_hash_bound_and_explicitly_limited() -> None:
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == FILE_SHA256
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    canonical = json.dumps(
        core,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == MANIFEST_HASH
    assert payload["manifest_hash"] == MANIFEST_HASH
    assert payload["evidence_kind"] == (
        "hash_bound_self_attested_boundary_ledger"
    )
    assert payload["independent_os_file_access_trace_available"] is False
    exposure = payload["explicit_prior_exposure"]
    assert exposure["source_family_exposed"] is True
    assert exposure["wscf_aggregate_comparator_summary_exposed"] is True
    assert exposure["globally_pristine_human_holdout_claimed"] is False
    counters = payload["fccm_boundary_counters"]
    assert counters["fccm_features_derived"] == 0
    assert counters["fccm_candidate_incidence_rows_derived"] == 0
    assert counters["raw_comparator_data_rows_decoded"] == 0
    assert counters["btc_market_rows_decoded"] == 0
    assert counters["realized_funding_rows_decoded"] == 0
    assert counters["future_return_rows_decoded"] == 0
    assert counters["pnl_cagr_mdd_values_decoded"] == 0
    assert counters[
        "post_2023_bitfinex_or_wbtc_source_value_rows_decoded"
    ] == 0
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", str(ARTIFACT.relative_to(ROOT))],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert tracked.returncode == 0
    for binding in payload["direct_selection_artifacts_seen"]:
        path = ROOT / binding["path"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == binding["sha256"]
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", binding["path"]],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert tracked.returncode == 0, binding["path"]
