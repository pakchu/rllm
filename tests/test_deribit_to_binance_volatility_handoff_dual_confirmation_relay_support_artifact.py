import hashlib
import json

from training import build_deribit_to_binance_volatility_handoff_dual_confirmation_relay_support as support


def test_dbvhdr_support_is_frozen_pass_before_novelty_and_economics():
    assert hashlib.sha256(support.RESULT.read_bytes()).hexdigest() == "af5ea6006193a37808750f0f3a4ef56a972735589e7c438633fdc54a85c3ea58"
    data = json.loads(support.RESULT.read_text())
    core = {key: value for key, value in data.items() if key != "manifest_hash"}
    assert data["manifest_hash"] == support.canonical_hash(core) == "cfb2ee42d46fdaf9561ae31c147d710f308e103d776fa2005263424224c1d029"
    assert data["clock"]["sha256"] == hashlib.sha256(support.CLOCK.read_bytes()).hexdigest()
    assert data["clock"]["rows"] == 391
    assert data["support_passed"] is True
    assert all(data["support_checks"].values())
    assert data["advance_to_gross9_novelty"] is True
    assert data["advance_to_economic_outcomes"] is False
