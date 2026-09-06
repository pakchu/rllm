import numpy as np
from training import evaluate_g9_macro_historical as h


def test_fixed_combinations_and_full_h1():
    assert h.WINDOWS['2026H1']==('2026-01-01','2026-07-01')
    assert np.allclose(h.WEIGHTS[1,:5],h.WEIGHTS[0,:5]*.5)
    assert h.WEIGHTS[1,5]==1
    assert np.array_equal(h.WEIGHTS[2,:5],h.WEIGHTS[0,:5])
    assert h.WEIGHTS[2,5]==.5
    assert h.DESIGN['selection'].startswith('none')


def test_registration_json_roundtrip_is_idempotent(tmp_path,monkeypatch):
    monkeypatch.setattr(h,'OUT',tmp_path)
    monkeypatch.setattr(h.exporter,'validate_frozen_inputs',lambda: {})
    monkeypatch.setattr(h.exporter,'INPUT_BINDINGS',{})
    monkeypatch.setattr(h.base,'sha',lambda path:'fixed')
    assert h.register()==h.register()
