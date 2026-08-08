import hashlib
import json
from pathlib import Path

import pandas as pd

from training import train_high_volatility_microstructure_ridge_relay as train


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_hvmrr_model_is_frozen_before_oos_incidence_or_outcomes():
    model = json.loads(train.MODEL.read_text())
    core = {key: value for key, value in model.items() if key != "manifest_hash"}
    assert model["manifest_hash"] == train.canonical_hash(core)
    assert model["policy_id"] == "HVMRR-6"
    assert model["training_rows"] == 2724
    assert model["training_high_volatility_rows"] == 871
    assert model["prediction_strength_threshold"] == 0.0009743008286605152
    assert model["oos_incidence_opened"] is False
    assert model["oos_outcomes_opened"] is False
    assert model["refit_authorized"] is False


def test_hvmrr_model_hashes_bind_preregistration_sources_and_training_rows():
    model = json.loads(train.MODEL.read_text())
    for key in ("preregistration", "source_manifest", "feature_panel", "pretraining_rows"):
        item = model[key]
        assert item["sha256"] == sha(Path(item["path"]))
    manifest = json.loads(train.SOURCE_MANIFEST.read_text())
    core = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    assert manifest["manifest_hash"] == train.canonical_hash(core)
    assert manifest["oos_incidence_opened"] is False
    assert manifest["oos_outcomes_opened"] is False


def test_hvmrr_training_rows_end_strictly_before_oos_train():
    rows = pd.read_csv(train.TRAINING, compression="gzip", usecols=["decision_time", "entry_time", "exit_time"])
    for column in rows:
        rows[column] = pd.to_datetime(rows[column], utc=True)
    assert rows.decision_time.min() >= train.LABEL_START
    assert rows.exit_time.max() < train.LABEL_END
    assert len(rows) == 2724
