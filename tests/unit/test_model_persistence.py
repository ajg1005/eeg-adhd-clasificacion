import json
from pathlib import Path

import pandas as pd

from backend.core.config import BASE_DIR
from backend.training.persistence import persist_final_model
from backend.training.data import prepare_epochs


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else BASE_DIR / path


def test_persist_final_model_ml_creates_artifacts(tmp_path, monkeypatch, eeg_dataframe_factory):
    from backend.training import persistence as model_persistence

    monkeypatch.setattr(model_persistence, "TRAINED_MODELS_DIR", tmp_path / "trained")
    eeg_params = {
        "epoch_size": 16,
        "step_size": 16,
        "sfreq": 128,
        "nperseg": 16,
        "feature_mode": "temporal",
        "use_filtering": False,
    }
    df = pd.DataFrame(eeg_dataframe_factory(samples_per_patient=32))
    prepared = prepare_epochs(df, eeg_params)

    record = persist_final_model(
        experiment_id=123,
        model_type="ml",
        model_name="random_forest",
        eeg_params=eeg_params,
        model_params={"n_estimators": 5, "max_depth": 2},
        training_params={},
        prepared=prepared,
    )

    artifact_path = _resolve(record["artifact_path"])
    feature_columns_path = _resolve(record["feature_columns_path"])
    metadata_path = artifact_path.parent / "metadata.json"

    assert artifact_path.exists()
    assert feature_columns_path.exists()
    assert metadata_path.exists()
    assert record["model_type"] == "ml"
    assert record["model_family"] == "machine_learning"
    assert record["n_features"] > 0
    assert record["n_epochs_training"] == len(prepared.y_epochs)
    assert record["n_subjects_training"] == len(set(prepared.groups_epochs))
    assert record["file_size_bytes"] == artifact_path.stat().st_size

    feature_columns = json.loads(feature_columns_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert len(feature_columns) == record["n_features"]
    assert metadata["model_name"] == "random_forest"
    assert metadata["feature_mode"] == "temporal"
    assert metadata["class_mapping"] == {"0": "Control", "1": "ADHD"}