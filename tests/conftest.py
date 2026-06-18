import os
import tempfile
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_DIR = Path(__file__).resolve().parents[1]
TEST_DATABASE_PATH = Path(tempfile.gettempdir()) / f"eeg_adhd_tests_{uuid.uuid4().hex}.db"
TEST_DATASETS_DIR = Path(tempfile.gettempdir()) / f"eeg_adhd_test_datasets_{uuid.uuid4().hex}"
os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DATABASE_PATH.as_posix()}"
os.environ["DATASETS_DIR"] = TEST_DATASETS_DIR.as_posix()

TESTS_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = TESTS_DIR / "fixtures"
MODELS_DIR = PROJECT_DIR / "models"

ML_MODEL_PATH = MODELS_DIR / "ml" / "final_model.joblib"
DL_MODEL_PATH = MODELS_DIR / "dl" / "final_model.keras"


# Decoradores convenientes para tests que necesitan los modelos exportados.
requires_ml_model = pytest.mark.skipif(
    not ML_MODEL_PATH.exists(),
    reason=f"Falta {ML_MODEL_PATH}. Ejecuta python -m scripts.export_model.",
)
requires_dl_model = pytest.mark.skipif(
    not DL_MODEL_PATH.exists(),
    reason=f"Falta {DL_MODEL_PATH}. Ejecuta python -m scripts.export_model_dl.",
)


@pytest.fixture(scope="session")
def client():
    from backend.db import init_db
    from backend.main import app

    init_db()
    return TestClient(app)


@pytest.fixture(scope="session")
def fixtures_dir():
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def sample_prediction_csv_path(fixtures_dir):
    return fixtures_dir / "sample_eeg_prediction.csv"


@pytest.fixture(scope="session")
def valid_eeg_dataset_csv_path(fixtures_dir):
    return fixtures_dir / "valid_eeg_dataset.csv"


@pytest.fixture(scope="session")
def invalid_missing_columns_csv_path(fixtures_dir):
    return fixtures_dir / "invalid_missing_columns.csv"


@pytest.fixture
def post_csv():
    """Helper para enviar un CSV a un endpoint via TestClient."""

    def _post(client, path, url, data=None):
        with path.open("rb") as csv_file:
            return client.post(
                url,
                data=data or {},
                files={"file": (path.name, csv_file, "text/csv")},
            )

    return _post


@pytest.fixture
def eeg_dataframe_factory():
    from backend.constants import REQUIRED_EEG_COLUMNS

    def _factory(patients=None, samples_per_patient=32):
        patients = patients or [
            ("control_1", 0),
            ("control_2", 0),
            ("adhd_1", 1),
            ("adhd_2", 1),
        ]
        rows = []

        for subject_index, (subject_id, label) in enumerate(patients):
            for sample_index in range(samples_per_patient):
                row = {
                    channel: float(label * 0.5 + subject_index * 0.01 + sample_index * 0.001 + channel_index * 0.0001)
                    for channel_index, channel in enumerate(REQUIRED_EEG_COLUMNS)
                }
                row["ID"] = subject_id
                row["Class"] = label
                rows.append(row)

        return rows

    return _factory
