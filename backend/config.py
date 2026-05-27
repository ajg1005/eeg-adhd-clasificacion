from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.getenv("MODELS_DIR", BASE_DIR / "models"))
FIGURES_DIR = Path(os.getenv("FIGURES_DIR", BASE_DIR / "Figuras"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", BASE_DIR / "results"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://eeg_app:eeg_app_password@localhost:5432/eeg_adhd",
)


CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
