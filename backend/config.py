from pathlib import Path
from urllib.parse import quote
import os


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.getenv("MODELS_DIR", BASE_DIR / "models"))
FIGURES_DIR = Path(os.getenv("FIGURES_DIR", BASE_DIR / "Figuras"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", BASE_DIR / "results"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    pg_pass = os.getenv("POSTGRES_PASSWORD")
    if pg_pass:
        pg_user = quote(os.getenv("POSTGRES_USER", "eeg_app"), safe="")
        pg_host = os.getenv("POSTGRES_HOST", "localhost")
        pg_port = os.getenv("POSTGRES_PORT", "5432")
        pg_db = os.getenv("POSTGRES_DB", "eeg_adhd")
        DATABASE_URL = f"postgresql+psycopg://{pg_user}:{quote(pg_pass, safe='')}@{pg_host}:{pg_port}/{pg_db}"
    else:
        DATABASE_URL = f"sqlite:///{(BASE_DIR / 'experiments.db').as_posix()}"


CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
