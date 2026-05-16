from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.getenv("MODELS_DIR", BASE_DIR / "models"))
FIGURES_DIR = Path(os.getenv("FIGURES_DIR", BASE_DIR / "Figuras"))


CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
