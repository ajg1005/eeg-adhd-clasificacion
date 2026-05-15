from pathlib import Path
import os
import sys


BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"
MODELS_DIR = Path(os.getenv("MODELS_DIR", BASE_DIR / "models"))
FIGURES_DIR = Path(os.getenv("FIGURES_DIR", BASE_DIR / "Figuras"))


CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


def configure_import_paths() -> None:
    scripts_path = str(SCRIPTS_DIR)
    if scripts_path not in sys.path:
        sys.path.append(scripts_path)
