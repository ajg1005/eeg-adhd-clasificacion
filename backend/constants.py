"""Re-exporta las constantes/helpers compartidos definidos en scripts.constants.

Existe por compatibilidad con los imports del backend (que ya usaban
`from backend.constants import ...`). El source of truth real vive en
scripts/constants.py para que research y app no diverjan.
"""
from scripts.constants import (  # noqa: F401
    CLASS_ADHD,
    CLASS_CONTROL,
    CLASS_TO_LABEL,
    CLASS_UNKNOWN,
    RANDOM_STATE,
    REQUIRED_COLUMNS,
    REQUIRED_EEG_COLUMNS,
    normalize_class_to_int,
    normalize_class_to_label,
)
