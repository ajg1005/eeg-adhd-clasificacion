"""Constantes compartidas entre los scripts y el backend.

Fuente unica de verdad para semillas y otros valores transversales que
necesitan ser identicos en research y en la app para garantizar
reproducibilidad.
"""

# Semilla unica usada en todos los splits, modelos y remuestreos del proyecto.
# Cambiarla aqui afecta automaticamente a research scripts y a backend.
RANDOM_STATE = 42
