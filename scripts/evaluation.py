"""Helpers de evaluacion compartidos entre research (scripts) y app (backend).

Centraliza las metricas de clasificacion binaria y la busqueda de threshold
optimo. Ambas funciones se usan en:

- scripts.train_dl y scripts.train_ml (CV en research).
- backend/services/training_runners.py (entrenamiento interactivo desde la UI).
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


# Devuelve un diccionario con las metricas estandar de clasificacion binaria
def metrics_dict(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


# Busca el threshold que maximiza balanced_accuracy (desempate por F1)
def find_best_threshold(y_true, y_score, lo=0.2, hi=0.8, n_points=61):
    best_threshold = 0.5
    best_balanced = -np.inf
    best_f1 = -np.inf

    for threshold in np.linspace(lo, hi, n_points):
        y_pred = (y_score >= threshold).astype(int)
        balanced = balanced_accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        if balanced > best_balanced or (np.isclose(balanced, best_balanced) and f1 > best_f1):
            best_threshold = float(threshold)
            best_balanced = float(balanced)
            best_f1 = float(f1)

    return best_threshold
