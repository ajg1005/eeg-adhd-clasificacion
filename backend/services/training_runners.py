from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance

from backend.modeling.dl_factory import create_dl_model, create_early_stopping
from backend.modeling.model_factory import create_ml_model
from backend.services.training_data import (
    CLASS_TO_LABEL,
    PreparedEpochs,
    features_for_mode,
    n_splits_for_groups,
)
from scripts.evaluation import find_best_threshold, metrics_dict
from scripts.split import make_group_kfold_splits, make_group_shuffle_split


FEATURE_IMPORTANCE_MAX_EPOCHS = 80
FEATURE_IMPORTANCE_N_REPEATS = 1
FEATURE_IMPORTANCE_TOP_FEATURES = 15
FEATURE_IMPORTANCE_TOP_CHANNELS = 10
FEATURE_IMPORTANCE_SCORING = "f1_weighted"


def patient_results(groups: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    result_df = pd.DataFrame({"patient_id": groups, "true": y_true, "pred": y_pred})

    for patient_id, patient_df in result_df.groupby("patient_id", sort=False):
        pred_counts = patient_df["pred"].value_counts(normalize=True)
        predicted_label = int(patient_df["pred"].mode().iloc[0])
        true_label = int(patient_df["true"].mode().iloc[0])
        rows.append(
            {
                "patient_id": str(patient_id),
                "true_label": CLASS_TO_LABEL[true_label],
                "predicted_label": CLASS_TO_LABEL[predicted_label],
                "n_epochs": int(len(patient_df)),
                "control_epoch_percentage": float(pred_counts.get(0, 0.0)),
                "adhd_epoch_percentage": float(pred_counts.get(1, 0.0)),
                "correct": bool(true_label == predicted_label),
            }
        )

    return rows


def run_ml_cross_subject_cv(
    model_name: str,
    model_params: dict[str, Any],
    eeg_params: dict[str, Any],
    prepared: PreparedEpochs,
) -> dict[str, Any]:
    features = features_for_mode(prepared.x_epochs, prepared.eeg_columns, eeg_params)
    n_splits = n_splits_for_groups(prepared.y_epochs, prepared.groups_epochs)
    cv_splits = make_group_kfold_splits(features, prepared.y_epochs, prepared.groups_epochs, n_splits=n_splits)
    base_model = create_ml_model(model_name, model_params)

    fold_results = []
    feature_importance = None
    y_true_all = []
    y_pred_all = []
    groups_all = []

    for split_data in cv_splits:
        fitted_model = clone(base_model)
        fitted_model.fit(split_data["X_train"], split_data["y_train"])
        y_pred = fitted_model.predict(split_data["X_test"]).astype(int)
        y_true = np.asarray(split_data["y_test"]).astype(int)
        groups_test = np.asarray(split_data["groups_test"]).astype(str)

        fold_metrics = metrics_dict(y_true, y_pred)
        fold_metrics["fold"] = int(split_data["fold"])
        fold_metrics["n_train_subjects"] = int(len(set(split_data["groups_train"])))
        fold_metrics["n_test_subjects"] = int(len(set(groups_test)))
        fold_results.append(fold_metrics)

        if feature_importance is None:
            feature_importance = _safe_feature_importance_for_fold(
                model=fitted_model,
                X_test=split_data["X_test"],
                y_test=y_true,
                eeg_columns=prepared.eeg_columns,
                fold=int(split_data["fold"]),
            )

        y_true_all.extend(y_true.tolist())
        y_pred_all.extend(y_pred.tolist())
        groups_all.extend(groups_test.tolist())

    return {
        "y_true": np.asarray(y_true_all, dtype=int),
        "y_pred": np.asarray(y_pred_all, dtype=int),
        "groups": np.asarray(groups_all, dtype=str),
        "fold_results": fold_results,
        "evaluation_mode": f"{n_splits}-fold StratifiedGroupKFold cross-subject CV",
        "feature_importance": feature_importance,
    }


def run_dl_cross_subject_cv(
    model_name: str,
    model_params: dict[str, Any],
    training_params: dict[str, Any],
    prepared: PreparedEpochs,
) -> dict[str, Any]:
    n_splits = n_splits_for_groups(prepared.y_epochs, prepared.groups_epochs)
    cv_splits = make_group_kfold_splits(
        prepared.x_epochs,
        prepared.y_epochs,
        prepared.groups_epochs,
        n_splits=n_splits,
    )

    fold_results = []
    y_true_all = []
    y_pred_all = []
    groups_all = []

    for split_data in cv_splits:
        fold = int(split_data["fold"])
        x_train, x_val, y_train, y_val, groups_train, groups_val = make_group_shuffle_split(
            split_data["X_train"],
            split_data["y_train"],
            split_data["groups_train"],
            test_size=0.2,
            random_state=42 + fold,
        )

        x_train = np.asarray(x_train).astype(np.float32)
        x_val = np.asarray(x_val).astype(np.float32)
        X_test = np.asarray(split_data["X_test"]).astype(np.float32)
        y_train = np.asarray(y_train).astype(np.float32)
        y_val = np.asarray(y_val).astype(np.float32)
        y_test = np.asarray(split_data["y_test"]).astype(int)
        groups_test = np.asarray(split_data["groups_test"]).astype(str)

        model = create_dl_model(
            model_name=model_name,
            input_shape=x_train.shape[1:],
            model_params=model_params,
            training_params=training_params,
        )
        model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=int(training_params.get("epochs", 25)),
            batch_size=int(training_params.get("batch_size", 32)),
            callbacks=_dl_callbacks(training_params),
            verbose=0,
        )

        batch_size = int(training_params.get("batch_size", 32))
        y_val_score = model.predict(x_val, batch_size=batch_size, verbose=0).reshape(-1)
        threshold = find_best_threshold(y_val.astype(int), y_val_score)
        y_score = model.predict(X_test, batch_size=batch_size, verbose=0).reshape(-1)
        y_pred = (y_score >= threshold).astype(int)

        fold_metrics = metrics_dict(y_test, y_pred)
        fold_metrics["fold"] = fold
        fold_metrics["best_threshold"] = float(threshold)
        fold_metrics["n_train_subjects"] = int(len(set(groups_train)))
        fold_metrics["n_val_subjects"] = int(len(set(groups_val)))
        fold_metrics["n_test_subjects"] = int(len(set(groups_test)))
        fold_results.append(fold_metrics)

        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())
        groups_all.extend(groups_test.tolist())

    return {
        "y_true": np.asarray(y_true_all, dtype=int),
        "y_pred": np.asarray(y_pred_all, dtype=int),
        "groups": np.asarray(groups_all, dtype=str),
        "fold_results": fold_results,
        "evaluation_mode": f"{n_splits}-fold StratifiedGroupKFold cross-subject CV with inner validation",
    }


def _safe_feature_importance_for_fold(
    model,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    eeg_columns: list[str],
    fold: int,
) -> dict[str, Any]:
    try:
        return _feature_importance_for_fold(model, X_test, y_test, eeg_columns, fold)
    except Exception as exc:
        return {
            "method": "permutation_importance",
            "scoring": FEATURE_IMPORTANCE_SCORING,
            "n_repeats": FEATURE_IMPORTANCE_N_REPEATS,
            "evaluated_epochs": 0,
            "source": f"fold {fold} test set sin solape de pacientes",
            "top_features": [],
            "by_channel": [],
            "error": str(exc),
        }


def _feature_importance_for_fold(
    model,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    eeg_columns: list[str],
    fold: int,
) -> dict[str, Any]:
    x_eval, y_eval = _stratified_subsample(X_test, y_test, FEATURE_IMPORTANCE_MAX_EPOCHS)

    result = permutation_importance(
        model,
        x_eval,
        y_eval,
        scoring=FEATURE_IMPORTANCE_SCORING,
        n_repeats=FEATURE_IMPORTANCE_N_REPEATS,
        random_state=42,
        n_jobs=1,
    )

    importance_df = pd.DataFrame(
        {
            "feature": list(x_eval.columns),
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    return {
        "method": "permutation_importance",
        "scoring": FEATURE_IMPORTANCE_SCORING,
        "n_repeats": FEATURE_IMPORTANCE_N_REPEATS,
        "evaluated_epochs": int(len(x_eval)),
        "source": f"fold {fold} test set sin solape de pacientes",
        "top_features": _importance_rows(importance_df, FEATURE_IMPORTANCE_TOP_FEATURES),
        "by_channel": _importance_rows(
            _aggregate_importance_by_channel(importance_df, eeg_columns),
            FEATURE_IMPORTANCE_TOP_CHANNELS,
        ),
    }


def _stratified_subsample(X: pd.DataFrame, y: np.ndarray, max_rows: int) -> tuple[pd.DataFrame, np.ndarray]:
    y = np.asarray(y).astype(int)
    if max_rows <= 0 or len(X) <= max_rows:
        return X.reset_index(drop=True), y

    _, x_sample, _, y_sample = _groupless_stratified_split(X, y, max_rows)
    return x_sample.reset_index(drop=True), np.asarray(y_sample).astype(int)


def _groupless_stratified_split(
    X: pd.DataFrame,
    y: np.ndarray,
    sample_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    stratify = y if sample_size >= len(np.unique(y)) else None
    try:
        return train_test_split(
            X,
            y,
            test_size=sample_size,
            stratify=stratify,
            random_state=42,
        )
    except ValueError:
        return train_test_split(
            X,
            y,
            test_size=sample_size,
            stratify=None,
            random_state=42,
        )


def _aggregate_importance_by_channel(importance_df: pd.DataFrame, channels: list[str]) -> pd.DataFrame:
    rows = []
    for channel in channels:
        mask = importance_df["feature"].str.startswith(f"{channel}_")
        rows.append(
            {
                "feature": channel,
                "importance_mean": float(importance_df.loc[mask, "importance_mean"].sum()),
                "importance_std": 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False)


def _importance_rows(importance_df: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    rows = []
    for row in importance_df.head(limit).itertuples(index=False):
        rows.append(
            {
                "feature": str(row.feature),
                "importance_mean": float(row.importance_mean),
                "importance_std": float(row.importance_std),
            }
        )
    return rows


def _dl_callbacks(training_params: dict[str, Any]):
    import keras

    patience = int(training_params.get("early_stopping_patience", 5))
    return [
        create_early_stopping(patience),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=1,
            min_lr=1e-6,
            verbose=0,
        ),
    ]
