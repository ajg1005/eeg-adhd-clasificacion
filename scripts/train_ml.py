"""Entrenamiento y evaluacion CV cross-subject de los modelos ML clasicos."""

import json
import pandas as pd
from sklearn.base import clone
from sklearn.utils.class_weight import compute_sample_weight

from scripts.data_load import load_dataset
from scripts.epochs import create_epochs
from scripts.evaluation import metrics_dict
from scripts.feature_pipeline import build_features_from_epochs
from scripts.pipeline import get_models
from scripts.paths import CSV_PATH, FIGURES_DIR, ML_BEST_CONFIG_PATH, RESULTS_DIR
from scripts.preprocessing import preprocess_dataset
from scripts.split import make_group_kfold_splits
from scripts.visual import (
    plot_confusion_matrix,
    plot_model_metric_bar,
    plot_roc_curve,
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


EPOCH_SIZE = 1920
STEP_SIZE = 960
SFREQ = 128
NPERSEG = 960
N_SPLITS = 5


def _fit_model(model_name, model, X_train, y_train):
    # XGBoost no soporta class_weight nativo: balanceamos via sample_weight por fold.
    # El resto de modelos sklearn ya llevan class_weight="balanced" en su Pipeline.
    fitted_model = clone(model)
    fit_kwargs = {}
    if model_name == "xgboost":
        fit_kwargs["model__sample_weight"] = compute_sample_weight("balanced", y_train)
    fitted_model.fit(X_train, y_train, **fit_kwargs)
    return fitted_model


def _model_score(fitted_model, X_test):
    if hasattr(fitted_model, "predict_proba"):
        return fitted_model.predict_proba(X_test)[:, 1]
    if hasattr(fitted_model, "decision_function"):
        return fitted_model.decision_function(X_test)
    return None


def _evaluate_model_on_fold(model_name, model, split_data, oof_predictions):
    X_train = split_data["X_train"]
    X_test = split_data["X_test"]
    y_train = split_data["y_train"]
    y_test = split_data["y_test"]
    fold = split_data["fold"]

    fitted_model = _fit_model(model_name, model, X_train, y_train)
    y_pred = fitted_model.predict(X_test)
    y_score = _model_score(fitted_model, X_test)

    oof_predictions[model_name]["y_true"].extend(y_test)
    oof_predictions[model_name]["y_pred"].extend(y_pred)
    if y_score is not None:
        oof_predictions[model_name]["y_score"].extend(y_score)

    m = metrics_dict(y_test, y_pred)
    print(
        f"{model_name} - Fold {fold} | "
        f"Accuracy epoch: {m['accuracy']:.4f} | "
        f"Balanced Acc: {m['balanced_accuracy']:.4f} | "
        f"F1 epoch: {m['f1_score']:.4f}"
    )
    return {
        "Modelo": model_name,
        "Fold": fold,
        "Accuracy_epoch": m["accuracy"],
        "BalancedAccuracy_epoch": m["balanced_accuracy"],
        "Precision_epoch": m["precision"],
        "Recall_epoch": m["recall"],
        "F1_epoch": m["f1_score"],
    }


def _evaluate_fold(split_data, models, oof_predictions):
    fold = split_data["fold"]
    groups_train = split_data["groups_train"]
    groups_test = split_data["groups_test"]

    print(f"\nFold {fold}")
    print("Sujetos train:", len(set(groups_train)))
    print("Sujetos test:", len(set(groups_test)))
    overlap_fold = set(groups_train) & set(groups_test)
    print("Solapamiento train/test en fold:", len(overlap_fold))

    return [
        _evaluate_model_on_fold(model_name, model, split_data, oof_predictions)
        for model_name, model in models.items()
    ]


def _build_summary_df(results):
    results_df = pd.DataFrame(results)
    summary_df = results_df.groupby("Modelo").agg({
        "Accuracy_epoch": ["mean", "std"],
        "BalancedAccuracy_epoch": ["mean", "std"],
        "Precision_epoch": ["mean", "std"],
        "Recall_epoch": ["mean", "std"],
        "F1_epoch": ["mean", "std"],
    }).round(4)
    return results_df, summary_df


def _build_best_model_config(best_model_name, summary_df, x_features, groups_epochs, eeg_cols):
    def metric(name, stat):
        return float(summary_df.loc[best_model_name, (name, stat)])

    return {
        "best_model": best_model_name,
        "feature_mode": "combined",
        "sfreq": SFREQ,
        "epoch_size": EPOCH_SIZE,
        "step_size": STEP_SIZE,
        "nperseg": NPERSEG,
        "apply_zscore": False,
        "apply_filtering": False,
        "channels": list(eeg_cols),
        "selection_metric": "F1_epoch_mean_cv",
        "cv_metrics": {
            "accuracy_epoch_mean": metric("Accuracy_epoch", "mean"),
            "accuracy_epoch_std": metric("Accuracy_epoch", "std"),
            "balanced_accuracy_epoch_mean": metric("BalancedAccuracy_epoch", "mean"),
            "balanced_accuracy_epoch_std": metric("BalancedAccuracy_epoch", "std"),
            "precision_epoch_mean": metric("Precision_epoch", "mean"),
            "precision_epoch_std": metric("Precision_epoch", "std"),
            "recall_epoch_mean": metric("Recall_epoch", "mean"),
            "recall_epoch_std": metric("Recall_epoch", "std"),
            "f1_epoch_mean": metric("F1_epoch", "mean"),
            "f1_epoch_std": metric("F1_epoch", "std"),
        },
        "dataset_summary": {
            "n_epochs_total": int(len(x_features)),
            "n_features": int(x_features.shape[1]),
            "n_subjects": int(len(set(groups_epochs))),
        },
        "training_strategy": f"CV cross-subject StratifiedGroupKFold de {N_SPLITS} folds",
        "note": "Este archivo lo usa export_model.py para entrenar y exportar el modelo final.",
    }


def _save_best_model_figures(best_model_name, oof_predictions):
    best_y_true = oof_predictions[best_model_name]["y_true"]
    best_y_pred = oof_predictions[best_model_name]["y_pred"]
    best_y_score = oof_predictions[best_model_name]["y_score"]

    plot_confusion_matrix(
        best_y_true,
        best_y_pred,
        save_path=FIGURES_DIR / f"{best_model_name}_cv_confusion_matrix.png",
    )

    has_scores = len(best_y_score) == len(best_y_true) and len(best_y_score) > 0
    if not has_scores:
        print(f"\nEl modelo {best_model_name} no dispone de scores continuos para ROC.")
        return

    plot_roc_curve(
        best_y_true,
        best_y_score,
        save_path=FIGURES_DIR / f"{best_model_name}_cv_roc_curve.png",
    )


def main():
    # Cargar el dataset, limpiar y preprocesar
    df = load_dataset(CSV_PATH)
    df_clean, eeg_cols = preprocess_dataset(df)

    # Segmentar en epochs
    x_epochs, y_epochs, groups_epochs = create_epochs(
        df=df_clean,
        eeg_columns=eeg_cols,
        label_column="Class",
        group_column="ID",
        epoch_size=EPOCH_SIZE,
        step_size=STEP_SIZE,
    )

    print("Shape X_epochs:", x_epochs.shape)
    print("Shape y_epochs:", y_epochs.shape)
    print("Shape groups_epochs:", groups_epochs.shape)

    x_features = build_features_from_epochs(
        x_epochs=x_epochs,
        channel_names=eeg_cols,
        feature_mode="combined",
        sfreq=SFREQ,
        nperseg=NPERSEG,
    )

    print("Shape X_features:", x_features.shape)

    # CV cross-subject sobre todo el dataset
    cv_splits = make_group_kfold_splits(
        x_features,
        y_epochs,
        groups_epochs,
        n_splits=N_SPLITS,
    )
    models = get_models()

    oof_predictions = {
        model_name: {"y_true": [], "y_pred": [], "y_score": []}
        for model_name in models.keys()
    }

    results = []
    for split_data in cv_splits:
        results.extend(_evaluate_fold(split_data, models, oof_predictions))

    results_df, summary_df = _build_summary_df(results)

    print("\nRESUMEN CV CROSS-SUBJECT")
    print(summary_df)

    results_df.to_csv(RESULTS_DIR / "ml_cv_fold_results.csv", index=False)
    summary_df.to_csv(RESULTS_DIR / "ml_cv_summary.csv")

    # Elegir mejor modelo usando la media de CV
    best_model_name = summary_df[("F1_epoch", "mean")].idxmax()
    print(f"\nMejor modelo según media de F1 por epoch en CV: {best_model_name}")

    best_model_config = _build_best_model_config(
        best_model_name, summary_df, x_features, groups_epochs, eeg_cols
    )

    with open(ML_BEST_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(best_model_config, f, indent=4)

    print(f"\nConfiguración del mejor modelo guardada en: {ML_BEST_CONFIG_PATH}")

    # Figuras de comparación entre modelos
    plot_model_metric_bar(
        summary_df,
        metric_name="BalancedAccuracy_epoch",
        save_path=FIGURES_DIR / "cv_model_comparison_balanced_accuracy.png",
    )
    plot_model_metric_bar(
        summary_df,
        metric_name="F1_epoch",
        save_path=FIGURES_DIR / "cv_model_comparison_f1.png",
    )

    # Figuras del mejor modelo (matriz de confusión + ROC OOF)
    _save_best_model_figures(best_model_name, oof_predictions)


if __name__ == "__main__":
    main()
