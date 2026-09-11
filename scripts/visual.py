"""Funciones de visualizacion (matrices de confusion, ROC, comparativas)."""

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, auc, roc_curve

# Crear figuras y guardarlas
def _save_fig(fig, save_path):
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")


def plot_confusion_matrix(y_test, y_score, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, y_score, ax=ax)
    plt.tight_layout()
    _save_fig(fig, save_path)
    plt.close(fig)


def plot_roc_curve(y_test, y_score, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, y_score, ax=ax)
    plt.tight_layout()
    _save_fig(fig, save_path)
    plt.close(fig)


def plot_roc_curves_comparison(oof_predictions, model_names, labels=None, save_path=None):
    labels = labels or {}
    colors = {
        "random_forest": "#4C72B0",
        "rbf_svc": "#55A868",
        "xgboost": "#DD8452",
    }

    fig, ax = plt.subplots(figsize=(6.8, 5.4))

    for model_name in model_names:
        y_true = oof_predictions[model_name]["y_true"]
        y_score = oof_predictions[model_name]["y_score"]
        if len(y_true) == 0 or len(y_score) != len(y_true):
            continue

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        label = labels.get(model_name, model_name)
        ax.plot(
            fpr,
            tpr,
            linewidth=2.2,
            color=colors.get(model_name),
            label=f"{label} (AUC={roc_auc:.3f})",
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="0.55", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves - cross-subject CV")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    _save_fig(fig, save_path)
    plt.close(fig)

def plot_model_metric_bar(summary_df, metric_name, save_path=None):
    plot_df = summary_df.copy()
    means = plot_df[(metric_name, "mean")]
    stds = plot_df[(metric_name, "std")]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(means.index, means.values, yerr=stds.values, capsize=4)
    ax.set_title(f"Comparación de modelos - {metric_name}")
    ax.set_ylabel(metric_name)
    ax.set_xlabel("Modelo")
    plt.xticks(rotation=20)
    plt.tight_layout()
    _save_fig(fig, save_path)
    plt.close(fig)

