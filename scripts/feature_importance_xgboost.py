"""
Importancia de caracteristicas para XGBoost.

Este script complementa scripts.feature_importance, que se mantiene centrado en
el mejor modelo ML exportado. Aqui se entrena XGBoost con el mismo esquema de
features y el mismo split cross-subject, se calcula permutation_importance.

Ejecucion:
- python -m scripts.feature_importance_xgboost
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_sample_weight

from scripts.constants import RANDOM_STATE
from scripts.data_load import load_dataset
from scripts.epochs import create_epochs
from scripts.feature_importance import aggregate_by_channel, load_json, stratified_subsample
from scripts.feature_pipeline import align_feature_columns, build_features_from_config
from scripts.paths import (
    CSV_PATH,
    FIGURES_DIR,
    ML_FEATURE_COLUMNS_PATH as FEATURE_COLUMNS_PATH,
    ML_METADATA_PATH as METADATA_PATH,
    RESULTS_DIR,
)
from scripts.pipeline import create_ml_model
from scripts.preprocessing import preprocess_dataset
from scripts.split import make_group_shuffle_split

TEST_SIZE = 0.2
TEST_SAMPLE_SIZE = 0
N_REPEATS = 10
N_JOBS = 1
SCORING = "f1_weighted"
TOP_N = 20
DRY_RUN = False

XGBOOST_COLOR = "#DD8452"

def save_xgboost_figures(importance_df, channel_df, scoring, top_n):
    top = importance_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(
        top["feature"],
        top["importance_mean"],
        xerr=top["importance_std"],
        color=XGBOOST_COLOR,
        edgecolor="black",
    )
    ax.set_xlabel(f"Caida media de {scoring} al permutar la caracteristica")
    ax.set_title(f"Top {top_n} caracteristicas mas importantes (XGBoost)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    top_fig = FIGURES_DIR / f"xgboost_feature_importance_top{top_n}.png"
    fig.savefig(top_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        channel_df["channel"],
        channel_df["importance_sum"],
        color=XGBOOST_COLOR,
        edgecolor="black",
    )
    ax.set_ylabel("Suma de importancia por canal")
    ax.set_title("Importancia agregada por canal EEG (XGBoost)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    channel_fig = FIGURES_DIR / "xgboost_feature_importance_by_channel.png"
    fig.savefig(channel_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return top_fig, channel_fig


def main():

    print("Cargando metadata y dataset...")
    metadata = load_json(METADATA_PATH)
    feature_columns = load_json(FEATURE_COLUMNS_PATH)

    df = load_dataset(CSV_PATH)
    df_clean, eeg_cols = preprocess_dataset(df)

    x_epochs, y_epochs, groups_epochs = create_epochs(
        df=df_clean,
        eeg_columns=eeg_cols,
        epoch_size=metadata["epoch_size"],
        step_size=metadata["step_size"],
    )

    x_features = align_feature_columns(
        build_features_from_config(x_epochs, eeg_cols, metadata),
        feature_columns,
    )
    y_epochs = np.asarray(y_epochs).astype(int)
    groups_epochs = np.asarray(groups_epochs).astype(str)

    print(f"Total epochs: {len(x_features)} | pacientes: {len(set(groups_epochs))}")

    X_train, X_test, y_train, y_test, groups_train, groups_test = make_group_shuffle_split(
        x_features,
        y_epochs,
        groups_epochs,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    overlap = len(set(groups_train) & set(groups_test))
    print(
        f"Split: {len(X_train)} train ({len(set(groups_train))} pacientes) | "
        f"{len(X_test)} test ({len(set(groups_test))} pacientes) | overlap pacientes={overlap}"
    )

    if DRY_RUN:
        print("Dry-run OK: XGBoost puede entrenarse y evaluarse con este split cross-subject.")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Entrenando XGBoost en train cross-subject...")
    model = create_ml_model("xgboost")
    if "model__n_jobs" in model.get_params():
        model.set_params(model__n_jobs=1)
    model.fit(
        X_train,
        y_train,
        model__sample_weight=compute_sample_weight("balanced", y_train),
    )

    x_test_used, y_test_used = stratified_subsample(X_test, y_test, TEST_SAMPLE_SIZE)
    print(
        f"Calculando permutation_importance para XGBoost "
        f"({N_REPEATS} repeticiones, {len(x_test_used)} epochs, "
        f"scoring={SCORING}, n_jobs={N_JOBS})..."
    )
    result = permutation_importance(
        model,
        x_test_used,
        y_test_used,
        scoring=SCORING,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )

    importance_df = (
        pd.DataFrame(
            {
                "feature": feature_columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    importance_csv = RESULTS_DIR / "xgboost_feature_importance.csv"
    importance_df.to_csv(importance_csv, index=False)

    channel_df = aggregate_by_channel(importance_df, metadata.get("channels", []))
    channel_csv = RESULTS_DIR / "xgboost_feature_importance_by_channel.csv"
    channel_df.to_csv(channel_csv, index=False)

    top_fig, channel_fig = save_xgboost_figures(
        importance_df,
        channel_df,
        SCORING,
        TOP_N,
    )


    print(f"\nTabla XGBoost por feature : {importance_csv}")
    print(f"Tabla XGBoost por canal   : {channel_csv}")
    print(f"Top features XGBoost      : {top_fig}")
    print(f"Por canal XGBoost         : {channel_fig}")

    print("\nTop caracteristicas XGBoost:")
    print(importance_df.head(max(TOP_N // 4, 5)).to_string(index=False))


if __name__ == "__main__":
    main()