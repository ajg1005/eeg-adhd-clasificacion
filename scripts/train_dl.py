from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from tensorflow import keras

from data_load import load_dataset
from epochs import create_epochs
from preprocessing import preprocess_dataset
from signal_preprocessing import apply_basic_filtering, zscore_per_subject
from split import make_group_kfold_splits, make_group_shuffle_split
from tf_models import build_model
from visual import plot_confusion_matrix, plot_roc_curve


CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "adhdata.csv"
OUTPUT_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = OUTPUT_DIR / "Figuras"
TRAINING_CURVES_DIR = FIGURES_DIR / "training_curves_tf"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_CURVES_DIR.mkdir(parents=True, exist_ok=True)


MODELS_TO_RUN = ["eegnet", "cnn_1d", "cnn_lstm"]

EPOCH_SIZE = 512
STEP_SIZE = 256

BATCH_SIZE = 32
LEARNING_RATE = 3e-4
N_EPOCHS = 40
PATIENCE = 4
DROPOUT = 0.4
LR_REDUCE_FACTOR = 0.5
LR_REDUCE_PATIENCE = 1
LR_MIN = 1e-6

N_SPLITS = 5
RANDOM_STATE = 42


def set_seed(seed):
    keras.utils.set_random_seed(seed)


def compute_metrics(y_true, y_pred):
    return {
        "Accuracy_epoch": accuracy_score(y_true, y_pred),
        "BalancedAccuracy_epoch": balanced_accuracy_score(y_true, y_pred),
        "Precision_epoch": precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "Recall_epoch": recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "F1_epoch": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
    }


def find_best_threshold(y_true, y_score):
    """
    Ajusta el umbral usando la validacion interna.
    Maximiza balanced accuracy y usa F1 como desempate.
    """
    best_threshold = 0.5
    best_bal_acc = -np.inf
    best_f1 = -np.inf

    for threshold in np.linspace(0.2, 0.8, 61):
        y_pred = (y_score >= threshold).astype(int)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        if bal_acc > best_bal_acc or (np.isclose(bal_acc, best_bal_acc) and f1 > best_f1):
            best_threshold = float(threshold)
            best_bal_acc = float(bal_acc)
            best_f1 = float(f1)

    return best_threshold


def plot_training_history(history, model_name, fold):
    """
    Guarda curvas de entrenamiento y validacion para loss, accuracy y AUC.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (metric_name, metric_label) in zip(
        axes,
        [("loss", "Loss"), ("accuracy", "Accuracy"), ("auc", "AUC")],
    ):
        train_values = history.history.get(metric_name)
        val_values = history.history.get(f"val_{metric_name}")

        if train_values is None or val_values is None:
            ax.axis("off")
            continue

        ax.plot(train_values, label=f"train_{metric_name}")
        ax.plot(val_values, label=f"val_{metric_name}")
        ax.set_title(metric_label)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle(f"{model_name} - Fold {fold} - Training Curves")
    fig.tight_layout()
    fig.savefig(
        TRAINING_CURVES_DIR / f"{model_name}_fold_{fold}_training_curves.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def build_callbacks():
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=LR_REDUCE_FACTOR,
            patience=LR_REDUCE_PATIENCE,
            min_lr=LR_MIN,
            verbose=1,
        ),
    ]


def build_summary_df(results):
    results_df = pd.DataFrame(results)
    return results_df.groupby("Modelo").agg({
        "Accuracy_epoch": ["mean", "std"],
        "BalancedAccuracy_epoch": ["mean", "std"],
        "Precision_epoch": ["mean", "std"],
        "Recall_epoch": ["mean", "std"],
        "F1_epoch": ["mean", "std"],
        "TestLoss": ["mean", "std"],
    }).round(4)


def prepare_dl_arrays(X_train, X_val, X_test, y_train, y_val, y_test):
    return (
        np.asarray(X_train).astype(np.float32),
        np.asarray(X_val).astype(np.float32),
        np.asarray(X_test).astype(np.float32),
        np.asarray(y_train).astype(np.float32),
        np.asarray(y_val).astype(np.float32),
        np.asarray(y_test).astype(np.float32),
    )


def build_compiled_model(model_name, input_shape, seed):
    set_seed(seed)

    model = build_model(
        model_name=model_name,
        input_shape=input_shape,
        dropout=DROPOUT,
    )

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=LEARNING_RATE,
            clipnorm=1.0,
        ),
        loss=keras.losses.BinaryCrossentropy(label_smoothing=0.05),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
        ],
    )

    return model


def fit_model(model_name, X_train, y_train, X_val, y_val, seed, fold):
    model = build_compiled_model(model_name, X_train.shape[1:], seed)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=build_callbacks(),
        verbose=1,
    )

    plot_training_history(history, model_name, fold)

    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    best_val_loss = float(np.min(history.history["val_loss"]))
    best_val_acc = float(np.max(history.history["val_accuracy"]))

    print(
        f"{model_name} | best_epoch={best_epoch:02d} | "
        f"best_val_loss={best_val_loss:.4f} | "
        f"best_val_acc={best_val_acc:.4f}"
    )

    return model


def evaluate_model(model, X_test, y_test, threshold=0.5):
    eval_outputs = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=0)
    y_score = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0).ravel()
    y_pred = (y_score >= threshold).astype(int)

    metrics = compute_metrics(y_test, y_pred)
    metrics["TestLoss"] = float(eval_outputs[0])
    metrics["BestThreshold"] = float(threshold)

    return metrics, y_test, y_pred, y_score


def run_single_model_fold(model_name, X_train, y_train, X_val, y_val, X_test, y_test, seed, fold):
    model = fit_model(
        model_name=model_name,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        seed=seed,
        fold=fold,
    )

    y_val_score = model.predict(X_val, batch_size=BATCH_SIZE, verbose=0).ravel()
    best_threshold = find_best_threshold(y_val, y_val_score)

    print(f"{model_name} | threshold ajustado en validacion: {best_threshold:.2f}")

    return evaluate_model(model, X_test, y_test, threshold=best_threshold)


def save_best_tf_outputs(summary_df, oof_predictions):
    best_model_name = summary_df[("F1_epoch", "mean")].idxmax()
    print(f"\nMejor modelo segun F1 medio: {best_model_name}")

    best_y_true = oof_predictions[best_model_name]["y_true"]
    best_y_pred = oof_predictions[best_model_name]["y_pred"]
    best_y_score = oof_predictions[best_model_name]["y_score"]

    plot_confusion_matrix(
        best_y_true,
        best_y_pred,
        save_path=FIGURES_DIR / f"{best_model_name}_tf_cv_confusion_matrix.png",
    )

    if len(best_y_true) == len(best_y_score) and len(best_y_score) > 0:
        plot_roc_curve(
            best_y_true,
            best_y_score,
            save_path=FIGURES_DIR / f"{best_model_name}_tf_cv_roc_curve.png",
        )
    else:
        print(f"\nEl modelo {best_model_name} no dispone de scores continuos para ROC.")


def main():
    print("TensorFlow version:", tf.__version__)

    df = load_dataset(CSV_PATH)
    df_clean, eeg_cols = preprocess_dataset(df)
    df_filtered = apply_basic_filtering(df_clean, eeg_cols, subject_col="ID")
    df_normalized = zscore_per_subject(df_filtered, eeg_cols, subject_col="ID")

    X_epochs, y_epochs, groups_epochs = create_epochs(
        df=df_normalized,
        eeg_columns=eeg_cols,
        label_column="Class",
        group_column="ID",
        epoch_size=EPOCH_SIZE,
        step_size=STEP_SIZE,
    )

    print("Shape X_epochs:", X_epochs.shape)
    print("Shape y_epochs:", y_epochs.shape)
    print("Shape groups_epochs:", groups_epochs.shape)

    cv_splits = make_group_kfold_splits(
        X_epochs,
        y_epochs,
        groups_epochs,
        n_splits=N_SPLITS,
    )

    results = []
    oof_predictions = {
        model_name: {"y_true": [], "y_pred": [], "y_score": []}
        for model_name in MODELS_TO_RUN
    }

    for split_data in cv_splits:
        fold = split_data["fold"]
        X_train_full = split_data["X_train"]
        X_test = split_data["X_test"]
        y_train_full = split_data["y_train"]
        y_test = split_data["y_test"]
        groups_train_full = split_data["groups_train"]
        groups_test = split_data["groups_test"]

        print(f"\n===== Fold {fold} =====")
        print("Sujetos train outer:", len(set(groups_train_full)))
        print("Sujetos test outer:", len(set(groups_test)))
        print("Solapamiento outer train/test:", len(set(groups_train_full) & set(groups_test)))

        X_train, X_val, y_train, y_val, groups_train, groups_val = make_group_shuffle_split(
            X_train_full,
            y_train_full,
            groups_train_full,
            test_size=0.2,
            random_state=RANDOM_STATE + fold,
        )

        print("Sujetos train inner:", len(set(groups_train)))
        print("Sujetos val inner:", len(set(groups_val)))
        print("Solapamiento inner train/val:", len(set(groups_train) & set(groups_val)))

        X_train, X_val, X_test_ready, y_train, y_val, y_test_ready = prepare_dl_arrays(
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
        )

        for model_name in MODELS_TO_RUN:
            print(f"\nEntrenando modelo TF: {model_name}")

            metrics, y_true_fold, y_pred_fold, y_score_fold = run_single_model_fold(
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test_ready,
                y_test=y_test_ready,
                seed=RANDOM_STATE + fold,
                fold=fold,
            )

            metrics["Fold"] = fold
            metrics["Modelo"] = model_name
            results.append(metrics)

            oof_predictions[model_name]["y_true"].extend(y_true_fold)
            oof_predictions[model_name]["y_pred"].extend(y_pred_fold)
            oof_predictions[model_name]["y_score"].extend(y_score_fold)

            print(
                f"{model_name} - Fold {fold} | "
                f"Accuracy={metrics['Accuracy_epoch']:.4f} | "
                f"BalancedAcc={metrics['BalancedAccuracy_epoch']:.4f} | "
                f"F1={metrics['F1_epoch']:.4f}"
            )

    summary_df = build_summary_df(results)
    print("\nRESUMEN TF CROSS-SUBJECT")
    print(summary_df)

    save_best_tf_outputs(summary_df, oof_predictions)


if __name__ == "__main__":
    main()