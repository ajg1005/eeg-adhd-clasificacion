"""
Importancia de features del modelo ML exportado.

Sigue la recomendacion de los docs de sklearn: aplica permutation_importance
sobre un test set held-out, no sobre los datos de entrenamiento.

1. Genera epochs y features del dataset completo.
2. Hace un split cross-subject (make_group_shuffle_split): los pacientes del
   test no aparecen en train.
3. Entrena un modelo fresco identico al exportado sobre el train.
4. Aplica permutation_importance en el test: baraja cada feature y mide
   cuanto cae el F1.

Asi la importancia refleja que features ayudan a generalizar a pacientes
nuevos, no que features memoriza el modelo entrenado.

Outputs:
- results/feature_importance.csv : importancia por feature (media y std).
- results/feature_importance_by_channel.csv : importancia agregada por canal.
- Figuras/feature_importance_top20.png : top 20 features mas importantes.
- Figuras/feature_importance_by_channel.png : importancia agregada por canal.

Ejemplos:
- python scripts/feature_importance.py --dry-run
- python scripts/feature_importance.py --test-sample-size 100 --n-repeats 3
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from data_load import load_dataset  # noqa: E402
from epochs import create_epochs  # noqa: E402
from features import extract_epoch_features  # noqa: E402
from preprocessing import preprocess_dataset  # noqa: E402
from spectral_features import extract_spectral_features  # noqa: E402
from split import make_group_shuffle_split  # noqa: E402


CSV_PATH = BASE_DIR / "data" / "adhdata.csv"
MODELS_DIR = BASE_DIR / "models" / "ml"
MODEL_PATH = MODELS_DIR / "final_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.json"

FIGURES_DIR = BASE_DIR / "Figuras"
RESULTS_DIR = BASE_DIR / "results"

DEFAULT_TEST_SAMPLE_SIZE = 0  # 0 = usar todo el test set
DEFAULT_N_REPEATS = 10
DEFAULT_TEST_SIZE = 0.2
RANDOM_STATE = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Calcula importancia de features por permutacion.")
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE,
                        help="Proporcion de pacientes en el test held-out (default 0.2).")
    parser.add_argument("--test-sample-size", type=int, default=DEFAULT_TEST_SAMPLE_SIZE,
                        help="Limitar epochs del test set para acelerar (0 = todo).")
    parser.add_argument("--n-repeats", type=int, default=DEFAULT_N_REPEATS,
                        help="Permutaciones por feature (default 10).")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--scoring", default="f1_weighted",
                        help="Metrica usada por sklearn.permutation_importance.")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true",
                        help="Valida carga, features y split sin calcular importancias.")
    return parser.parse_args()


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Ejecuta scripts/export_model.py primero.")
    return json.loads(path.read_text(encoding="utf-8"))


def build_features(metadata, eeg_cols, x_epochs):
    feature_mode = metadata["feature_mode"].lower()
    sfreq = int(metadata["sfreq"])
    nperseg = int(metadata.get("nperseg", metadata["epoch_size"]))

    if feature_mode in {"time", "temporal"}:
        return extract_epoch_features(x_epochs, eeg_cols)
    if feature_mode == "spectral":
        return extract_spectral_features(x_epochs, eeg_cols, sfreq=sfreq, nperseg=nperseg)
    if feature_mode == "combined":
        temporal = extract_epoch_features(x_epochs, eeg_cols)
        spectral = extract_spectral_features(x_epochs, eeg_cols, sfreq=sfreq, nperseg=nperseg)
        return pd.concat([temporal.reset_index(drop=True), spectral.reset_index(drop=True)], axis=1)

    raise ValueError(f"feature_mode no soportado: {metadata['feature_mode']}")


def align_feature_columns(x_features, feature_columns):
    missing = [feature for feature in feature_columns if feature not in x_features.columns]
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"Faltan features esperadas por el modelo: {preview}")

    return x_features.loc[:, feature_columns]


def stratified_subsample(X, y, sample_size):
    if sample_size <= 0 or len(X) <= sample_size:
        return X.reset_index(drop=True), np.asarray(y).astype(int)

    _, x_sample, _, y_sample = train_test_split(
        X,
        y,
        test_size=sample_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    print(f"Submuestreado test set a {sample_size} epochs para acelerar.")
    return x_sample.reset_index(drop=True), np.asarray(y_sample).astype(int)


def aggregate_by_channel(importance_df, channels):
    rows = []
    for channel in channels:
        prefix = f"{channel}_"
        mask = importance_df["feature"].str.startswith(prefix)
        rows.append({
            "channel": channel,
            "importance_sum": float(importance_df.loc[mask, "importance_mean"].sum()),
            "n_features": int(mask.sum()),
        })
    return pd.DataFrame(rows).sort_values("importance_sum", ascending=False).reset_index(drop=True)


def main():
    args = parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No existe {MODEL_PATH}. Ejecuta scripts/export_model.py primero.")

    print("Cargando modelo, metadata y dataset...")
    pipeline = joblib.load(MODEL_PATH)
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
        build_features(metadata, eeg_cols, x_epochs),
        feature_columns,
    )
    y_epochs = np.asarray(y_epochs).astype(int)
    groups_epochs = np.asarray(groups_epochs).astype(str)

    print(f"Total epochs: {len(x_features)} | pacientes: {len(set(groups_epochs))}")

    # Split cross-subject: ningun paciente se solapa entre train y test
    X_train, X_test, y_train, y_test, groups_train, groups_test = make_group_shuffle_split(
        x_features,
        y_epochs,
        groups_epochs,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
    )
    overlap = len(set(groups_train) & set(groups_test))
    print(f"Split: {len(X_train)} train ({len(set(groups_train))} pacientes) | "
          f"{len(X_test)} test ({len(set(groups_test))} pacientes) | overlap pacientes={overlap}")

    if args.dry_run:
        print("Dry-run OK: datos, features y split cross-subject preparados correctamente.")
        return

    # Entrenamos un modelo fresco identico al exportado sobre el train split.
    # clone() conserva hiperparametros y preprocesado (scaler) sin pesos aprendidos.
    print(f"Entrenando modelo fresco ({metadata['model_name']}) en train...")
    fresh_model = clone(pipeline)
    fresh_model.fit(X_train, y_train)

    # Opcional: submuestrear test si es muy grande
    x_test_used, y_test_used = stratified_subsample(X_test, y_test, args.test_sample_size)

    print(f"Calculando permutation_importance en test ({args.n_repeats} repeticiones, "
          f"{len(x_test_used)} epochs, scoring={args.scoring}, n_jobs={args.n_jobs})...")
    result = permutation_importance(
        fresh_model,
        x_test_used,
        y_test_used,
        scoring=args.scoring,
        n_repeats=args.n_repeats,
        random_state=RANDOM_STATE,
        n_jobs=args.n_jobs,
    )

    importance_df = pd.DataFrame({
        "feature": feature_columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    importance_csv = RESULTS_DIR / "feature_importance.csv"
    importance_df.to_csv(importance_csv, index=False)

    channels = metadata.get("channels", [])
    channel_df = aggregate_by_channel(importance_df, channels)
    channel_csv = RESULTS_DIR / "feature_importance_by_channel.csv"
    channel_df.to_csv(channel_csv, index=False)

    # Top N features
    top = importance_df.head(args.top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"],
            color="#4C72B0", edgecolor="black")
    ax.set_xlabel(f"Caida media de {args.scoring} al permutar la feature (test held-out)")
    ax.set_title(f"Top {args.top_n} features mas importantes ({metadata['model_name']})")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    top_fig = FIGURES_DIR / "feature_importance_top20.png"
    fig.savefig(top_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Importancia agregada por canal
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(channel_df["channel"], channel_df["importance_sum"], color="#4C72B0", edgecolor="black")
    ax.set_ylabel("Suma de la importancia de sus features")
    ax.set_title("Importancia agregada por canal EEG (test held-out)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    channel_fig = FIGURES_DIR / "feature_importance_by_channel.png"
    fig.savefig(channel_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nTabla por feature : {importance_csv}")
    print(f"Tabla por canal   : {channel_csv}")
    print(f"Top features      : {top_fig}")
    print(f"Por canal         : {channel_fig}")
    print(f"\nTop {args.top_n // 4 or 5} features:")
    print(importance_df.head(args.top_n // 4 or 5).to_string(index=False))


if __name__ == "__main__":
    main()
