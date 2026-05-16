"""
Comparacion estadistica entre modelos ML y DL.

Lee los CSVs de resultados por fold (results/ml_cv_fold_results.csv y
results/dl_cv_fold_results.csv) y produce:

- Tabla por modelo con media, desviacion y CI bootstrap 95% para F1 y
  balanced accuracy.
- Tabla pareada con t-test pareado (ttest_rel) sobre los folds para F1 y
  balanced accuracy, incluyendo p-valor crudo y corregido por Bonferroni.
- Figura con barras y CI 95% bootstrap por modelo.

Los outputs se guardan en results/ y Figuras/ para citarlos directamente en
la memoria del TFG.
"""
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import bootstrap, ttest_rel


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "Figuras"

ML_CSV = RESULTS_DIR / "ml_cv_fold_results.csv"
DL_CSV = RESULTS_DIR / "dl_cv_fold_results.csv"

METRICS = ["F1_epoch", "BalancedAccuracy_epoch"]
BOOTSTRAP_RESAMPLES = 10000
CI_LEVEL = 0.95
RANDOM_STATE = 42


def load_fold_results():
    if not ML_CSV.exists():
        raise FileNotFoundError(f"Falta {ML_CSV}. Ejecuta scripts/train_ml.py primero.")
    if not DL_CSV.exists():
        raise FileNotFoundError(f"Falta {DL_CSV}. Ejecuta scripts/train_dl.py primero.")

    ml = pd.read_csv(ML_CSV)
    dl = pd.read_csv(DL_CSV)
    ml["Familia"] = "ML"
    dl["Familia"] = "DL"

    keep = ["Familia", "Modelo", "Fold"] + METRICS
    return pd.concat([ml[keep], dl[keep]], ignore_index=True)


def bootstrap_ci(values):
    """CI bootstrap 95% para la media. Devuelve (low, high)."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(RANDOM_STATE)
    res = bootstrap(
        (values,),
        np.mean,
        confidence_level=CI_LEVEL,
        n_resamples=BOOTSTRAP_RESAMPLES,
        method="basic",
        random_state=rng,
    )
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def per_model_summary(df):
    rows = []
    for model_name, group in df.groupby("Modelo", sort=False):
        familia = group["Familia"].iloc[0]
        row = {"Familia": familia, "Modelo": model_name, "n_folds": int(len(group))}
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            low, high = bootstrap_ci(values)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        rows.append(row)
    summary = pd.DataFrame(rows)
    return summary.sort_values("F1_epoch_mean", ascending=False).reset_index(drop=True)


def paired_tests(df):
    models = list(df["Modelo"].unique())
    n_pairs = len(list(combinations(models, 2)))
    rows = []

    for metric in METRICS:
        pivot = df.pivot_table(index="Fold", columns="Modelo", values=metric)
        pivot = pivot[models]  # mantener orden

        for model_a, model_b in combinations(models, 2):
            paired = pivot[[model_a, model_b]].dropna()
            if len(paired) < 2:
                continue
            stat, pvalue = ttest_rel(paired[model_a], paired[model_b])
            diff = paired[model_a].to_numpy() - paired[model_b].to_numpy()
            rows.append({
                "metric": metric,
                "model_a": model_a,
                "model_b": model_b,
                "n_folds": int(len(paired)),
                "mean_diff": float(np.mean(diff)),
                "t_stat": float(stat),
                "p_value": float(pvalue),
                "p_value_bonferroni": float(min(1.0, pvalue * n_pairs)),
                "significant_bonferroni_5pct": bool(pvalue * n_pairs < 0.05),
            })

    return pd.DataFrame(rows)


def plot_summary(summary, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric in zip(axes, METRICS):
        means = summary[f"{metric}_mean"].to_numpy()
        lows = summary[f"{metric}_ci_low"].to_numpy()
        highs = summary[f"{metric}_ci_high"].to_numpy()
        errors = np.vstack([means - lows, highs - means])

        colors = ["#4C72B0" if familia == "ML" else "#DD8452" for familia in summary["Familia"]]
        ax.bar(summary["Modelo"], means, yerr=errors, capsize=4, color=colors, edgecolor="black")
        ax.set_title(metric.replace("_epoch", "").replace("_", " "))
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Media (CV) con CI 95% bootstrap")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.3)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#4C72B0", label="ML"),
        plt.Rectangle((0, 0), 1, 1, color="#DD8452", label="DL"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def format_summary(summary):
    formatted = summary.copy()
    numeric_cols = [col for col in formatted.columns if col not in {"Familia", "Modelo", "n_folds"}]
    for col in numeric_cols:
        formatted[col] = formatted[col].map(lambda v: f"{v:.4f}")
    return formatted


def format_paired(paired):
    formatted = paired.copy()
    formatted["mean_diff"] = formatted["mean_diff"].map(lambda v: f"{v:+.4f}")
    formatted["t_stat"] = formatted["t_stat"].map(lambda v: f"{v:+.3f}")
    # p-values: notacion cientifica para que p=1e-9 no se vea como 0
    formatted["p_value"] = formatted["p_value"].map(lambda v: f"{v:.2e}")
    formatted["p_value_bonferroni"] = formatted["p_value_bonferroni"].map(lambda v: f"{v:.2e}")
    return formatted


def write_markdown(summary, paired, path):
    lines = ["# Comparacion estadistica de modelos", ""]
    lines.append("## Resumen por modelo (CV cross-subject)")
    lines.append("")
    cols = ["Familia", "Modelo", "n_folds"]
    for metric in METRICS:
        cols += [f"{metric}_mean", f"{metric}_std", f"{metric}_ci_low", f"{metric}_ci_high"]
    lines.append(format_summary(summary[cols]).to_markdown(index=False))
    lines.append("")
    lines.append(f"CI 95% bootstrap con {BOOTSTRAP_RESAMPLES} resamples (seed={RANDOM_STATE}).")
    lines.append("")
    lines.append("## Tests pareados (ttest_rel sobre folds)")
    lines.append("")
    lines.append(format_paired(paired).to_markdown(index=False))
    lines.append("")
    lines.append(
        "p_value_bonferroni = p crudo * numero de pares; "
        "significant_bonferroni_5pct = True si p_bonf < 0.05."
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    df = load_fold_results()
    print(f"Cargados {len(df)} resultados de fold ({df['Modelo'].nunique()} modelos).")

    summary = per_model_summary(df)
    paired = paired_tests(df)

    summary_csv = RESULTS_DIR / "model_comparison_summary.csv"
    paired_csv = RESULTS_DIR / "model_comparison_paired.csv"
    summary.to_csv(summary_csv, index=False)
    paired.to_csv(paired_csv, index=False)

    markdown_path = RESULTS_DIR / "model_comparison.md"
    write_markdown(summary, paired, markdown_path)

    figure_path = FIGURES_DIR / "model_comparison_ci.png"
    plot_summary(summary, figure_path)

    print("\n=== Resumen por modelo ===")
    print(format_summary(summary).to_string(index=False))

    print("\n=== Top pares mas significativos (F1) ===")
    f1_paired = paired[paired["metric"] == "F1_epoch"].sort_values("p_value").head(10)
    print(format_paired(f1_paired).to_string(index=False))

    print(f"\nResumen por modelo : {summary_csv}")
    print(f"Tests pareados     : {paired_csv}")
    print(f"Markdown memoria   : {markdown_path}")
    print(f"Figura barras+CI   : {figure_path}")


if __name__ == "__main__":
    main()
