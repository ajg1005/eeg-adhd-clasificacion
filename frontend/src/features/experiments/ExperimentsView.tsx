import { useEffect, useMemo, useState } from "react";
import type { TFunction } from "i18next";
import { useTranslation } from "react-i18next";

import {
  getBestAvailableModel,
  getExperimentDetail,
  getExperiments,
} from "./api";
import type { JsonValue } from "../../shared/types";
import type {
  BestAvailableModel,
  ExperimentDetail,
  ExperimentSummary,
} from "./types";
import { formatMetric } from "../../shared/utils/formatters";
import {
  evaluationModeLabel,
  modelParamLabel,
  optionValueLabel,
  signalParamLabel,
  trainingParamLabel,
} from "../training/trainingLabels";
import { errorMessage } from "../../shared/utils/errors";

type ParameterLabel = (t: TFunction, name: string) => string;

interface ParameterGroupProps {
  labelFor: ParameterLabel;
  params: Record<string, JsonValue>;
  t: TFunction;
  title: string;
}

// El listado llega por fecha; interesa mas por metrica. Se ordena al recibirlo
// (y no al pintar) para que la fila preseleccionada sea la primera que se ve.
function sortByBalancedAccuracy(
  items: ExperimentSummary[],
): ExperimentSummary[] {
  const score = (item: ExperimentSummary): number =>
    Number.isFinite(item.balanced_accuracy) ? item.balanced_accuracy : -Infinity;

  return [...items].sort((a, b) => score(b) - score(a));
}

function formatDate(value: string | null | undefined, language?: string): string {
  if (!value) {
    return "N/A";
  }

  return new Date(value).toLocaleString(language === "en" ? "en-US" : "es-ES");
}

function formatParameterValue(t: TFunction, value: JsonValue): string {
  if (Array.isArray(value)) {
    return value.map((item) => formatParameterValue(t, item)).join(", ");
  }

  if (value !== null && typeof value === "object") {
    return Object.entries(value)
      .map(([key, item]) => `${key}: ${formatParameterValue(t, item)}`)
      .join(", ");
  }

  return optionValueLabel(t, value);
}

function ParameterGroup({
  labelFor,
  params,
  t,
  title,
}: ParameterGroupProps) {
  const entries = Object.entries(params);

  if (entries.length === 0) {
    return null;
  }

  return (
    <section className="experiment-parameter-group">
      <h4>{title}</h4>
      <dl className="experiment-parameter-list">
        {entries.map(([name, value]) => (
          <div key={name}>
            <dt>{labelFor(t, name)}</dt>
            <dd>{formatParameterValue(t, value)}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

interface ExperimentsViewProps {
  // Ids de modelo realmente cargables (los que devuelve /models): un
  // trained_model_id cuyo artefacto falte no sirve para inferencia.
  availableModelIds: string[];
  onUseForInference: (modelId: string) => void;
}

export function ExperimentsView({
  availableModelIds,
  onUseForInference,
}: ExperimentsViewProps) {
  const { i18n, t } = useTranslation();
  const [experiments, setExperiments] = useState<ExperimentSummary[]>([]);
  const [bestAvailableModel, setBestAvailableModel] =
    useState<BestAvailableModel | null>(null);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [selectedExperiment, setSelectedExperiment] =
    useState<ExperimentDetail | null>(null);
  const [loadingList, setLoadingList] = useState(true);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [error, setError] = useState("");

  async function loadExperiments(): Promise<void> {
    setLoadingList(true);
    setError("");

    try {
      const [items, bestModel] = await Promise.all([
        getExperiments(),
        getBestAvailableModel(),
      ]);
      const sorted = sortByBalancedAccuracy(items);
      const nextSelectedId = selectedId ?? sorted[0]?.id ?? null;

      setExperiments(sorted);
      setBestAvailableModel(bestModel);
      setSelectedId(nextSelectedId);

      if (nextSelectedId !== null && nextSelectedId !== selectedId) {
        setSelectedExperiment(null);
        setLoadingDetail(true);
      }
    } catch (caughtError) {
      setError(
        errorMessage(caughtError, "errors.experiments.list"),
      );
    } finally {
      setLoadingList(false);
    }
  }

  useEffect(() => {
    let cancelled = false;

    void Promise.all([getExperiments(), getBestAvailableModel()])
      .then(([items, bestModel]) => {
        if (cancelled) {
          return;
        }

        const sorted = sortByBalancedAccuracy(items);
        const initialSelectedId = sorted[0]?.id ?? null;
        setExperiments(sorted);
        setBestAvailableModel(bestModel);
        setSelectedId(initialSelectedId);
        setLoadingDetail(initialSelectedId !== null);
      })
      .catch((caughtError: unknown) => {
        if (!cancelled) {
          setError(
            errorMessage(caughtError, "errors.experiments.list"),
          );
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoadingList(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (selectedId === null) {
      return;
    }

    let cancelled = false;

    void getExperimentDetail(selectedId)
      .then((experiment) => {
        if (!cancelled) {
          setSelectedExperiment(experiment);
        }
      })
      .catch((caughtError: unknown) => {
        if (!cancelled) {
          setError(
            errorMessage(caughtError, "errors.experiments.detail"),
          );
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoadingDetail(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [selectedId]);

  const selectedSummary = useMemo(
    () => experiments.find((experiment) => experiment.id === selectedId),
    [experiments, selectedId],
  );

  function selectExperiment(experimentId: number): void {
    if (experimentId === selectedId) {
      return;
    }

    setSelectedExperiment(null);
    setLoadingDetail(true);
    setSelectedId(experimentId);
  }

  return (
    <section className="training-layout">
      {error && <div className="alert alert-error">{error}</div>}

      <div className="panel">
        <div className="section-heading-row">
          <div>
            <h2>{t("experiments.bestAvailableTitle")}</h2>
            <p className="muted">{t("experiments.bestAvailableDescription")}</p>
          </div>
        </div>

        {loadingList && !bestAvailableModel ? (
          <p className="muted">{t("common.loading")}</p>
        ) : bestAvailableModel ? (
          <>
            <div className="best-model-identity">
              <h3>{bestAvailableModel.display_name}</h3>
              <p className="muted">
                {bestAvailableModel.model_type.toUpperCase()} /{" "}
                {t("experiments.experiment", {
                  id: bestAvailableModel.experiment_id,
                })}{" "}
                / {formatDate(bestAvailableModel.created_at, i18n.resolvedLanguage)}
              </p>
            </div>

            <div className="metric-grid best-model-summary-grid">
              <div>
                <span>{t("metrics.balancedAccuracy")}</span>
                <strong>{formatMetric(bestAvailableModel.balanced_accuracy)}</strong>
              </div>
              <div>
                <span>{t("metrics.f1")}</span>
                <strong>{formatMetric(bestAvailableModel.f1_score)}</strong>
              </div>
              <div>
                <span>{t("common.dataset")}</span>
                <strong>{bestAvailableModel.dataset_filename}</strong>
              </div>
              <div>
                <span>{t("common.patients")}</span>
                <strong>{bestAvailableModel.n_subjects}</strong>
              </div>
            </div>
          </>
        ) : (
          <p className="muted">{t("experiments.bestAvailableEmpty")}</p>
        )}
      </div>

      <div className="panel">
        <div className="section-heading-row">
          <h2>{t("experiments.title")}</h2>
          <button
            className="primary-button compact-button"
            disabled={loadingList}
            onClick={() => {
              void loadExperiments();
            }}
            type="button"
          >
            {loadingList ? t("common.loading") : t("common.refresh")}
          </button>
        </div>

        {experiments.length === 0 ? (
          <p className="muted">
            {loadingList ? t("experiments.loadingList") : t("experiments.empty")}
          </p>
        ) : (
          <div className="patient-table-wrap">
            <table className="patient-table experiments-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>{t("experiments.date")}</th>
                  <th>{t("common.model")}</th>
                  <th>{t("experiments.modelType")}</th>
                  <th>{t("common.dataset")}</th>
                  <th>{t("metrics.balanced")}</th>
                  <th>F1</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {experiments.map((experiment) => {
                  const isSelected = experiment.id === selectedId;
                  const isBestAvailable =
                    experiment.id === bestAvailableModel?.experiment_id;
                  const rowClass = [
                    isSelected ? "selected-row" : "",
                    isBestAvailable ? "best-row" : "",
                  ]
                    .filter(Boolean)
                    .join(" ");
                  const modelId =
                    experiment.trained_model_id != null
                      ? `trained_model_${experiment.trained_model_id}`
                      : null;
                  const usable = modelId !== null &&
                    availableModelIds.includes(modelId);

                  return (
                    <tr
                      aria-selected={isSelected}
                      className={rowClass}
                      key={experiment.id}
                      onClick={() => {
                        selectExperiment(experiment.id);
                      }}
                      onKeyDown={(event) => {
                        if (event.key === "Enter" || event.key === " ") {
                          event.preventDefault();
                          selectExperiment(experiment.id);
                        }
                      }}
                      tabIndex={0}
                    >
                      <td>#{experiment.id}</td>
                      <td>{formatDate(experiment.created_at, i18n.resolvedLanguage)}</td>
                      <td className="experiment-model-cell">
                        <strong>{experiment.display_name}</strong>
                        {isBestAvailable && (
                          <span className="best-row-badge">
                            {t("experiments.bestBadge")}
                          </span>
                        )}
                      </td>
                      <td>
                        <span className="model-type-label">
                          {experiment.model_type.toUpperCase()}
                        </span>
                      </td>
                      <td>{experiment.dataset.filename}</td>
                      <td>{formatMetric(experiment.balanced_accuracy)}</td>
                      <td>{formatMetric(experiment.f1_score)}</td>
                      <td className="experiment-action-cell">
                        {usable && (
                          <button
                            className="row-action"
                            onClick={(event) => {
                              // La fila entera selecciona: que el boton no lo dispare.
                              event.stopPropagation();
                              onUseForInference(modelId);
                            }}
                            type="button"
                          >
                            {t("experiments.useForInference")}
                          </button>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {(selectedSummary || loadingDetail) && (
        <div className="panel">
          <div className="section-heading-row">
            <h2>{t("experiments.detail")}</h2>
            {selectedId !== null && (
              <span className="importance-meta">
                {t("experiments.experiment", { id: selectedId })}
              </span>
            )}
          </div>

          {loadingDetail && <p className="muted">{t("experiments.loadingDetail")}</p>}

          {selectedExperiment && (
            <>
              <div className="experiment-detail-identity">
                <h3>{selectedExperiment.display_name}</h3>
                <p className="muted">
                  {selectedExperiment.model_type.toUpperCase()} /{" "}
                  {evaluationModeLabel(t, selectedExperiment.evaluation_mode)}
                </p>
              </div>

              <div className="metric-grid metrics-wide experiment-metrics">
                <div>
                  <span>{t("metrics.accuracy")}</span>
                  <strong>{formatMetric(selectedExperiment.accuracy)}</strong>
                </div>
                <div>
                  <span>{t("metrics.balancedAccuracy")}</span>
                  <strong>{formatMetric(selectedExperiment.balanced_accuracy)}</strong>
                </div>
                <div>
                  <span>{t("metrics.precision")}</span>
                  <strong>{formatMetric(selectedExperiment.precision)}</strong>
                </div>
                <div>
                  <span>{t("metrics.recall")}</span>
                  <strong>{formatMetric(selectedExperiment.recall)}</strong>
                </div>
                <div>
                  <span>F1</span>
                  <strong>{formatMetric(selectedExperiment.f1_score)}</strong>
                </div>
                <div>
                  <span>{t("common.time")}</span>
                  <strong>{selectedExperiment.training_time_seconds.toFixed(2)}s</strong>
                </div>
              </div>

              <h3>{t("common.dataset")}</h3>
              <p className="muted">
                {selectedExperiment.dataset.filename} - {selectedExperiment.dataset.rows}{" "}
                {t("common.rows").toLowerCase()} - {selectedExperiment.dataset.columns}{" "}
                {t("common.columns").toLowerCase()} -{" "}
                {selectedExperiment.dataset.n_subjects}{" "}
                {t("common.patients").toLowerCase()}
              </p>

              <div className="experiment-configuration">
                <h3>{t("common.configuration")}</h3>
                <div className="experiment-parameter-groups">
                  <ParameterGroup
                    labelFor={signalParamLabel}
                    params={selectedExperiment.eeg_params}
                    t={t}
                    title={t("experiments.signalConfiguration")}
                  />
                  <ParameterGroup
                    labelFor={modelParamLabel}
                    params={selectedExperiment.model_params}
                    t={t}
                    title={t("experiments.modelConfiguration")}
                  />
                  <ParameterGroup
                    labelFor={trainingParamLabel}
                    params={selectedExperiment.training_params}
                    t={t}
                    title={t("experiments.trainingConfiguration")}
                  />
                </div>
              </div>
            </>
          )}
        </div>
      )}
    </section>
  );
}
