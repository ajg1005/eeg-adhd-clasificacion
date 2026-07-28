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

    const [experimentsResult, bestModelResult] = await Promise.allSettled([
      getExperiments(),
      getBestAvailableModel(),
    ]);

    if (experimentsResult.status === "fulfilled") {
      const sorted = sortByBalancedAccuracy(experimentsResult.value);
      const nextSelectedId = selectedId ?? sorted[0]?.id ?? null;

      setExperiments(sorted);
      setSelectedId(nextSelectedId);

      if (nextSelectedId !== null && nextSelectedId !== selectedId) {
        setSelectedExperiment(null);
        setLoadingDetail(true);
      }
    } else {
      setError(
        errorMessage(experimentsResult.reason, "errors.experiments.list"),
      );
    }

    if (bestModelResult.status === "fulfilled") {
      setBestAvailableModel(bestModelResult.value);
    } else if (experimentsResult.status === "fulfilled") {
      setBestAvailableModel(null);
      setError(
        errorMessage(bestModelResult.reason, "errors.experiments.bestModel"),
      );
    }

    setLoadingList(false);
  }

  useEffect(() => {
    let cancelled = false;

    void Promise.allSettled([getExperiments(), getBestAvailableModel()]).then(
      ([experimentsResult, bestModelResult]) => {
        if (cancelled) {
          return;
        }

        if (experimentsResult.status === "fulfilled") {
          const sorted = sortByBalancedAccuracy(experimentsResult.value);
          const initialSelectedId = sorted[0]?.id ?? null;

          setExperiments(sorted);
          setSelectedId(initialSelectedId);
          setLoadingDetail(initialSelectedId !== null);
        } else {
          setError(
            errorMessage(experimentsResult.reason, "errors.experiments.list"),
          );
        }

        if (bestModelResult.status === "fulfilled") {
          setBestAvailableModel(bestModelResult.value);
        } else if (experimentsResult.status === "fulfilled") {
          setError(
            errorMessage(bestModelResult.reason, "errors.experiments.bestModel"),
          );
        }

        setLoadingList(false);
      },
    );

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
      {error && <div className="alert alert-error" role="alert">{error}</div>}
      <div className="panel best-model-row">
        {loadingList && !bestAvailableModel ? (
          <p className="muted">{t("common.loading")}</p>
        ) : bestAvailableModel ? (
          <>
            <div className="best-model-identity">
              <span className="eyebrow accent">
                {t("experiments.bestAvailableTitle")}
              </span>
              <h2>
                {bestAvailableModel.display_name} ·{" "}
                {t("experiments.experiment", {
                  id: bestAvailableModel.experiment_id,
                })}
              </h2>
              <p className="muted">
                {[
                  bestAvailableModel.model_type.toUpperCase(),
                  bestAvailableModel.dataset_filename,
                  `${String(bestAvailableModel.n_subjects)} ${t("common.patients").toLowerCase()}`,
                  formatDate(
                    bestAvailableModel.created_at,
                    i18n.resolvedLanguage,
                  ),
                ].join(" · ")}
              </p>
            </div>

            <div className="best-model-figures">
              <div className="headline-metric accent">
                <span>{t("metrics.balancedAccuracy")}</span>
                <strong>
                  {formatMetric(bestAvailableModel.balanced_accuracy)}
                </strong>
              </div>
              <div className="headline-metric">
                <span>{t("metrics.f1")}</span>
                <strong>{formatMetric(bestAvailableModel.f1_score)}</strong>
              </div>
              {availableModelIds.includes(bestAvailableModel.model_id) && (
                <button
                  className="primary-button compact-button"
                  onClick={() => {
                    onUseForInference(bestAvailableModel.model_id);
                  }}
                  type="button"
                >
                  {t("experiments.useForInference")}
                </button>
              )}
            </div>
          </>
        ) : (
          <div>
            <span className="eyebrow">
              {t("experiments.bestAvailableTitle")}
            </span>
            <p className="muted">{t("experiments.bestAvailableEmpty")}</p>
          </div>
        )}
      </div>

      <div className="panel">
        <div className="section-heading-row">
          <span className="eyebrow">{t("experiments.title")}</span>
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
                  <th>{t("common.model")}</th>
                  <th>{t("experiments.modelType")}</th>
                  <th className="metric-column-accent">
                    {t("metrics.balanced")}
                  </th>
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
                      <td className="experiment-model-cell">
                        <strong className={isBestAvailable ? "best-row-name" : undefined}>
                          {experiment.display_name}
                        </strong>
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
                      <td
                        className={
                          isBestAvailable ? "metric-column-accent" : undefined
                        }
                      >
                        {formatMetric(experiment.balanced_accuracy)}
                      </td>
                      <td>{formatMetric(experiment.f1_score)}</td>
                      <td className="experiment-action-cell">
                        {usable && (
                          <button
                            className="row-action"
                            onClick={(event) => {
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
