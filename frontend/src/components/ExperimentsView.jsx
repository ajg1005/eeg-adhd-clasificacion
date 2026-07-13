
import { useCallback, useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";

import {
  getBestAvailableModel,
  getExperimentDetail,
  getExperiments,
} from "../api";
import { formatMetric } from "../utils/formatters";

function formatDate(value, language) {
  if (!value) {
    return "N/A";
  }

  return new Date(value).toLocaleString(language === "en" ? "en-US" : "es-ES");
}

export function ExperimentsView() {
  const { i18n, t } = useTranslation();
  const [experiments, setExperiments] = useState([]);
  const [bestAvailableModel, setBestAvailableModel] = useState(null);
  const [selectedId, setSelectedId] = useState(null);
  const [selectedExperiment, setSelectedExperiment] = useState(null);
  const [loadingList, setLoadingList] = useState(true);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [error, setError] = useState("");

  const loadExperiments = useCallback(async () => {
    setLoadingList(true);
    setError("");

    try {
      const [items, bestModel] = await Promise.all([
        getExperiments(),
        getBestAvailableModel(),
      ]);
      setExperiments(items);
      setBestAvailableModel(bestModel);
      setSelectedId((current) => current ?? items[0]?.id ?? null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoadingList(false);
    }
  }, []);

  useEffect(() => {
    let mounted = true;

    async function loadInitialExperiments() {
      try {
        const [items, bestModel] = await Promise.all([
          getExperiments(),
          getBestAvailableModel(),
        ]);
        if (!mounted) {
          return;
        }

        setExperiments(items);
        setBestAvailableModel(bestModel);
        setSelectedId((current) => current ?? items[0]?.id ?? null);
      } catch (err) {
        if (mounted) {
          setError(err.message);
        }
      } finally {
        if (mounted) {
          setLoadingList(false);
        }
      }
    }

    loadInitialExperiments();

    return () => {
      mounted = false;
    };
  }, []);

  useEffect(() => {
    if (!selectedId) {
      return;
    }

    async function loadDetail() {
      setLoadingDetail(true);
      setError("");

      try {
        setSelectedExperiment(await getExperimentDetail(selectedId));
      } catch (err) {
        setError(err.message);
      } finally {
        setLoadingDetail(false);
      }
    }

    loadDetail();
  }, [selectedId]);

  const selectedSummary = useMemo(
    () => experiments.find((experiment) => experiment.id === selectedId),
    [experiments, selectedId],
  );


  return (
    <section className="training-layout">
      {error && <div className="alert alert-error">{error}</div>}

      <div className="panel best-model-overview">
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
                {bestAvailableModel.model_type.toUpperCase()} / {t("experiments.experiment", {
                  id: bestAvailableModel.experiment_id,
                })} / {formatDate(bestAvailableModel.created_at, i18n.resolvedLanguage)}
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
          <div>
            <h2>{t("experiments.title")}</h2>
            <p className="muted">{t("experiments.description")}</p>
          </div>
          <button
            className="primary-button compact-button"
            disabled={loadingList}
            onClick={loadExperiments}
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
            <table className="patient-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>{t("experiments.date")}</th>
                  <th>{t("common.model")}</th>
                  <th>{t("experiments.modelType")}</th>
                  <th>{t("common.dataset")}</th>
                  <th>{t("metrics.balanced")}</th>
                  <th>F1</th>

                </tr>
              </thead>
              <tbody>
                {experiments.map((experiment) => {
                  const isSelected = experiment.id === selectedId;
                  const isBestAvailable = experiment.id === bestAvailableModel?.experiment_id;
                  const rowClass = [
                    isSelected ? "selected-row" : "",
                    isBestAvailable ? "best-row" : "",
                  ]
                    .filter(Boolean)
                    .join(" ");
                  const selectExperiment = () => setSelectedId(experiment.id);

                  return (
                    <tr
                      aria-selected={isSelected}
                      className={rowClass}
                      key={experiment.id}
                      onClick={selectExperiment}
                      onKeyDown={(event) => {
                        if (event.key === "Enter" || event.key === " ") {
                          event.preventDefault();
                          selectExperiment();
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
            {selectedId && (
              <span className="importance-meta">
                {t("experiments.experiment", { id: selectedId })}
              </span>
            )}
          </div>

          {loadingDetail && <p className="muted">{t("experiments.loadingDetail")}</p>}

          {selectedExperiment && (
            <>
              <div className="metric-grid metrics-wide">
                <div>
                  <span>{t("metrics.accuracy")}</span>
                  <strong>{formatMetric(selectedExperiment.accuracy)}</strong>
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
                {t("common.rows").toLowerCase()} - {selectedExperiment.dataset.n_subjects}{" "}
                {t("common.patients").toLowerCase()}
              </p>

              <h3>{t("common.configuration")}</h3>
              <pre className="training-log">
                {JSON.stringify(
                  {
                    eeg_params: selectedExperiment.eeg_params,
                    model_params: selectedExperiment.model_params,
                    training_params: selectedExperiment.training_params,
                  },
                  null,
                  2,
                )}
              </pre>
            </>
          )}
        </div>
      )}
    </section>
  );
}
