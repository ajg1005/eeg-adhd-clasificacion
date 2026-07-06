import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

import { useInferenceController } from "./hooks/useInferenceController";
import { useTrainingDataset } from "./hooks/useTrainingDataset";
import { TAB_GROUPS } from "./config/tabs";
import { AppHeader } from "./components/AppHeader";
import { DatasetView } from "./components/DatasetView";
import { ExperimentsView } from "./components/ExperimentsView";
import { ModelSelector } from "./components/ModelSelector";
import { ModelView } from "./components/ModelView";
import { PredictionView } from "./components/PredictionView";
import { Tabs } from "./components/Tabs";
import { TrainingView } from "./components/TrainingView";
import "./App.css";

function App() {
  const { t } = useTranslation();
  const controller = useInferenceController();
  const trainingDataset = useTrainingDataset();
  const [trainingInProgress, setTrainingInProgress] = useState(false);

  useEffect(() => {
    function handleBeforeUnload(event) {
      if (!trainingInProgress) {
        return;
      }

      event.preventDefault();
      event.returnValue = "";
    }

    window.addEventListener("beforeunload", handleBeforeUnload);
    return () => window.removeEventListener("beforeunload", handleBeforeUnload);
  }, [trainingInProgress]);

  function handleTabChange(nextTab) {
    if (trainingInProgress && nextTab !== controller.activeTab) {
      return;
    }

    controller.setActiveTab(nextTab);
  }

  function handleTrainingFinished(trainingResult) {
    const trainedModelId = trainingResult?.trained_model_id
      ? `trained_model_${trainingResult.trained_model_id}`
      : null;

    controller.refreshModels(trainedModelId).catch(() => {});
  }

  return (
    <main className="app-shell">
      <AppHeader apiStatus={controller.apiStatus} />

      <Tabs
        activeTab={controller.activeTab}
        disabled={trainingInProgress}
        onTabChange={handleTabChange}
        tabGroups={TAB_GROUPS}
      />

      {trainingInProgress && (
        <div className="alert alert-warning">
          {t("app.trainingInProgress")}
        </div>
      )}

      {controller.error && <div className="alert alert-error">{controller.error}</div>}

      {controller.activeTab === "model" && (
        <>
          <ModelSelector
            modelInfo={controller.modelInfo}
            models={controller.models}
            onModelChange={controller.handleModelChange}
            selectedModelId={controller.selectedModelId}
          />
          <ModelView
            metrics={controller.metrics}
            metricsChartData={controller.metricsChartData}
            modelFigures={controller.modelFigures}
            modelInfo={controller.modelInfo}
          />
        </>
      )}

      {controller.activeTab === "dataset" && (
        <DatasetView
          classFilter={trainingDataset.classFilter}
          error={trainingDataset.error}
          file={trainingDataset.file}
          handleAnalyzeDataset={trainingDataset.handleAnalyzeDataset}
          handleClassFilterChange={trainingDataset.handleClassFilterChange}
          handleFileChange={trainingDataset.handleFileChange}
          handleMaxPatientsChange={trainingDataset.handleMaxPatientsChange}
          handleSavedDatasetChange={trainingDataset.handleSavedDatasetChange}
          loadingDatasets={trainingDataset.loadingDatasets}
          loadingStats={trainingDataset.loadingStats}
          maxPatients={trainingDataset.maxPatients}
          savedDatasets={trainingDataset.savedDatasets}
          selectedDataset={trainingDataset.selectedDataset}
          stats={trainingDataset.stats}
        />
      )}

      {controller.activeTab === "training" && (
        <TrainingView
          file={trainingDataset.file}
          onTrainingFinished={handleTrainingFinished}
          onTrainingStateChange={setTrainingInProgress}
          selectedDataset={trainingDataset.selectedDataset}
          stats={trainingDataset.stats}
        />
      )}

      {controller.activeTab === "experiments" && <ExperimentsView />}

      {controller.activeTab === "prediction" && (
        <>
          <ModelSelector
            modelInfo={controller.modelInfo}
            models={controller.models}
            onModelChange={controller.handleModelChange}
            selectedModelId={controller.selectedModelId}
          />
          <PredictionView
            decisionScore={controller.decisionScore}
            file={controller.file}
            loadingPrediction={controller.loadingPrediction}
            loadingValidation={controller.loadingValidation}
            modelInfo={controller.modelInfo}
            onFileChange={controller.handleFileChange}
            onPredict={controller.handlePrediction}
            prediction={controller.prediction}
            validation={controller.validation}
          />
        </>
      )}
    </main>
  );
}

export default App;
