import { lazy, Suspense } from "react";

import { AppHeader } from "./components/AppHeader";
import { DatasetView } from "../features/datasets/DatasetView";
import { ExperimentsView } from "../features/experiments/ExperimentsView";
import { ModelSelector } from "../features/inference/ModelSelector";
import { PredictionView } from "../features/inference/PredictionView";
import { Tabs } from "../shared/components/Tabs";
import { TrainingView } from "../features/training/TrainingView";
import { TAB_GROUPS } from "./tabs";
import { useInferenceController } from "../features/inference/useInferenceController";
import { useTrainingDataset } from "../features/datasets/useTrainingDataset";
import { useTrainingTask } from "../features/training/useTrainingTask";
import type { TrainingResult } from "../features/training/types";
import "./App.css";

const ModelView = lazy(async () => {
  const module = await import("../features/inference/ModelView");
  return { default: module.ModelView };
});

function App() {
  const controller = useInferenceController();
  const trainingDataset = useTrainingDataset();
  const trainingTask = useTrainingTask(handleTrainingFinished);

  function handleTrainingFinished(trainingResult: TrainingResult): void {
    const trainedModelId = trainingResult.trained_model_id
      ? `trained_model_${trainingResult.trained_model_id}`
      : null;

    void controller.refreshModels(trainedModelId).catch(() => undefined);
  }

  return (
    <main className="app-shell">
      <AppHeader />

      <Tabs
        activeTab={controller.activeTab}
        onTabChange={controller.setActiveTab}
        tabGroups={TAB_GROUPS}
      />

      {controller.error && (
        <div className="alert alert-error">{controller.error}</div>
      )}

      {controller.activeTab === "model" && (
        <>
          <ModelSelector
            modelInfo={controller.modelInfo}
            models={controller.models}
            onModelChange={controller.handleModelChange}
            selectedModelId={controller.selectedModelId}
          />
          <Suspense fallback={null}>
            <ModelView
              metrics={controller.metrics}
              metricsChartData={controller.metricsChartData}
              modelFigures={controller.modelFigures}
              modelInfo={controller.modelInfo}
            />
          </Suspense>
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
          loadingTraining={trainingTask.trainingInProgress}
          onStartTraining={trainingTask.startTraining}
          result={trainingTask.result}
          selectedDataset={trainingDataset.selectedDataset}
          stats={trainingDataset.stats}
          taskError={trainingTask.error}
          taskStatus={trainingTask.status}
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
            modelAvailable={Boolean(controller.selectedModelId)}
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
