import { useInferenceController } from "./hooks/useInferenceController";
import { useTrainingDataset } from "./hooks/useTrainingDataset";
import { useTrainingTask } from "./hooks/useTrainingTask";
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
  const controller = useInferenceController();
  const trainingDataset = useTrainingDataset();
  const trainingTask = useTrainingTask(handleTrainingFinished);
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
        onTabChange={controller.setActiveTab}
        tabGroups={TAB_GROUPS}
      />

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
