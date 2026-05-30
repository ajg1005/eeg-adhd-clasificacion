import { useEffect, useState } from "react";

import { useInferenceController } from "./hooks/useInferenceController";
import { useTrainingDataset } from "./hooks/useTrainingDataset";
import { TABS } from "./config/tabs";
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

  return (
    <main className="app-shell">
      <AppHeader apiStatus={controller.apiStatus} />

      <Tabs
        activeTab={controller.activeTab}
        disabled={trainingInProgress}
        onTabChange={handleTabChange}
        tabs={TABS}
      />

      {trainingInProgress && (
        <div className="alert alert-warning">
          Entrenamiento en curso. Espera a que termine antes de cambiar de pestaña o recargar.
        </div>
      )}

      {controller.error && <div className="alert alert-error">{controller.error}</div>}

      {controller.activeTab === "Modelo" && (
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
            modelCatalog={controller.modelCatalog}
            modelFigures={controller.modelFigures}
            modelInfo={controller.modelInfo}
          />
        </>
      )}

      {controller.activeTab === "Dataset entrenamiento" && (
        <DatasetView
          classFilter={trainingDataset.classFilter}
          error={trainingDataset.error}
          file={trainingDataset.file}
          handleAnalyzeDataset={trainingDataset.handleAnalyzeDataset}
          handleClassFilterChange={trainingDataset.handleClassFilterChange}
          handleFileChange={trainingDataset.handleFileChange}
          handleMaxPatientsChange={trainingDataset.handleMaxPatientsChange}
          loadingStats={trainingDataset.loadingStats}
          maxPatients={trainingDataset.maxPatients}
          stats={trainingDataset.stats}
        />
      )}

      {controller.activeTab === "Entrenamiento" && (
        <TrainingView
          file={trainingDataset.file}
          onTrainingStateChange={setTrainingInProgress}
          stats={trainingDataset.stats}
        />
      )}

      {controller.activeTab === "Experimentos" && <ExperimentsView />}

      {controller.activeTab === "Predicción" && (
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
            predictionChartData={controller.predictionChartData}
            validation={controller.validation}
          />
        </>
      )}
    </main>
  );
}

export default App;
