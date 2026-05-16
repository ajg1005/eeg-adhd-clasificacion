import { useInferenceController } from "./hooks/useInferenceController";
import { TABS } from "./config/tabs";
import { AppHeader } from "./components/AppHeader";
import { DataView } from "./components/DataView";
import { ModelSelector } from "./components/ModelSelector";
import { ModelView } from "./components/ModelView";
import { PredictionView } from "./components/PredictionView";
import { SignalView } from "./components/SignalView";
import { Tabs } from "./components/Tabs";
import { TrainingView } from "./components/TrainingView";
import "./App.css";

function App() {
  const controller = useInferenceController();

  return (
    <main className="app-shell">
      <AppHeader apiStatus={controller.apiStatus} />

      <Tabs
        activeTab={controller.activeTab}
        onTabChange={controller.setActiveTab}
        tabs={TABS}
      />

      <ModelSelector
        modelInfo={controller.modelInfo}
        models={controller.models}
        onModelChange={controller.handleModelChange}
        selectedModelId={controller.selectedModelId}
      />

      {controller.error && <div className="alert alert-error">{controller.error}</div>}

      {controller.activeTab === TABS[0] && (
        <DataView
          classFilter={controller.classFilter}
          datasetSummary={controller.datasetSummary}
          file={controller.file}
          loadingDatasetSummary={controller.loadingDatasetSummary}
          loadingValidation={controller.loadingValidation}
          maxPatients={controller.maxPatients}
          modelInfo={controller.modelInfo}
          onClassFilterChange={controller.handleClassFilterChange}
          onFileChange={controller.handleFileChange}
          onMaxPatientsChange={controller.handleMaxPatientsChange}
          validation={controller.validation}
        />
      )}

      {controller.activeTab === TABS[1] && (
        <ModelView
          metrics={controller.metrics}
          metricsChartData={controller.metricsChartData}
          modelCatalog={controller.modelCatalog}
          modelFigures={controller.modelFigures}
          modelInfo={controller.modelInfo}
        />
      )}

      {controller.activeTab === TABS[2] && <TrainingView />}

      {controller.activeTab === TABS[3] && (
        <SignalView
          file={controller.file}
          loadingPreview={controller.loadingPreview}
          maxPoints={controller.maxPoints}
          modelInfo={controller.modelInfo}
          onLoadPreview={controller.handleLoadPreview}
          selectedChannel={controller.selectedChannel}
          setMaxPoints={controller.setMaxPoints}
          setSelectedChannel={controller.setSelectedChannel}
          signalPreview={controller.signalPreview}
        />
      )}

      {controller.activeTab === TABS[4] && (
        <PredictionView
          adhdEpochs={controller.adhdEpochs}
          controlEpochs={controller.controlEpochs}
          decisionScore={controller.decisionScore}
          file={controller.file}
          finalClassEpochPercentage={controller.finalClassEpochPercentage}
          loadingPrediction={controller.loadingPrediction}
          onPredict={controller.handlePrediction}
          prediction={controller.prediction}
          predictionChartData={controller.predictionChartData}
          thresholdUsed={controller.thresholdUsed}
          validation={controller.validation}
        />
      )}
    </main>
  );
}

export default App;

