import PropTypes from "prop-types";
import { useTranslation } from "react-i18next";

import { modelInfoShape } from "../propTypes";
import { ModelSelectField } from "./ModelSelectField";

export function ModelSelector({
  modelInfo,
  models,
  onModelChange,
  selectedModelId,
}) {
  const { t } = useTranslation();
  const hasEnabledModels = models.some((model) => model.enabled !== false);
  const options = [
    ...(!selectedModelId
      ? [{ disabled: true, label: t("model.noAvailableModels"), value: "" }]
      : []),
    ...models.map((model) => ({
      disabled: model.enabled === false,
      label:
        model.enabled === false
          ? `${model.display_name} (${t("model.unavailable")})`
          : model.display_name,
      value: model.model_id,
    })),
  ];

  return (
    <section className="model-selector panel">
      <ModelSelectField
        disabled={!hasEnabledModels}
        label={t("model.inferenceSelector")}
        onChange={onModelChange}
        options={options}
        value={selectedModelId}
      />

      {modelInfo && (
        <p className="muted">
          {[modelInfo.display_name, modelInfo.model_name, modelInfo.model_family]
            .filter(Boolean)
            .join(" - ")}
        </p>
      )}
    </section>
  );
}

ModelSelector.propTypes = {
  modelInfo: modelInfoShape,
  models: PropTypes.arrayOf(
    PropTypes.shape({
      display_name: PropTypes.string.isRequired,
      enabled: PropTypes.bool,
      model_id: PropTypes.string.isRequired,
    }),
  ).isRequired,
  onModelChange: PropTypes.func.isRequired,
  selectedModelId: PropTypes.string.isRequired,
};