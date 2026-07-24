import type { ChangeEventHandler } from "react";
import { useTranslation } from "react-i18next";

import type { ModelInfo, ModelRegistryItem, SelectOption } from "../types";
import { ModelSelectField } from "./ModelSelectField";

interface ModelSelectorProps {
  modelInfo: ModelInfo | null;
  models: ModelRegistryItem[];
  onModelChange: ChangeEventHandler<HTMLSelectElement>;
  selectedModelId: string;
}

export function ModelSelector({
  modelInfo,
  models,
  onModelChange,
  selectedModelId,
}: ModelSelectorProps) {
  const { t } = useTranslation();
  const hasEnabledModels = models.some((model) => model.enabled !== false);
  const options: SelectOption[] = [
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
