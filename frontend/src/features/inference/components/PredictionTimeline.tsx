import { useMemo } from "react";
import { useTranslation } from "react-i18next";

import type { ModelInfo } from "../types";

interface PredictionTimelineProps {
  epochPredictions: string[];
  modelInfo: ModelInfo | null;
}

// Con cientos de ventanas una barra por ventana no se ve: se agrupan en tramos
// y cada tramo toma la clase mayoritaria de las ventanas que contiene.
const MAX_SLOTS = 48;

function classVariant(label: string): "adhd" | "control" | "other" {
  const normalized = label.trim().toLowerCase();

  if (normalized === "adhd" || normalized === "tdah") {
    return "adhd";
  }

  return normalized === "control" ? "control" : "other";
}

function majorityLabel(labels: string[]): string {
  const counts = new Map<string, number>();

  for (const label of labels) {
    counts.set(label, (counts.get(label) ?? 0) + 1);
  }

  let winner = labels[0] ?? "";
  let best = 0;

  for (const [label, count] of counts) {
    if (count > best) {
      winner = label;
      best = count;
    }
  }

  return winner;
}

// Duracion cubierta por las ventanas: la ultima acaba step*(n-1)+size muestras
// despues del inicio. Sin los parametros del modelo no se puede afirmar nada,
// asi que el eje se omite.
function coveredSeconds(
  epochs: number,
  modelInfo: ModelInfo | null,
): number | null {
  const sfreq = modelInfo?.sfreq;
  const epochSize = modelInfo?.epoch_size;
  const stepSize = modelInfo?.step_size;

  if (!sfreq || !epochSize || !stepSize || epochs <= 0) {
    return null;
  }

  return ((epochs - 1) * stepSize + epochSize) / sfreq;
}

export function PredictionTimeline({
  epochPredictions,
  modelInfo,
}: PredictionTimelineProps) {
  const { t } = useTranslation();
  const slots = useMemo(() => {
    const total = epochPredictions.length;
    const slotCount = Math.min(total, MAX_SLOTS);
    const perSlot = Math.ceil(total / slotCount);

    return Array.from({ length: slotCount }, (_, index) => {
      const chunk = epochPredictions.slice(
        index * perSlot,
        (index + 1) * perSlot,
      );

      return { label: majorityLabel(chunk), size: chunk.length };
    }).filter((slot) => slot.size > 0);
  }, [epochPredictions]);

  if (epochPredictions.length === 0) {
    return null;
  }

  const seconds = coveredSeconds(epochPredictions.length, modelInfo);
  const windowsPerSlot = slots[0]?.size ?? 1;

  return (
    <div className="prediction-timeline">
      <span className="eyebrow">{t("prediction.timelineTitle")}</span>

      <div
        aria-label={t("prediction.timelineTitle")}
        className="timeline-track"
        role="img"
      >
        {slots.map((slot, index) => (
          <span
            className={`timeline-slot ${classVariant(slot.label)}`}
            key={index}
            title={slot.label}
          />
        ))}
      </div>

      <div className="timeline-axis">
        <span>{seconds === null ? "1" : "0 s"}</span>
        <span>
          {windowsPerSlot > 1
            ? t("prediction.timelineScale", { count: windowsPerSlot })
            : t("prediction.timelineScaleSingle")}
        </span>
        <span>
          {seconds === null
            ? epochPredictions.length
            : `${Math.round(seconds)} s`}
        </span>
      </div>
    </div>
  );
}
