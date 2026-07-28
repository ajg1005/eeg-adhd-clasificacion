import { requestJson } from "../../shared/api/client";
import { translate } from "../../shared/utils/errors";
import type {
  BestAvailableModel,
  ExperimentDetail,
  ExperimentSummary,
} from "./types";

export function getBestAvailableModel(): Promise<BestAvailableModel> {
  return requestJson<BestAvailableModel>(
    { route: "bestModel" },
    undefined,
    translate("errors.experiments.bestModel"),
  );
}

export async function getExperiments(): Promise<ExperimentSummary[]> {
  const data = await requestJson<{ experiments: ExperimentSummary[] }>(
    { route: "experiments" },
    undefined,
    translate("errors.experiments.history"),
  );
  return data.experiments;
}

export function getExperimentDetail(
  experimentId: number,
): Promise<ExperimentDetail> {
  return requestJson<ExperimentDetail>(
    { route: "experimentDetail", id: experimentId },
    undefined,
    translate("errors.experiments.detail"),
  );
}
