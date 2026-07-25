export type TabId =
  | "dataset"
  | "training"
  | "experiments"
  | "model"
  | "prediction";

export interface TabGroup {
  id: string;
  labelKey: string;
  tabs: readonly TabId[];
}

export const TAB_GROUPS = [
  {
    id: "trainingFlow",
    labelKey: "tabs.groups.training",
    tabs: ["dataset", "training", "experiments"],
  },
  {
    id: "inferenceFlow",
    labelKey: "tabs.groups.inference",
    tabs: ["model", "prediction"],
  },
] as const satisfies readonly TabGroup[];
