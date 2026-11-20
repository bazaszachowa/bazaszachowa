import type { ReactNode } from "react";

import type { PredictionMap } from "../playerPredictions";
import type { StatsItem } from "../PositionsMoves";

import type { ColumnDefinition } from "./types";

const percentageCell = (value: number): ReactNode => {
  const percentage = Math.max(0, Math.min(100, value * 100));

  return (
    <>
      {percentage.toFixed(2)}%
      <br />
      <meter max={100} min={0} value={percentage} />
    </>
  );
};

export const createPredictionColumns = (
  predictions: PredictionMap,
): ColumnDefinition[] => {
  const getPrediction = (row: StatsItem) => predictions.get(row.move);

  return [
    {
      calculate: (row: StatsItem) =>
        getPrediction(row)?.absolute.count.toFixed(2) ?? "-",
      id: "ai-absolute-count",
      label: "M1C",
      visibleByDefault: true,
    },
    {
      calculate: (row: StatsItem) =>
        getPrediction(row)?.absolute.points.toFixed(2) ?? "-",

      id: "ai-absolute-rate",
      label: "M1P",
      visibleByDefault: true,
    },
    {
      calculate: (row: StatsItem) => {
        const value = getPrediction(row)?.percentage.gameShare;

        return value === undefined ? "-" : percentageCell(value);
      },
      id: "ai-game-share",
      label: "M2O",
      visibleByDefault: true,
    },
    {
      calculate: (row: StatsItem) => {
        const value = getPrediction(row)?.percentage.pointScore;

        return value === undefined ? "-" : percentageCell(value);
      },
      id: "ai-point-score",
      label: "M2P",
      visibleByDefault: true,
    },
  ];
};
