import { countBy, flatMap, sumBy } from "es-toolkit";

import {
  calculatePlayerPredictions,
  type PredictionMap,
} from "../playerPredictions";
import type { StatsItem } from "../PositionsMoves";

import { defineColumn } from "./columnFactory";
import type { ColumnContext } from "./types";

type AIValues = PredictionMap;
interface MoveStats {
  count: number;
  points: number;
}

interface TrendValues {
  alfa: number[];
  beta: number[];
}

const SYSTEM_BASE = 0.5;
const EPSILON = 0.1;
const YEAR_BOUND = 10;

const transformAI = (
  rows: readonly StatsItem[],
  context: ColumnContext,
): AIValues => calculatePlayerPredictions(context.games, rows);

const calcProbability = ({
  minYear,
  moveStats,
  year,
  yearsMap,
}: {
  minYear: number;
  moveStats: Record<number, MoveStats>;
  year: number;
  yearsMap: Record<number, number>;
}): number => {
  let probability = 1;

  for (let currentYear = minYear; currentYear <= year; currentYear++) {
    probability *= SYSTEM_BASE;

    const stats = moveStats[currentYear];
    const yearCount = yearsMap[currentYear];

    const yearPercentage =
      yearCount && stats?.count
        ? Math.max(stats.count / yearCount, EPSILON) *
          Math.max(stats.points / stats.count, EPSILON)
        : EPSILON ** 2;

    probability += yearPercentage * SYSTEM_BASE;
  }

  return probability;
};

function calculateTrends(
  stats: readonly StatsItem[],
  currentYear: number,
): TrendValues {
  const yearsMap = countBy(
    flatMap(stats, (stat) => stat.years),
    (year) => year,
  );

  const yearValues = Object.keys(yearsMap).map(Number);

  if (yearValues.length === 0) {
    return {
      alfa: stats.map(() => 0),
      beta: stats.map(() => 0),
    };
  }

  const base = Math.pow(YEAR_BOUND, 1 / YEAR_BOUND);

  const total = sumBy(
    stats,
    (item) => item.years.length * Math.pow(base, YEAR_BOUND),
  );

  const values = stats.map((item) =>
    sumBy(item.years, (year) =>
      Math.pow(
        base,
        year <= currentYear - YEAR_BOUND - 1
          ? 1
          : YEAR_BOUND - (currentYear - year),
      ),
    ),
  );

  const maxValue = Math.max(...values);
  const maxYear = Math.max(...yearValues);
  const minYear = Math.min(...yearValues);

  const denominator =
    1 +
    (maxYear <= currentYear - YEAR_BOUND
      ? YEAR_BOUND
      : currentYear - maxYear + 1);

  let scaleFactor = 1;

  if (maxValue !== 0 && maxValue < total / denominator) {
    scaleFactor = total / maxValue / denominator;
  }

  const alfa =
    total === 0
      ? stats.map(() => 0)
      : values.map((value) => (value * scaleFactor * 100) / total);

  const probabilities = stats.map((item) =>
    calcProbability({
      minYear,
      moveStats: item.stats,
      year: currentYear,
      yearsMap,
    }),
  );

  const maxProbability = Math.max(...probabilities);

  let scaleFactor2 = 1;

  if (maxProbability !== 0 && maxProbability < 1 / denominator) {
    scaleFactor2 = 1 / maxProbability / denominator;
  }

  const beta = probabilities.map((value) => value * scaleFactor2 * 100);

  return { alfa, beta };
}

const renderPercentage = (value: number) => (
  <>
    {value.toFixed(2)}%
    <br />
    <meter max={100} value={value} />
  </>
);

const transformTrends = (
  rows: readonly StatsItem[],
  context: ColumnContext,
): TrendValues => calculateTrends(rows, context.currentYear);

export const trendColumns = [
  defineColumn<TrendValues, number>({
    calculate: (_row, index, trends) => trends.alfa[index],
    id: "alfa",
    label: "Fα",
    render: renderPercentage,
    transform: transformTrends,
    transformKey: "trends",
    visibleByDefault: true,
  }),

  defineColumn<TrendValues, number>({
    calculate: (_row, index, trends) => trends.beta[index],
    id: "beta",
    label: "Fβ",
    render: renderPercentage,
    transform: transformTrends,
    transformKey: "trends",
    visibleByDefault: true,
  }),
  defineColumn<AIValues, number>({
    calculate: (row, _index, values) =>
      values.get(row.move)?.absolute.count ?? 0,
    id: "ai-count",
    label: "M1C",
    render: (value) => value.toFixed(2),
    transform: transformAI,
    transformKey: "ai",
    visibleByDefault: true,
  }),

  defineColumn<AIValues, number>({
    calculate: (row, _index, values) =>
      values.get(row.move)?.absolute.points ?? 0,
    id: "ai-rate",
    label: "M1P",
    render: (value) => value.toFixed(2),
    transform: transformAI,
    transformKey: "ai",
    visibleByDefault: true,
  }),

  defineColumn<AIValues, number>({
    calculate: (row, _index, values) =>
      values.get(row.move)?.percentage.gameShare ?? 0,
    id: "ai-game-share",
    label: "M2O",
    render: (value) => renderPercentage(value * 100),
    transform: transformAI,
    transformKey: "ai",
    visibleByDefault: true,
  }),

  defineColumn<AIValues, number>({
    calculate: (row, _index, values) =>
      values.get(row.move)?.percentage.pointScore ?? 0,
    id: "ai-point-score",
    label: "M2P",
    render: (value) => renderPercentage(value * 100),
    transform: transformAI,
    transformKey: "ai",
    visibleByDefault: true,
  }),
];
