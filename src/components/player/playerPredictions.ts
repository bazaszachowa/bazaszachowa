import type { GameData } from "../../ChessEditor";
import {
  type ModelInput,
  type PlayerPrediction,
  predictPlayer,
} from "../../models/predictor";

import type { StatsItem } from "./PositionsMoves";

export type PredictionMap = ReadonlyMap<string, PlayerPrediction>;

const getYear = (game: GameData): number | undefined =>
  typeof game.Year === "number" ? game.Year : undefined;

export const createYearGameCounts = (
  games: readonly GameData[],
): Map<number, number> => {
  const result = new Map<number, number>();

  for (const game of games) {
    const year = getYear(game);

    if (year === undefined) {
      continue;
    }

    result.set(year, (result.get(year) ?? 0) + 1);
  }

  return result;
};

const getYears = (yearGameCounts: ReadonlyMap<number, number>): number[] => {
  const years = [...yearGameCounts.keys()].toSorted((a, b) => a - b);

  if (years.length === 0) {
    return [];
  }

  const first = years[0];
  const last = years.at(-1);

  if (first === undefined || last === undefined) {
    return [];
  }

  return Array.from({ length: last - first + 1 }, (_, index) => first + index);
};

const getPointScore = (points: number, count: number): number =>
  count > 0 ? Math.max(0, Math.min(1, points / count)) : 0;

const createSequences = (
  item: StatsItem,
  yearGameCounts: ReadonlyMap<number, number>,
): {
  absolute: ModelInput[];
  percentage: ModelInput[];
} => {
  const years = getYears(yearGameCounts);
  const absolute: ModelInput[] = [];
  const percentage: ModelInput[] = [];

  for (const year of years) {
    const totalGames = yearGameCounts.get(year) ?? 0;
    const stats = item.stats[year];

    if (totalGames === 0) {
      absolute.push([0, 0, 0]);
      percentage.push([0, 0, 0]);
      continue;
    }

    if (!stats || stats.count === 0) {
      absolute.push([0, 0, 1]);
      percentage.push([0, 0, 1]);
      continue;
    }

    const score = getPointScore(stats.points, stats.count);

    absolute.push([Math.log1p(stats.count), score, 1]);

    percentage.push([Math.min(1, stats.count / totalGames), score, 1]);
  }

  return { absolute, percentage };
};

export const calculatePlayerPredictions = (
  games: readonly GameData[],
  positions: readonly StatsItem[],
): PredictionMap => {
  const yearGameCounts = createYearGameCounts(games);
  const predictions = new Map<string, PlayerPrediction>();

  for (const item of positions) {
    const sequences = createSequences(item, yearGameCounts);

    if (sequences.absolute.length === 0) {
      continue;
    }

    predictions.set(
      item.move,
      predictPlayer(sequences.absolute, sequences.percentage),
    );
  }

  return predictions;
};
