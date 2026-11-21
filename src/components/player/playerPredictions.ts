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
  if (yearGameCounts.size === 0) {
    return [];
  }

  const years = [...yearGameCounts.keys()];
  years.sort((a, b) => a - b);

  const first = years[0];
  const last = years.at(-1);

  if (first === undefined || last === undefined) {
    return [];
  }

  return Array.from({ length: last - first + 1 }, (_, index) => first + index);
};

const getPointScore = (points: number, count: number): number =>
  count > 0 ? Math.max(0, Math.min(1, points / count)) : 0;

interface PlayerSequences {
  absolute: ModelInput[];
  percentage: ModelInput[];
}

const createSequences = (
  item: StatsItem,
  years: readonly number[],
  totalGamesByYear: readonly number[],
): PlayerSequences => {
  const absolute: ModelInput[] = [];
  const percentage: ModelInput[] = [];

  for (const [index, year] of years.entries()) {
    const totalGames = totalGamesByYear[index];
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

  return {
    absolute,
    percentage,
  };
};

export const calculatePlayerPredictions = (
  games: readonly GameData[],
  positions: readonly StatsItem[],
): PredictionMap => {
  const predictions = new Map<string, PlayerPrediction>();

  if (games.length === 0 || positions.length === 0) {
    return predictions;
  }

  const yearGameCounts = createYearGameCounts(games);

  const years = getYears(yearGameCounts);

  if (years.length === 0) {
    return predictions;
  }

  const totalGamesByYear = years.map((year) => yearGameCounts.get(year) ?? 0);

  for (const item of positions) {
    const sequences = createSequences(item, years, totalGamesByYear);

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
