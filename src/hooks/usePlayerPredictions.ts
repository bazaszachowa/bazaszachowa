import { useEffect, useState } from "react";

import type { GameData } from "../ChessEditor";
import type { StatsItem } from "../components/player/PositionsMoves.tsx";
import type { PlayerPrediction } from "../models/predictor";

interface PredictionState {
  error: null | string;
  loading: boolean;
  predictions: Map<string, PlayerPrediction>;
}

type PredictionWorkerResponse =
  | {
      error: string;
      predictions?: never;
    }
  | {
      error?: null;
      predictions: readonly (readonly [string, PlayerPrediction])[];
    };

const createInitialState = (): PredictionState => ({
  error: null,
  loading: false,
  predictions: new Map(),
});

export const usePlayerPredictions = (
  games: readonly GameData[],
  positions: readonly StatsItem[],
): PredictionState => {
  const [state, setState] = useState<PredictionState>(createInitialState);

  useEffect(() => {
    let active = true;
    let worker: undefined | Worker;

    const handleMessage = (event: MessageEvent<PredictionWorkerResponse>) => {
      if (!active) {
        return;
      }

      const { data } = event;

      if (data.error) {
        setState({
          error: data.error,
          loading: false,
          predictions: new Map(),
        });
      } else {
        setState({
          error: null,
          loading: false,
          predictions: new Map(data.predictions),
        });
      }

      worker?.terminate();
    };

    const handleError = (event: ErrorEvent) => {
      if (!active) {
        return;
      }

      setState({
        error: event.message,
        loading: false,
        predictions: new Map(),
      });

      worker?.terminate();
    };

    if (games.length === 0 || positions.length === 0) {
      setState(createInitialState());
    } else {
      worker = new Worker(
        new URL("../models/prediction.worker.ts", import.meta.url),
        { type: "module" },
      );

      setState((previous) => ({
        ...previous,
        error: null,
        loading: true,
      }));

      worker.addEventListener("message", handleMessage);
      worker.addEventListener("error", handleError);

      worker.postMessage({ games, positions });
    }

    return () => {
      active = false;

      if (worker) {
        worker.removeEventListener("message", handleMessage);
        worker.removeEventListener("error", handleError);
        worker.terminate();
      }
    };
  }, [games, positions]);

  return state;
};
