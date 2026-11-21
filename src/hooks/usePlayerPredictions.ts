import { useEffect, useRef, useState } from "react";

import type { GameData } from "../ChessEditor";
import type { StatsItem } from "../components/player/PositionsMoves";
import type { PlayerPrediction } from "../models/predictor";

interface PredictionState {
  error: null | string;
  loading: boolean;
  predictions: Map<string, PlayerPrediction>;
}

interface PredictionWorkerRequest {
  games: readonly GameData[];
  positions: readonly StatsItem[];
  requestId: number;
}

type PredictionWorkerResponse =
  | {
      error: string;
      requestId: number;
    }
  | {
      predictions: readonly [string, PlayerPrediction][];
      requestId: number;
    };

const createInitialState = (): PredictionState => ({
  error: null,
  loading: true,
  predictions: new Map(),
});

export const usePlayerPredictions = (
  games: readonly GameData[],
  positions: readonly StatsItem[],
): PredictionState => {
  const [state, setState] = useState<PredictionState>(createInitialState);

  const workerReference = useRef<null | Worker>(null);
  const requestIdReference = useRef(0);

  useEffect(() => {
    const worker = new Worker(
      new URL("../models/prediction.worker.ts", import.meta.url),
      { type: "module" },
    );

    workerReference.current = worker;

    const handleMessage = (event: MessageEvent<PredictionWorkerResponse>) => {
      const data = event.data;

      if (data.requestId !== requestIdReference.current) {
        return;
      }

      if ("error" in data) {
        setState({
          error: data.error,
          loading: false,
          predictions: new Map(),
        });

        return;
      }

      setState({
        error: null,
        loading: false,
        predictions: new Map(data.predictions),
      });
    };

    const handleError = (event: ErrorEvent) => {
      setState({
        error: event.message || "Błąd Workera",
        loading: false,
        predictions: new Map(),
      });
    };

    worker.addEventListener("message", handleMessage);
    worker.addEventListener("error", handleError);

    return () => {
      worker.removeEventListener("message", handleMessage);
      worker.removeEventListener("error", handleError);

      worker.terminate();

      if (workerReference.current === worker) {
        workerReference.current = null;
      }
    };
  }, []);

  useEffect(() => {
    const worker = workerReference.current;
    const requestId = ++requestIdReference.current;

    if (games.length === 0 || positions.length === 0) {
      setState(createInitialState());
      return;
    }

    if (!worker) {
      return;
    }

    setState((previous) => ({
      ...previous,
      error: null,
      loading: true,
    }));

    const request: PredictionWorkerRequest = {
      games,
      positions,
      requestId,
    };

    worker.postMessage(request);
  }, [games, positions]);

  return state;
};
