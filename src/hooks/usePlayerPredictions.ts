import { useEffect, useState } from "react";

import type { GameData } from "../ChessEditor";
import type { PlayerPrediction } from "../models/predictor";
import type { StatsItem } from "../components/player/PositionsMoves.tsx";

interface PredictionState {
  predictions: Map<string, PlayerPrediction>;
  loading: boolean;
  error: string | null;
}

const initialState: PredictionState = {
  predictions: new Map(),
  loading: false,
  error: null,
};

export const usePlayerPredictions = (
  games: readonly GameData[],
  positions: readonly StatsItem[],
): PredictionState => {
  const [state, setState] = useState(initialState);

  useEffect(() => {
    if (games.length === 0 || positions.length === 0) {
      setState(initialState);
      return;
    }

    const worker = new Worker(
      new URL("../models/prediction.worker.ts", import.meta.url),
      { type: "module" },
    );

    let active = true;

    setState((previous) => ({
      ...previous,
      loading: true,
      error: null,
    }));

    worker.onmessage = ({ data }) => {
      if (!active) return;

      if (data.error) {
        setState({
          predictions: new Map(),
          loading: false,
          error: data.error,
        });
      } else {
        setState({
          predictions: new Map(data.predictions),
          loading: false,
          error: null,
        });
      }

      worker.terminate();
    };

    worker.onerror = (event) => {
      if (!active) return;

      setState({
        predictions: new Map(),
        loading: false,
        error: event.message,
      });

      worker.terminate();
    };

    worker.postMessage({ games, positions });

    return () => {
      active = false;
      worker.terminate();
    };
  }, [games, positions]);

  return state;
};
