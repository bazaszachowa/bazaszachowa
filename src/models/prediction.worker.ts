import type { GameData } from "../ChessEditor";
import { calculatePlayerPredictions } from "../components/player/playerPredictions";
import type { StatsItem } from "../components/player/PositionsMoves";

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
      predictions: readonly [string, import("./predictor").PlayerPrediction][];
      requestId: number;
    };

// eslint-disable-next-line sonarjs/post-message
globalThis.addEventListener(
  "message",
  (event: MessageEvent<PredictionWorkerRequest>) => {
    const { games, positions, requestId } = event.data;

    try {
      const predictions = calculatePlayerPredictions(games, positions);

      const response: PredictionWorkerResponse = {
        predictions: [...predictions.entries()],
        requestId,
      };

      globalThis.postMessage(response);
    } catch (error) {
      const response: PredictionWorkerResponse = {
        error: error instanceof Error ? error.message : String(error),
        requestId,
      };

      globalThis.postMessage(response);
    }
  },
);
