import type { GameData } from "../ChessEditor";
import { calculatePlayerPredictions } from "../components/player/playerPredictions";
import type { StatsItem } from "../components/player/PositionsMoves";

interface WorkerRequest {
  games: GameData[];
  positions: StatsItem[];
}

// eslint-disable-next-line unicorn/prefer-global-this
const worker = self as DedicatedWorkerGlobalScope;

worker.addEventListener("message", ({ data }: MessageEvent<WorkerRequest>) => {
  try {
    const predictions = calculatePlayerPredictions(data.games, data.positions);

    worker.postMessage({
      predictions: [...predictions.entries()],
    });
  } catch (error) {
    worker.postMessage({
      error: error instanceof Error ? error.message : String(error),
    });
  }
});
