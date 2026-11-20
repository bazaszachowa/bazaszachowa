import { map } from "es-toolkit/compat";

import absoluteWeightsJson from "./absolute.json";
import percentageWeightsJson from "./percentage.json";

export interface AbsolutePrediction {
  count: number;
  points: number;
}

export type ModelInput = readonly [number, number, number];
export interface PercentagePrediction {
  gameShare: number;
  pointScore: number;
}

export interface PlayerPrediction {
  absolute: AbsolutePrediction;
  percentage: PercentagePrediction;
}

interface AbsoluteOutputWeights {
  count: DenseWeights;
  rate: DenseWeights;
  type: "absolute";
}

interface DenseWeights {
  bias: Vector;
  kernel: Matrix;
}

interface GruWeights {
  bias: [Vector, Vector];
  kernel: Matrix;
  recurrent_kernel: Matrix;
}

type Matrix = number[][];

interface ModelWeights {
  dense: DenseWeights;
  gru: GruWeights;
  gru_units: number;
  input_size: number;
  output: AbsoluteOutputWeights | PercentageOutputWeights;
}

interface PercentageOutputWeights {
  bias: Vector;
  kernel: Matrix;
  type: "percentage";
}

type Vector = number[];

const absoluteWeights = absoluteWeightsJson as unknown as ModelWeights;

const percentageWeights = percentageWeightsJson as unknown as ModelWeights;

const sigmoid = (x: number): number => {
  if (x >= 0) {
    return 1 / (1 + Math.exp(-x));
  }

  const exp = Math.exp(x);
  return exp / (1 + exp);
};

const softplus = (x: number): number =>
  Math.max(x, 0) + Math.log1p(Math.exp(-Math.abs(x)));

const vectorMatrix = (vector: readonly number[], matrix: Matrix): Vector => {
  const outputSize = matrix[0]?.length ?? 0;

  return Array.from({ length: outputSize }, (_, column) =>
    vector.reduce((sum, value, row) => sum + value * matrix[row][column], 0),
  );
};

const addVectors = (a: readonly number[], b: readonly number[]): Vector =>
  a.map((value, index) => value + b[index]);

const sliceVector = (
  values: readonly number[],
  start: number,
  end: number,
): Vector => values.slice(start, end);

const dense = (
  input: readonly number[],
  weights: DenseWeights,
  activation: "linear" | "relu" = "linear",
): Vector => {
  const output = weights.kernel[0].map((_, column) => {
    let value = weights.bias[column];

    for (const [row, element] of input.entries()) {
      value += element * weights.kernel[row][column];
    }

    return activation === "relu" ? Math.max(0, value) : value;
  });

  return output;
};

/**
 * Keras GRU:
 * - gate order: z, r, h
 * - reset_after: true
 * - input bias and recurrent bias are stored separately
 *
 * Timesteps with [-999, -999, -999] are treated as padding.
 */
const runGru = (
  sequence: readonly ModelInput[],
  weights: GruWeights,
  units: number,
): Vector => {
  let state = Array.from({ length: units }, () => 0);

  const inputBias = weights.bias[0];
  const recurrentBias = weights.bias[1];

  for (const input of sequence) {
    if (input.every((value) => value === -999)) {
      continue;
    }

    const inputProjection = vectorMatrix(input, weights.kernel);
    const recurrentProjection = vectorMatrix(state, weights.recurrent_kernel);

    const inputZ = sliceVector(inputProjection, 0, units);
    const inputR = sliceVector(inputProjection, units, 2 * units);
    const inputH = sliceVector(inputProjection, 2 * units, 3 * units);

    const recurrentZ = sliceVector(recurrentProjection, 0, units);
    const recurrentR = sliceVector(recurrentProjection, units, 2 * units);
    const recurrentH = sliceVector(recurrentProjection, 2 * units, 3 * units);

    const inputBiasZ = sliceVector(inputBias, 0, units);
    const inputBiasR = sliceVector(inputBias, units, 2 * units);
    const inputBiasH = sliceVector(inputBias, 2 * units, 3 * units);

    const recurrentBiasZ = sliceVector(recurrentBias, 0, units);
    const recurrentBiasR = sliceVector(recurrentBias, units, 2 * units);
    const recurrentBiasH = sliceVector(recurrentBias, 2 * units, 3 * units);

    const z = map(
      addVectors(
        addVectors(inputZ, inputBiasZ),
        addVectors(recurrentZ, recurrentBiasZ),
      ),
      sigmoid,
    );

    const r = map(
      addVectors(
        addVectors(inputR, inputBiasR),
        addVectors(recurrentR, recurrentBiasR),
      ),
      sigmoid,
    );

    const candidate = inputH.map((value, index) => {
      const recurrentValue =
        (recurrentH[index] + recurrentBiasH[index]) * r[index];

      return Math.tanh(value + inputBiasH[index] + recurrentValue);
    });

    state = state.map(
      (previous, index) =>
        z[index] * previous + (1 - z[index]) * candidate[index],
    );
  }

  return state;
};

const runHiddenLayers = (
  sequence: readonly ModelInput[],
  weights: ModelWeights,
): Vector => {
  const state = runGru(sequence, weights.gru, weights.gru_units);

  return dense(state, weights.dense, "relu");
};

const predictAbsolute = (
  sequence: readonly ModelInput[],
): AbsolutePrediction => {
  if (absoluteWeights.output.type !== "absolute") {
    throw new Error("Nieprawidłowy typ wyjścia modelu absolutnego");
  }

  const hidden = runHiddenLayers(sequence, absoluteWeights);

  const predictedLogCount = softplus(
    dense(hidden, absoluteWeights.output.count)[0],
  );

  const predictedRate = sigmoid(dense(hidden, absoluteWeights.output.rate)[0]);

  const count = Math.max(0, Math.expm1(predictedLogCount));
  const rate = Math.max(0, Math.min(1, predictedRate));

  return {
    count,
    points: count * rate,
  };
};

const predictPercentage = (
  sequence: readonly ModelInput[],
): PercentagePrediction => {
  if (percentageWeights.output.type !== "percentage") {
    throw new Error("Nieprawidłowy typ wyjścia modelu procentowego");
  }

  const hidden = runHiddenLayers(sequence, percentageWeights);
  const output = dense(hidden, percentageWeights.output);

  return {
    gameShare: sigmoid(output[0]),
    pointScore: sigmoid(output[1]),
  };
};

export const predictPlayer = (
  absoluteSequence: readonly ModelInput[],
  percentageSequence: readonly ModelInput[],
): PlayerPrediction => ({
  absolute: predictAbsolute(absoluteSequence),
  percentage: predictPercentage(percentageSequence),
});
