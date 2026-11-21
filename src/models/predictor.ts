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

interface PercentageOutputWeights extends DenseWeights {
  type: "percentage";
}

interface PreparedAbsoluteOutput {
  count: PreparedDense;
  rate: PreparedDense;
  type: "absolute";
}

interface PreparedDense {
  bias: Float32Array;
  inputSize: number;
  kernel: Float32Array;
  outputSize: number;
}

interface PreparedGru {
  inputBias: Float32Array;
  inputSize: number;
  kernel: Float32Array;
  recurrentBias: Float32Array;
  recurrentKernel: Float32Array;
  units: number;
}

interface PreparedModel {
  dense: PreparedDense;
  gru: PreparedGru;
  output: PreparedAbsoluteOutput | PreparedPercentageOutput;
}

interface PreparedPercentageOutput {
  dense: PreparedDense;
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

/**
 * Spłaszcza macierz w kolejności wierszowej.
 * Wagi są przygotowywane jeden raz podczas inicjalizacji modułu.
 */
const flattenMatrix = (
  matrix: Matrix,
  expectedRows: number,
  expectedColumns: number,
): Float32Array => {
  if (
    matrix.length !== expectedRows ||
    matrix.some((row) => row.length !== expectedColumns)
  ) {
    throw new Error(
      `Invalid matrix shape. Expected ${expectedRows}x${expectedColumns}.`,
    );
  }

  const result = new Float32Array(expectedRows * expectedColumns);
  let offset = 0;

  for (let row = 0; row < expectedRows; row++) {
    for (let column = 0; column < expectedColumns; column++) {
      result[offset++] = matrix[row][column];
    }
  }

  return result;
};

const prepareDense = (
  weights: DenseWeights,
  expectedInputSize?: number,
): PreparedDense => {
  const inputSize = weights.kernel.length;
  const outputSize = weights.kernel[0]?.length ?? 0;

  if (
    inputSize === 0 ||
    outputSize === 0 ||
    weights.kernel.some((row) => row.length !== outputSize) ||
    weights.bias.length !== outputSize ||
    (expectedInputSize !== undefined && inputSize !== expectedInputSize)
  ) {
    throw new Error("Invalid Dense weights shape");
  }

  return {
    bias: Float32Array.from(weights.bias),
    inputSize,
    kernel: flattenMatrix(weights.kernel, inputSize, outputSize),
    outputSize,
  };
};

const prepareGru = (
  weights: GruWeights,
  units: number,
  inputSize: number,
): PreparedGru => {
  const width = units * 3;

  if (weights.bias[0].length !== width || weights.bias[1].length !== width) {
    throw new Error("Invalid GRU bias shape");
  }

  return {
    inputBias: Float32Array.from(weights.bias[0]),
    inputSize,
    kernel: flattenMatrix(weights.kernel, inputSize, width),
    recurrentBias: Float32Array.from(weights.bias[1]),
    recurrentKernel: flattenMatrix(weights.recurrent_kernel, units, width),
    units,
  };
};

const prepareModel = (weights: ModelWeights): PreparedModel => {
  const gru = prepareGru(weights.gru, weights.gru_units, weights.input_size);

  const denseLayer = prepareDense(weights.dense, weights.gru_units);

  const output: PreparedModel["output"] =
    weights.output.type === "absolute"
      ? {
          count: prepareDense(weights.output.count, denseLayer.outputSize),
          rate: prepareDense(weights.output.rate, denseLayer.outputSize),
          type: "absolute",
        }
      : {
          dense: prepareDense(weights.output, denseLayer.outputSize),
          type: "percentage",
        };

  return {
    dense: denseLayer,
    gru,
    output,
  };
};

const absoluteModel = prepareModel(absoluteWeights);
const percentageModel = prepareModel(percentageWeights);

/**
 * Keras GRU:
 * - gate order: z, r, h
 * - reset_after: true
 * - osobny input bias i recurrent bias
 *
 * Padding [-999, -999, -999] jest pomijany.
 */
const runGru = (
  sequence: readonly ModelInput[],
  weights: PreparedGru,
): Float32Array => {
  const {
    inputBias,
    inputSize,
    kernel,
    recurrentBias,
    recurrentKernel,
    units,
  } = weights;

  const width = units * 3;

  let state = new Float32Array(units);
  let nextState = new Float32Array(units);

  const inputProjection = new Float32Array(width);
  const recurrentProjection = new Float32Array(width);

  for (const input of sequence) {
    if (input[0] === -999 && input[1] === -999 && input[2] === -999) {
      continue;
    }

    inputProjection.set(inputBias);

    for (let feature = 0; feature < inputSize; feature++) {
      const value = input[feature];
      const offset = feature * width;

      for (let column = 0; column < width; column++) {
        inputProjection[column] += value * kernel[offset + column];
      }
    }

    recurrentProjection.set(recurrentBias);

    for (let row = 0; row < units; row++) {
      const value = state[row];
      const offset = row * width;

      for (let column = 0; column < width; column++) {
        recurrentProjection[column] += value * recurrentKernel[offset + column];
      }
    }

    for (let index = 0; index < units; index++) {
      const z = sigmoid(inputProjection[index] + recurrentProjection[index]);

      const r = sigmoid(
        inputProjection[units + index] + recurrentProjection[units + index],
      );

      /*
       * reset_after=true:
       * (recurrent_h + recurrent_bias_h) * r
       */
      const candidate = Math.tanh(
        inputProjection[2 * units + index] +
          recurrentProjection[2 * units + index] * r,
      );

      nextState[index] = z * state[index] + (1 - z) * candidate;
    }

    const previousState = state;
    state = nextState;
    nextState = previousState;
  }

  return state;
};

const dense = (
  input: ArrayLike<number>,
  weights: PreparedDense,
  activation: "linear" | "relu" = "linear",
): Float32Array => {
  const { bias, inputSize, kernel, outputSize } = weights;

  if (input.length !== inputSize) {
    throw new Error(
      `Invalid Dense input size: expected ${inputSize}, got ${input.length}`,
    );
  }

  const output = Float32Array.from(bias);

  for (let row = 0; row < inputSize; row++) {
    const value = input[row];
    const offset = row * outputSize;

    for (let column = 0; column < outputSize; column++) {
      output[column] += value * kernel[offset + column];
    }
  }

  if (activation === "relu") {
    for (let index = 0; index < outputSize; index++) {
      output[index] = Math.max(0, output[index]);
    }
  }

  return output;
};

const runHiddenLayers = (
  sequence: readonly ModelInput[],
  model: PreparedModel,
): Float32Array => {
  const state = runGru(sequence, model.gru);

  return dense(state, model.dense, "relu");
};

const predictAbsolute = (
  sequence: readonly ModelInput[],
): AbsolutePrediction => {
  if (absoluteModel.output.type !== "absolute") {
    throw new Error("Nieprawidłowy typ wyjścia modelu absolutnego");
  }

  const hidden = runHiddenLayers(sequence, absoluteModel);
  const output = absoluteModel.output;

  const predictedLogCount = softplus(dense(hidden, output.count)[0]);

  const predictedRate = sigmoid(dense(hidden, output.rate)[0]);

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
  if (percentageModel.output.type !== "percentage") {
    throw new Error("Nieprawidłowy typ wyjścia modelu procentowego");
  }

  const hidden = runHiddenLayers(sequence, percentageModel);

  const output = dense(hidden, percentageModel.output.dense);

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
