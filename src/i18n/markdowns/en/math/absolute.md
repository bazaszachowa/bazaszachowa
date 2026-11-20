The model is designed to predict the absolute number of games in which a given chess move will occur, as well as the number of points scored in the following year. It uses historical yearly data and a GRU (Gated Recurrent Unit) neural network to identify temporal dependencies.

## Network Architecture

The model consists of the following layers:

1. **Input** — accepts a sequence of historical data containing three features for each year.
2. **Masking** — ignores padding values used to equalize sequences of different lengths (`-999`).
3. **GRU** — recurrent layer with 32 units and a dropout rate of `0.1`.
4. **Dense** — fully connected layer with 32 neurons and ReLU activation.
5. **Dropout** — reduces the risk of overfitting using a rate of `0.2`.
6. **Output** — two independent output neurons: the first uses Softplus activation, while the second uses Sigmoid activation.

### Training Parameters

| Parameter | Value |
|---|---:|
| Recurrent network type | GRU |
| Number of GRU units | 32 |
| GRU dropout | 0.1 |
| Number of Dense neurons | 32 |
| Dense activation | ReLU |
| Dense dropout | 0.2 |
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Loss function | MSE |
| Metric | MAE |
| Batch size | 128 |
| Default number of epochs | 30 |
| Validation split | 20% |
| Gradient clipping | `clipnorm = 1.0` |

## Input Data Encoding

For each year, the model receives information about the number of occurrences of a given move and its scoring performance.

Definitions:

- **$g_y$** — number of games in which the given move occurred.
- **$p_y$** — total number of points scored in games in which the given move occurred.

Three input features are derived from these values.

### Logarithmic Representation of the Number of Games

$$
q_y = \ln(1+g_y)
$$

Where:

- **$q_y$** — logarithmically encoded number of games.
- **$g_y$** — number of games containing the given move.

Logarithmic transformation reduces the influence of very large values and decreases the scale differences between small and large occurrence counts.

This allows the model to learn more effectively from data containing substantially different numbers of games.

### Scoring Performance

$$
r_y =
\begin{cases}
\frac{p_y}{g_y}, & \text{if } g_y > 0, \\
0, & \text{if } g_y = 0.
\end{cases}
$$

Where:

- **$r_y$** — average scoring performance of the move.
- **$p_y$** — total number of points scored.
- **$g_y$** — number of games containing the given move.

Scoring performance allows the model to account for the quality of a move independently of its number of occurrences.

Under the standard chess scoring system (1 point for a win, 0.5 points for a draw, and 0 points for a loss), the scoring performance lies within the range $[0,1]$.

### Move Occurrence Indicator

$$
h_y =
\begin{cases}
1, & \text{if } g_y > 0, \\
0, & \text{if } g_y = 0.
\end{cases}
$$

Where:

- **$h_y$** — indicates whether the move occurred in the given year.

This additional feature allows the model to distinguish between a move that did not occur and a move that occurred but achieved a scoring performance of zero.

### Input Vector

$$
X_y = [q_y,\ r_y,\ h_y]
$$

Where:

- **$X_y$** — vector of three input features describing a given year.

The logarithmic representation of the number of games reduces scale differences between small and large values. Scoring performance and the move occurrence indicator provide additional information about the historical data.

## Output Layer

The model uses two independent output neurons, each with its own activation function.

### Predicted Number of Games

$$
z_1 = \operatorname{softplus}(a_1)
$$

Where:

- **$z_1$** — predicted logarithmically encoded number of games.
- **$a_1$** — value of the first neuron before applying the activation function.

The Softplus function ensures a non-negative output, which is appropriate for the predicted logarithmically encoded number of games.

### Predicted Scoring Performance

$$
z_2 = \operatorname{sigmoid}(a_2)
$$

Where:

- **$z_2$** — predicted scoring performance.
- **$a_2$** — value of the second neuron before applying the activation function.

The Sigmoid function restricts the output to the range $[0,1]$, allowing it to represent scoring performance.

## Output Decoding

The network outputs are transformed back to their original scale to obtain the predicted number of games and the number of points scored.

### Predicted Number of Games

$$
\hat{g} = \max(0,\ e^{z_1}-1)
$$

Where:

- **$\hat{g}$** — predicted number of games.
- **$z_1$** — output of the first neuron.

Decoding reverses the logarithmic transformation applied during data preparation:

$$
q_y = \ln(1+g_y)
$$

This transforms the predicted value from logarithmic space back to the original number-of-games scale.

### Predicted Number of Points

$$
\hat{p} =
\hat{g} \times \operatorname{clip}(z_2,0,1)
$$

Where:

- **$\hat{p}$** — predicted number of points scored.
- **$\hat{g}$** — predicted number of games.
- **$z_2$** — predicted scoring performance.
- **$\operatorname{clip}$** — function restricting a value to the range $[0,1]$.

The predicted number of points is calculated using the predicted number of games and the predicted scoring performance.

This approach ensures consistency between the two predicted quantities.

## Final Output

$$
f_{\text{absolute}}(x) = [\hat{g},\hat{p}]
$$

Where:

- **$f_{\text{absolute}}(x)$** — model prediction output.
- **$\hat{g}$** — predicted number of games.
- **$\hat{p}$** — predicted number of points scored.

The output describes the absolute scale of the move's occurrence and its predicted performance expressed as the number of points scored.

## Training Data Preparation

Historical data is transformed into training samples based on consecutive years. Each sample contains a sequence of data from previous years and target values from the following year.

For historical data covering the years 2020–2023, the following samples are generated:

| Input data | Expected output |
|---|---|
| 2020 | 2021 |
| 2020–2021 | 2022 |
| 2020–2022 | 2023 |

The model learns to predict the number of games and the number of points scored in the following year based on the available historical data.

The target values consist of the logarithmically encoded number of games and scoring performance. After prediction, the results are transformed back to their original scale.

Sequences of different lengths are padded with the value `-999`, which is ignored by the Masking layer.

### Data Split

The data is divided into training and validation sets at the player–role series level.

- **Training set** — used to update the network weights.
- **Validation set** — used to evaluate prediction quality during training.

The default validation set proportion is 20%.

## Loss Function

The model uses the Mean Squared Error (MSE) loss function:

$$
MSE =
\frac{1}{N}
\sum_{i=1}^{N}
(y_i-\hat{y}_i)^2
$$

Where:

- **$MSE$** — mean squared prediction error.
- **$N$** — number of predicted values.
- **$y_i$** — actual value.
- **$\hat{y}_i$** — predicted value.

The loss function accounts for errors in both predicted values: the logarithmically encoded number of games and scoring performance.

This means that the error for the number of games is calculated in logarithmic space rather than on the original occurrence-count scale.

The Adam optimizer updates the network weights to minimize the prediction error.

## Training Control Mechanisms

Additional mechanisms are used to control the training process:

- **TerminateOnNaN** — stops training if a NaN value occurs.
- **ModelCheckpoint** — if a save path is provided, saves the model with the best validation performance.
- **ReduceLROnPlateau** — reduces the learning rate when the validation error stops improving.
- **EarlyStopping** — terminates training after a specified number of epochs without improvement in the validation error.



## Model M1 — Absolute: Output Notation

Model M1 (Absolute) predicts absolute values describing the popularity of a move and the number of points scored in games in which that move occurred.

| Notation | Meaning |
| ---------- | ----------------------------------------------------------------------------------------- |
| **M1C** | Predicted number of games in which a given move will occur (*Games Count*). |
| **M1P** | Predicted total number of points scored in games in which a given move occurred (*Points*). |

### Interpretation of the outputs

- **M1C (Games Count)** — the estimated number of games in which a given move will occur. This value describes the predicted popularity of the move in absolute terms.
- **M1P (Points)** — the estimated total number of points scored in games in which a given move occurred.

Both values are predicted by model M1. M1C describes how frequently the move is expected to occur, while M1P represents the predicted number of points scored in games featuring that move.
