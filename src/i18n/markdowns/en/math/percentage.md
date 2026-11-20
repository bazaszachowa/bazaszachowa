# Model Percentage

The model is designed to predict the percentage share of a given chess move among all games played, as well as its scoring performance in the following year. It uses historical yearly data and a GRU (Gated Recurrent Unit) neural network to identify temporal dependencies.

## Network Architecture

The model consists of the following layers:

1. **Input** — accepts a sequence of historical data containing three features for each year.
2. **Masking** — ignores padding values used to equalize sequences of different lengths (`-999`).
3. **GRU** — recurrent layer with 32 units and a dropout rate of `0.1`.
4. **Dense** — fully connected layer with 32 neurons and ReLU activation.
5. **Dropout** — reduces the risk of overfitting using a rate of `0.2`.
6. **Output** — Dense output layer with two neurons, each using the Sigmoid activation function.

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

For each year, the model receives information about the occurrence of a given move and its scoring performance.

Definitions:

- **$g_y$** — number of games in which the given move occurred.
- **$G_y$** — total number of games played in the given year.
- **$p_y$** — total number of points scored in games in which the given move occurred.

Three input features are derived from these values.

### Move Share Among Games

$$
u_y =
\begin{cases}
\frac{g_y}{G_y}, & \text{if } G_y > 0, \\
0, & \text{if } G_y = 0.
\end{cases}
$$

Where:

- **$u_y$** — share of games containing the given move among all games.
- **$g_y$** — number of games containing the given move.
- **$G_y$** — total number of games played in the given year.

Normalization makes it possible to compare the popularity of a move across different years, even when the total number of games varies significantly.

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

This value represents the average number of points scored per game in which the move occurred.

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
X_y = [u_y,\ r_y,\ h_y]
$$

Where:

- **$X_y$** — vector of three input features describing a given year.

All input features lie within the range $[0,1]$. As a result, the model operates on normalized values and does not require additional logarithmic transformation of the number of games.

## Output Layer

The model uses a Dense layer with two independent output neurons, each employing the Sigmoid activation function.

$$
z_1 = \operatorname{sigmoid}(a_1)
$$

$$
z_2 = \operatorname{sigmoid}(a_2)
$$

Where:

- **$z_1$** — predicted share of the move among all games.
- **$z_2$** — predicted scoring performance of the move.
- **$a_1, a_2$** — neuron values before applying the activation function.

The Sigmoid function restricts both outputs to the range $[0,1]$, allowing them to represent the percentage share and scoring performance, respectively.

## Output Decoding

The network outputs are transformed into the valid range for the predicted values.

### Predicted Move Share

$$
\hat{u} = \operatorname{clip}(z_1,0,1)
$$

Where:

- **$\hat{u}$** — predicted share of the move among games.
- **$z_1$** — output of the first neuron.

### Predicted Scoring Performance

$$
\hat{r} = \operatorname{clip}(z_2,0,1)
$$

Where:

- **$\hat{r}$** — predicted scoring performance of the move.
- **$z_2$** — output of the second neuron.

The $\operatorname{clip}$ function restricts the value to the range $[0,1]$.

When presenting the results as percentages, both values are multiplied by 100.

## Final Output

$$
f_{\text{percentage}}(x) = [\hat{u},\hat{r}]
$$

Where:

- **$f_{\text{percentage}}(x)$** — model prediction output.
- **$\hat{u}$** — predicted share of the move among games.
- **$\hat{r}$** — predicted scoring performance.

The output describes the relative popularity of the move and its predicted scoring performance in the following year.

Both values are predicted independently. The model does not use a normalization mechanism across different moves. Therefore, the sum of the predicted shares of all moves is not required to equal 100%.

## Training Data Preparation

Historical data is transformed into training samples based on consecutive years. Each sample contains a sequence of data from previous years and target values from the following year.

For historical data covering the years 2020–2023, the following samples are generated:

| Input data | Expected output |
|---|---|
| 2020 | 2021 |
| 2020–2021 | 2022 |
| 2020–2022 | 2023 |

The model learns to predict the move share among games and its scoring performance in the following year based on the available historical data.

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

The loss function accounts for errors in both predicted values: the move share among games and its scoring performance.

The Adam optimizer updates the network weights to minimize the prediction error.

## Training Control Mechanisms

Additional mechanisms are used to control the training process:

- **TerminateOnNaN** — stops training if a NaN value occurs.
- **ModelCheckpoint** — if a save path is provided, saves the model with the best validation performance.
- **ReduceLROnPlateau** — reduces the learning rate when the validation error stops improving.
- **EarlyStopping** — terminates training after a specified number of epochs without improvement in the validation error.



## Model M2 — Percentage: Output Notation

Model M2 (Percentage) predicts percentage values describing the popularity of a move and its scoring performance.

| Notation | Meaning |
| ---------- | -------------------------------------------------------------------------------------------------------------------- |
| **M2O** | Predicted percentage of games featuring a given move out of all games played in a given year (*Occurrence*). |
| **M2P** | Predicted scoring performance of a move, expressed as a percentage (*Points Percentage*). |

### Interpretation of the outputs

- **M2O (Occurrence)** — the estimated proportion of games in which a given move occurred relative to the total number of games played in a given year.
- **M2P (Points Percentage)** — the estimated average number of points scored per game, expressed as a percentage. For example, a value of 65% represents an average of 0.65 points per game.

M2P does not represent the move's share of all points scored. It measures the average scoring performance in games in which the given move occurs.

M2O describes the popularity of a move in percentage terms, whereas M2P represents its scoring performance.
