# My Torch: Neural Network-Based Chessboard Analyzer

## Overview

This project involves creating two binaries:
1. **Neural Network Generator**: Builds customizable neural network structures.
2. **Chessboard Analyzer**: Trains the network or predicts chess game states based on input chessboard positions.

The analyzer uses a supervised learning approach with data formatted in the **Forsyth–Edwards Notation (FEN)** to classify game states into:
- **Check White**
- **Check Black**
- **Checkmate White**
- **Checkmate Black**
- **Stalemate**
- **Nothing**

## Results

### Accuracy

The chessboard analyzer has shown good accuracy in predicting game states. The overall accuracy is **91.26%**.

Here is the accuracy for each game state:

- **Nothing**: **94.50%**
- **Check Black**: **85.93%**
- **Checkmate Black**: **91.86%**
- **Checkmate White**: **83.46%**
- **Check White**: **93.80%**
- **Stalemate**: **84.38%**

These results show the model's effectiveness in recognizing different game states.

## Neural Network Generator

### Purpose
The generator creates a blank neural network structure from a configuration file. The network is saved in a JSON format containing:
- **Weights**: Randomly initialized connections between neurons.
- **Biases**: Initialized to 0.

### Input
A configuration file specifying:
- Number of layers.
- Number of neurons per layer.
- Additional hyperparameters (if applicable).

### Output
A JSON file containing:
- `weights`: Representing the connections between neurons.
- `biases`: Representing the biases for each layer.

### Example Usage
```bash
./my_torch_generator config.conf
```

### Example Output
```json
{
  "weights": [...],
  "biases": [...]
}
```

### Configuration File Format
The configuration file must be written in JSON format and include the following fields:
- `input_size`: Must be set to 768, representing the 768 input features derived from the FEN string.
- `hidden_layers`: A list of integers where each integer represents the number of neurons in a hidden layer. You can customize the number and size of hidden layers as desired.
- `output_size`: Must be set to 6, representing the 6 possible game states.
- `activation`: The activation function to use (e.g., "sigmoid").
- `learning_rate`: The learning rate for the training process.

### Example Configuration File
```json
{
    "input_size": 768,
    "hidden_layers": [256, 128, 64],
    "output_size": 6,
    "activation": "sigmoid",
    "learning_rate": 0.1
}
```

---

## Chessboard Analyzer

### Purpose
The analyzer processes chessboard states to:
1. Train the neural network to recognize game states (training mode).
2. Predict the game state for input chessboards (prediction mode).

### Modes
- **Training Mode**: Updates the network using FEN data and expected outputs.
- **Prediction Mode**: Predicts the game state for given FEN inputs.

### Input
- **Network file**: A JSON file containing weights and biases.
- **Data file**: A text file with chessboard positions in FEN format.

### Output
- **Training Mode**: Saves the updated network back to a JSON file.
- **Prediction Mode**: Outputs the predicted game states.

### Example Usage
- Training:
    ```bash
    ./my_torch_analyzer --train network.json training_data.txt
    ```
- Prediction:
    ```bash
    ./my_torch_analyzer --predict network.json chessboards.txt
    ```

---

## Training Implementation

### How Neural Network Training Works
A neural network is a computational model inspired by the way biological neural networks in the human brain process information. It consists of layers of interconnected nodes (neurons) where each connection has an associated weight. The network learns by adjusting these weights based on the error between the predicted output and the actual output.

### Steps in Training
1. **Forward Pass**: The input data is passed through the network layer by layer to obtain the output predictions. Each neuron applies a weighted sum of its inputs followed by an activation function.
2. **Loss Calculation**: The error (loss) between the predicted outputs and the actual outputs is calculated using a loss function. In this implementation, we use the **cross-entropy loss** function.
3. **Backward Pass**: The gradients of the loss with respect to the weights and biases are calculated using backpropagation. This involves propagating the error backward through the network.
4. **Parameter Update**: The weights and biases are updated using the gradients to minimize the loss. This is done using the **gradient descent** optimization algorithm.

### Activation Functions
Activation functions introduce non-linearity into the network, allowing it to learn complex patterns. The activation function used in this implementation is the **sigmoid** function.

#### Sigmoid Function
The sigmoid function is defined as:
\[ \sigma(x) = \frac{1}{1 + e^{-x}} \]
It maps any real-valued number to a value between 0 and 1, making it useful for binary classification tasks. The sigmoid function has a smooth gradient, which helps in the backpropagation process.

The derivative of the sigmoid function, which is used in the backward pass, is:
\[ \sigma'(x) = \sigma(x) \cdot (1 - \sigma(x)) \]
This derivative is used to calculate the gradient of the loss with respect to the weights and biases during backpropagation.

### Training Function
The `train_network` function in `src/analyzer/train_network.py` handles the training process. It takes the following parameters:
- `training_data`: The training data consisting of input-output pairs.
- `weights`: The initial weights of the network.
- `biases`: The initial biases of the network.
- `learning_rate`: The learning rate for the gradient descent optimization.
- `epochs`: The number of epochs (iterations) to train the network.

The function performs the forward pass, loss calculation, backward pass, and parameter update for each epoch and prints the loss and time taken for each epoch.

---

## Current Implementation

### Completed
1. **Neural Network Generator**:
   - Generates network weights and biases in JSON format.

2. **Chessboard Analyzer**:
   - `parse_fen`: Converts FEN strings into numerical input.
   - `load_training_data`: Reads FEN data and maps outputs to one-hot encodings.
   - `train_network`: Trains the neural network using the provided training data.
   - `predict`: Predicts the game state for given FEN inputs.

### Files and Directories
- `src/generator/generator.py`: Contains the code for generating neural networks.
- `src/analyzer/analyzer.py`: Contains the main code for training and predicting using the neural network.
- `src/analyzer/data_loader.py`: Contains functions for loading and parsing training data.
- `src/analyzer/network_loader.py`: Contains functions for loading and saving neural network weights and biases.
- `src/analyzer/predict.py`: Contains the prediction function.
- `src/analyzer/train_network.py`: Contains the training function.
- `src/utils/config_loader.py`: Contains the function for loading configuration files.
- `configs/torch_network.conf`: Example configuration file for generating a neural network.
- `README.md`: This documentation file.

### Dependencies
- `numpy`: Used for numerical operations.

### Installation
To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

### Running the Project
1. **Generate a Neural Network**:
    ```bash
    ./my_torch_generator configs/torch_network.conf 1
    ```

2. **Train the Neural Network**:
    ```bash
    ./my_torch_analyzer --train network.json training_data.txt
    ```

3. **Predict Using the Neural Network**:
    ```bash
    ./my_torch_analyzer --predict network.json chessboards.txt
    ```

### Additional Scripts
- `scripts/compare_networks.py`: Compares two neural networks.
- `scripts/prediction_percentage.py`: Calculates the accuracy of neural network predictions.
- `scripts/shuffle_lines.py`: Shuffles the lines of a file randomly.
- `scripts/graph_nn.py`: Visualizes the structure of a neural network.

### Makefile
The `Makefile` contains targets for building, running, cleaning, and installing the project.

### .gitignore
The `.gitignore` file specifies files and directories to be ignored by Git.

---

## Future Work
- Implement additional activation functions.
- Add support for different optimization algorithms.
- Improve the visualization of neural network structures.
- Enhance the training process with more advanced techniques.

---

## License
This project is licensed under the MIT License.