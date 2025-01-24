import numpy as np
import time
from src.analyzer.predict import sigmoid

def format_time(seconds):
    mins, secs = divmod(seconds, 60)
    hrs, mins = divmod(mins, 60)
    time_str = ""
    if hrs > 0:
        time_str += f"{int(hrs)}h "
    if mins > 0 or hrs > 0:
        time_str += f"{int(mins)}m "
    time_str += f"{int(secs)}s"
    return time_str.strip()

def cross_entropy_loss(predictions, targets):
    return -np.sum(targets * np.log(predictions + 1e-9)) / targets.shape[0]

def sigmoid_derivative(x):
    return x * (1 - x)


def forward_pass(inputs, weights, biases):
    activations = [np.array(inputs)]
    for w, b in zip(weights, biases):
        z = np.dot(activations[-1], np.array(w)) + np.array(b)
        a = sigmoid(z)
        activations.append(a)
    return activations

def backward_pass(activations, targets, weights):
    deltas = [activations[-1] - np.array(targets)]
    for i in range(len(weights) - 1, 0, -1):
        delta = np.dot(deltas[-1], np.array(weights[i]).T) * sigmoid_derivative(activations[i])
        deltas.append(delta)
    deltas.reverse()
    return deltas

def update_parameters(weights, biases, activations, deltas, learning_rate):
    for i in range(len(weights)):
        weights[i] -= learning_rate * np.outer(activations[i], deltas[i])
        biases[i] -= learning_rate * deltas[i]
    return weights, biases

def train_network(training_data, weights, biases, learning_rate=0.1, epochs=1000):
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()
        total_loss = 0

        for inputs, targets in training_data:
            activations = forward_pass(inputs, weights, biases)
            loss = cross_entropy_loss(activations[-1], np.array(targets))
            total_loss += loss

            deltas = backward_pass(activations, targets, weights)
            weights, biases = update_parameters(weights, biases, activations, deltas, learning_rate)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        total_duration = epoch_end_time - start_time
        expected_total_time = total_duration / (epoch + 1) * epochs
        remaining_time = expected_total_time - total_duration

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(training_data):.4f}")
        print(f"Time for last epoch: {format_time(epoch_duration)}, Total time elapsed: {format_time(total_duration)}, Expected time remaining: {format_time(remaining_time)}")

    return weights, biases
