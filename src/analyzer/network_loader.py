import json
import numpy as np

def load_network(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    weights = [np.array(w) for w in data['weights']]
    biases = [np.array(b) for b in data['biases']]
    return weights, biases

def save_network(filepath, weights, biases):
    data = {
        "weights": [w.tolist() for w in weights],
        "biases": [b.tolist() for b in biases]
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)