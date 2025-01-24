import argparse
import numpy as np
import json

def load_network(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    weights = [np.array(w) for w in data['weights']]
    biases = [np.array(b) for b in data['biases']]
    return weights, biases

def compare_networks(network1, network2):
    weights1, biases1 = load_network(network1)
    weights2, biases2 = load_network(network2)

    if len(weights1) != len(weights2) or len(biases1) != len(biases2):
        print("The networks have different structures.")
        return

    for i, (w1, w2, b1, b2) in enumerate(zip(weights1, weights2, biases1, biases2)):
        if not np.array_equal(w1, w2):
            print(f"Layer {i} weights are different.")
        else:
            print(f"Layer {i} weights are the same.")

        if not np.array_equal(b1, b2):
            print(f"Layer {i} biases are different.")
        else:
            print(f"Layer {i} biases are the same.")

def main():
    parser = argparse.ArgumentParser(description="Compare two neural networks.")
    parser.add_argument("network1", type=str, help="Path to the first network JSON file.")
    parser.add_argument("network2", type=str, help="Path to the second network JSON file.")
    args = parser.parse_args()

    compare_networks(args.network1, args.network2)

if __name__ == "__main__":
    main()
