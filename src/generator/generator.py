import argparse
import os
import json
from src.utils.config_loader import load_config_file
from src.neural_network import NeuralNetwork

def generate_network(config_file, count, output_dir="."):
    config = load_config_file(config_file)

    input_size = config["input_size"]
    hidden_layers = config["hidden_layers"]
    output_size = config["output_size"]
    activation = config.get("activation", "sigmoid")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(1, count + 1):
        nn = NeuralNetwork(input_size, hidden_layers, output_size, activation)
        output_file = os.path.join(output_dir, f"basic_network_{i}.nn")
        save_network_struct(nn, output_file)
        print(f"Neural network generated and saved to {output_file}")

def save_network_struct(nn, output_file):
    structure = {
        "weights": [w.tolist() for w in nn.weights],
        "biases": [b.tolist() for b in nn.biases]
    }
    with open(output_file, "w") as f:
        json.dump(structure, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a neural network from a configuration file.")
    parser.add_argument("config_file", type=str, help="Path to the configuration file.")
    parser.add_argument("count", type=int, help="Number of networks to generate.")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory to save the generated network structure (default: current directory).")

    args = parser.parse_args()

    generate_network(args.config_file, args.count, args.output_dir)
