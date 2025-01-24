import json
import networkx as nx
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("network_file", type=str, help="Path to the neural network file.")
args = parser.parse_args()

with open(args.network_file, 'r') as f:
    nn_structure = json.load(f)

G = nx.DiGraph()

layer_sizes = [len(layer) for layer in nn_structure['weights']]
layer_sizes.append(len(nn_structure['weights'][-1][0]))  # Add the output layer size
for layer_idx, layer in enumerate(nn_structure['weights']):
    for node_idx, node_weights in enumerate(layer):
        for weight_idx, weight in enumerate(node_weights):
            G.add_edge(f"L{layer_idx}_N{node_idx}", f"L{layer_idx+1}_N{weight_idx}", weight=weight)

def layered_layout(G, layer_sizes):
    pos = {}
    x_offset = 0
    for layer_idx, layer_size in enumerate(layer_sizes):
        y_offset = -(layer_size - 1) / 2
        for node_idx in range(layer_size):
            pos[f"L{layer_idx}_N{node_idx}"] = (x_offset, y_offset + node_idx)
        x_offset += 1
    return pos

pos = layered_layout(G, layer_sizes)
edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, font_weight="bold", arrows=True)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

plt.title("Neural Network Structure")
plt.show()
