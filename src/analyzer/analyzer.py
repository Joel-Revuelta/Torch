import argparse
import numpy as np
from src.analyzer.data_loader import load_training_data, parse_fen, OUTPUT_ENCODING
from src.analyzer.network_loader import load_network, save_network
from src.analyzer.predict import predict
from src.analyzer.train_network import train_network

def main():
    parser = argparse.ArgumentParser(description="Neural network analyzer for chess positions.")
    parser.add_argument("--train", action="store_true", help="Launch the neural network in training mode. Each chessboard in FILE must contain inputs to send to the neural network in FEN notation and the expected output separated by space. If specified, the newly trained neural network will be saved in SAVEFILE. Otherwise, it will be saved in the original LOADFILE.")
    parser.add_argument("--predict", action="store_true", help="Launch the neural network in prediction mode. Each chessboard in FILE must contain inputs to send to the neural network in FEN notation, and optionally an expected output.")
    parser.add_argument("--save", type=str, help="Save neural network into SAVEFILE. Only works in train mode.")
    parser.add_argument("LOADFILE", type=str, help="File containing an artificial neural network.")
    parser.add_argument("FILE", type=str, help="File containing chessboards.")
    args = parser.parse_args()

    if args.train and args.predict:
        parser.error("Cannot train and predict at the same time.")
    elif not args.train and not args.predict:
        parser.error("Must specify either --train or --predict.")

    weights, biases = load_network(args.LOADFILE)

    if args.train:
        training_data = load_training_data(args.FILE)
        weights, biases = train_network(training_data, weights, biases, learning_rate=0.1, epochs=100)

        save_path = args.save if args.save else args.LOADFILE
        save_network(save_path, weights, biases)
    elif args.predict:
        with open(args.FILE, "r") as f:
            fens = f.readlines()
        inputs = [parse_fen(fen) for fen in fens]
        predictions = [predict(input, weights, biases) for input in inputs]
        output_labels = list(OUTPUT_ENCODING.keys())
        for _, prediction in zip(fens, predictions):
            predicted_label = output_labels[np.argmax(prediction)]
            print(predicted_label)

if __name__ == "__main__":
    main()
