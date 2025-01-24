import argparse
import numpy as np
from collections import defaultdict

def load_predictions(filepath):
    with open(filepath, 'r') as f:
        predictions = [line.strip() for line in f.readlines()]
    return predictions

def load_expected(filepath, full_match):
    expected = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if full_match:
                label = " ".join(parts[6:])
            else:
                label = parts[6]
            expected.append(label)
    return expected

def calculate_accuracy(predictions, expected, full_match):
    correct = 0
    total = len(expected)
    label_counts = defaultdict(int)
    correct_counts = defaultdict(int)

    for pred, exp in zip(predictions, expected):
        label_counts[exp] += 1
        if full_match:
            match = pred == exp
        else:
            match = pred.split()[0] == exp.split()[0]
        if match:
            correct += 1
            correct_counts[exp] += 1

    overall_accuracy = (correct / total) * 100
    label_accuracies = {label: (correct_counts[label] / count) * 100 for label, count in label_counts.items()}

    return overall_accuracy, label_accuracies

def color_text(text, accuracy):
    if accuracy > 90:
        return f"\033[92m{text}\033[0m"
    elif 75 <= accuracy <= 90:
        return f"\033[93m{text}\033[0m"
    else:
        return f"\033[91m{text}\033[0m"

def main():
    parser = argparse.ArgumentParser(description="Calculate the accuracy of neural network predictions.")
    parser.add_argument("predictions_file", type=str, help="Path to the file containing the predictions.")
    parser.add_argument("expected_file", type=str, help="Path to the file containing the expected outputs.")
    parser.add_argument("--full", action="store_true", help="Match the entire string instead of just the first word.")
    args = parser.parse_args()

    predictions = load_predictions(args.predictions_file)
    expected = load_expected(args.expected_file, args.full)

    overall_accuracy, label_accuracies = calculate_accuracy(predictions, expected, args.full)
    max_label_length = max(len(label) for label in label_accuracies.keys())

    print(color_text(f"Overall Accuracy: {' ' * (max_label_length - 1)}{overall_accuracy:6.2f}%", overall_accuracy))
    for label, accuracy in label_accuracies.items():
        print(color_text(f"Accuracy for '{label}': {' ' * (max_label_length - len(label))}{accuracy:6.2f}%", accuracy))

if __name__ == "__main__":
    main()