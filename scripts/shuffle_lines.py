import argparse
import random

def shuffle_file_lines(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    random.shuffle(lines)

    with open(filepath, 'w') as file:
        file.writelines(lines)

def main():
    parser = argparse.ArgumentParser(description="Shuffle the lines of a file randomly.")
    parser.add_argument("filepath", type=str, help="Path to the file to shuffle.")
    args = parser.parse_args()

    shuffle_file_lines(args.filepath)

if __name__ == "__main__":
    main()
