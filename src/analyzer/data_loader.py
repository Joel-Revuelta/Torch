
OUTPUT_ENCODING = {
    "Check White": [1, 0, 0, 0, 0, 0],
    "Check Black": [0, 1, 0, 0, 0, 0],
    "Checkmate White": [0, 0, 1, 0, 0, 0],
    "Checkmate Black": [0, 0, 0, 1, 0, 0],
    "Nothing": [0, 0, 0, 0, 1, 0],
    "Stalemate": [0, 0, 0, 0, 0, 1],
}

def parse_fen(fen):
    piece_to_index = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    board_state = [0] * 768
    fen_parts = fen.split()
    board_part = fen_parts[0]
    row = 0
    col = 0
    for char in board_part:
        if char.isdigit():
            col += int(char)
        elif char == '/':
            row += 1
            col = 0
        else:
            index = piece_to_index[char]
            board_state[row * 96 + col * 12 + index] = 1
            col += 1
    return board_state

def load_training_data(filepath):
    training_data = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            fen = " ".join(parts[:6])
            label = " ".join(parts[6:])
            expected_output = OUTPUT_ENCODING.get(label, [0, 0, 0, 0, 0, 0])
            inputs = parse_fen(fen)
            training_data.append((inputs, expected_output))
    return training_data