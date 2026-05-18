import chess
import random
import numpy as np
from tqdm import tqdm

from .stockfish_eval import evaluate_board
from .data_loader import board_to_tensor


def generate_random_position(max_plies=30):
    board = chess.Board()

    for _ in range(random.randint(5, max_plies)):
        if board.is_game_over():
            break
        board.push(random.choice(list(board.legal_moves)))

    return board


def generate_dataset(num_samples=10000):
    X = []
    y = []

    print("Generating synthetic positions...")

    for _ in tqdm(range(num_samples)):
        board = generate_random_position()

        X.append(board_to_tensor(board))
        y.append(evaluate_board(board))

    return np.array(X), np.array(y)