import chess
import numpy as np
from tqdm import tqdm
# import zstandard as zstd

from .stockfish_eval import evaluate_board


def board_to_tensor(board):
    tensor = np.zeros((12, 8, 8), dtype=np.float32)

    piece_map = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            idx = piece_map[piece.piece_type]
            if piece.color == chess.WHITE:
                idx += 6

            row = square // 8
            col = square % 8
            tensor[idx, row, col] = 1

    return tensor


from tqdm import tqdm

def extract_positions(games, max_positions=20000):
    X = []
    y = []

    print("Extracting positions...")

    for game in tqdm(games):
        board = game.board()

        for move in game.mainline_moves():
            board.push(move)

            X.append(board_to_tensor(board))
            y.append(evaluate_board(board))  # STOCKFISH

            if len(X) >= max_positions:
                return X, y

    return X, y