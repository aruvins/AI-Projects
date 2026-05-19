import chess
from stockfish import Stockfish

# Adjust path if needed:
# macOS brew usually:
STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"

stockfish = Stockfish(path=STOCKFISH_PATH, depth=18)


def evaluate_board(board: chess.Board) -> float:
    """
    Returns Stockfish evaluation in centipawns.
    """
    stockfish.set_fen_position(board.fen())
    eval = stockfish.get_evaluation()

    if eval["type"] == "mate":
        # convert mate to large value
        return 10000 if eval["value"] > 0 else -10000

    return eval["value"] / 100.0  # normalize to pawns