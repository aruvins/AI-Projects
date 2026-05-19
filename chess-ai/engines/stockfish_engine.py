import chess
import torch
from stockfish import Stockfish

class StockfishAI:
    def __init__(self, binary_path: str = "/opt/homebrew/bin/stockfish", depth: int = 20):
        """
        Initializes your custom Chess AI agent backed by the Stockfish engine.
        """
        # Configurations optimized for your Mac's hardware
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        self.engine = Stockfish(
            path=binary_path,
            depth=depth,
            parameters={
                "Threads": 4,          # Dedicate CPU performance cores
                "Hash": 2048,          # 2GB RAM cache to recall positions
                "Skill Level": 20      # Maximum capability (Elo ~3650+)
            }
        )

    def get_static_evaluation(self, board: chess.Board) -> float:
        """
        Analyzes the current board state and returns a score from White's perspective.
        Useful if you are building your own custom alpha-beta minimax search tree.
        """
        self.engine.set_fen_position(board.fen())
        evaluation = self.engine.get_evaluation()
        
        if evaluation["type"] == "mate":
            # Return heavily weighted metrics for forced mate sequences
            return 10000.0 if evaluation["value"] > 0 else -10000.0
            
        return evaluation["value"] / 100.0  # Normalized to standard pawn units

    def choose_move(self, board: chess.Board) -> chess.Move:
        """
        Asks the neural net to instantly calculate the absolute best move.
        """
        self.engine.set_fen_position(board.fen())
        best_move_san = self.engine.get_best_move()
        return chess.Move.from_uci(best_move_san)