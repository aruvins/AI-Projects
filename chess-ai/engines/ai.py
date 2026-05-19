import torch
import chess
import numpy as np
from train.mcts import MCTSNode, run_mcts

from .model import ChessNet


class ChessAI:
    def __init__(self, simulations=200):
        # Prefer MPS (Apple Silicon), then CUDA, then CPU
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.simulations = simulations

        self.model = ChessNet().to(self.device)
        self.model.load_state_dict(
            torch.load("model/chess_net.pth", map_location=self.device)
        )
        self.model.eval()

    def choose_move(self, board: chess.Board) -> chess.Move:
        """
        Uses MCTS to select the best move. Falls back to a random legal move
        if MCTS produces no children (shouldn't happen but is defensive).
        """
        legal = list(board.legal_moves)

        if len(legal) == 0:
            return None

        if len(legal) == 1:
            return legal[0]

        root = MCTSNode(board.copy())

        with torch.no_grad():
            root = run_mcts(
                root, self.model,
                simulations=self.simulations,
                device=self.device
            )

        if not root.children:
            return legal[0]

        # Pick the move with the highest visit count (most robust policy)
        best_move = max(root.children.items(), key=lambda item: item[1].visits)[0]
        return best_move