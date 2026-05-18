import torch
import chess
import numpy as np

from .model import ChessNet
from .data_loader import board_to_tensor

class ChessAI:
    def __init__(self):
        self.model = ChessNet()
        self.model.load_state_dict(torch.load("model/chess_net.pth"))
        self.model.eval()

    def evaluate(self, board):
        x = torch.tensor(board_to_tensor(board)).unsqueeze(0)
        with torch.no_grad():
            return self.model(x).item()

    def choose_move(self, board):
        best_move = None
        best_value = -999

        for move in board.legal_moves:
            board.push(move)
            value = self.evaluate(board)
            board.pop()

            if value > best_value:
                best_value = value
                best_move = move

        return best_move