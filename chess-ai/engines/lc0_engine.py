import chess
import chess.engine
import subprocess

import torch

class LC0Engine:
    def __init__(self, lc0_path="lc0", weights_path=None):
        self.engine = chess.engine.SimpleEngine.popen_uci(lc0_path)

        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        if weights_path:
            self.engine.configure({
                "WeightsFile": weights_path
            })

    def choose_move(self, board: chess.Board, time_limit=0.1):
        result = self.engine.play(
            board,
            chess.engine.Limit(time=time_limit)
        )
        return result.move
    


    def close(self):
        self.engine.quit()