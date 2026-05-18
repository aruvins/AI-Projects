import chess

# 64 squares × 64 destinations = 4096 possible moves
MOVE_SPACE = 64 * 64

def move_to_index(move: chess.Move):
    return move.from_square * 64 + move.to_square


def index_to_move(index: int):
    from_sq = index // 64
    to_sq = index % 64
    return chess.Move(from_sq, to_sq)