import numpy as np
import chess
import torch
from train.move_encoding import move_to_index
from utils.data_loader import board_to_tensor


class MCTSNode:
    def __init__(self, board, parent=None, prior=0):
        self.board = board
        self.parent = parent
        self.prior = prior

        self.children = {}
        self.visits = 0
        self.value_sum = 0


def ucb_score(parent, child, c=1.4):
    if child.visits == 0:
        return float("inf")

    # Q is from the CHILD's perspective (child.value_sum was accumulated
    # with sign already flipped during backprop, so we negate here to get
    # the value from the PARENT's perspective when comparing children).
    q = -child.value_sum / child.visits
    u = c * child.prior * np.sqrt(parent.visits) / (1 + child.visits)

    return q + u


def select(node):
    """
    BUG FIX: The original code reassigned `node` inside the loop, so the
    `parent` argument to ucb_score was always the child being evaluated
    (not the actual parent). We fix this by tracking parent explicitly.
    """
    while node.children:
        parent = node
        node = max(node.children.values(), key=lambda n: ucb_score(parent, n))
    return node


def expand(node, policy):
    """
    Only expand if the node is not terminal and has not already been expanded.
    """
    if node.board.is_game_over() or node.children:
        return

    legal_moves = list(node.board.legal_moves)

    # Mask and renormalize policy to legal moves only
    legal_priors = {}
    prior_sum = 0.0
    for move in legal_moves:
        idx = move_to_index(move)
        p = float(policy[idx])
        legal_priors[move] = p
        prior_sum += p

    # Avoid division by zero; fall back to uniform if all priors are zero
    if prior_sum < 1e-8:
        prior_sum = len(legal_moves)
        legal_priors = {m: 1.0 for m in legal_moves}

    for move, prior in legal_priors.items():
        b = node.board.copy()
        b.push(move)
        node.children[move] = MCTSNode(b, node, prior / prior_sum)


def backpropagate(node, value):
    """
    Backpropagates the value. `value` is from the perspective of the player
    whose turn it is at `node`. Each step up the tree flips the sign.
    """
    while node:
        node.visits += 1
        node.value_sum += value
        value = -value
        node = node.parent


def run_mcts(root, model, simulations=200, device="cpu"):
    """
    Increased default simulations from 50 → 200 for substantially better play.
    Also handles terminal nodes correctly (no expansion, use game result as value).
    """
    # Expand root immediately so selection has children to explore
    if not root.children and not root.board.is_game_over():
        board_tensor = torch.tensor(
            board_to_tensor(root.board), dtype=torch.float32
        ).unsqueeze(0).to(device)
        with torch.no_grad():
            policy, value = model(board_tensor)
        policy = torch.softmax(policy, dim=1).detach().cpu().numpy()[0]
        expand(root, policy)

    for _ in range(simulations):
        node = select(root)

        if node.board.is_game_over():
            # Use the actual game result as the value signal
            result = node.board.result()
            if result == "1-0":
                # White wins: +1 if it's black's turn (they just lost), -1 for white
                value = 1.0 if node.board.turn == chess.BLACK else -1.0
            elif result == "0-1":
                value = 1.0 if node.board.turn == chess.WHITE else -1.0
            else:
                value = 0.0
        else:
            board_tensor = torch.tensor(
                board_to_tensor(node.board), dtype=torch.float32
            ).unsqueeze(0).to(device)

            with torch.no_grad():
                policy, value_tensor = model(board_tensor)

            policy = torch.softmax(policy, dim=1).detach().cpu().numpy()[0]
            value = value_tensor.item()

            expand(node, policy)

        backpropagate(node, value)

    return root