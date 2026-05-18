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

    q = child.value_sum / child.visits
    u = c * child.prior * np.sqrt(parent.visits) / (1 + child.visits)

    return q + u


def select(node):
    while node.children:
        node = max(node.children.values(), key=lambda n: ucb_score(node, n))
    return node


def expand(node, policy):
    legal_moves = list(node.board.legal_moves)

    for move in legal_moves:
        b = node.board.copy()
        b.push(move)

        idx = move_to_index(move)
        prior = policy[idx]

        node.children[move] = MCTSNode(b, node, prior)


def backpropagate(node, value):
    while node:
        node.visits += 1
        node.value_sum += value
        node = node.parent


def run_mcts(root, model, simulations=50, device="cpu"):
    for _ in range(simulations):

        node = select(root)

        board_tensor = board_to_tensor(node.board)
        board_tensor = torch.tensor(board_tensor).unsqueeze(0).to(device)

        policy, value = model(board_tensor)
        policy = torch.softmax(policy, dim=1).detach().cpu().numpy()[0]
        value = value.item()

        expand(node, policy)
        backpropagate(node, value)

    return root