import torch
import numpy as np
import chess
from tqdm import tqdm

from utils.model import ChessNet
from train.mcts import MCTSNode, run_mcts
from utils.data_loader import board_to_tensor
from train.move_encoding import move_to_index


# -----------------------------
# SELF PLAY
# -----------------------------
def self_play_game(model):
    board = chess.Board()

    game_states = []
    game_moves = []

    while not board.is_game_over():

        root = MCTSNode(board.copy())
        root = run_mcts(root, model)

        # convert visit counts → policy target
        visits = np.zeros(64 * 64)

        for move, child in root.children.items():
            idx = move_to_index(move)
            visits[idx] = child.visits

        visits = visits / (np.sum(visits) + 1e-8)

        best_move = max(root.children.items(),
                        key=lambda item: item[1].visits)[0]

        game_states.append(board.copy())
        game_moves.append((visits, best_move))

        board.push(best_move)

    # final game result
    result = board.result()

    if result == "1-0":
        value = 1
    elif result == "0-1":
        value = -1
    else:
        value = 0

    # build training samples
    samples = []

    for state, (policy, _) in zip(game_states, game_moves):
        samples.append((state, policy, value))

    return samples


# -----------------------------
# TRAIN LOOP
# -----------------------------
def train():
    model = ChessNet()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    policy_loss_fn = torch.nn.CrossEntropyLoss()
    value_loss_fn = torch.nn.MSELoss()

    replay_buffer = []

    for iteration in range(10):

        print(f"\n=== Iteration {iteration} ===")

        # -------------------------
        # 1. SELF PLAY
        # -------------------------
        for _ in tqdm(range(20), desc="Self-play"):
            game_data = self_play_game(model)
            replay_buffer.extend(game_data)

        print(f"Replay buffer size: {len(replay_buffer)}")

        # -------------------------
        # 2. TRAINING
        # -------------------------
        model.train()

        for epoch in range(3):
            total_loss = 0

            for state, policy_target, value_target in replay_buffer:

                board_tensor = torch.tensor(
                    board_to_tensor(state),
                    dtype=torch.float32
                ).unsqueeze(0)

                policy_logits, value = model(board_tensor)

                # policy loss (move distribution)
                policy_target = torch.tensor(policy_target, dtype=torch.float32).unsqueeze(0)
                policy_loss = torch.nn.functional.mse_loss(policy_logits, policy_target)

                # value loss (game result)
                value_target = torch.tensor([[value_target]], dtype=torch.float32)
                v_loss = value_loss_fn(value, value_target)

                loss = policy_loss + v_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch} loss: {total_loss:.4f}")

        # -------------------------
        # 3. SAVE MODEL
        # -------------------------
        torch.save(model.state_dict(), "model/chess_net.pth")

        print("Model saved.")

if __name__ == "__main__":
    train()