import torch
import torch.nn.functional as F
import numpy as np
import chess
import random
from tqdm import tqdm

from engines.model import ChessNet
from train.mcts import MCTSNode, run_mcts
from utils.data_loader import board_to_tensor
from train.move_encoding import move_to_index


# -----------------------------
# SELF PLAY
# -----------------------------
def self_play_game(model, device="cpu", simulations=200):
    board = chess.Board()

    game_states = []
    game_policies = []

    while not board.is_game_over():
        root = MCTSNode(board.copy())
        root = run_mcts(root, model, simulations=simulations, device=device)

        # Convert visit counts → policy target (normalized)
        visits = np.zeros(64 * 64, dtype=np.float32)
        for move, child in root.children.items():
            idx = move_to_index(move)
            visits[idx] = child.visits

        visit_sum = visits.sum()
        if visit_sum > 0:
            visits /= visit_sum

        best_move = max(root.children.items(), key=lambda item: item[1].visits)[0]

        # Store the player-to-move so we can assign value correctly later
        game_states.append((board.copy(), board.turn))
        game_policies.append(visits)

        board.push(best_move)

    # Determine game result from each player's perspective
    result = board.result()
    if result == "1-0":
        white_value = 1.0
    elif result == "0-1":
        white_value = -1.0
    else:
        white_value = 0.0

    samples = []
    for (state, player_turn), policy in zip(game_states, game_policies):
        # Value is from the perspective of the player who was to move
        value = white_value if player_turn == chess.WHITE else -white_value
        samples.append((state, policy, value))

    return samples


# -----------------------------
# TRAIN LOOP
# -----------------------------
def train():
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    model = ChessNet().to(device)

    # Try loading existing weights to continue training
    try:
        model.load_state_dict(torch.load("model/chess_net.pth", map_location=device))
        print("Loaded existing model weights — continuing training.")
    except FileNotFoundError:
        print("No existing weights found — training from scratch.")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    value_loss_fn = torch.nn.MSELoss()

    # Cap replay buffer size to avoid unbounded memory growth
    MAX_BUFFER_SIZE = 50_000
    replay_buffer = []

    BATCH_SIZE = 128

    for iteration in range(10):
        print(f"\n=== Iteration {iteration + 1}/10 ===")

        # -------------------------
        # 1. SELF PLAY
        # -------------------------
        model.eval()
        new_samples = []
        for _ in tqdm(range(20), desc="Self-play games"):
            game_data = self_play_game(model, device=device)
            new_samples.extend(game_data)

        replay_buffer.extend(new_samples)

        # Trim buffer to max size (keep most recent)
        if len(replay_buffer) > MAX_BUFFER_SIZE:
            replay_buffer = replay_buffer[-MAX_BUFFER_SIZE:]

        print(f"Replay buffer size: {len(replay_buffer)}")

        # -------------------------
        # 2. TRAINING (mini-batches)
        # -------------------------
        model.train()

        for epoch in range(5):
            # Shuffle each epoch
            random.shuffle(replay_buffer)

            total_policy_loss = 0.0
            total_value_loss = 0.0
            num_batches = 0

            for batch_start in range(0, len(replay_buffer), BATCH_SIZE):
                batch = replay_buffer[batch_start: batch_start + BATCH_SIZE]
                if len(batch) < 2:
                    continue

                states, policy_targets, value_targets = zip(*batch)

                board_tensors = torch.tensor(
                    np.stack([board_to_tensor(s) for s in states]),
                    dtype=torch.float32
                ).to(device)

                policy_targets_t = torch.tensor(
                    np.stack(policy_targets), dtype=torch.float32
                ).to(device)

                value_targets_t = torch.tensor(
                    np.array(value_targets, dtype=np.float32)
                ).unsqueeze(1).to(device)

                policy_logits, value_pred = model(board_tensors)

                # Policy loss: KL divergence (soft targets)
                # Equivalent to cross-entropy when targets sum to 1
                log_probs = F.log_softmax(policy_logits, dim=1)
                # Avoid log(0) by clamping targets
                safe_targets = policy_targets_t.clamp(min=1e-8)
                policy_loss = -(safe_targets * log_probs).sum(dim=1).mean()

                # Value loss: MSE
                v_loss = value_loss_fn(value_pred, value_targets_t)

                loss = policy_loss + v_loss

                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += v_loss.item()
                num_batches += 1

            if num_batches > 0:
                print(
                    f"  Epoch {epoch + 1} | "
                    f"Policy Loss: {total_policy_loss / num_batches:.4f} | "
                    f"Value Loss: {total_value_loss / num_batches:.4f}"
                )

        # -------------------------
        # 3. SAVE MODEL
        # -------------------------
        import os
        os.makedirs("model", exist_ok=True)
        torch.save(model.state_dict(), "model/chess_net.pth")
        print("Model saved.")


if __name__ == "__main__":
    train()