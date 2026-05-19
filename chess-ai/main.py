import sys
import os
import torch
import chess.engine
from tqdm import tqdm
import math

from utils.train import train
from utils.gui import ChessGUI
from engines.ai import ChessAI
from utils.stockfish_eval import evaluate_board, stockfish  # Reusing your existing Stockfish wrapper
from engines.lc0_engine import LC0Engine
from engines.stockfish_engine import StockfishAI

def calculate_elo(wins, losses, draws, baseline_elo=1500):
    """
    Calculates an approximate Elo rating based on the game scores
    against a fixed-rated baseline engine.
    """
    total_games = wins + losses + draws
    if total_games == 0:
        return 0
    
    # Calculate score percentage (Wins = 1 pt, Draws = 0.5 pt, Losses = 0 pt)
    score_percentage = (wins + 0.5 * draws) / total_games
    
    # Handle extreme edge cases to prevent logarithm math errors
    if score_percentage >= 1.0:
        return baseline_elo + 400
    if score_percentage <= 0.0:
        return baseline_elo - 400
        
    # Standard Bradley-Terry Elo inversion formula
    elo_diff = -400 * math.log10((1.0 / score_percentage) - 1.0)
    return round(baseline_elo + elo_diff)

def run_elo_evaluation(num_games=10):
    """
    Pits the trained MCTS ChessAI against Stockfish across a series of games,
    alternating colors to eliminate side bias.
    """
    print(f"\n--- Starting Elo Evaluation ({num_games} Games) ---")
    
    # Initialize your trained neural network player
    try:
        ai_agent = StockfishAI(depth=18)
        print(f"Loaded ChessAI on device: {ai_agent.device}")
    except FileNotFoundError:
        print("Error: Trained model weights ('model/chess_net.pth') not found. Train the model first.")
        return

    # Configure Stockfish to a reasonable intermediate baseline (e.g., Level 5 / ~1500 Elo)
    # Your stockfish_eval.py sets depth=12, which roughly operates around 1500-1800 Elo depending on hardware.
    baseline_elo = 4000
    print(f"Baseline Opponent: Stockfish (Depth 18, Anchored at ~{baseline_elo} Elo)")

    stats = {"wins": 0, "losses": 0, "draws": 0}

    for game_idx in range(num_games):
        board = chess.Board()
        
        # Alternate colors: AI is White on even games, Black on odd games
        ai_is_white = (game_idx % 2 == 0)
        ai_color_str = "White" if ai_is_white else "Black"
        
        print(f"\nGame {game_idx + 1}/{num_games} | AI is playing as {ai_color_str}...")
        
        while not board.is_game_over():
            if (board.turn == chess.WHITE and ai_is_white) or (board.turn == chess.BLACK and not ai_is_white):
                # Neural Network Move selection via MCTS
                move = ai_agent.choose_move(board)
            else:
                # Stockfish move selection via your engine wrapper
                stockfish.set_fen_position(board.fen())
                best_move_san = stockfish.get_best_move()
                move = board.parse_san(best_move_san) if best_move_san else list(board.legal_moves)[0]
                
            board.push(move)

        # Game Over Evaluation
        result = board.result()
        print(f"Game Result: {result} ({board.outcome().termination.name})")
        
        if result == "1-0":
            if ai_is_white:
                stats["wins"] += 1
                print("🏆 AI Won!")
            else:
                stats["losses"] += 1
                print("❌ Stockfish Won.")
        elif result == "0-1":
            if not ai_is_white:
                stats["wins"] += 1
                print("🏆 AI Won!")
            else:
                stats["losses"] += 1
                print("❌ Stockfish Won.")
        else:
            stats["draws"] += 1
            print("🤝 Game ended in a Draw.")

    # Final Summary Reports
    print("\n=========================================")
    print("           EVALUATION SUMMARY            ")
    print("=========================================")
    print(f"Record: {stats['wins']} Wins | {stats['losses']} Losses | {stats['draws']} Draws")
    
    calculated_rating = calculate_elo(stats["wins"], stats["losses"], stats["draws"], baseline_elo=baseline_elo)
    print(f"Approximate Agent Elo: {calculated_rating}")
    print("=========================================\n")


def main():
    while True:
        print("♟️ Neural Chess AI (V1)")
        print("----------------------")
        print("1. Train model")
        print("2. Play vs AI")
        print("3. Evaluate Model Elo vs Stockfish")
        print("4. Exit")

        choice = input("\nSelect option: ").strip()

        if choice == "1":
            train()
        elif choice == "2":
            gui = ChessGUI()
            gui.run()
        elif choice == "3":
            try:
                games_input = input("Enter number of evaluation games (default 10): ").strip()
                games = int(games_input) if games_input else 10
            except ValueError:
                games = 10
            run_elo_evaluation(num_games=games)
        elif choice == "4":
            print("Goodbye!")
            sys.exit()
        else:
            print("Invalid option. Please choose 1, 2, 3, or 4.\n")

if __name__ == "__main__":
    main()