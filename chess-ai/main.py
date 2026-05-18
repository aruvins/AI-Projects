import chess
import pygame

from gui import ChessGUI
from engine import find_best_move


AI_DEPTH = 3


class ChessApp(ChessGUI):

    def __init__(self):

        super().__init__()

    def ai_move(self):

        if (
            not self.board.is_game_over()
            and self.board.turn == chess.BLACK
        ):

            move = find_best_move(
                self.board,
                AI_DEPTH
            )

            if move:
                self.board.push(move)

    def run(self):

        running = True

        while running:

            self.clock.tick(60)

            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    running = False

                elif (
                    event.type == pygame.MOUSEBUTTONDOWN
                    and self.board.turn == chess.WHITE
                ):

                    self.handle_click(
                        pygame.mouse.get_pos()
                    )

            # AI move
            self.ai_move()

            # Draw board
            self.draw_game_state()

            pygame.display.flip()

        pygame.quit()


if __name__ == "__main__":

    app = ChessApp()
    app.run()