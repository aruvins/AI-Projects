import pygame
import chess
from .ai import ChessAI



# ----------------------------
# Configuration
# ----------------------------

WIDTH = 640
HEIGHT = 640

DIMENSION = 8
SQ_SIZE = WIDTH // DIMENSION

LIGHT_COLOR = (240, 217, 181)
DARK_COLOR = (181, 136, 99)

HIGHLIGHT_COLOR = (246, 246, 105)

FPS = 60

# Unicode chess pieces
PIECE_UNICODE = {
    "P": "♙",
    "N": "♘",
    "B": "♗",
    "R": "♖",
    "Q": "♕",
    "K": "♔",
    "p": "♟",
    "n": "♞",
    "b": "♝",
    "r": "♜",
    "q": "♛",
    "k": "♚",
}


# ----------------------------
# GUI Class
# ----------------------------

class ChessGUI:

    def __init__(self):
        self.ai = ChessAI()
        self.ai_enabled = True

        self.game_over = False
        self.result = None

        pygame.init()

        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Chess AI")

        self.clock = pygame.time.Clock()
        self.coord_font = pygame.font.SysFont("arial", 18)
        self.board = chess.Board()

        self.selected_square = None
        self.valid_moves = []

        # Font for pieces
        self.font = pygame.font.SysFont("arialunicode", 64)

    # ----------------------------
    # Draw Board
    # ----------------------------

    def draw_board(self):

        for row in range(DIMENSION):
            for col in range(DIMENSION):

                color = LIGHT_COLOR if (row + col) % 2 == 0 else DARK_COLOR

                pygame.draw.rect(
                    self.screen,
                    color,
                    (
                        col * SQ_SIZE,
                        row * SQ_SIZE,
                        SQ_SIZE,
                        SQ_SIZE
                    )
                )

    def draw_coordinates(self):
        letters = ["a", "b", "c", "d", "e", "f", "g", "h"]

        # Draw files (a–h) at bottom
        for col in range(8):
            text = self.coord_font.render(letters[col], True, (0, 0, 0))

            x = col * SQ_SIZE + SQ_SIZE - 12
            y = HEIGHT - 18

            self.screen.blit(text, (x, y))

        # Draw ranks (1–8) on left side
        for row in range(8):
            rank = str(8 - row)
            text = self.coord_font.render(rank, True, (0, 0, 0))

            x = 5
            y = row * SQ_SIZE + 5

            self.screen.blit(text, (x, y))
    # ----------------------------
    # Highlight Squares
    # ----------------------------

    def highlight_squares(self):

        if self.selected_square is not None:

            row = 7 - (self.selected_square // 8)
            col = self.selected_square % 8

            pygame.draw.rect(
                self.screen,
                HIGHLIGHT_COLOR,
                (
                    col * SQ_SIZE,
                    row * SQ_SIZE,
                    SQ_SIZE,
                    SQ_SIZE
                )
            )

            for move in self.valid_moves:

                target = move.to_square

                target_row = 7 - (target // 8)
                target_col = target % 8

                pygame.draw.circle(
                    self.screen,
                    (50, 50, 50),
                    (
                        target_col * SQ_SIZE + SQ_SIZE // 2,
                        target_row * SQ_SIZE + SQ_SIZE // 2
                    ),
                    10
                )

    # ----------------------------
    # Draw Pieces
    # ----------------------------

    def draw_pieces(self):

        for square in chess.SQUARES:

            piece = self.board.piece_at(square)

            if piece:

                piece_symbol = PIECE_UNICODE[piece.symbol()]

                row = 7 - (square // 8)
                col = square % 8

                text = self.font.render(piece_symbol, True, (0, 0, 0))

                text_rect = text.get_rect(
                    center=(
                        col * SQ_SIZE + SQ_SIZE // 2,
                        row * SQ_SIZE + SQ_SIZE // 2
                    )
                )

                self.screen.blit(text, text_rect)

    # ----------------------------
    # Draw Entire Game State
    # ----------------------------

    def draw_game_state(self):

        self.draw_board()
        self.highlight_squares()
        self.draw_pieces()
        self.draw_coordinates()
        
        if self.game_over:
            self.draw_game_over()

    # ----------------------------
    # Draw Game Over Screen
    # ----------------------------

    def draw_game_over(self):
        if self.game_over:
            font = pygame.font.SysFont("arial", 48)

            text = font.render(self.result, True, (200, 0, 0))

            text_rect = text.get_rect(
                center=(WIDTH // 2, HEIGHT // 2)
            )

            self.screen.blit(text, text_rect)
    # ----------------------------
    # Convert Mouse Position
    # ----------------------------

    def get_square_from_mouse(self, pos):

        x, y = pos

        col = x // SQ_SIZE
        row = y // SQ_SIZE

        square = chess.square(col, 7 - row)

        return square

    # ----------------------------
    # Handle Mouse Click
    # ----------------------------

    def handle_click(self, pos):

        square = self.get_square_from_mouse(pos)

        # Select piece
        if self.selected_square is None:

            piece = self.board.piece_at(square)

            if piece and piece.color == self.board.turn:

                self.selected_square = square

                self.valid_moves = [
                    move
                    for move in self.board.legal_moves
                    if move.from_square == square
                ]

        # Attempt move
        else:

            move = chess.Move(
                self.selected_square,
                square
            )

            # Auto queen promotion
            piece = self.board.piece_at(self.selected_square)

            if (
                piece
                and piece.piece_type == chess.PAWN
                and (square // 8 == 0 or square // 8 == 7)
            ):
                move = chess.Move(
                    self.selected_square,
                    square,
                    promotion=chess.QUEEN
                )

            if move in self.board.legal_moves:
                self.board.push(move)

                if self.ai_enabled and not self.board.is_game_over():
                    ai_move = self.ai.choose_move(self.board)
                    if ai_move:
                        self.board.push(ai_move)
                # CHECK GAME STATE AFTER MOVE
                if self.board.is_checkmate():
                    self.game_over = True
                    self.result = "Checkmate!"

                elif self.board.is_stalemate():
                    self.game_over = True
                    self.result = "Stalemate!"

                elif self.board.is_insufficient_material():
                    self.game_over = True
                    self.result = "Draw (Insufficient material)"

                elif self.board.is_fifty_moves():
                    self.game_over = True
                    self.result = "Draw (50-move rule)"

            self.selected_square = None
            self.valid_moves = []

    # ----------------------------
    # Main Loop
    # ----------------------------

    def run(self):

        running = True

        while running:

            self.clock.tick(FPS)

            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:

                    if not self.board.is_game_over():

                        self.handle_click(
                            pygame.mouse.get_pos()
                        )

            self.draw_game_state()

            pygame.display.flip()

        pygame.quit()


# ----------------------------
# Run Game
# ----------------------------

if __name__ == "__main__":

    gui = ChessGUI()
    gui.run()