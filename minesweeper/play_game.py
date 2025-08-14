import os
from .minesweeper import GameState, Minesweeper, CellState, BOARD_SIZE
from .model import load_model
import numpy as np
from .print_board import print_board

models_dir = os.path.join(os.path.dirname(__file__), 'models')

def main():
    r = np.random.RandomState(2 ** 31 - 7)
    game = Minesweeper(rows=BOARD_SIZE, cols=BOARD_SIZE, mines=10, seed=r.randint(2 ** 32 - 1))

    # Load the latest model chronologically from the models directory
    print("Loading model...")
    try:
        # List all model files in the models directory
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
        if not model_files:
            raise FileNotFoundError("No model files found in 'models' directory.")
        # Sort files by modification time
        model_files.sort(key=lambda f: os.path.getmtime(os.path.join(models_dir, f)))
        latest_model_file = model_files[-1]  # Get the most recent model file
        model_name = os.path.splitext(latest_model_file)[0]  # Remove the .h5 extension
        print(f"Loading model: {model_name}")
        # Load the model
        model = load_model(model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    i = 0
    while game.game_state == GameState.PLAYING:
        visible_board = game.get_visible_board()
        actions = model.predict(visible_board.flatten().reshape(1, -1))
        actions = actions.reshape(BOARD_SIZE, BOARD_SIZE, 2)
        valid_moves = []
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if visible_board[row, col] == -1:  # Cell is hidden
                    valid_moves.append((actions[row, col, 0], (row, col, CellState.REVEALED)))
                    valid_moves.append((actions[row, col, 1], (row, col, CellState.FLAGGED)))
        valid_moves.sort(reverse=True, key=lambda x: x[0])  # Sort by action value
        row, col, state = valid_moves[0][1]
        print(f"\n\nMove {i}:")

        if state == CellState.REVEALED:
            print(f"Revealing cell at ({row}, {col})")
            game.reveal(row, col)
        else:
            print(f"Flagging cell at ({row}, {col})")
            game.flag(row, col)

        print_board(game)
        i += 1

if __name__ == "__main__":
    main()