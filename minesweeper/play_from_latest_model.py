import numpy as np
from .model import load_model
from .minesweeper import Minesweeper, GameState
from .print_board import print_board
import os

def main():
    # Load the latest model from the models directory
    # First list the models directory to find all the available models
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    if not os.path.exists(models_dir):
        print("No models directory found. Please train a model first.")
        return
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
    if not model_files:
        print("No models found in the models directory. Please train a model first.")
        return
    # Sort the model files by modification time to get the latest one
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)))
    latest_model_file = model_files[-1]
    print(f"Loading the latest model: {latest_model_file}")
    model_name = os.path.splitext(latest_model_file)[0]  # Remove the .h5 extension
    # Load the model
    try:
        model = load_model(model_name)
        print("Model loaded successfully.")
    except FileNotFoundError as e:
        print(e)
        return

    game_seed = 999999998  # Example seed. Much larger than used for generating the game data.
    r = np.random.RandomState(game_seed)
    game_seed = r.randint(0, 2**32 - 1)
    game = Minesweeper(rows=9, cols=9, mines=10, seed=game_seed)

    move = 0
    while game.get_game_state() == GameState.PLAYING:
        move += 1
        print("Current board:")
        print_board(game)
        # Infer the next best move using the model
        visible_board = game.get_visible_board()
        action = model.predict(visible_board.reshape(1, -1))  # Reshape to match model input
        reshaped_action = action.reshape(9, 9)

        if move == 2:
            reshaped_action = reshaped_action * 1000.0
            reshaped_action = reshaped_action.astype(np.int32)
            print("Predicted action values for each cell:")
            print(reshaped_action)
            return

        # Find the un-revealed cell with the highest predicted value
        move_scores = []
        for row in range(game.rows):
            for col in range(game.cols):
                if visible_board[row, col] == -1:  # Only consider hidden cells
                    move_scores.append((reshaped_action[row, col], row, col))
        move_scores.sort(reverse=True, key=lambda x: x[0])

        predicted_row, predicted_col = move_scores[0][1], move_scores[0][2]
        game.reveal(predicted_row, predicted_col)
        print(f"Revealed cell at ({predicted_row}, {predicted_col})")

    print("Ending board:")
    print_board(game)

if __name__ == "__main__":
    main()