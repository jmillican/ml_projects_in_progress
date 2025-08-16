import os
import numpy as np
from .minesweeper import GameState, Minesweeper, CellState, BOARD_SIZE
from .model import load_latest_model
from .print_board import print_board
import tensorflow as tf
from tensorflow.keras import Model as TfKerasModel # type: ignore
from tqdm import tqdm
from .profile import profile_start, profile_end, print_profiles
from .basic_config import suppress_tensorflow_logging, force_tensorflow_cpu

suppress_tensorflow_logging()
force_tensorflow_cpu()


models_dir = os.path.join(os.path.dirname(__file__), 'models')

def produce_model_predictions(game: Minesweeper, model: TfKerasModel) -> np.ndarray:
    profile_start("Predict")
    profile_start("PredictGetVisible")
    visible_board = game.get_visible_board()
    profile_end("PredictGetVisible")

    reshaped_input = visible_board.reshape(1, BOARD_SIZE, BOARD_SIZE, 1)
    # Use model() instead of model.predict() for single predictions - much faster!
    actions = model(reshaped_input, training=False).numpy()
    reshaped = actions.reshape(BOARD_SIZE, BOARD_SIZE, 2)
    profile_end("Predict")
    return reshaped

def decide_next_move_from_prediction(game: Minesweeper, actions: np.ndarray) -> tuple[int, int, CellState]:
    profile_start("DecideNextMoveFromPrediction")
    visible_board = game.get_visible_board()
    valid_moves = []
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if visible_board[row, col] == -1:  # Cell is hidden
                valid_moves.append((actions[row, col, 0], (row, col, CellState.REVEALED)))
                valid_moves.append((actions[row, col, 1], (row, col, CellState.FLAGGED)))

    valid_moves.sort(reverse=True, key=lambda x: x[0])  # Sort by action value
    profile_end("DecideNextMoveFromPrediction")
    return valid_moves[0][1]   # Return the top-ranked next move

def decide_next_move_with_model(game: Minesweeper, model) -> tuple[int, int, CellState]:
    """
    Use the model to decide the next move based on the current game state.
    
    Args:
        game: The current Minesweeper game instance.
        model: The trained model to predict the next move.
    
    Returns:
        A tuple containing the row, column, and cell state (REVEALED or FLAGGED).
    """
    actions = produce_model_predictions(game, model)
    return decide_next_move_from_prediction(game, actions)

def decide_next_move_with_rng(game: Minesweeper, rng: np.random.RandomState) -> tuple[int, int, CellState]:
    """
    Decide the next move using a random number generator.
    
    Args:
        game: The current Minesweeper game instance.
        rng: A random number generator instance.
    
    Returns:
        A tuple containing the row, column, and cell state (REVEALED or FLAGGED).
    """
    visible_board = game.get_visible_board()
    row = rng.randint(0, BOARD_SIZE)
    col = rng.randint(0, BOARD_SIZE)

    while visible_board[row, col] != -1:  # Find a hidden cell
        row = rng.randint(0, BOARD_SIZE)
        col = rng.randint(0, BOARD_SIZE)

    # Figure out the proportion of remaining cells that are mines.
    remaining_cells = np.sum(visible_board == -1)
    remaining_mines = game.get_remaining_mines()
    mine_proportion = remaining_mines / remaining_cells

    if remaining_cells == BOARD_SIZE * BOARD_SIZE:
        # If all cells are hidden, any will work.
        state = CellState.REVEALED
    else:
        state = CellState.REVEALED if rng.rand() < mine_proportion else CellState.FLAGGED

    return row, col, state

def play_game_with_model(game_seed: int, model: TfKerasModel) -> tuple[int, GameState]:
    """
    Play a Minesweeper game using the provided model to decide moves.
    
    Args:
        game: The current Minesweeper game instance.
        model: The trained model to predict the next move.
    
    Returns:
        A tuple containing the number of moves made and the final game state.
    """
    game = Minesweeper(rows=BOARD_SIZE, cols=BOARD_SIZE, mines=10, seed=game_seed)
    i = 0
    while game.get_game_state() == GameState.PLAYING:
        row, col, state = decide_next_move_with_model(game, model)

        if state == CellState.REVEALED:
            game.reveal(row, col)
        else:
            game.flag(row, col)

        i += 1

    return i, game.get_game_state()

def main():
    r = np.random.RandomState(2 ** 31 - 1)

    # Load the latest model chronologically from the models directory
    model1 = load_latest_model()
    # Load the second-latest model chronologically from the models directory
    model2 = load_latest_model(offset=1)

    model1_prevails = 0
    model2_prevails = 0
    results = {
        'model1': {'wins': 0, 'losses': 0},
        'model2': {'wins': 0, 'losses': 0},
    }
    draw = 0
    for i in tqdm(range(5000)):
        game_seed = r.randint(2 ** 32 - i)

        model1_result = play_game_with_model(game_seed, model1)
        model2_result = play_game_with_model(game_seed, model2)

        model1_prevailed = (model1_result[1] == GameState.WON and model2_result[1] == GameState.LOST) or \
                    (model1_result[1] == GameState.WON and model2_result[1] == GameState.WON and model1_result[0] < model2_result[0]) or \
                    (model1_result[1] == GameState.LOST and model2_result[1] == GameState.LOST and model1_result[0] > model2_result[0])
        model2_prevailed = (model2_result[1] == GameState.WON and model1_result[1] == GameState.LOST) or \
                (model2_result[1] == GameState.WON and model1_result[1] == GameState.WON and model2_result[0] < model1_result[0]) or \
                (model2_result[1] == GameState.LOST and model1_result[1] == GameState.LOST and model2_result[0] > model1_result[0])
        
        if model1_result[1] == GameState.WON:
            results['model1']['wins'] += 1
        else:
            results['model1']['losses'] += 1
        if model2_result[1] == GameState.WON:
            results['model2']['wins'] += 1
        else:
            results['model2']['losses'] += 1

        if model1_prevailed:
            model1_prevails += 1
        elif model2_prevailed:
            model2_prevails += 1
        else:
            draw += 1

    print(f"Model 1 is better: {model1_prevails}, Model 2 better: {model2_prevails}, Draws: {draw}")
    print(results)

if __name__ == "__main__":
    main()