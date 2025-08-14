import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
import numpy as np
from .minesweeper import GameState, Minesweeper, CellState, BOARD_SIZE
from .model import load_model
from .print_board import print_board
from tensorflow.keras import Model as TfKerasModel # type: ignore
from tqdm import tqdm
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)  # Suppress warnings


models_dir = os.path.join(os.path.dirname(__file__), 'models')

def decide_next_move_with_model(game: Minesweeper, model) -> tuple[int, int, CellState]:
    """
    Use the model to decide the next move based on the current game state.
    
    Args:
        game: The current Minesweeper game instance.
        model: The trained model to predict the next move.
    
    Returns:
        A tuple containing the row, column, and cell state (REVEALED or FLAGGED).
    """
    visible_board = game.get_visible_board()
    actions = model.predict(visible_board.flatten().reshape(1, -1), verbose=0)
    actions = actions.reshape(BOARD_SIZE, BOARD_SIZE, 2)

    # if True:
    #     # Print a grid of the actions, normalised into a 0-9 range of integer values.
    #     print("\nReveal values (0-9):")
    #     for row in range(BOARD_SIZE):
    #         row_values = []
    #         for col in range(BOARD_SIZE):
    #             if visible_board[row, col] == -1:
    #                 row_values.append(int(actions[row, col, 0] * 5 + 5))
    #             else:
    #                 row_values.append(" ")
    #         print(" ".join(f"{val}" for val in row_values))
    #     print("\nFlag values (0-9):")
    #     for row in range(BOARD_SIZE):
    #         row_values = []
    #         for col in range(BOARD_SIZE):
    #             if visible_board[row, col] == -1:
    #                 row_values.append(int(actions[row, col, 1] * 5 + 5))
    #             else:
    #                 row_values.append(" ")
    #         print(" ".join(f"{val}" for val in row_values))

    valid_moves = []
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if visible_board[row, col] == -1:  # Cell is hidden
                valid_moves.append((actions[row, col, 0], (row, col, CellState.REVEALED)))
                valid_moves.append((actions[row, col, 1], (row, col, CellState.FLAGGED)))

    valid_moves.sort(reverse=True, key=lambda x: x[0])  # Sort by action value
    return valid_moves[0][1]   # Return the top-ranked next move

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

def does_model_beat_random(game_seed: int, model: TfKerasModel) -> tuple[bool, bool]:
    # Use fixed seeds for reproducibility
    game_start_seed = game_seed
    r = np.random.RandomState(game_seed + 1)
    
    game = Minesweeper(rows=BOARD_SIZE, cols=BOARD_SIZE, mines=10, seed=game_start_seed)
    i = 0
    while game.game_state == GameState.PLAYING:
        row, col, state = decide_next_move_with_model(game, model)

        if state == CellState.REVEALED:
            game.reveal(row, col)
        else:
            game.flag(row, col)

        # print_board(game)
        i += 1
    model_result = (i, game.get_game_state())

    game = Minesweeper(rows=BOARD_SIZE, cols=BOARD_SIZE, mines=10, seed=game_start_seed)
    i = 0
    while game.game_state == GameState.PLAYING:
        row, col, state = decide_next_move_with_rng(game, r)

        if state == CellState.REVEALED:
            game.reveal(row, col)
        else:
            game.flag(row, col)

        # print_board(game)
        i += 1
    rng_result = (i, game.get_game_state())

    # Model wins if it wins the game faster than the RNG, or if it takes longer to lose.
    model_won = (model_result[1] == GameState.WON and rng_result[1] == GameState.LOST) or \
                (model_result[1] == GameState.WON and rng_result[1] == GameState.WON and model_result[0] < rng_result[0]) or \
                (model_result[1] == GameState.LOST and rng_result[1] == GameState.LOST and model_result[0] > rng_result[0])
    rng_won = (rng_result[1] == GameState.WON and model_result[1] == GameState.LOST) or \
               (rng_result[1] == GameState.WON and model_result[1] == GameState.WON and rng_result[0] < model_result[0]) or \
               (rng_result[1] == GameState.LOST and model_result[1] == GameState.LOST and rng_result[0] > model_result[0])

    if model_won and rng_won:
        raise ValueError("Both model and RNG won, which should not happen.")

    return (model_won, rng_won)


def main():
    r = np.random.RandomState(2 ** 31 - 1)
    # game = Minesweeper(rows=BOARD_SIZE, cols=BOARD_SIZE, mines=10, seed=r.randint(2 ** 32 - 1))

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

    model_wins = 0
    rng_wins = 0
    draw = 0
    for i in tqdm(range(1000)):
        game_seed = r.randint(2 ** 32 - i)
        model_win, rng_win = does_model_beat_random(game_seed, model)
        if model_win:
            model_wins += 1
        elif rng_win:
            rng_wins += 1
        else:
            draw += 1

    print(f"Model wins: {model_wins}, RNG wins: {rng_wins}, Draws: {draw}")


    # i = 0
    # while game.game_state == GameState.PLAYING:
    #     print(f"\n\nMove {i}:")
            
    #     row, col, state = decide_next_move_with_model(game, model)

    #     if state == CellState.REVEALED:
    #         print(f"Revealing cell at ({row}, {col})")
    #         game.reveal(row, col)
    #     else:
    #         print(f"Flagging cell at ({row}, {col})")
    #         game.flag(row, col)

    #     print_board(game)
    #     i += 1

if __name__ == "__main__":
    main()