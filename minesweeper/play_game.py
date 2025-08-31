import os
import numpy as np
from .minesweeper import GameState, Minesweeper, CellState, BOARD_ROWS, BOARD_COLS, BOARD_MINES, INPUT_CHANNELS
from .model import load_latest_model, load_model
from .print_board import print_board
import torch
import torch.nn as nn
from tqdm import tqdm
from .profile import profile_start, profile_end, print_profiles
# Device configuration for Apple Silicon
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


models_dir = os.path.join(os.path.dirname(__file__), 'models')

def produce_model_predictions(game: Minesweeper, model: nn.Module) -> np.ndarray:
    input_board = game.get_input_board()
    
    # Convert to PyTorch tensor and change from NHWC to NCHW format
    reshaped_input = input_board.reshape(1, BOARD_ROWS, BOARD_COLS, INPUT_CHANNELS)
    tensor_input = torch.from_numpy(reshaped_input).float().permute(0, 3, 1, 2).to(device)
    
    with torch.no_grad():
        actions = model(tensor_input)
        # Convert back from NCHW to NHWC format
        actions = actions.permute(0, 2, 3, 1).cpu().numpy()

    reshaped = actions.reshape(BOARD_ROWS, BOARD_COLS, 2)
    return reshaped

def produce_model_predictions_batch(games: list[Minesweeper], model: nn.Module) -> np.ndarray:
    model_inputs = [game.get_input_board().reshape(1, BOARD_ROWS, BOARD_COLS, INPUT_CHANNELS) for game in games]

    # Stack inputs and convert to PyTorch tensor in NCHW format
    stacked_inputs = np.vstack(model_inputs)
    tensor_inputs = torch.from_numpy(stacked_inputs).float().permute(0, 3, 1, 2).to(device)
    
    with torch.no_grad():
        actions = model(tensor_inputs)
        # Convert back from NCHW to NHWC format
        actions = actions.permute(0, 2, 3, 1).cpu().numpy()

    reshaped = actions.reshape(len(games), BOARD_ROWS, BOARD_COLS, 2)
    return reshaped

def decide_next_move_from_prediction(game: Minesweeper, actions: np.ndarray) -> tuple[int, int, CellState]:

    valid_moves_mask = game.valid_moves_mask
    masked_actions = actions * valid_moves_mask

    # It's possible that all moves have a negative value, in which case the max will now be a masked value,
    # because the mask is 0. Let's force all masked actions to be negative infinity.
    masked_actions[valid_moves_mask == 0] = -np.inf

    # # Get the indices of the maximum action value
    max_indices = np.unravel_index(np.argmax(masked_actions, axis=None), masked_actions.shape)
    mask_implementation_row, mask_implementation_col, mask_implementation_type_index = max_indices
    mask_implementation_type = CellState.REVEALED if mask_implementation_type_index == 0 else CellState.FLAGGED
    result = (int(mask_implementation_row), int(mask_implementation_col), mask_implementation_type)

    return result

def decide_next_move_with_model(game: Minesweeper, model: nn.Module) -> tuple[int, int, CellState]:
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
    row = rng.randint(0, BOARD_ROWS)
    col = rng.randint(0, BOARD_COLS)

    while visible_board[row, col] != -1:  # Find a hidden cell
        row = rng.randint(0, BOARD_ROWS)
        col = rng.randint(0, BOARD_COLS)

    # Figure out the proportion of remaining cells that are mines.
    remaining_cells = np.sum(visible_board == -1)
    remaining_mines = game.get_remaining_mines()
    mine_proportion = remaining_mines / remaining_cells

    if remaining_cells == BOARD_ROWS * BOARD_COLS:
        # If all cells are hidden, any will work.
        state = CellState.REVEALED
    else:
        state = CellState.REVEALED if rng.rand() < mine_proportion else CellState.FLAGGED

    return row, col, state

def play_games_with_model(game_seeds: list[int], model: nn.Module) -> list[tuple[int, GameState]]:
    """
    Play a Minesweeper game using the provided model to decide moves.
    
    Args:
        game: The current Minesweeper game instance.
        model: The trained model to predict the next move.
    
    Returns:
        A tuple containing the number of moves made and the final game state.
    """
    games = [
        Minesweeper(rows=BOARD_ROWS, cols=BOARD_COLS, mines=BOARD_MINES, seed=game_seed)
        for game_seed in game_seeds
    ]
    turn = 1
    num_running_games = len(games)

    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'

    while num_running_games > 0:
        print(f" - Turn {turn}, Games left: {num_running_games}")
        predictions = produce_model_predictions_batch(games, model)

        for i, game in enumerate(games):
            if game.get_game_state() == GameState.PLAYING:
                row, col, state = decide_next_move_from_prediction(game, predictions[i])

                if state == CellState.REVEALED:
                    game.reveal(row, col)
                else:
                    game.flag(row, col)

                if game.get_game_state() != GameState.PLAYING:
                    num_running_games -= 1
        turn += 1
        print(LINE_UP, end=LINE_CLEAR)

    print(f" - Turn {turn}, Games left: {num_running_games}")

    return [(game.num_moves, game.get_game_state()) for game in games]

def main():
    offsets_to_load = 1

    for offset in sorted(range(offsets_to_load), reverse=True):
        r = np.random.RandomState(2 ** 31 - 1)
        # Load the latest model chronologically from the models directory
        model, model_name = load_latest_model(offset=offset)
        model = model.to(device)
        model.eval()  # Set to evaluation mode

        # # Show model architecture and parameter count
        # print("\nModel Summary:")
        # model.summary()

        results = {
            'model1': {'wins': 0, 'losses': 0, 'total_moves_in_winning_games': 0, 'total_moves_in_losing_games': 0},
        }

        game_seeds = [r.randint(2**32 - 1) for _ in range(5000)]
        print(f"\n\nPlaying 5000 games with the model...")
        print("Playing model 1...")
        model_results = play_games_with_model(game_seeds, model)

        for i in range(len(game_seeds)):
            model_result = model_results[i]
            
            if model_result[1] == GameState.WON:
                results['model1']['wins'] += 1
                results['model1']['total_moves_in_winning_games'] += model_result[0]
            else:
                results['model1']['losses'] += 1
                results['model1']['total_moves_in_losing_games'] += model_result[0]

        avg_moves_to_win = results['model1']['total_moves_in_winning_games'] / results['model1']['wins'] if results['model1']['wins'] > 0 else 0.0
        avg_moves_to_lose = results['model1']['total_moves_in_losing_games'] / results['model1']['losses'] if results['model1']['losses'] > 0 else 0.0
        model1_results = {
            'wins': results['model1']['wins'],
            'losses': results['model1']['losses'],
            'avg_moves_to_win': f"{avg_moves_to_win:.2f}",
            'avg_moves_to_lose': f"{avg_moves_to_lose:.2f}",
        }
        print(f"{model_name}: {model1_results}")

if __name__ == "__main__":
    main()