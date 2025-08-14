from .minesweeper import Minesweeper, CellState, GameState, BOARD_SIZE
import os
import numpy as np
from tqdm import tqdm

NUM_MINES = 10

training_data_dir = os.path.join(os.path.dirname(__file__), 'training_data')
if not os.path.exists(training_data_dir):
    os.makedirs(training_data_dir)

training_data_file = os.path.join(training_data_dir, 'training_data.npz')

def get_output_vector(visible_board: np.ndarray, mine_board: np.ndarray) -> np.ndarray:
    ## Create a vector covering all possible actions. This means:
    ##  - rows * cols grid for revealing cells.
    ##  - rows * cols grid for flagging cells.
    ## In theory we could probably do this with a rows * cols grid for all moves, and
    ## a single value for whether or not to flag a cell; but this seems likely to be
    ## harder to train.
    output_vector = np.zeros((BOARD_SIZE, BOARD_SIZE, 2), dtype=np.float32)
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if visible_board[row, col] == -1:
                if mine_board[row, col]:
                    output_vector[row, col, 0] = -100.0  # Revealing a mine is a losing move.
                    output_vector[row, col, 1] = 100.0   # Flagging a mine is a great move.
                else:
                    output_vector[row, col, 0] = 10.0   # Revealing a safe cell is a slightly good move.
                    output_vector[row, col, 1] = -100.0  # Flagging a safe cell is a losing move.
    return output_vector

def get_move(rng: np.random.RandomState, visible_board: np.ndarray, mine_board: np.ndarray) -> tuple[tuple[int, int], CellState]:
    row = rng.randint(0, BOARD_SIZE)
    col = rng.randint(0, BOARD_SIZE)

    while visible_board[row, col] != -1:
        row = rng.randint(0, BOARD_SIZE)
        col = rng.randint(0, BOARD_SIZE)

    if mine_board[row, col]:
        return (row, col), CellState.FLAGGED
    else:
        return (row, col), CellState.REVEALED

# Check if the file exists initially.
if os.path.exists(training_data_file):
    raise Exception("Training data file already exists. Please delete it before running this script again.")

boards_moves = []

for i in tqdm(range(10 ** 5)):
    # Create a PRNG, from which we can generate
    # random seeds for the various PRNGs used in the game.
    r = np.random.RandomState(i)
    game_seed = r.randint(0, 2**32 - 1)
    moves_seed = r.randint(0, 2**32 - 1)
    game = Minesweeper(rows=BOARD_SIZE, cols=BOARD_SIZE, mines=NUM_MINES, seed=game_seed)

    r = np.random.RandomState(moves_seed)

    while game.get_game_state() == GameState.PLAYING:
        visible_board = game.get_visible_board()
        safe_move_vector = get_output_vector(visible_board, game.mine_board)
        ((row, col), move) = get_move(r, visible_board, game.mine_board)
        boards_moves.append((visible_board, safe_move_vector))
        if move == CellState.REVEALED:
            game.reveal(row, col)
        elif move == CellState.FLAGGED:
            game.flag(row, col)
        else:
            raise ValueError(f"Invalid move: {move}")

# Save the training data into a single file for Tensorflow training.
if os.path.exists(training_data_file):
    raise Exception("Training data file was created mid-run.")

boards_to_save = np.array([bm[0] for bm in boards_moves], dtype=np.float32)
safe_moves_to_save = np.array([bm[1] for bm in boards_moves], dtype=np.float32)

np.savez_compressed(training_data_file, boards=boards_to_save, safe_moves=safe_moves_to_save)
print(f"Training data saved to {training_data_file}.")