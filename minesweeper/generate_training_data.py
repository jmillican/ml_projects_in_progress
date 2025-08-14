from .minesweeper import Minesweeper, GameState
import os
import numpy as np

# # Open the training_data directory, and list its contents. Use a relative path to the current script
# # file

# training_data_dir = os.path.join(os.path.dirname(__file__), 'training_data')

# # List the contents of the training_data directory
# training_data_files = os.listdir(training_data_dir)

# print("Contents of training_data directory:")
# for filename in training_data_files:
#     print(f"- {filename}")

BOARD_SIZE = 9
NUM_MINES = 10

def get_random_move_and_safe_move_vector(visible_board: np.ndarray, mine_board: np.ndarray) -> tuple[tuple[int, int], np.ndarray]:
    mRow, mCol = r.randint(BOARD_SIZE), r.randint(BOARD_SIZE)
    while visible_board[mRow, mCol] != -1 or mine_board[mRow, mCol]:  # Ensure we select a hidden non-mine cell
        mRow, mCol = r.randint(BOARD_SIZE), r.randint(BOARD_SIZE)
    move = (mRow, mCol)
    safe_moves = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    safe_moves[visible_board == -1] = 1.0  # Mark hidden cells
    safe_moves[mine_board] = 0.0  # Mark non-mine cells as not possible
    return move, safe_moves

training_data_dir = os.path.join(os.path.dirname(__file__), 'training_data')
if not os.path.exists(training_data_dir):
    os.makedirs(training_data_dir)

for i in range(100000):
    # Create a PRNG, from which we can generate
    # random seeds for the various PRNGs used in the game.
    r = np.random.RandomState(i)
    game_seed = r.randint(0, 2**32 - 1)
    moves_seed = r.randint(0, 2**32 - 1)
    game = Minesweeper(rows=BOARD_SIZE, cols=BOARD_SIZE, mines=NUM_MINES, seed=game_seed)

    r = np.random.RandomState(moves_seed)

    boards_moves = []

    while game.get_game_state() == GameState.PLAYING:
        visible_board = game.get_visible_board()
        move, safe_move_vector = get_random_move_and_safe_move_vector(visible_board, game.mine_board)
        boards_moves.append((visible_board, safe_move_vector))
        game.reveal(*move)
    
    # Save the training data to a series of files - each containing one board and its safe moves vector
    # use the training_data directory, and index it relative to this script file. Each game should be
    # within its own subdirectory, and each move should be in its own file.
    game_subdir = os.path.join(training_data_dir, f'game_{i}')
    if not os.path.exists(game_subdir):
        os.makedirs(game_subdir)
    for j, (board, safe_moves) in enumerate(boards_moves):
        np.savez_compressed(f"{game_subdir}/move_{j}.npz", board=board, safe_moves=safe_moves)