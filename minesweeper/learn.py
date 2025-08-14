import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense  # type: ignore
from .model import create_model, save_model
from datetime import datetime

NUM_TO_LOAD = 10000

training_data_dir = os.path.join(os.path.dirname(__file__), 'training_data')
models_dir = os.path.join(os.path.dirname(__file__), 'models')

def list_training_data_directory():
    # Check if the directory exists
    if os.path.exists(training_data_dir):
        # List all files in the directory
        files = os.listdir(training_data_dir)
        # Filter to only directories
        files = [f for f in files if os.path.isdir(os.path.join(training_data_dir, f))]
        # Filter to only directories with the correct name prefix
        files = [f for f in files if f.startswith('game_')]
        if not files:
            return "No training data found in the directory."
        # Sort files by their names (which are game indices)
        files.sort(key=lambda x: int(x.split('_')[1]))
        # Return the sorted list of files
        return files
    else:
        return "Training data directory does not exist."

def load_training_game(game_directory_name: str) -> list[tuple[str, np.ndarray, np.ndarray]]:
    game_subdir = os.path.join(training_data_dir, game_directory_name)
    if not os.path.exists(game_subdir):
        raise FileNotFoundError(f"Game directory {game_subdir} does not exist.")
    game_files = os.listdir(game_subdir)

    boards = []
    safe_moves = []
    for game_file in game_files:
        file_path = os.path.join(game_subdir, game_file)
        if not file_path.endswith('.npz'):
            continue
        data = np.load(file_path)
        boards.append(data['board'])
        safe_moves.append(data['safe_moves'])

    return list(zip(game_files, boards, safe_moves))

def main():
    all_games = list_training_data_directory()
    all_boards = []
    all_safe_moves = []
    i = 0

    found_shape = False
    safe_moves_shape_vector = np.zeros((0, 0), dtype=np.float32)
    safe_moves_shape_flattened = np.zeros((0, 0), dtype=np.float32)

    for game in all_games:
        if i >= NUM_TO_LOAD:
            break
        game_data = load_training_game(game)
        for _, board, safe_moves in game_data:
            if not found_shape:
                safe_moves_shape_vector = np.zeros_like(safe_moves)
                safe_moves_shape_flattened = safe_moves_shape_vector.flatten()
                found_shape = True

            all_boards.append(board.flatten())
            all_safe_moves.append(safe_moves.flatten())
        i += 1
        if i % 100 == 0:
            print(f"Loaded {i} games.")
    
    print(f"Loaded {len(all_boards)} boards and {len(all_safe_moves)} safe move vectors from training data.")

    model = create_model(
        input_shape=all_boards[0].shape,
        output_shape=safe_moves_shape_flattened.shape)
    
    # Show model architecture and parameter count
    print("\nModel Summary:")
    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}")

    model.fit(
        np.array(all_boards),
        np.array(all_safe_moves),
        epochs=10,
        verbose=1)
    
    # Save model as minesweeper_model_<yy-mm-dd_hh-mm>.h5
    model_name = f"minesweeper_model_{datetime.now().strftime('%y-%m-%d_%H-%M')}"
    print(f"Saving model as {model_name}.h5")
    save_model(model, model_name)

if __name__ == "__main__":
    main()