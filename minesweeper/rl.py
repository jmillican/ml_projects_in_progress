from .play_game import produce_model_predictions, decide_next_move_from_prediction, decide_next_move_with_rng
from .model import load_latest_model, save_model
from .minesweeper import Minesweeper, GameState, CellState, BOARD_SIZE
from tqdm import tqdm
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import Model as TfKerasModel  # type: ignore
from .profile import profile_start, profile_end, get_profile, print_profiles

discount_factor = 0.7  # Discount factor for future rewards

training_data_dir = os.path.join(os.path.dirname(__file__), 'rl_training_data')
if not os.path.exists(training_data_dir):
    os.makedirs(training_data_dir)

def save_rl_training_data(boards, reward_vectors, filename_prefix='rl_training_data', iteration=0):

    filename = f"{filename_prefix}_{iteration}.npz"
    training_data_file = os.path.join(training_data_dir, filename)

    # Save the training data into a single file for Tensorflow training.
    if os.path.exists(training_data_file):
        raise Exception("Training data file was created mid-run.")

    boards_to_save = np.array(boards, dtype=np.float32)
    reward_vectors = np.array(reward_vectors, dtype=np.float32)

    np.savez_compressed(training_data_file, boards=boards_to_save, reward_vectors=reward_vectors)
    print(f"Training data saved to {training_data_file}.")

def load_rl_training_data(filename_prefix='rl_training_data', iteration=0) -> tuple[np.ndarray, np.ndarray]:
    filename = f"{filename_prefix}_{iteration}.npz"
    training_data_file = os.path.join(training_data_dir, filename)

    if not os.path.exists(training_data_file):
        raise Exception("Training data file does not exist.")

    data = np.load(training_data_file)
    boards = data['boards']
    reward_vectors = data['reward_vectors']
    return boards, reward_vectors

def main():
    model = load_latest_model(verbose=True)
    rng = np.random.RandomState(12345)  # Fixed seed for reproducibility

    boards = []
    reward_vectors = []

    for rl_run in range(10):
        print(f"Running RL iteration {rl_run + 1}...")
        for i in tqdm(range(5000)):
            profile_start("RL Game")
            game = Minesweeper(rows=BOARD_SIZE, cols=BOARD_SIZE, mines=10, seed=rng.randint(2**32 - 1))

            prediction = produce_model_predictions(game, model)

            while game.get_game_state() == GameState.PLAYING:
                start_board = game.get_visible_board()
                reward = 0.0
                row, col, state = decide_next_move_from_prediction(game, prediction)

                if state == CellState.REVEALED:
                    game.reveal(row, col)
                else:
                    game.flag(row, col)
                
                if game.get_game_state() == GameState.LOST:
                    reward = -10.0
                elif game.get_game_state() == GameState.WON:
                    reward = 10.0
                else:
                    if state == CellState.REVEALED:
                        # Reward for revealing a cell
                        reward = 0.1
                    elif state == CellState.FLAGGED:
                        # Reward for flagging a mine
                        reward = 1.0
                    else:
                        raise Exception(f"Invalid state: {state}")

                    visible_board = game.get_visible_board()

                    prediction = produce_model_predictions(game, model)

                    max_next_q = None
                    for r in range(BOARD_SIZE):
                        for c in range(BOARD_SIZE):
                            if visible_board[r, c] == -1:
                                if max_next_q is None or prediction[r, c, 0] > max_next_q:
                                    max_next_q = prediction[r, c, 0]
                                if prediction[r, c, 1] > max_next_q:
                                    max_next_q = prediction[r, c, 1]
                    if max_next_q is None:
                        raise Exception("No valid actions found in the model's predictions.")
                    # Discounted future reward
                    reward += max_next_q * discount_factor
                boards.append(start_board)
                reward_vector = np.zeros((BOARD_SIZE, BOARD_SIZE, 2), dtype=np.float32)
                if state == CellState.REVEALED:
                    reward_vector[row, col, 0] = reward  # Reward for revealing a cell
                elif state == CellState.FLAGGED:
                    reward_vector[row, col, 1] = reward  # Reward for flagging a mine
                reward_vectors.append(reward_vector)
            profile_end("RL Game")

        print_profiles()
        print(f"Collected {len(boards)} training examples.")

        save_rl_training_data(boards, reward_vectors, filename_prefix='rl_training_data', iteration=rl_run)
        # boards, reward_vectors = load_rl_training_data(filename_prefix='rl_training_data', iteration=rl_run)

        # Add early stopping to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(  # type: ignore
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        )
        
        # Add ReduceLROnPlateau to reduce learning rate when loss plateaus
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(   # type: ignore
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        )

        model.fit(
            np.array(boards).reshape(-1, 9, 9, 1),
            np.array(reward_vectors).reshape(-1, 9, 9, 2),
            epochs=5,
            callbacks=[early_stopping, reduce_lr],
            verbose=1)
        
        save_model(model, f"rl_conv_model_iteration_{rl_run + 1}")

if __name__ == "__main__":
    main()