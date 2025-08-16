from .play_game import decide_next_move_from_prediction, decide_next_move_with_rng
from .model import load_latest_model, save_model
from .minesweeper import Minesweeper, GameState, CellState, BOARD_SIZE
from tqdm import tqdm
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import Model as TfKerasModel  # type: ignore
from .profile import profile_start, profile_end, get_profile, print_profiles

discount_factor = 0.9  # Discount factor for future rewards

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

def produce_model_predictions_batch(games: list[Minesweeper], model: TfKerasModel) -> np.ndarray:
    profile_start("PredictBatch")
    model_inputs = [game.get_visible_board().reshape(1, BOARD_SIZE, BOARD_SIZE, 1) for game in games]

    actions = model.predict(np.vstack(model_inputs), verbose=0)
    reshaped = actions.reshape(len(games), BOARD_SIZE, BOARD_SIZE, 2)
    profile_end("PredictBatch")
    return reshaped

NUM_IN_RUN = 50000
BATCH_SIZE = 1000

def main():
    model = load_latest_model(verbose=True)
    rng = np.random.RandomState(123456)  # Fixed seed for reproducibility

    boards = []
    reward_vectors = []

    for rl_run in range(10, 20):
        print(f"Running RL iteration {rl_run + 1}...")

        if NUM_IN_RUN % BATCH_SIZE != 0:
            raise ValueError("NUM_IN_RUN must be divisible by BATCH_SIZE for this setup.")
        iterations = NUM_IN_RUN // BATCH_SIZE

        for _ in tqdm(range(iterations)):
            profile_start("RL Game")

            games = [
                Minesweeper(rows=BOARD_SIZE, cols=BOARD_SIZE, mines=10, seed=rng.randint(2**32 - 1))
                for _ in range(BATCH_SIZE)
            ]

            predictions = produce_model_predictions_batch(games, model)

            move = 0
            while games:
                move += 1
                rewards = [0.0] * len(games)
                start_boards = []
                visible_boards = []
                rows = [0] * len(games)
                cols = [0] * len(games)
                states = [CellState.HIDDEN] * len(games)

                for i, game in enumerate(games):
                    start_boards.append(game.get_visible_board())
                    reward = 0.0
                    row, col, state = decide_next_move_from_prediction(game, predictions[i])
                    rows[i] = row
                    cols[i] = col
                    states[i] = state

                    if state == CellState.REVEALED:
                        game.reveal(row, col)
                    else:
                        game.flag(row, col)
                    
                    if game.get_game_state() == GameState.LOST:
                        rewards[i] = -10.0
                    elif game.get_game_state() == GameState.WON:
                        rewards[i] = 10.0
                    else:
                        if state == CellState.REVEALED:
                            # Reward for revealing a cell
                            rewards[i] = 0.1
                        elif state == CellState.FLAGGED:
                            # Reward for flagging a mine
                            rewards[i] = 1.0
                        else:
                            raise Exception(f"Invalid state: {state}")

                    visible_boards.append(game.get_visible_board())

                predictions = produce_model_predictions_batch(games, model)

                for i, game in enumerate(games):
                    if game.get_game_state() == GameState.PLAYING:
                        max_next_q = None   
                        for r in range(BOARD_SIZE):
                            for c in range(BOARD_SIZE):
                                if visible_boards[i][r, c] == -1:
                                    if max_next_q is None or predictions[i][r, c, 0] > max_next_q:
                                        max_next_q = predictions[i][r, c, 0]
                                    if predictions[i][r, c, 1] > max_next_q:
                                        max_next_q = predictions[i][r, c, 1]
                        if max_next_q is None:
                            raise Exception("No valid actions found in the model's predictions.")
                        # Discounted future reward
                        rewards[i] += max_next_q * discount_factor
                    boards.append(start_boards[i])
                    reward_vector = np.zeros((BOARD_SIZE, BOARD_SIZE, 2), dtype=np.float32)
                    if states[i] == CellState.REVEALED:
                        reward_vector[rows[i], cols[i], 0] = rewards[i]  # Reward for revealing a cell
                    elif states[i] == CellState.FLAGGED:
                        reward_vector[rows[i], cols[i], 1] = rewards[i]  # Reward for flagging a mine
                    else:
                        raise Exception(f"Invalid state: {states[i]}")
                    reward_vectors.append(reward_vector)
                games = [game for game in games if game.get_game_state() == GameState.PLAYING]
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