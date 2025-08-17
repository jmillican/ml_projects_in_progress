from .play_game import decide_next_move_from_prediction, decide_next_move_with_rng, produce_model_predictions_batch
from .model import load_latest_model, save_model, create_model
from .minesweeper import Minesweeper, GameState, CellState, BOARD_SIZE
from tqdm import tqdm
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import Model as TfKerasModel  # type: ignore
import tensorflow.keras # type: ignore
from .profile import profile_start, profile_end, get_profile, print_profiles
from datetime import datetime
from .print_board import print_board

discount_factor = 0.92  # Discount factor for future rewards
RANDOM_PROBABILITY = 0.03  # Probability of making a random move instead of the model's prediction

def save_rl_training_data(boards, target_vectors, filename_prefix='rl_training_data', iteration=0):
    training_data_dir = os.path.join(os.path.dirname(__file__), 'rl_training_data')
    if not os.path.exists(training_data_dir):
        os.makedirs(training_data_dir)

    filename = f"{filename_prefix}_{iteration}.npz"
    training_data_file = os.path.join(training_data_dir, filename)

    # Save the training data into a single file for Tensorflow training.
    if os.path.exists(training_data_file):
        raise Exception("Training data file was created mid-run.")

    boards_to_save = np.array(boards, dtype=np.float32)
    target_vectors = np.array(target_vectors, dtype=np.float32)

    np.savez_compressed(training_data_file, boards=boards_to_save, target_vectors=target_vectors)
    print(f"Training data saved to {training_data_file}.")

# def load_rl_training_data(filename_prefix='rl_training_data', iteration=0) -> tuple[np.ndarray, np.ndarray]:
#     filename = f"{filename_prefix}_{iteration}.npz"
#     training_data_file = os.path.join(training_data_dir, filename)

#     if not os.path.exists(training_data_file):
#         raise Exception("Training data file does not exist.")

#     data = np.load(training_data_file)
#     boards = data['boards']
#     reward_vectors = data['reward_vectors']
#     return boards, reward_vectors

NUM_IN_RUN = 4500
BATCH_SIZE = 750
TARGET_SAMPLES_PER_ITERATION = 20000

def main():
    # model = load_latest_model(verbose=True)
    
    model = create_model(
        input_shape=(9, 9, 3,),
        output_shape=(9*9*2,))

    rng = np.random.RandomState(123456)  # Fixed seed for reproducibility


    if NUM_IN_RUN % BATCH_SIZE != 0:
        raise ValueError("NUM_IN_RUN must be divisible by BATCH_SIZE for this setup.")
    iterations = NUM_IN_RUN // BATCH_SIZE
    
    for rl_run in range(10000):
        boards = []
        reward_vectors = []
        target_vectors = []
        # Save the model after every 30 iterations
        if (rl_run < 300 and rl_run % 30 == 0) or (rl_run >= 300 and rl_run < 600 and rl_run % 50 == 0) or (rl_run >= 600 and rl_run % 150 == 0):
            print(f"Saving model after iteration {rl_run}...")

            model_name = "rl_model_{}_iteration_{}".format(datetime.now().strftime('%y-%m-%d_%H-%M'), rl_run)

            save_model(model, model_name)

        print(f"\nRunning RL iteration {rl_run + 1}...")
        for batch_num in tqdm(range(iterations)):
            profile_start("RL Game")

            profile_start("RL Game: Create Games")
            games = [
                Minesweeper(rows=BOARD_SIZE, cols=BOARD_SIZE, mines=10, seed=rng.randint(2**32 - 1))
                for _ in range(BATCH_SIZE)
            ]
            profile_end("RL Game: Create Games")

            predictions = produce_model_predictions_batch(games, model)

            move = 0
            while games:
                move += 1
                rewards = [0.0] * len(games)
                start_input_boards = []
                rows = [0] * len(games)
                cols = [0] * len(games)
                states = [CellState.HIDDEN] * len(games)

                for i, game in enumerate(games):
                    start_input_boards.append(game.get_input_board())
                    if rng.rand() < RANDOM_PROBABILITY:
                        # Make a random move
                        row, col, state = decide_next_move_with_rng(game, rng)
                    else:
                        # Use the model's prediction
                        row, col, state = decide_next_move_from_prediction(game, predictions[i])
                    rows[i] = row
                    cols[i] = col
                    states[i] = state

                    profile_start("RL Game: Make Move")
                    if state == CellState.REVEALED:
                        game.reveal(row, col)
                    else:
                        game.flag(row, col)
                    profile_end("RL Game: Make Move")

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

                predictions = produce_model_predictions_batch(games, model)

                for i, game in enumerate(games):
                    target = rewards[i]
                    if game.get_game_state() == GameState.PLAYING:
                        profile_start("Calculate Next Q")

                        # Calculate the maximum Q value for the next state
                        # This is the max Q value for the next state, which is used to calculate the
                        # target for the current state.

                        valid_moves_mask = game.valid_moves_mask
                        masked_predictions = predictions[i] * valid_moves_mask
                        # It's possible that all moves have a negative value, in which case the max will now be a masked value,
                        # because the mask is 0. Let's force all masked actions to be negative infinity.
                        masked_predictions[valid_moves_mask == 0] = -np.inf
                        max_next_q = np.max(masked_predictions)
                        if max_next_q == -np.inf:
                            raise Exception("No valid actions found in the model's predictions.")

                        target += max_next_q * discount_factor
                        profile_end("Calculate Next Q")


                    boards.append(start_input_boards[i])
                    target_vector = np.zeros((BOARD_SIZE, BOARD_SIZE, 2), dtype=np.float32)
                    if states[i] == CellState.REVEALED:
                        target_vector[rows[i], cols[i], 0] = target  # Reward for revealing a cell
                    elif states[i] == CellState.FLAGGED:
                        target_vector[rows[i], cols[i], 1] = target  # Reward for flagging a mine
                    else:
                        raise Exception(f"Invalid state: {states[i]}")
                    target_vectors.append(target_vector)
                games = [game for game in games if game.get_game_state() == GameState.PLAYING]
            profile_end("RL Game")

            if len(boards) >= TARGET_SAMPLES_PER_ITERATION:
                break

        print(f"Collected {len(boards)} training examples.")

        # save_rl_training_data(boards, reward_vectors, filename_prefix='rl_training_data', iteration=rl_run)
        # boards, reward_vectors = load_rl_training_data(filename_prefix='rl_training_data', iteration=rl_run)

        # Update learning rate for RL (lower than pre-training)
        tensorflow.keras.backend.set_value(
            model.optimizer.learning_rate,
            0.000001  # 1e-6, adjust as needed
        )
        
        profile_start("Model Fit")
        model.fit(
            np.array(boards).reshape(-1, 9, 9, 3),
            np.array(target_vectors).reshape(-1, 9, 9, 2),
            epochs=1,
            verbose=1)
        profile_end("Model Fit")

        print("\nProfiling stats:\n")
        print_profiles()

if __name__ == "__main__":
    main()