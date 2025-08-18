from .play_game import decide_next_move_from_prediction, decide_next_move_with_rng, produce_model_predictions_batch
from .model import load_latest_model, save_model, create_model
from .minesweeper import Minesweeper, GameState, CellState, BOARD_SIZE, INPUT_CHANNELS
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

NUM_IN_RUN = 4500
BATCH_SIZE = 750
TARGET_SAMPLES_PER_ITERATION = 20000

def main():
    # model = load_latest_model(verbose=True)
    
    model = create_model(
        input_shape=(BOARD_SIZE, BOARD_SIZE, INPUT_CHANNELS,),
        output_shape=(BOARD_SIZE, BOARD_SIZE, 2,))
    # Update learning rate for RL (lower than pre-training)
    tensorflow.keras.backend.set_value(
        model.optimizer.learning_rate,
        0.000001  # 1e-6, adjust as needed
    )

    rng = np.random.RandomState(123456)  # Fixed seed for reproducibility


    if NUM_IN_RUN % BATCH_SIZE != 0:
        raise ValueError("NUM_IN_RUN must be divisible by BATCH_SIZE for this setup.")
    iterations = NUM_IN_RUN // BATCH_SIZE
    
    for rl_run in range(10000):
        profile_start("Overall RL Run")

        profile_start("Allocate Boards and Rewards")
        boards = []
        reward_vectors = []
        target_vectors = []
        profile_end("Allocate Boards and Rewards")

        profile_start("Save Model")
        # Save the model after every 30 iterations
        if (rl_run < 300 and rl_run % 30 == 0) or (rl_run >= 300 and rl_run < 600 and rl_run % 50 == 0) or (rl_run >= 600 and rl_run % 150 == 0):
            print(f"Saving model after iteration {rl_run}...")

            model_name = "rl_model_{}_iteration_{}".format(datetime.now().strftime('%y-%m-%d_%H-%M'), rl_run)

            save_model(model, model_name)
        profile_end("Save Model")

        
        print(f"\nRunning RL iteration {rl_run + 1}...")
        for batch_num in tqdm(range(iterations)):
            profile_start("Create Games")
            games = [
                Minesweeper(rows=BOARD_SIZE, cols=BOARD_SIZE, mines=10, seed=rng.randint(2**32 - 1))
                for _ in range(BATCH_SIZE)
            ]
            profile_end("Create Games")

            profile_start("Produce Predictions")
            predictions = produce_model_predictions_batch(games, model)
            profile_end("Produce Predictions")

            move = 0
            while games:
                profile_start("Start Move Manipulations")
                move += 1
                rewards = [0.0] * len(games)
                start_input_boards = []
                rows = [0] * len(games)
                cols = [0] * len(games)
                states = [CellState.HIDDEN] * len(games)
                profile_end("Start Move Manipulations")

                for i, game in enumerate(games):
                    profile_start("Store Input Board")
                    start_input_boards.append(game.get_input_board())
                    profile_end("Store Input Board")
                    profile_start("Generate Random")
                    rand = rng.rand()
                    profile_end("Generate Random")

                    if rand < RANDOM_PROBABILITY:
                        profile_start("Decide Next Move With RNG")
                        # Make a random move
                        row, col, state = decide_next_move_with_rng(game, rng)
                        profile_end("Decide Next Move With RNG")
                    else:
                        profile_start("Decide Next Move With Model")
                        # Use the model's prediction
                        row, col, state = decide_next_move_from_prediction(game, predictions[i])
                        profile_end("Decide Next Move With Model")

                    rows[i] = row
                    cols[i] = col
                    states[i] = state

                    if state == CellState.REVEALED:
                        game.reveal(row, col)
                    else:
                        game.flag(row, col)

                    profile_start("Determine Rewards")
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
                    profile_end("Determine Rewards")

                profile_start("Produce Predictions")
                predictions = produce_model_predictions_batch(games, model)
                profile_end("Produce Predictions")

                for i, game in enumerate(games):
                    target = rewards[i]
                    if game.get_game_state() == GameState.PLAYING:

                        profile_start("Calculate Target Value")
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
                        profile_end("Calculate Target Value")

                    profile_start("Store Target Vector and Input Board")
                    boards.append(start_input_boards[i])
                    target_vector = np.zeros((BOARD_SIZE, BOARD_SIZE, 2), dtype=np.float32)
                    if states[i] == CellState.REVEALED:
                        target_vector[rows[i], cols[i], 0] = target  # Reward for revealing a cell
                    elif states[i] == CellState.FLAGGED:
                        target_vector[rows[i], cols[i], 1] = target  # Reward for flagging a mine
                    else:
                        raise Exception(f"Invalid state: {states[i]}")
                    target_vectors.append(target_vector)
                    profile_end("Store Target Vector and Input Board")
                profile_start("Remove Finished Games")
                games = [game for game in games if game.get_game_state() == GameState.PLAYING]
                profile_end("Remove Finished Games")

            if len(boards) >= TARGET_SAMPLES_PER_ITERATION:
                break

        print(f"Collected {len(boards)} training examples.")
        
        profile_start("Model Fit")
        model.fit(
            np.array(boards).reshape(-1, BOARD_SIZE, BOARD_SIZE, INPUT_CHANNELS),
            np.array(target_vectors).reshape(-1, BOARD_SIZE, BOARD_SIZE, 2),
            epochs=1,
            verbose=1)
        profile_end("Model Fit")

        profile_end("Overall RL Run")
        print("\nProfiling stats:\n")
        print_profiles()

if __name__ == "__main__":
    main()