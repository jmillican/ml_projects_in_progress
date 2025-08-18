from .game import Breakout, produce_model_predictions_batch
from .model import load_latest_model, save_model, create_model
from tqdm import tqdm
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import Model as TfKerasModel  # type: ignore
import tensorflow.keras # type: ignore
from datetime import datetime
from .profile import profile_start, profile_end, print_profiles

discount_factor = 0.96  # Discount factor for future rewards
RANDOM_PROBABILITY = 0.03  # Probability of making a random move instead of the model's prediction

NUM_IN_RUN = 100
BATCH_SIZE = 20
TARGET_SAMPLES_PER_ITERATION = 20000

LOSE_REWARD = -20.0
WIN_REWARD = 20.0
FLAG_MINE_REWARD = 0.0
REVEAL_CELL_REWARD = 0.0
INITIAL_LEARNING_RATE = 0.000001  # 1e-6, adjust as needed
LEARNING_RATE_DECAY = 0.999

MOVE_LIMIT = 1000

def main():
    # model = load_latest_model(verbose=True)
    
    model = create_model(
        input_shape=(210, 160, 6,),
        output_shape=(4,))

    rng = np.random.RandomState(123456)  # Fixed seed for reproducibility

    if NUM_IN_RUN % BATCH_SIZE != 0:
        raise ValueError("NUM_IN_RUN must be divisible by BATCH_SIZE for this setup.")
    iterations = NUM_IN_RUN // BATCH_SIZE

    learning_rate = INITIAL_LEARNING_RATE
    
    for rl_run in range(10000):
        profile_start("Overall RL Run")

        profile_start("Allocate Boards and Rewards")
        boards = []
        reward_vectors = []
        target_vectors = []
        profile_end("Allocate Boards and Rewards")

        profile_start("Save Model")
        # Save the model after every 30 iterations
        if (rl_run < 30 and rl_run % 10 == 0) or (rl_run >= 30 and rl_run < 300 and rl_run % 30 == 0) or (rl_run >= 300 and rl_run < 600 and rl_run % 50 == 0) or (rl_run >= 600 and rl_run % 150 == 0):
            print(f"Saving model after iteration {rl_run}...")

            model_name = "rl_model_{}_iteration_{}".format(datetime.now().strftime('%y-%m-%d_%H-%M'), rl_run)

            save_model(model, model_name)
        profile_end("Save Model")

        
        print(f"\nRunning RL iteration {rl_run + 1}...")
        for batch_num in tqdm(range(iterations)):
            profile_start("Create Games")
            games = [
                Breakout(seed=rng.randint(2**32 - 1))
                for _ in range(BATCH_SIZE)
            ]
            profile_end("Create Games")

            profile_start("Produce Predictions")
            predictions = produce_model_predictions_batch(games, model)
            profile_end("Produce Predictions")

            move = 0
            while games:
                move += 1

                if move >= MOVE_LIMIT:
                    print(f"Stopping after {MOVE_LIMIT} moves")
                    break

                print("Taking move", move, "for batch", batch_num + 1, "of", iterations)
                rewards = [0.0] * len(games)
                start_input_matrixes = []
                actions = [0] * len(games)

                for i, game in enumerate(games):
                    profile_start("Store Input Matrix")
                    start_input_matrixes.append(game.getModelInput())
                    profile_end("Store Input Matrix")
                    profile_start("Generate Random")
                    rand = rng.rand()
                    profile_end("Generate Random")

                    if rand < RANDOM_PROBABILITY:
                        profile_start("Decide Next Move With RNG")
                        # Make a random move
                        action = game.decide_next_move_with_rng(rng)
                        profile_end("Decide Next Move With RNG")
                    else:
                        profile_start("Decide Next Move With Model")
                        # Use the model's prediction
                        action = game.decide_next_move_from_prediction(predictions[i])
                        profile_end("Decide Next Move With Model")
                    actions[i] = action

                    rewards[i] = game.take_action(action)
                    if rewards[i] != 0.0:
                        print(f"Game {i} received a reward of {rewards[i]} for action {action}")

                profile_start("Produce Predictions")
                predictions = produce_model_predictions_batch(games, model)
                profile_end("Produce Predictions")

                for i, game in enumerate(games):
                    target = np.float32(rewards[i])
                    if not game.isGameOver():

                        profile_start("Calculate Target Value")
                        max_next_q = np.max(predictions[i])
                        target += max_next_q * discount_factor
                        profile_end("Calculate Target Value")

                    profile_start("Store Target Vector and Input Board")
                    boards.append(start_input_matrixes[i])
                    target_vector = np.zeros((4), dtype=np.float32)
                    target_vector[actions[i]] = target
                    target_vectors.append(target_vector)
                    profile_end("Store Target Vector and Input Board")
                profile_start("Remove Finished Games")
                games = [game for game in games if not game.isGameOver()]
                profile_end("Remove Finished Games")

            if len(boards) >= TARGET_SAMPLES_PER_ITERATION:
                break

        print(f"Collected {len(boards)} training examples.")
        print(f"Training with learning rate: {learning_rate}")


        # Update learning rate for RL (lower than pre-training)
        tensorflow.keras.backend.set_value(
            model.optimizer.learning_rate,
            learning_rate
        )
        learning_rate *= LEARNING_RATE_DECAY

        profile_start(f"Model Fit")
        model.fit(
            np.array(boards).reshape(-1, 210, 160, 6),
            np.array(target_vectors).reshape(-1, 4),
            epochs=1,
            verbose=1)
        profile_end("Model Fit")

        profile_end("Overall RL Run")
        print("\nProfiling stats:\n")
        print_profiles()

if __name__ == "__main__":
    main()