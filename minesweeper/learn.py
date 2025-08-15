import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense  # type: ignore
from .model import create_model, save_model
from datetime import datetime

NUM_TO_LOAD = 5 * (10 ** 5)

training_data_dir = os.path.join(os.path.dirname(__file__), 'training_data')
if not os.path.exists(training_data_dir):
    os.makedirs(training_data_dir)
training_data_file = os.path.join(training_data_dir, 'training_data.npz')

models_dir = os.path.join(os.path.dirname(__file__), 'models')

def load_consolidated_data(max_samples=None) -> tuple[np.ndarray, np.ndarray]:
    """Load training data from consolidated file."""
    full_path = os.path.join(os.path.dirname(__file__), training_data_file)

    if not os.path.exists(full_path):
        raise Exception(f"Consolidated data file not found at {full_path}")

    print(f"Loading consolidated data from {full_path}...")
    start_time = datetime.now()
    
    data = np.load(full_path)
    all_boards = data['boards']
    move_ratings = data['safe_moves'] / 100.0 ## I started in the range -100 to 100, but now I want to normalize it to -1 to 1.
    
    if max_samples and max_samples < len(all_boards):
        all_boards = all_boards[:max_samples]
        move_ratings = move_ratings[:max_samples]

    load_time = (datetime.now() - start_time).total_seconds()
    print(f"Loaded {len(all_boards)} training examples in {load_time:.1f} seconds")

    return all_boards, move_ratings

def main():
    # Try to load consolidated data first
    all_boards, move_ratings = load_consolidated_data(max_samples=NUM_TO_LOAD)

    print(f"Loaded {len(all_boards)} boards and {len(move_ratings)} move vectors from training data.")
    move_ratings_zeros_shape = np.zeros_like(move_ratings[0])
    move_ratings_flattened_shape = move_ratings_zeros_shape.flatten()

    model = create_model(
        input_shape=(all_boards[0].shape[0], all_boards[0].shape[1], 1),
        output_shape=move_ratings_flattened_shape.shape)

    # Show model architecture and parameter count
    print("\nModel Summary:")
    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}")

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
        all_boards.reshape(-1, all_boards[0].shape[0], all_boards[0].shape[1], 1),
        move_ratings,
        epochs=10,
        validation_split=0.2,  # Use 20% of data for validation
        callbacks=[early_stopping, reduce_lr],
        verbose=1)
    
    # Save model as minesweeper_model_<yy-mm-dd_hh-mm>.h5
    model_name = f"minesweeper_model_{datetime.now().strftime('%y-%m-%d_%H-%M')}"
    print(f"Saving model as {model_name}.h5")
    save_model(model, model_name)

if __name__ == "__main__":
    main()