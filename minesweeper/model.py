import os
import tensorflow as tf
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Conv2D, Flatten  # type: ignore
import numpy as np
from tensorflow.keras import Model as TfKerasModel # type: ignore

models_dir = os.path.join(os.path.dirname(__file__), 'models')

def loss_function(y_true, y_pred):
    """
    Custom loss function that masks the loss for cells that are not visible.
    """
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)  # Mask where y_true is not zero
    masked_loss = tf.reduce_mean(tf.square(y_true - y_pred) * mask)
    return masked_loss

def create_model(input_shape: tuple[int, ...], output_shape: tuple[int, ...]) -> TfKerasModel:
    tf.random.set_seed(1234)
    tf.config.run_functions_eagerly(False)
    model = Sequential(
        [
            tf.keras.Input(shape=input_shape, name='input_layer'),  # type: ignore
            Conv2D(24, (6, 6), activation='leaky_relu', padding='same', name='conv1'),
            Conv2D(20, (5, 5), activation='leaky_relu', padding='same', name='conv2'),
            Conv2D(16, (4, 4), activation='leaky_relu', padding='same', name='conv3'),
            Conv2D(12, (3, 3), activation='leaky_relu', padding='same', name='conv4'),
            Conv2D(2, (1, 1), activation='linear', padding='same', name='output_layer'),
        ]
    )
    # Use modern Adam optimizer with learning rate schedule
    initial_learning_rate = 0.00003 # 3e-5, adjust as needed
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay( # type: ignore
        initial_learning_rate,
        decay_steps=10000,
        decay_rate=0.96,
        staircase=True)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)  # type: ignore
    model.compile(optimizer=optimizer, loss=loss_function)
    return model

def getModelPath(model_name: str) -> str:
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    return os.path.join(models_dir, f"{model_name}.h5")

def save_model(model: TfKerasModel, model_name: str):
    model_path = getModelPath(model_name)
    model.save(model_path)
    print(f"Model saved to {model_path}")

def load_model(model_name: str) -> TfKerasModel:
    model_path = getModelPath(model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model {model_name} does not exist at {model_path}.")
    return tf.keras.models.load_model(model_path, custom_objects={'loss_function': loss_function})   # type: ignore

def load_latest_model(offset: int = 0, verbose: bool = True) -> tuple[TfKerasModel, str]:
    # List all model files in the models directory
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
    if not model_files:
        raise FileNotFoundError("No model files found in 'models' directory.")
    
    # Sort files by modification time
    model_files.sort(key=lambda f: os.path.getmtime(os.path.join(models_dir, f)))
    latest_model_file = model_files[-(1 + offset)]  # Get the most recent model file, offset by the parameter
    model_name = os.path.splitext(latest_model_file)[0]  # Remove the .h5 extension
    if verbose:
        print(f"Loading model: {model_name}")
    # Load the model
    model = load_model(model_name)
    return model, model_name