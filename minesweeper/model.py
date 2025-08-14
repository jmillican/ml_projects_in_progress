import os
import tensorflow as tf
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense  # type: ignore
import numpy as np
from tensorflow.keras import Model as TfKerasModel # type: ignore

def create_model(input_shape: tuple[int, ...], output_shape: tuple[int, ...]) -> TfKerasModel:
    tf.random.set_seed(1234)
    tf.config.run_functions_eagerly(False)
    model = Sequential(
        [
            tf.keras.Input(shape=input_shape, name='input_layer'),  # type: ignore
            Dense(2 ** 7, activation='relu', name='layer1'),
            Dense(2 ** 6, activation='relu', name='layer2'),
            Dense(2 ** 6, activation='relu', name='layer3'),
            Dense(output_shape[0], activation='sigmoid', name='output_layer')
        ]
    )
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.00001)  # type: ignore
    model.compile(optimizer=optimizer, loss='binary_crossentropy')
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
    return tf.keras.models.load_model(model_path)   # type: ignore
