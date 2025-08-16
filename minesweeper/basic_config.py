import os
import absl.logging
import tensorflow as tf

def suppress_tensorflow_logging():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
    absl.logging.set_verbosity(absl.logging.ERROR)  # Suppress warnings

def force_tensorflow_cpu():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    tf.config.set_visible_devices([], 'GPU')

def print_model_hardware_device(model):
    print(f"Available GPUs: {tf.config.list_physical_devices('GPU')}")
    print(f"Visible devices: {tf.config.get_visible_devices()}")
    # Check if model is on GPU
    print(f"Model device: {model.weights[0].device if model.weights else 'No weights'}")