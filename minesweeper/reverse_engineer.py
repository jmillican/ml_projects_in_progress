from .model import load_latest_model
import random

def main():
    ## Pick a neuron randomly; and then identify the input values which maximise its activation.

    # First load the model
    model, model_name = load_latest_model()

    print(f"Loaded model: {model_name}")

    # Now we can inspect the model's layers and weights
    # let's start by constructing a list of all layers
    layers = model.layers
    print(f"Model has {len(layers)} layers.")

    # # List of all neurons, including their layer, index within the layer, and whatever other
    # # objects represent them.
    # neurons = []
    # # Now let's list all of the neurons in each layer
    # for i, layer in enumerate(layers):
    #     # Identify the output shape of the layer.
    #     output_shape = layer.output_shape
    
    layer_number, layer = random.choice([l for l in enumerate(layers)])
    print(f"Selected layer {layer_number} with name {layer.name} and output shape {layer.output_shape}.")

    # print(f"Model has {len(neurons)} neurons.")
    # print(f"Neurons: {neurons}")

    # print(f"Model has {len(neurons)} neurons.")


if __name__ == "__main__":
    main()