import argparse
import tensorflow as tf
from tensorflow import keras

def load_model(file_path):
    model = keras.models.load_model(file_path)
    return model

def get_layers_and_weights(model):
    layers_and_weights = []
    for layer in model.layers:
        layer_name = layer.name
        layer_weights = layer.get_weights()
        layers_and_weights.append((layer_name, layer_weights))
    return layers_and_weights

def main():
    parser = argparse.ArgumentParser(description='Load a machine learning model and return layers with associated weights.')
    parser.add_argument('--file_path', type=str, help='Path to the pre-trained model file', required=True)
    args = parser.parse_args()

    file_path = args.file_path

    # Load the model
    model = load_model(file_path)
    model.summary()

    # Get layers and weights
    layers_and_weights = get_layers_and_weights(model)

    # Print layers and associated weights
    for layer_name, layer_weights in layers_and_weights:
        print(f"Layer: {layer_name}")
        for i, weight_matrix in enumerate(layer_weights):
            print(f"  Weight Matrix {i + 1}:")
            print(weight_matrix)
            print()

if __name__ == "__main__":
    main()
