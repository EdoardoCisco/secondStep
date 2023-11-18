import argparse
import tensorflow as tf
from tensorflow import lite

def load_model(file_path):
    interpreter = lite.Interpreter(model_path=file_path)
    interpreter.allocate_tensors()
    return interpreter

def get_layers_and_weights(interpreter):
    layers_and_weights = []
    for i in range(len(interpreter.get_tensor_details())):
        layer_name = interpreter.get_tensor_details()[i]['name']
        # layer_weights = [interpreter.get_tensor(i)]
        # layers_and_weights.append((layer_name, layer_weights))
        layers_and_weights.append(layer_name)
    return layers_and_weights

def print_model_info(interpreter):
    print("Model Information:")
    for i, tensor_info in enumerate(interpreter.get_tensor_details()):
        print(f"Layer {i + 1}: {tensor_info['name']} - Shape: {tensor_info['shape']} - Type: {tensor_info['dtype']}")

def main():
    parser = argparse.ArgumentParser(description='Load a TensorFlow Lite model and return layers with associated weights.')
    parser.add_argument('--file_path', type=str, help='Path to the TensorFlow Lite model file', required=True)
    args = parser.parse_args()

    file_path = args.file_path

    # Load the model
    interpreter = load_model(file_path)

    # Print model information
    print_model_info(interpreter)

    # interpreter.model_content()
    # Get layers
    layers = get_layers_and_weights(interpreter)

    # Print layers
    print("\nLayers:")
    for layer_name in layers:
        print(f"  {layer_name}")

    # for layer_name, layer_weights in layers:
        # print(f"Layer: {layer_name}")
        # for i, weight_matrix in enumerate(layer_weights):
            # print(f"  Weight Matrix {i + 1}:")
            # print(weight_matrix)
            # print()

if __name__ == "__main__":
    main()

