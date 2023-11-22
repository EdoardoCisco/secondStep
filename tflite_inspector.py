import numpy as np
import tensorflow as tf
import argparse
import pandas as pd

def load_and_inspect_tflite_model(tflite_file_path):
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_file_path)
    interpreter.allocate_tensors()
    return interpreter

def print_in_out_details(interpreter):
    # Get information about the input and output
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("Input Tensor Details:", input_details)
    print("\nOutput Tensor Details:", output_details)

def get_layers(interpreter):
    # Get formatted layers
    for i, layer in enumerate(interpreter.get_tensor_details()):
        print(f"\nLayer {i + 1}: {layer['name']}")
        print("Type:", layer['dtype'])
        print("Shape:", layer['shape'])
    
def get_weights(interpreter):
    for i, layer in enumerate(interpreter.get_tensor_details()):
        if 'quantization' in layer:
            quantization, scale, zero_point = layer['quantization'], layer['quantization_parameters']['scales'], layer['quantization_parameters']['zero_points']
            # weights = interpreter.tensor(layer['index'])
            # real_weights = (weights - zero_point) * scale
            # print(f"Weights (Quantized):\n{weights}\n")
            # print(f"Weights (Real Values):\n{real_weights}\n")
            print(f"Layer {i}:")
            print(f"Quantization:\t{quantization}")
            print(f"Scale:\t{scale}")
            print(f"Zero point:\t{zero_point}\n")

def main():
    parser = argparse.ArgumentParser(description='Load a TensorFlow Lite model and return layers with associated weights.')
    parser.add_argument('--file_path', type=str, help='Path to the TensorFlow Lite model file', required=True)
    args = parser.parse_args()

    file_path = args.file_path

    # Load the model
    interpreter = load_and_inspect_tflite_model(file_path)

    # Print model information
    print_in_out_details(interpreter)
    get_layers(interpreter)
    print("\n=================================================\n")
    get_weights(interpreter)

if __name__ == "__main__":
    main()