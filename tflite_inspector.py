import numpy as np
import tensorflow as tf
import argparse

def load_and_inspect_tflite_model(tflite_file_path):
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_file_path)
    interpreter.allocate_tensors()
    return interpreter

def print_in_out_details(interpreter):
    # Get information about the model
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("Input Tensor Details:")
    print(input_details)

    print("\nOutput Tensor Details:")
    print(output_details)

def get_layers(interpreter):
    # Get formatted layers and weights
    for i, layer in enumerate(interpreter.get_tensor_details()):
        print(f"\nLayer {i + 1}: {layer['name']}")
        print("Type:", layer['dtype'])
        print("Shape:", layer['shape'])
    
def get_weights(interpreter):
    for layer in enumerate(interpreter.get_tensor_details()):
        # Print the weights if available
        if 'quantization' in layer and 'scales' in layer['quantization']:
            scales = layer['quantization']['scales']
            zero_points = layer['quantization']['zero_points']
            quantized_values = interpreter.get_tensor(layer['index'])

            # Dequantize the weights
            dequantized_values = (quantized_values - zero_points) * scales

            print("\nWeights:")
            print(dequantized_values)

def main():
    parser = argparse.ArgumentParser(description='Load a TensorFlow Lite model and return layers with associated weights.')
    parser.add_argument('--file_path', type=str, help='Path to the TensorFlow Lite model file', required=True)
    args = parser.parse_args()

    file_path = args.file_path

    # Load the model
    interpreter = load_and_inspect_tflite_model(file_path)

    # Print model information
    # print_in_out_details(interpreter)

    # print(interpreter.model_content())
    print("----------------")
    print(interpreter.get_tensor_details())
    print("----------------")
    get_weights(interpreter)
    print("----------------")

if __name__ == "__main__":
    main()