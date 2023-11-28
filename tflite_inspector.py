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
    
def get_quantizaion_values(interpreter):
    for i, layer in enumerate(interpreter.get_tensor_details()):
        if 'quantization' in layer:
            quantization, scale, zero_point = layer['quantization'], layer['quantization_parameters']['scales'], layer['quantization_parameters']['zero_points']
            print(f"Layer {i}:")
            print(f"Quantization:\t{quantization}")
            print(f"Scale:\t{scale}")
            print(f"Zero point:\t{zero_point}\n")

def get_weights(interpreter):
    for i, layer in enumerate(interpreter.get_tensor_details()):
        if 'Conv2D' in layer['name']:
            print("\nLayer ", layer['index'],interpreter.get_tensor(layer['index']))


def get_detail(interpreter):
    for layer in enumerate(interpreter.get_tensor_details()):
        print(layer, "\n=============================\n")

def get_structure(interpreter):
    ops_details = interpreter._get_ops_details()
    tensors_details = interpreter.get_tensor_details()
    for layer in ops_details:
        print(layer['index'], layer['op_name'])
        input_indices = layer['inputs']
        output_indices = layer['outputs']
        for input_index in input_indices:
            input_details = next((details for details in tensors_details if details['index'] == input_index), None)
            if input_details:
                print(f"Input Index {input_index}: {input_details['name']}")
        for output_index in output_indices:
            output_details = next((details for details in tensors_details if details['index'] == output_index), None)
            if output_details:
                print(f"Output Index {output_index}: {output_details['name']}")
        print("="*30)


        # input_tensors = [tensors_details['index'] for input_number in input_indices]
        # print("Input tensor: ", input_tensors)
        # output_indices = layer['outputs']
        # output_tensors = [interpreter.tensor(output_number) for output_number in output_indices]
        # print("Output tensor:", output_tensors)


            

def main():
    parser = argparse.ArgumentParser(description='Load a TensorFlow Lite model and return layers with associated weights.')
    parser.add_argument('--file_path', type=str, help='Path to the TensorFlow Lite model file', required=True)
    args = parser.parse_args()

    file_path = args.file_path

    # Load the model
    interpreter = load_and_inspect_tflite_model(file_path)

    # Print model information
    # print_in_out_details(interpreter)
    # get_layers(interpreter)
    # print("\n=================================================\n")
    #get_quantizaion_values(interpreter)
    # print("\n=================================================\n")
    # conv_layer = interpreter.get_tensor(8)
    # print(conv_layer)
    # get_weights(interpreter)
    # print("\n=================================================\n")
    # test(interpreter)
    # print(interpreter._get_ops_details())
    # model = tf.lite.experimental.Analyzer.analyze(model_content=interpreter)
    # print(model)
    # Get the filters
    # filters, biases = conv_layer.get_weights()
    # print(interpreter.get_tensor_details())
    # get_structure(interpreter)
    # print("\n=================================================\n")
    # print(biases)
    get_detail(interpreter)
    # get_structure(interpreter)
    # interpreter.tensor

    # print(interpreter.get_tensor_details())

if __name__ == "__main__":
    main()