import numpy as np
import tensorflow as tf
import argparse
import pandas as pd
import json

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
    structures = []
    for i, layer in enumerate(interpreter.get_tensor_details()):
        structure = {
            "index" : [layer['index']],
            "Layer name" : [layer['name']],
           "Shape:" : [layer['shape'].tolist()]
        }
        structures.append(structure)
    saveJason(structures, "layersShapes.json")

def get_weights(interpreter):
    structures = []
    ops_details = interpreter._get_ops_details()
    vector = np.vectorize(np.int_)
    for operation in ops_details:
        if operation['op_name'] == "CONV_2D" or operation['op_name'] == "DEPTHWISE_CONV_2D" or operation['op_name'] == "FULLY_CONNECTED":
            input_indices = operation['inputs']
            weights = interpreter.get_tensor(input_indices[1])
            bias = interpreter.get_tensor(input_indices[2])
            structure = {
                "name": [operation['op_name']],
                    "weights" : [vector(weights).tolist()],
                    "bias": [vector(bias).tolist()]
                }
            structures.append(structure)
    saveJason(structures, "weights.json")

def get_quantizaion_values(interpreter):
    structures = []
    ops_details = interpreter._get_ops_details()
    tensors_details = interpreter.get_tensor_details()
    for operation in ops_details:
        if operation['op_name'] == "CONV_2D" or operation['op_name'] == "DEPTHWISE_CONV_2D" or operation['op_name'] == "FULLY_CONNECTED":
            input_indices = operation['inputs']
            filters = next((details for details in tensors_details if details['index'] == input_indices[1]),None)
            bias = next((details for details in tensors_details if details['index'] == input_indices[2]),None)
            structure = {
                "name": [operation['op_name']],
                "filters": {
                    "index" : [filters['index']],
                    "qantization": [filters['quantization']],
                    "scale": [filters['quantization_parameters']['scales'].tolist()],
                    "zero point": [filters['quantization_parameters']['zero_points'].tolist()]
                },
                "bias": {
                    "index" : [bias['index']],
                    "qantization": [bias['quantization']],
                    "scale": [bias['quantization_parameters']['scales'].tolist()],
                    "zero point": [bias['quantization_parameters']['zero_points'].tolist()]
                }
            }
            structures.append(structure)           
    saveJason(structures, "quantizationValues.json")

def get_structure(interpreter):
    structures = []
    ops_details = interpreter._get_ops_details()
    tensors_details = interpreter.get_tensor_details()
    for layer in ops_details: 
        if not 'DELEGATE' in layer['op_name']:
            input_indices = layer['inputs']
            output_indices = layer['outputs']
            input_details = next((details for details in tensors_details if details['index'] == input_indices[0]),None)
            output_details = next((details for details in tensors_details if details['index'] == output_indices[0]),None)
            print(input_details)
            print(output_details)
            structure = {
                "index": [layer['index']],
                "op_name": [layer['op_name']],
                "input index": [input_details['index']],
                "input name": [input_details['name']],
                "output index": [output_details['index']],
                "output name": [output_details['name']]
            }
            structures.append(structure)
    saveJason(structures, "modelStructure.json")

def saveJason(structure, filename):
    with open(filename, 'w') as json_file:
        json.dump(structure, json_file, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Load a TensorFlow Lite model and return layers with associated weights.')
    parser.add_argument('--file_path', type=str, help='Path to the TensorFlow Lite model file', required=True)
    args = parser.parse_args()

    file_path = args.file_path

    # Load the model
    interpreter = load_and_inspect_tflite_model(file_path)

    # get_structure(interpreter)
    # get_layers(interpreter)
    # get_quantizaion_values(interpreter)
    get_weights(interpreter)

if __name__ == "__main__":
    main()