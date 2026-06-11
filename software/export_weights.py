"""
Export the trained float model to Q4.12 fixed point.

Loads models/model_float.pt, turns every weight and bias into a Q4.12 int,
writes one .mif and one .hex file per layer into models/weights_q4_12, then
checks the fixed point accuracy against the captured data. Last, it saves a set
of golden vectors: real inputs paired with the exact class and logits the fixed
point pass produces. The FPGA must reproduce these numbers bit for bit.

The weight files store the weights flattened in the order the FPGA reads them:
neuron by neuron, and inside each neuron, input by input. Keep this order the
same on the Verilog side.
"""

import json
import os

import torch

from fixed_point import classify, to_fixed
from preprocess import GESTURES, NUM_FEATURES
from train import GestureNet, load_samples

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model_float.pt")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "weights_q4_12")
GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "golden_vectors.json")

# How many real samples to freeze as golden vectors.
NUM_GOLDEN = 40

# Extra right shift after each layer, past the normal 12. The activations grow
# past the Q4.12 range of +-8 between layers, so each layer divides its result
# by a power of two to stay in range. These values were found by sweeping the
# captured data: they recover full float accuracy. The Verilog must use the
# same shifts. Order matches layer1, layer2, layer3.
OUTPUT_SHIFTS = [0, 1, 2]


def quantize_matrix(matrix):
    """Turn a 2D float tensor into rows of Q4.12 ints, one row per neuron."""
    rows = []
    for neuron in matrix:
        rows.append([to_fixed(value.item()) for value in neuron])
    return rows


def quantize_vector(vector):
    """Turn a 1D float tensor into a list of Q4.12 ints."""
    return [to_fixed(value.item()) for value in vector]


def build_layers(model):
    """
    Pull the weights out of the model as fixed point layers.

    Returns a list of (weights, biases, use_relu, output_shift) tuples in run
    order. The first two layers use ReLU. The output layer does not.
    """
    layers = []
    parts = [
        (model.layer1, True),
        (model.layer2, True),
        (model.layer3, False),
    ]
    for (linear, use_relu), shift in zip(parts, OUTPUT_SHIFTS):
        weights = quantize_matrix(linear.weight.data)
        biases = quantize_vector(linear.bias.data)
        layers.append((weights, biases, use_relu, shift))
    return layers


def to_hex_word(value):
    """16-bit two's complement hex string, no prefix, lower case, 4 digits."""
    return format(value & 0xFFFF, "04x")


def flatten_layer(weights, biases):
    """
    Flatten one layer into the order the FPGA reads it.

    All weights first, neuron by neuron and input by input, then all biases in
    neuron order. The ROM holds weights and biases in this single stream.
    """
    stream = []
    for neuron_row in weights:
        stream.extend(neuron_row)
    stream.extend(biases)
    return stream


def write_hex_file(path, values):
    """One 16-bit hex word per line, for Verilog $readmemh."""
    with open(path, "w") as handle:
        for value in values:
            handle.write(to_hex_word(value) + "\n")


def write_mif_file(path, values):
    """Quartus .mif, 16-bit words, hex radix, one address per word."""
    width = 16
    depth = len(values)
    with open(path, "w") as handle:
        handle.write("WIDTH=" + str(width) + ";\n")
        handle.write("DEPTH=" + str(depth) + ";\n")
        handle.write("ADDRESS_RADIX=DEC;\n")
        handle.write("DATA_RADIX=HEX;\n")
        handle.write("CONTENT BEGIN\n")
        for address, value in enumerate(values):
            handle.write("    " + str(address) + " : " + to_hex_word(value) + ";\n")
        handle.write("END;\n")


def write_layer_files(layers):
    """Write a .mif and .hex per layer. Returns the flattened stream per layer."""
    os.makedirs(OUT_DIR, exist_ok=True)
    streams = []
    for index, (weights, biases, _, _) in enumerate(layers, start=1):
        stream = flatten_layer(weights, biases)
        streams.append(stream)
        name = "layer" + str(index)
        write_hex_file(os.path.join(OUT_DIR, name + ".hex"), stream)
        write_mif_file(os.path.join(OUT_DIR, name + ".mif"), stream)
        print("wrote", name, "with", len(stream), "words")
    return streams


def fixed_accuracy(layers, inputs, labels):
    """Run the fixed point pass on every sample and return the accuracy."""
    correct = 0
    for features, label in zip(inputs, labels):
        fixed_input = [to_fixed(value) for value in features]
        predicted, _ = classify(fixed_input, layers)
        if predicted == label:
            correct = correct + 1
    return correct / len(inputs)


def save_golden_vectors(layers, inputs, labels):
    """
    Freeze the first NUM_GOLDEN samples as golden vectors.

    Each entry holds the Q4.12 input ints, the class the fixed point pass picks,
    and the Q4.12 logits. The FPGA testbench drives these inputs and checks the
    class and logits match exactly.
    """
    vectors = []
    for features, label in list(zip(inputs, labels))[:NUM_GOLDEN]:
        fixed_input = [to_fixed(value) for value in features]
        predicted, logits = classify(fixed_input, layers)
        vectors.append({
            "input_q4_12": fixed_input,
            "true_class": label,
            "expected_class": predicted,
            "expected_logits_q4_12": logits,
        })

    os.makedirs(os.path.dirname(GOLDEN_PATH), exist_ok=True)
    with open(GOLDEN_PATH, "w") as handle:
        json.dump({"format": "Q4.12", "output_shifts": OUTPUT_SHIFTS,
                   "gestures": GESTURES, "vectors": vectors},
                  handle, indent=2)
    print("saved", len(vectors), "golden vectors to", GOLDEN_PATH)


def main():
    if not os.path.exists(MODEL_PATH):
        print("No model found. Run train.py first.")
        return

    model = GestureNet(len(GESTURES))
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    layers = build_layers(model)
    write_layer_files(layers)

    inputs, labels = load_samples()
    if len(inputs) == 0:
        print("No samples found, skipping accuracy check and golden vectors.")
        return

    accuracy = fixed_accuracy(layers, inputs, labels)
    print("fixed point accuracy on all samples", round(accuracy, 4))

    save_golden_vectors(layers, inputs, labels)


if __name__ == "__main__":
    main()
