"""
Tests for the fixed point forward pass.

These check the integer math on its own, then check that the saved golden
vectors still reproduce through fixed_point.py. The golden check is the real
guard: it catches any change that would make the Python reference and the FPGA
disagree.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "software"))

from fixed_point import (
    INT16_MAX,
    INT16_MIN,
    SCALE,
    argmax,
    classify,
    dense_layer,
    relu,
    saturate,
    to_fixed,
)

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "golden_vectors.json")


def test_to_fixed_rounds_and_scales():
    assert to_fixed(1.0) == SCALE
    assert to_fixed(0.0) == 0
    assert to_fixed(-1.0) == -SCALE


def test_to_fixed_saturates_past_range():
    # 100.0 is far past the +-8 range, so it clamps to the 16-bit max
    assert to_fixed(100.0) == INT16_MAX
    assert to_fixed(-100.0) == INT16_MIN


def test_saturate_clamps():
    assert saturate(40000) == INT16_MAX
    assert saturate(-40000) == INT16_MIN
    assert saturate(5) == 5


def test_relu_floors_at_zero():
    assert relu(-3) == 0
    assert relu(0) == 0
    assert relu(7) == 7


def test_argmax_ties_go_low():
    assert argmax([5, 5, 1]) == 0
    assert argmax([1, 9, 3]) == 1


def test_dense_layer_matches_hand_math():
    # one neuron, two inputs of 1.0, weights 0.5 and 0.25, bias 0.0, no shift.
    # the result should be 0.75 in Q4.12.
    inputs = [to_fixed(1.0), to_fixed(1.0)]
    weights = [[to_fixed(0.5), to_fixed(0.25)]]
    biases = [to_fixed(0.0)]
    out = dense_layer(inputs, weights, biases, use_relu=False, output_shift=0)
    assert out[0] == to_fixed(0.75)


def test_output_shift_divides_by_power_of_two():
    # same as above but with a shift of 1, so the result halves to 0.375
    inputs = [to_fixed(1.0), to_fixed(1.0)]
    weights = [[to_fixed(0.5), to_fixed(0.25)]]
    biases = [to_fixed(0.0)]
    out = dense_layer(inputs, weights, biases, use_relu=False, output_shift=1)
    assert out[0] == to_fixed(0.375)


def load_golden():
    with open(GOLDEN_PATH) as handle:
        return json.load(handle)


def build_layers_from_files():
    """
    Rebuild the fixed point layers from the trained model, the same way
    export_weights does. Skipped if torch or the model file is missing.
    """
    import torch

    from export_weights import build_layers
    from preprocess import GESTURES
    from train import GestureNet

    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "model_float.pt")
    model = GestureNet(len(GESTURES))
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return build_layers(model)


def test_golden_vectors_reproduce():
    # the saved golden vectors must come back out of the fixed point pass with
    # the same class and logits. this is what keeps Python and the FPGA in sync.
    if not os.path.exists(GOLDEN_PATH):
        return  # nothing exported yet, skip quietly

    import pytest

    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "model_float.pt")
    if not os.path.exists(model_path):
        pytest.skip("no trained model, run train.py and export_weights.py first")

    golden = load_golden()
    layers = build_layers_from_files()

    for vector in golden["vectors"]:
        predicted, logits = classify(vector["input_q4_12"], layers)
        assert predicted == vector["expected_class"]
        assert logits == vector["expected_logits_q4_12"]
