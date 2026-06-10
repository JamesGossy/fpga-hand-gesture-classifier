"""
Tests for the model shape and dtype.

These need torch but not a webcam. They guard the dtype bug where float64
inputs would not match the float32 weights.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "software"))

from preprocess import GESTURES, NUM_FEATURES
from train import GestureNet, to_tensors


def test_model_outputs_one_logit_per_gesture():
    model = GestureNet(len(GESTURES))
    x = torch.zeros(1, NUM_FEATURES)
    out = model(x)
    assert out.shape == (1, len(GESTURES))


def test_float_input_runs_through_model():
    # to_tensors must give float32 so it matches the float32 weights. this
    # guards against a float64 array slipping in later and breaking the model.
    model = GestureNet(len(GESTURES))
    rows = [[0.1] * NUM_FEATURES]
    labels = [0]
    x, y = to_tensors(list(zip(rows, labels)))
    assert x.dtype == torch.float32
    # this line is the real test, it would raise if the dtype was wrong
    out = model(x)
    assert out.shape == (1, len(GESTURES))


def test_labels_are_long_for_loss():
    # cross entropy wants integer class labels
    rows = [[0.0] * NUM_FEATURES]
    labels = [3]
    _, y = to_tensors(list(zip(rows, labels)))
    assert y.dtype == torch.int64
