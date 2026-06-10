"""
Tests for the shared landmark preprocessing.

These only touch preprocess.py, which has no heavy dependencies, so they run
fast and need no webcam.
"""

import os
import sys

# let the tests import the modules in software/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "software"))

from preprocess import GESTURES, NUM_FEATURES, NUM_POINTS, normalize


def make_points(offset=0.0):
    """21 points in a simple line, the wrist first."""
    points = [(0.5 + offset, 0.5 + offset)]
    for i in range(1, NUM_POINTS):
        points.append((0.5 + offset + 0.01 * i, 0.5 + offset + 0.02 * i))
    return points


def test_output_length_is_42():
    out = normalize(make_points())
    assert len(out) == NUM_FEATURES


def test_wrist_maps_to_origin():
    out = normalize(make_points())
    # the wrist is the first point, so its x and y come out as zero
    assert out[0] == 0.0
    assert out[1] == 0.0


def test_translation_invariance():
    # the same hand shape in two screen spots gives the same numbers
    a = normalize(make_points(offset=0.0))
    b = normalize(make_points(offset=0.2))
    for value_a, value_b in zip(a, b):
        assert abs(value_a - value_b) < 1e-9


def test_scale_invariance():
    # a bigger version of the same shape gives the same numbers
    small = [(x, y) for x, y in make_points()]
    big = [(x * 2.0, y * 2.0) for x, y in small]
    out_small = normalize(small)
    out_big = normalize(big)
    for value_small, value_big in zip(out_small, out_big):
        assert abs(value_small - value_big) < 1e-9


def test_values_stay_in_minus_one_to_one():
    # dividing by the largest reach means nothing goes past one
    out = normalize(make_points())
    for value in out:
        assert -1.0 <= value <= 1.0


def test_all_points_on_wrist_does_not_crash():
    # the divide by zero guard should keep this safe
    points = [(0.3, 0.3)] * NUM_POINTS
    out = normalize(points)
    assert len(out) == NUM_FEATURES
    for value in out:
        assert value == 0.0


def test_gesture_list_matches_feature_count():
    assert NUM_FEATURES == NUM_POINTS * 2
    assert len(GESTURES) == 5
