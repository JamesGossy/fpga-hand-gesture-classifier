"""
Shared landmark preprocessing.

MediaPipe gives 21 hand points, each with an x and a y. That is 42 numbers.
The raw points depend on where the hand is on screen and how big it looks.
We want the gesture to mean the same thing no matter where the hand is or how
far away it is. So we move the points to sit around the wrist, then divide by
the hand size. Train, demo, and capture all call this so they agree.
"""

# The gesture names in class order. The position in this list is the class
# number the model learns and the FPGA sends back. Do not reorder this.
GESTURES = ["fist", "open_palm", "peace", "thumbs_up", "point"]

# 21 points, each with x and y, gives 42 input numbers.
NUM_POINTS = 21
NUM_FEATURES = NUM_POINTS * 2

# The wrist is point 0 in MediaPipe.
WRIST_INDEX = 0


def normalize(points):
    """
    Turn 21 raw (x, y) points into 42 normalized numbers.

    points is a list of 21 (x, y) pairs from MediaPipe.
    Returns a flat list of 42 floats.
    """
    wrist_x, wrist_y = points[WRIST_INDEX]

    # wrist point is the origin so the gesture works anywhere on screen
    moved = []
    for x, y in points:
        moved.append((x - wrist_x, y - wrist_y))

    # hand size is the largest distance from the wrist on either axis
    largest = 0.0
    for x, y in moved:
        if abs(x) > largest:
            largest = abs(x)
        if abs(y) > largest:
            largest = abs(y)

    # guard against a divide by zero if all points landed on the wrist
    if largest == 0.0:
        largest = 1.0

    flat = []
    for x, y in moved:
        flat.append(x / largest)
        flat.append(y / largest)

    return flat
