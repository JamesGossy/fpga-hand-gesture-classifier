"""
Capture tool.

Opens the webcam, finds one hand with MediaPipe, and shows the live picture.
You pick a gesture with the number keys. When you press space it saves the
normalized landmarks for the current frame to a CSV. Record a few hundred
samples per gesture, moving your hand around so the model sees variety.

Keys:
  1..5    pick which gesture you are recording
  space   save the current frame
  q       quit

Each gesture is saved to its own file in data/raw, for example
data/raw/fist.csv. Each row is 42 numbers, no header.
"""

import csv
import os

import cv2
import mediapipe as mp

from preprocess import GESTURES, NUM_POINTS, normalize

# Save next to the repo, not next to this script.
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def open_writer(gesture_name):
    """Open a CSV file to append rows for one gesture."""
    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, gesture_name + ".csv")
    handle = open(path, "a", newline="")
    return handle, csv.writer(handle)


def read_points(hand_landmarks):
    """Pull the 21 (x, y) pairs out of a MediaPipe result."""
    points = []
    for mark in hand_landmarks.landmark:
        points.append((mark.x, mark.y))
    return points


def main():
    hands = mp.solutions.hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    drawer = mp.solutions.drawing_utils

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Could not open the webcam.")
        return

    # Start on the first gesture in the list.
    current = 0
    counts = {name: 0 for name in GESTURES}

    print("Press 1 to 5 to pick a gesture, space to save, q to quit.")

    while True:
        ok, frame = camera.read()
        if not ok:
            break

        # mirror so moving right on screen matches moving your real hand right
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        points = None
        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            drawer.draw_landmarks(
                frame, hand, mp.solutions.hands.HAND_CONNECTIONS
            )
            points = read_points(hand)

        name = GESTURES[current]
        label = "gesture: " + name + "   saved: " + str(counts[name])
        if points is None:
            label = label + "   (no hand)"
        cv2.putText(
            frame, label, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )

        cv2.imshow("capture", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        # number keys 1..5 pick the gesture
        if ord("1") <= key <= ord("5"):
            picked = key - ord("1")
            if picked < len(GESTURES):
                current = picked

        if key == ord(" ") and points is not None and len(points) == NUM_POINTS:
            handle, writer = open_writer(name)
            writer.writerow(normalize(points))
            handle.close()
            counts[name] = counts[name] + 1
            print("saved one", name, "now at", counts[name])

    camera.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()
