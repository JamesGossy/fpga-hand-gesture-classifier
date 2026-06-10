"""
Live float demo.

Loads the trained model and runs the full pipeline on the webcam. For each
frame it finds the hand, normalizes the landmarks, runs the model, and shows
the guessed gesture on screen. This is the ground truth we compare the fixed
point and FPGA versions against later.

The UART sender is added in Phase 3 once serial_link.py exists. The spot where
the landmarks are ready is marked below.

Keys:
  q   quit
"""

import os

import cv2
import mediapipe as mp
import torch

from preprocess import GESTURES, NUM_POINTS, normalize
from train import GestureNet

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model_float.pt")


def load_model():
    model = GestureNet(len(GESTURES))
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    return model


def read_points(hand_landmarks):
    points = []
    for mark in hand_landmarks.landmark:
        points.append((mark.x, mark.y))
    return points


def predict(model, features):
    """Run one 42 number input and return the gesture name and its score."""
    x = torch.tensor([features], dtype=torch.float32)
    with torch.no_grad():
        logits = model(x)[0]
    best = int(logits.argmax())
    return GESTURES[best], float(logits[best])


def main():
    if not os.path.exists(MODEL_PATH):
        print("No model found. Run train.py first.")
        return

    model = load_model()

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

    print("Showing live predictions. Press q to quit.")

    while True:
        ok, frame = camera.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        label = "no hand"
        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            drawer.draw_landmarks(
                frame, hand, mp.solutions.hands.HAND_CONNECTIONS
            )
            points = read_points(hand)
            if len(points) == NUM_POINTS:
                features = normalize(points)

                # Phase 3: send features over UART to the FPGA here.

                name, score = predict(model, features)
                label = name + "  " + str(round(score, 2))

        cv2.putText(
            frame, label, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        )
        cv2.imshow("demo", frame)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()
