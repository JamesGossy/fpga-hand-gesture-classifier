"""
Live float demo.

Loads the trained model and runs the full pipeline on the webcam. For each
frame it finds the hand, normalizes the landmarks, runs the model, and shows
the guessed gesture on screen. This is the ground truth we compare the fixed
point and FPGA versions against later.

The sender talks to the board over serial_link. The spot where the landmarks are
ready is marked below.

Keys:
  q   quit
  r   start/stop recording a GIF of the window (saved to demo.gif)
"""

import os
import time

import cv2
import imageio
import mediapipe as mp
import torch

from preprocess import GESTURES, NUM_POINTS, normalize
from serial_link import SerialLink
from train import GestureNet

GIF_PATH = os.path.join(os.path.dirname(__file__), "..", "docs", "demo.gif")
GIF_FPS = 12  # the webcam loop is faster, this keeps the file small

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

    # open the board link if it is plugged in, otherwise run float-only
    link = SerialLink()
    try:
        link.open()
        print("Board link open.")
    except Exception as problem:
        link = None
        print("No board link, running float only:", problem)

    # only one packet is in flight at a time. We send the next one after the
    # board has answered the last, so replies stay in step with the live frame
    # instead of piling up behind a slow round trip.
    waiting_for_reply = False
    fpga_label = ""

    # GIF recording: press r to start, r again to stop and save
    recording = False
    recorded_frames = []

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

                name, score = predict(model, features)
                label = name + "  " + str(round(score, 2))

                # send a new packet only when the last one has been answered, so
                # the board reply tracks the current gesture instead of lagging
                if link is not None:
                    if not waiting_for_reply:
                        link.send_features(features)
                        waiting_for_reply = True
                    else:
                        fpga_class = link.read_reply()
                        if fpga_class is not None and fpga_class < len(GESTURES):
                            fpga_label = "fpga:" + GESTURES[fpga_class]
                            waiting_for_reply = False

                if fpga_label:
                    label = label + "  " + fpga_label

        cv2.putText(
            frame, label, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        )

        # grab the frame for the GIF before drawing the REC dot so the dot does
        # not end up in the recording
        if recording:
            recorded_frames.append(
                (time.perf_counter(), cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            )
            cv2.circle(frame, (frame.shape[1] - 20, 20), 8, (0, 0, 255), -1)

        cv2.imshow("demo", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            recording = not recording
            if recording:
                recorded_frames = []
                print("recording...")
            else:
                save_gif(recorded_frames)

    # if the user quits mid-recording, still save what we have
    if recording and recorded_frames:
        save_gif(recorded_frames)

    camera.release()
    cv2.destroyAllWindows()
    hands.close()
    if link is not None:
        link.close()


def save_gif(stamped_frames):
    """
    Write the captured frames to demo.gif. Each frame is (timestamp, image). We
    subsample down to about GIF_FPS so the file stays small and plays at real
    speed regardless of how fast the webcam loop ran.
    """
    if not stamped_frames:
        print("no frames to save")
        return

    step = 1.0 / GIF_FPS
    kept = []
    next_time = stamped_frames[0][0]
    for stamp, image in stamped_frames:
        if stamp >= next_time:
            kept.append(image)
            next_time += step

    os.makedirs(os.path.dirname(GIF_PATH), exist_ok=True)
    imageio.mimsave(GIF_PATH, kept, fps=GIF_FPS, loop=0)
    print("saved", len(kept), "frames to", os.path.abspath(GIF_PATH))


if __name__ == "__main__":
    main()
