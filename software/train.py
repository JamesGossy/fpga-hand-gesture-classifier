"""
Train the gesture model.

Reads the per gesture CSV files in data/raw, splits them into a train set and
a small validation set, trains a small MLP, prints the validation accuracy, and
saves the weights to models/model_float.pt.

The model is 42 inputs, two hidden layers of 32 with ReLU, then one output per
gesture. ReLU is used on purpose. The FPGA can do ReLU easily and it matches the
fixed point version later.
"""

import csv
import os
import random

import torch
import torch.nn as nn

from preprocess import GESTURES, NUM_FEATURES

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model_float.pt")

HIDDEN_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 200
VAL_FRACTION = 0.2


class GestureNet(nn.Module):
    """42 in, two hidden layers of 32 with ReLU, one output per gesture."""

    def __init__(self, num_classes):
        super().__init__()
        self.layer1 = nn.Linear(NUM_FEATURES, HIDDEN_SIZE)
        self.layer2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.layer3 = nn.Linear(HIDDEN_SIZE, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        return self.layer3(x)


def load_samples():
    """
    Read every gesture CSV. Returns two lists, the input rows and the class
    number for each row. The class number is the gesture position in GESTURES.
    """
    inputs = []
    labels = []
    for class_number, name in enumerate(GESTURES):
        path = os.path.join(RAW_DIR, name + ".csv")
        if not os.path.exists(path):
            print("no file for", name, "at", path, "skipping")
            continue
        with open(path, newline="") as handle:
            for row in csv.reader(handle):
                if len(row) != NUM_FEATURES:
                    continue
                inputs.append([float(value) for value in row])
                labels.append(class_number)
    return inputs, labels


def split_train_val(inputs, labels):
    """Shuffle, then peel off the validation fraction."""
    pairs = list(zip(inputs, labels))
    random.shuffle(pairs)
    cut = int(len(pairs) * (1.0 - VAL_FRACTION))
    train_pairs = pairs[:cut]
    val_pairs = pairs[cut:]
    return train_pairs, val_pairs


def to_tensors(pairs):
    rows = [row for row, _ in pairs]
    classes = [number for _, number in pairs]
    # model weights are float32, so inputs must be float32 too
    return torch.tensor(rows, dtype=torch.float32), torch.tensor(classes)


def accuracy(model, x, y):
    model.eval()
    with torch.no_grad():
        guesses = model(x).argmax(dim=1)
    return (guesses == y).float().mean().item()


def main():
    random.seed(0)
    torch.manual_seed(0)

    inputs, labels = load_samples()
    if len(inputs) == 0:
        print("No samples found. Run capture.py first.")
        return
    print("loaded", len(inputs), "samples")

    train_pairs, val_pairs = split_train_val(inputs, labels)
    train_x, train_y = to_tensors(train_pairs)
    val_x, val_y = to_tensors(val_pairs)

    model = GestureNet(len(GESTURES))
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        loss = loss_function(model(train_x), train_y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            val_accuracy = accuracy(model, val_x, val_y)
            print("epoch", epoch + 1, "loss", round(loss.item(), 4),
                  "val accuracy", round(val_accuracy, 3))

    final_accuracy = accuracy(model, val_x, val_y)
    print("final val accuracy", round(final_accuracy, 3))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print("saved model to", MODEL_PATH)


if __name__ == "__main__":
    main()
