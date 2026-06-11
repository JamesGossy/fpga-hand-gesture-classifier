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
BATCH_SIZE = 32

# Train fits the weights, val picks when to stop, test is the honest number we
# report once at the end. The split is per class so every gesture keeps its
# share in all three sets.
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15

# Early stopping. Train until val accuracy has not improved for this many
# epochs, then keep the best weights. MAX_EPOCHS is just a safety cap.
PATIENCE = 200
MAX_EPOCHS = 3000


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


def split_stratified(inputs, labels):
    """
    Split into train, val, and test. The split is done per class so each gesture
    keeps its share in all three sets. This matters because the classes are not
    the same size, so a plain random split could starve a small class in one set.
    """
    by_class = {}
    for row, label in zip(inputs, labels):
        by_class.setdefault(label, []).append(row)

    train_pairs = []
    val_pairs = []
    test_pairs = []
    for label, rows in by_class.items():
        random.shuffle(rows)
        train_cut = int(len(rows) * TRAIN_FRACTION)
        val_cut = int(len(rows) * (TRAIN_FRACTION + VAL_FRACTION))
        for row in rows[:train_cut]:
            train_pairs.append((row, label))
        for row in rows[train_cut:val_cut]:
            val_pairs.append((row, label))
        for row in rows[val_cut:]:
            test_pairs.append((row, label))

    random.shuffle(train_pairs)
    return train_pairs, val_pairs, test_pairs


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


def train_one_epoch(model, optimizer, loss_function, train_x, train_y):
    """Run one pass over the training data in mini-batches. Returns the last loss."""
    model.train()
    count = train_x.shape[0]
    order = torch.randperm(count)  # reshuffle each epoch so batches vary
    last_loss = 0.0
    for start in range(0, count, BATCH_SIZE):
        index = order[start:start + BATCH_SIZE]
        optimizer.zero_grad()
        loss = loss_function(model(train_x[index]), train_y[index])
        loss.backward()
        optimizer.step()
        last_loss = loss.item()
    return last_loss


def main():
    random.seed(0)
    torch.manual_seed(0)

    inputs, labels = load_samples()
    if len(inputs) == 0:
        print("No samples found. Run capture.py first.")
        return
    print("loaded", len(inputs), "samples")

    train_pairs, val_pairs, test_pairs = split_stratified(inputs, labels)
    train_x, train_y = to_tensors(train_pairs)
    val_x, val_y = to_tensors(val_pairs)
    test_x, test_y = to_tensors(test_pairs)
    print("split:", len(train_pairs), "train,", len(val_pairs), "val,",
          len(test_pairs), "test")

    model = GestureNet(len(GESTURES))
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # early stopping: keep the weights that scored best on val, stop when val
    # has not improved for PATIENCE epochs
    best_val = 0.0
    best_state = None
    epochs_since_best = 0

    for epoch in range(MAX_EPOCHS):
        loss = train_one_epoch(model, optimizer, loss_function, train_x, train_y)
        val_accuracy = accuracy(model, val_x, val_y)

        if val_accuracy > best_val:
            best_val = val_accuracy
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best = epochs_since_best + 1

        if (epoch + 1) % 20 == 0:
            print("epoch", epoch + 1, "loss", round(loss, 4),
                  "val accuracy", round(val_accuracy, 3),
                  "best", round(best_val, 3))

        if epochs_since_best >= PATIENCE:
            print("no val gain for", PATIENCE, "epochs, stopping at epoch", epoch + 1)
            break

    # roll back to the best val weights before we report and save
    model.load_state_dict(best_state)
    print("best val accuracy", round(best_val, 3))

    # test is touched once, here, so it is an honest held out number
    test_accuracy = accuracy(model, test_x, test_y)
    print("test accuracy", round(test_accuracy, 3))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print("saved model to", MODEL_PATH)


if __name__ == "__main__":
    main()
