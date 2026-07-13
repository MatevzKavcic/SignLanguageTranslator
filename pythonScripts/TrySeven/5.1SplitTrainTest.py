import os
import sys
import random
import numpy as np
from collections import defaultdict

# ==========================================================
# CONFIG
# ==========================================================

Try_number = sys.argv[1]

INPUT_FOLDER = f"{Try_number}_npyData"
OUTPUT_FOLDER = f"{Try_number}_Testing"

TEST_PER_CLASS = 2
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================================
# LOAD
# ==========================================================

print("Loading data...")

X = np.load(f"{INPUT_FOLDER}/X_aug.npy", allow_pickle=True)
y = np.load(f"{INPUT_FOLDER}/y_aug.npy")
labels = np.load(f"{INPUT_FOLDER}/labels.npy", allow_pickle=True)

print(f"Sequences : {len(X)}")
print(f"Classes   : {len(labels)}")

# ==========================================================
# GROUP INDICES BY LABEL
# ==========================================================

groups = defaultdict(list)

for idx, label in enumerate(y):
    groups[label].append(idx)

# ==========================================================
# SPLIT
# ==========================================================

train_idx = []
test_idx = []

print("\nSplitting dataset...\n")

for label in sorted(groups.keys()):

    indices = groups[label]

    random.shuffle(indices)

    test = indices[:TEST_PER_CLASS]
    train = indices[TEST_PER_CLASS:]

    train_idx.extend(train)
    test_idx.extend(test)

    print(
        f"Class {label:4d} | "
        f"Train: {len(train):2d} | "
        f"Test: {len(test):2d}"
    )

# ==========================================================
# SHUFFLE
# ==========================================================

random.shuffle(train_idx)
random.shuffle(test_idx)

# ==========================================================
# CREATE DATASETS
# ==========================================================

X_train = X[train_idx]
y_train = y[train_idx]

X_test = X[test_idx]
y_test = y[test_idx]

# ==========================================================
# SAVE
# ==========================================================

np.save(f"{OUTPUT_FOLDER}/X_train.npy", X_train)
np.save(f"{OUTPUT_FOLDER}/y_train.npy", y_train)

np.save(f"{OUTPUT_FOLDER}/X_test.npy", X_test)
np.save(f"{OUTPUT_FOLDER}/y_test.npy", y_test)

np.save(f"{OUTPUT_FOLDER}/labels.npy", labels)

# ==========================================================
# SUMMARY
# ==========================================================

print("\n====================================")
print("Split complete")
print("====================================")
print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")
print(f"Classes          : {len(labels)}")
print("====================================")
