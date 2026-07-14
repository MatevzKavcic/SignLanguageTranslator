import os
import numpy as np
from sklearn.model_selection import train_test_split

# ==========================================================
# CONFIG
# ==========================================================

INPUT_FOLDER = "../TrySeven/TrySeven_npyData"
OUTPUT_FOLDER = "SplitBySigns"

TEST_SIZE = 0.10
RANDOM_STATE = 42

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================================
# LOAD
# ==========================================================

print("Loading data...\n")

X = np.load(
    f"{INPUT_FOLDER}/X_aug.npy",
    allow_pickle=True
)

y = np.load(
    f"{INPUT_FOLDER}/y_aug.npy"
)

labels = np.load(
    f"{INPUT_FOLDER}/labels.npy",
    allow_pickle=True
)

print(f"Sequences : {len(X)}")
print(f"Classes   : {len(labels)}")

# ==========================================================
# SPLIT CLASSES
# ==========================================================

all_classes = np.unique(y)

train_classes, test_classes = train_test_split(
    all_classes,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    shuffle=True
)

print()
print(f"Training classes : {len(train_classes)}")
print(f"Testing classes  : {len(test_classes)}")

# ==========================================================
# BUILD MASKS
# ==========================================================

train_mask = np.isin(y, train_classes)
test_mask = np.isin(y, test_classes)

# ==========================================================
# SPLIT DATA
# ==========================================================

X_train = X[train_mask]
y_train = y[train_mask]

X_test = X[test_mask]
y_test = y[test_mask]

# ==========================================================
# SAVE
# ==========================================================

np.save(f"{OUTPUT_FOLDER}/X_train.npy", X_train)
np.save(f"{OUTPUT_FOLDER}/y_train.npy", y_train)

np.save(f"{OUTPUT_FOLDER}/X_test.npy", X_test)
np.save(f"{OUTPUT_FOLDER}/y_test.npy", y_test)

np.save(f"{OUTPUT_FOLDER}/train_classes.npy", train_classes)
np.save(f"{OUTPUT_FOLDER}/test_classes.npy", test_classes)

np.save(f"{OUTPUT_FOLDER}/labels.npy", labels)

# ==========================================================
# SUMMARY
# ==========================================================

print("\n======================================")
print("Split complete")
print("======================================")

print(f"Training sequences : {len(X_train)}")
print(f"Testing sequences  : {len(X_test)}")

print()

print(f"Training classes : {len(train_classes)}")
print(f"Testing classes  : {len(test_classes)}")

print()

print("Saved to:")
print(OUTPUT_FOLDER)
print("======================================")
