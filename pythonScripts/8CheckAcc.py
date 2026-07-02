import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# =====================================
# CALLBACKS
# =====================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=100,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    "best_siamese.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

# =====================================
# TRAIN MODEL
# =====================================

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=1000,
    callbacks=[early_stop, checkpoint]
)

# =====================================
# SAVE TRAINING HISTORY
# =====================================

with open("training_history.json", "w") as f:
    json.dump(history.history, f, indent=4)

print("\nTraining history saved to training_history.json")

# =====================================
# PRINT STATISTICS
# =====================================

print("\n========================================")
print("TRAINING STATISTICS")
print("========================================")

for metric in history.history:

    values = history.history[metric]

    print(f"\nMetric: {metric}")
    print(f"Epochs          : {len(values)}")
    print(f"First value     : {values[0]:.6f}")
    print(f"Last value      : {values[-1]:.6f}")

    if "loss" in metric:
        print(f"Best (minimum)  : {min(values):.6f}")
        print(f"Worst (maximum) : {max(values):.6f}")
        print(f"Best epoch      : {np.argmin(values)+1}")

    else:
        print(f"Best (maximum)  : {max(values):.6f}")
        print(f"Worst (minimum) : {min(values):.6f}")
        print(f"Best epoch      : {np.argmax(values)+1}")

print("\n========================================")

# =====================================
# PLOT LOSSES
# =====================================

if "loss" in history.history:

    plt.figure(figsize=(10,5))

    plt.plot(history.history["loss"], label="Training Loss")

    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Validation Loss")

    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.show()

# =====================================
# PLOT ACCURACY
# =====================================

if "accuracy" in history.history:

    plt.figure(figsize=(10,5))

    plt.plot(history.history["accuracy"], label="Training Accuracy")

    if "val_accuracy" in history.history:
        plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.show()

print("\nDone.")
