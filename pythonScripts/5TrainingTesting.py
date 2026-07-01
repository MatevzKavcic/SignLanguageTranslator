import numpy as np
import time

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Conv1D, MaxPooling1D

# =========================
# LOAD DATA (light)
# =========================
print("Loading...")

X = np.load("SecondTryProcesedNPYdataTestingBeforeNormalization/X_aug.npy", allow_pickle=True)
y = np.load("SecondTryProcesedNPYdataTestingBeforeNormalization/y_aug.npy")

print("Loaded:", len(X))


# =========================
# SPLIT BEFORE PAD (IMPORTANT)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

num_classes = len(np.unique(y))
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)


# =========================
# SEQUENCE GENERATOR (KEY FIX)
# =========================

MAX_LEN = max(seq.shape[0] for seq in X)
NUM_FEATURES = X[0].shape[1]

print(MAX_LEN)
print(NUM_FEATURES)


class DataGenerator(Sequence):
    def __init__(self, X, y, batch_size=8):
        self.X = X
        self.y = y
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        batch_X = self.X[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]

        # pad ONLY this batch
        #        batch_X = pad_sequences(batch_X, padding="post", dtype="float32")

        batch_X = pad_sequences(
            batch_X, maxlen=MAX_LEN, padding="post", dtype="float32"
        )

        return np.array(batch_X), np.array(batch_y)


train_gen = DataGenerator(X_train, y_train, batch_size=8)
test_gen = DataGenerator(X_test, y_test, batch_size=8)

# =========================
# MODEL
# =========================
print(X.shape)
print("Building mododel")

model = Sequential()

model.add(
    Conv1D(
        64,
        kernel_size=3,
        activation="relu",
        input_shape=(MAX_LEN, NUM_FEATURES)
    )
)

model.add(MaxPooling1D(2))

model.add(
    Conv1D(
        128,
        kernel_size=3,
        activation="relu"
    )
)

model.add(
    LSTM(
        128
    )
)

model.add(Dropout(0.4))

model.add(Dense(128, activation="relu"))
model.add(Dense(num_classes, activation="softmax"))

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)









# =========================
# CALLBACKS
# =========================
early_stop = EarlyStopping(patience=100, restore_best_weights=True)

checkpoint = ModelCheckpoint(
    "SecondTry_noNorm_sign_language_CNN_lstm.h5",
    save_best_only=True,
    monitor="val_accuracy"
)


# =========================
# TRAIN
# =========================
print("Training...")

history = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=100,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

print("DONE")
