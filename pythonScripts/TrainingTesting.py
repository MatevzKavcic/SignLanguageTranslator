import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =========================
# LOAD DATA
# =========================
X = np.load("procesedNPYdata/X_aug.npy", allow_pickle=True)
y = np.load("procesedNPYdata/y_aug.npy")

print("Raw X shape:", X.shape)

# =========================
# FIX SHAPE (IMPORTANT)
# =========================
X = pad_sequences(X, padding="post", dtype="float32")

print("Fixed X shape:", X.shape)

# =========================
# ONE HOT ENCODING
# =========================
num_classes = len(np.unique(y))
y_cat = to_categorical(y, num_classes=num_classes)

# =========================
# SPLIT (NO STRATIFY)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_cat,
    test_size=0.2,
    random_state=42
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# =========================
# MODEL
# =========================
model = Sequential()

model.add(LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.3))

model.add(LSTM(64))
model.add(Dropout(0.3))

model.add(Dense(64, activation="relu"))
model.add(Dense(num_classes, activation="softmax"))

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# =========================
# TRAINING
# =========================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=16,
    callbacks=[early_stop]
)

# =========================
# SAVE
# =========================
model.save("sign_language_lstm.h5")

print("DONE")