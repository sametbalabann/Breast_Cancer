import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from utils import create_generators

train_dir = "data/train_small"
img_size = (50, 50)
batch_size = 32

train_gen, val_gen = create_generators(train_dir, img_size, batch_size)

model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(50, 50, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
checkpoint = ModelCheckpoint("models/cnn_model.h5", save_best_only=True, monitor="val_accuracy", mode="max")

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=[checkpoint]
)