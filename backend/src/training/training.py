import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM, TimeDistributed, Bidirectional
)
from tensorflow.keras.layers import BatchNormalization, Activation, Reshape
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import datetime

# Definir la arquitectura CNN + RNN
def crear_modelo_cnn_rnn(input_shape):
    inputs = Input(shape=input_shape)

    # Capa CNN
    x = Conv2D(16, kernel_size=(3, 3), padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(32, kernel_size=(3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Reorganizar la salida para la RNN
    x = Reshape((x.shape[1], -1))(x)

    # Capa RNN (LSTM bidireccional)
    x = Bidirectional(LSTM(32, return_sequences=False))(x)

    # Capa densa final
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.7)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# Crear el modelo
input_shape = (128, 128, 1)  # Espectrogramas de entrada con un canal
modelo = crear_modelo_cnn_rnn(input_shape)
modelo.summary()

# Cargar los datos preprocesados
data = np.load("../data/preprocessed_data.npz")
X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_val"], data["y_val"]

# Callbacks para guardar el mejor modelo, detener temprano y usar TensorBoard
checkpoint = ModelCheckpoint("cnn_rnn_deepfake_detectorV3.keras", save_best_only=True, monitor="val_accuracy", mode="max")
early_stopping = EarlyStopping(patience=5, monitor="val_accuracy", mode="max", restore_best_weights=True)

# Configurar TensorBoard
log_dir = f"logs/fit/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Entrenar el modelo
history = modelo.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=32,
    epochs=50,
    callbacks=[checkpoint, early_stopping, tensorboard_callback],
    verbose=1
)

