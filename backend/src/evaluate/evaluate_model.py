import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Cargar el modelo entrenado
modelo = tf.keras.models.load_model("../training/cnn_rnn_deepfake_detector.keras")

# Cargar los datos preprocesados de evaluación
data = np.load("../data/preprocessed_data_eval.npz")
X_val, y_val = data["X_val"], data["y_val"]

# Evaluar el modelo en el conjunto de validación
loss, accuracy = modelo.evaluate(X_val, y_val, verbose=1)
print(f"\nPérdida en validación: {loss:.4f}")
print(f"Precisión en validación: {accuracy:.4f}")

# Generar predicciones
predicciones = modelo.predict(X_val)

# Convertir probabilidades en clases binarias (0 o 1)
predicciones_binarias = (predicciones > 0.5).astype(int)

# Reporte de clasificación
print("\nReporte de clasificación:")
print(classification_report(y_val, predicciones_binarias))

# Matriz de confusión
print("\nMatriz de confusión:")
print(confusion_matrix(y_val, predicciones_binarias))

# Visualizar la matriz de confusión
plt.figure(figsize=(6, 6))
plt.imshow(confusion_matrix(y_val, predicciones_binarias), cmap='Blues')
plt.title("Matriz de confusión")
plt.xlabel("Predicción")
plt.ylabel("Etiqueta real")
plt.colorbar()
plt.show()

# Graficar la precisión durante el entrenamiento
history = np.load("history_logs.npz", allow_pickle=True)["history"][()]

plt.plot(history['accuracy'], label='Precisión de entrenamiento')
plt.plot(history['val_accuracy'], label='Precisión de validación')
plt.xlabel('Epocas')
plt.ylabel('Precisión')
plt.legend()
plt.title('Precisión durante el entrenamiento')
plt.show()

# Graficar la pérdida durante el entrenamiento
plt.plot(history['loss'], label='Pérdida de entrenamiento')
plt.plot(history['val_loss'], label='Pérdida de validación')
plt.xlabel('Epocas')
plt.ylabel('Pérdida')
plt.legend()
plt.title('Pérdida durante el entrenamiento')
plt.show()