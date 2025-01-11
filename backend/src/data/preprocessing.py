import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Ruta de los archivo
#audio_dir = "train_dev/" # preprocesamiento de datos para training
audio_dir = "eval/"
protocol_path = "protocol.txt"
output_shape = (128, 128)

# Leer el archivo de protocolo
protocol = pd.read_csv(protocol_path, sep=" ", header=None,
                       names=["Subject_id", "file_name", "dash", "spoof_type", "Label"])




# Filtrar entradas válidas
valid_entries = []
for _, row in protocol.iterrows():
    file_path = os.path.join(audio_dir, row["file_name"] + ".wav")

    if os.path.exists(file_path):
        valid_entries.append(row)


# Preprocesamiento
def preprocess_audio(file_path, target_shape):
    y, sr = librosa.load(file_path, sr=None)
    y = librosa.util.fix_length(y, size=sr * 4)  # Ajustar a 4 segundos
    y = librosa.effects.preemphasis(y)
    spectrogram = librosa.stft(y, n_fft=512, hop_length=256)
    spectrogram = np.abs(spectrogram)
    spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)
    spectrogram = np.resize(spectrogram, target_shape)
    return spectrogram

# Preparar datos
X, y = [], []
for row in valid_entries:
    file_path = os.path.join(audio_dir, row["file_name"] + ".wav")
    label = 1 if row["Label"] == "bonafide" else 0
    spectrogram = preprocess_audio(file_path, output_shape)
    X.append(spectrogram)
    y.append(label)

X = np.array(X)[..., np.newaxis]  # Añadir canal
y = np.array(y)


### LA PARTE DEL CODIGO PARA PROCESAR LOS DE DATOS DE TRAINING Y VALIDACIÓN

# # División en entrenamiento y validación
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# # Guardar datos preprocesados
# np.savez("preprocessed_data.npz", X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

### PARTE PARA SOLO PREPROCESAR LOS DE TESTING

np.savez("preprocessed_data_eval.npz", X_val=X, y_val=y)