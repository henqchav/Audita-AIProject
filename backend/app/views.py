import os
import librosa
import numpy as np
from django.http import JsonResponse
from django.conf import settings
from tensorflow.keras.models import load_model
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

# Cargar el modelo entrenado
MODEL_PATH = os.path.join(settings.BASE_DIR, 'app', 'models', 'cnn_rnn_deepfake_detectorV3.keras')
model = load_model(MODEL_PATH)

# Preprocesamiento del audio
def preprocess_audio(audio_path, target_shape=(128, 128)):
    y, sr = librosa.load(audio_path, sr=16000)
    y = librosa.util.fix_length(y, size=sr * 4)  # Ajustar a 4 segundos
    y = librosa.effects.preemphasis(y)
    spectrogram = librosa.stft(y, n_fft=512, hop_length=256)
    spectrogram = np.abs(spectrogram)
    spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)
    spectrogram = np.resize(spectrogram, target_shape)
    return spectrogram

# API para procesar el archivo de audio
@method_decorator(csrf_exempt, name='dispatch')
class AudioPredictionView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request, *args, **kwargs):
        # Verificar si se subió un archivo
        if 'audio' not in request.FILES:
            return JsonResponse({"error": "No se recibió ningún archivo"}, status=400)

        audio_file = request.FILES['audio']

        # Guardar temporalmente el archivo subido
        temp_path = "temp_audio.wav"
        with open(temp_path, 'wb') as f:
            for chunk in audio_file.chunks():
                f.write(chunk)

        # Preprocesar el audio
        spectrogram = preprocess_audio(temp_path)
        spectrogram = np.expand_dims(spectrogram, axis=0)  # Añadir dimensión de batch
        spectrogram = np.expand_dims(spectrogram, axis=-1)  # Añadir canal

        # Realizar la predicción
        prediction = model.predict(spectrogram)[0][0]
        result = 1 if prediction > 0.5 else 0  # 1: bonafide, 0: spoof

        # Eliminar el archivo temporal
        os.remove(temp_path)

        response = JsonResponse({"result": result})
        response["Access-Control-Allow-Origin"] = "*"
        return response
