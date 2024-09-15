import time
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import openvino as ov

# Fetch `notebook_utils` module
import requests
r = requests.get(
    url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)

open("notebook_utils.py", "w").write(r.text)

from notebook_utils import download_file, device_widget

base_artifacts_dir = Path("./artifacts").expanduser()

# --- Cambiar el nombre del modelo por uno de detección de emociones ---
model_name = "emotions-recognition-retail-0003"
model_xml_name = f"{model_name}.xml"
model_bin_name = f"{model_name}.bin"

model_xml_path = base_artifacts_dir / model_xml_name

base_url = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.4/models_bin/1/emotions-recognition-retail-0003/FP32/"

if not model_xml_path.exists():
    download_file(base_url + model_xml_name, model_xml_name, base_artifacts_dir)
    download_file(base_url + model_bin_name, model_bin_name, base_artifacts_dir)
else:
    print(f"{model_name} ya descargado en {base_artifacts_dir}")

core = ov.Core()
model = core.read_model(model=model_xml_path)
compiled_model = core.compile_model(model=model, device_name="CPU")

output_layer = compiled_model.output(0)

# Las clases de emociones que detecta el modelo
emotions = ["neutral", "happy", "sad", "surprise", "anger"]

# --- Captura de video en tiempo real desde la cámara ---
cap = cv2.VideoCapture(0)  # Usamos la cámara (dispositivo 0)

if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

# Mostrar el flujo de video en tiempo real y capturar una imagen cada 3 segundos
last_capture_time = time.time()

while True:
    # Leer un frame de la cámara
    ret, frame = cap.read()
    if not ret:
        print("No se pudo capturar la imagen")
        break

    # Mostrar el flujo de video en una ventana
    cv2.imshow('Camara en vivo', frame)

    # Capturar y procesar cada 3 segundos
    current_time = time.time()
    if current_time - last_capture_time >= 3:
        last_capture_time = current_time
        
        # Preprocesar la imagen capturada
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        input_image = cv2.resize(src=gray_frame, dsize=(64, 64))  # Tamaño requerido por el modelo de emociones
        input_image = input_image.reshape(1, 1, 64, 64)  # La forma esperada es 1x1x64x64
        input_image = input_image.astype(np.float32)

        # Ejecutar inferencia con el modelo de emociones
        result_infer = compiled_model([input_image])[output_layer]
        
        # Obtener la emoción con mayor probabilidad
        emotion_index = np.argmax(result_infer)
        prediccion = emotions[emotion_index]
        
        # Mostrar la clase predicha
        print(f"Emoción detectada: {prediccion}")

        # Mostrar la imagen y la predicción en una ventana separada
        cv2.putText(frame, f"Emocion: {prediccion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Imagen capturada', frame)

    # Romper el loop si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
