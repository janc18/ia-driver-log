import time
from pathlib import Path
import cv2
import numpy as np
import openvino as ov
import socket
import requests
from datetime import datetime
import random

# Fetch `notebook_utils` module
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

# --- Clasificadores de Haar para la detección de rostros y ojos ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# --- Variables de control ---
eye_closure_count = 0
start_time = time.time()
last_capture_time = time.time()
emotion_lock = False
emotion_lock_start_time = 0
prediccion = "Desconocido"

# Obtener información del dispositivo
hostname = socket.gethostname()
ip_address = socket.gethostbyname(hostname)

# --- Variables para controlar la actualización de la velocidad ---
velocidad = random.uniform(10, 100)  # Inicializamos la velocidad
last_speed_update_time = time.time()  # Guardamos el tiempo de la última actualización
update_interval = 1  # Intervalo de actualización en segundos


last_post_time = time.time()

# Obtener información del dispositivo
hostname = socket.gethostname()
ip_address = socket.gethostbyname(hostname)

# --- Captura de video en tiempo real desde la cámara ---
cap = cv2.VideoCapture("./video.mp4")  # Usamos la cámara (dispositivo 0)

if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

while True:
    # Leer un frame de la cámara
    ret, frame = cap.read()
    if not ret:
        print("No se pudo capturar la imagen")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostros
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

    # Detectar ojos dentro del rostro
    eyes_detected = 0
    for (x, y, w, h) in faces:
        roi_gray = gray_frame[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))

        # Filtrar detecciones de ojos pequeñas
        min_eye_size = 20
        eyes = [eye for eye in eyes if eye[2] > min_eye_size and eye[3] > min_eye_size]

        # Dibujar rectángulo alrededor del rostro
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Si detectamos ojos, actualizamos el conteo
        if len(eyes) > 0:
            eyes_detected = len(eyes)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        # Preprocesar el rostro para el modelo de detección de emociones
        face_blob = cv2.resize(roi_color, (64, 64))  # Ajusta al tamaño que necesita el modelo
        face_blob = np.transpose(face_blob, (2, 0, 1))  # Cambia a formato de canales
        face_blob = np.expand_dims(face_blob, 0)  # Añadir dimensión batch
        face_blob = face_blob.astype(np.float32)

        # Obtener la predicción de la emoción
        result = compiled_model([face_blob])[output_layer]
        emotion_index = np.argmax(result)
        prediccion = emotions[emotion_index]  # Actualizar la predicción de emoción

    # Si no se detectan ojos y un rostro está presente, sumamos al contador
    if len(faces) > 0 and eyes_detected == 0:
        eye_closure_count += 1

        # Verificar si ha pasado suficiente tiempo para actualizar la velocidad
    current_time = time.time()
    if current_time - last_speed_update_time >= update_interval:
        velocidad = random.uniform(90, 95)  # Actualizar la velocidad aleatoria
        last_speed_update_time = current_time  # Actualizar el tiempo de la última actualización

    # Mostrar el flujo de video en una ventana
    cv2.putText(frame, f"Velocidad: {velocidad:.2f} km/h", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Emocion: {prediccion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Distraccion: {eye_closure_count}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    # Mostrar información del dispositivo
    cv2.putText(frame, f"Dispositivo: {hostname} ({ip_address})", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

    # Mostrar el flujo de video con la predicción y la información del dispositivo superpuestos
    cv2.imshow('Camara en vivo - Emocion detectada', frame)

    if time.time() - last_post_time >= 20:
        last_post_time = time.time()
        
        # Obtener la fecha y la hora actuales
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        
        # Hacer el POST con los datos requeridos
        try:
            response = requests.post(
                url="https://fridaplatform.com/generate",
                json={
                    "inputs": f"escribe un resumen de 150 palabras de diagnostico sobre un conductor en carretera segun el estado de animo '{prediccion}' y los pts de distraccion en 30 segundos ('{eye_closure_count}'). Agrega la velocidad a la que iba ('{velocidad:.2f}') en el lugar ('{hostname}'), Ademas, has una muestra de que datos guardarias en una base de datos en mongo con la informacion dada.",
                    "parameters": {"max_new_tokens": 250}
                }
            )
            # Registrar el POST con la fecha y la hora
            print(f"{timestamp}: POST realizado con éxito, respuesta:", response.json())
        except Exception as e:
            # Registrar el POST con la fecha y la hora en caso de error
            print(f"{timestamp}: Error en el POST: {e}")

    # Romper el loop si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()