import time
from pathlib import Path
import cv2
import numpy as np
import openvino as ov
import socket
import requests

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

# --- Captura de video en tiempo real desde la cámara ---
cap = cv2.VideoCapture(0)  # Usamos la cámara (dispositivo 0)

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
        # Mostrar coordenadas del rostro
        cv2.putText(frame, f"Face: ({x}, {y})", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

        # Si detectamos ojos, actualizamos el conteo
        if len(eyes) > 0:
            eyes_detected = len(eyes)
            for (ex, ey, ew, eh) in eyes:
                # Dibujar rectángulo alrededor de los ojos
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                # Mostrar coordenadas de los ojos
                cv2.putText(frame, f"Eye: ({x+ex}, {y+ey})", (x+ex, y+ey-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    # Si no se detectan ojos y un rostro está presente, sumamos al contador
    if len(faces) > 0 and eyes_detected == 0:
        eye_closure_count += 1

    # Si pasan 60 segundos, reseteamos el contador de cierres de ojos
    if time.time() - start_time >= 60:
        if eye_closure_count > 5:
            prediccion = "adormilado"
            emotion_lock = True
            emotion_lock_start_time = time.time()
        eye_closure_count = 0
        start_time = time.time()

    # Mostrar el flujo de video en una ventana
    cv2.putText(frame, f"Ojos cerrados: {eye_closure_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    # Si no hay bloqueo de emoción, detectamos emociones
    if not emotion_lock:
        # Capturar y procesar cada 3 segundos
        current_time = time.time()
        if current_time - last_capture_time >= 3:
            last_capture_time = current_time

            # Preprocesar la imagen capturada
            input_image = cv2.resize(src=gray_frame, dsize=(64, 64))  # Tamaño requerido por el modelo de emociones

            # Repetimos el canal para que sea de 3 canales (RGB falso)
            input_image = np.stack([input_image] * 3, axis=-1)  # Forma (64, 64, 3)

            # Cambiamos la forma a la que espera el modelo: (1, 3, 64, 64)
            input_image = np.transpose(input_image, (2, 0, 1))  # De (64, 64, 3) a (3, 64, 64)
            input_image = np.expand_dims(input_image, 0)  # De (3, 64, 64) a (1, 3, 64, 64)
            input_image = input_image.astype(np.float32)

            # Ejecutar inferencia con el modelo de emociones
            result_infer = compiled_model([input_image])[output_layer]

            # Obtener la emoción con mayor probabilidad
            emotion_index = np.argmax(result_infer)
            prediccion = emotions[emotion_index]

    # Si la emoción está bloqueada por "adormilado", la mantenemos durante 1 minuto
    if emotion_lock and time.time() - emotion_lock_start_time >= 60:    
        emotion_lock = False  # Desbloqueamos la emoción después de 1 minuto

    # Mostrar la predicción en el cuadro de la cámara en vivo
    cv2.putText(frame, f"Emocion: {prediccion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv2.LINE_AA)

    # Mostrar información del dispositivo
    cv2.putText(frame, f"Dispositivo: {hostname} ({ip_address})", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

    # Mostrar el flujo de video con la predicción y la información del dispositivo superpuestos
    cv2.imshow('Camara en vivo - Emocion detectada', frame)

    # Romper el loop si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows() 
