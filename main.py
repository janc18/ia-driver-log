import time
from pathlib import Path
import cv2
import numpy as np
import openvino as ov
import requests

# Fetch `notebook_utils` module
r = requests.get(
    url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)
open("notebook_utils.py", "w").write(r.text)

from notebook_utils import download_file

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
last_post_time = time.time()  # Control para el POST cada 30 segundos
emotion_lock = False
emotion_lock_start_time = 0
prediccion = "Desconocido"

# --- Modelos de detección de objetos ---
object_model_name = "mobilenet-ssd"
object_model_xml_name = f"{object_model_name}.xml"
object_model_bin_name = f"{object_model_name}.bin"

object_model_xml_path = base_artifacts_dir / object_model_xml_name

object_base_url = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.4/models_bin/1/mobilenet-ssd/FP32/"

if not object_model_xml_path.exists():
    download_file(object_base_url + object_model_xml_name, object_model_xml_name, base_artifacts_dir)
    download_file(object_base_url + object_model_bin_name, object_model_bin_name, base_artifacts_dir)
else:
    print(f"{object_model_name} ya descargado en {base_artifacts_dir}")

# Cargar el modelo de detección de objetos
object_model = core.read_model(model=object_model_xml_path)
compiled_object_model = core.compile_model(model=object_model, device_name="CPU")

object_output_layer = compiled_object_model.output(0)

# Definir las clases de objetos para MobileNet SSD
object_classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Función para detectar objetos
def detect_objects(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (0, 0, 0), swapRB=True, crop=False)
    compiled_object_model.set_input(blob)
    detections = compiled_object_model.output()
    
    detected_objects = []
    for detection in detections[0, 0]:
        score = detection[2]
        if score > 0.5:  # Umbral de confianza
            class_id = int(detection[1])
            if class_id in [1, 8, 14]:  # IDs para televisión, laptop y celular
                detected_objects.append(object_classes[class_id])
    return detected_objects

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
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) > 0:
            eyes_detected = len(eyes)
    if len(faces) > 0 and eyes_detected == 0:
        eye_closure_count += 1

    # Detección de objetos
    detected_objects = detect_objects(frame)

    # Reseteo del contador de cierres de ojos y detección de emociones
    if time.time() - start_time >= 60:
        if eye_closure_count > 5:
            prediccion = "adormilado"
            emotion_lock = True
            emotion_lock_start_time = time.time()
        eye_closure_count = 0
        start_time = time.time()

    # Mostrar la información en la ventana de video
    cv2.putText(frame, f"Ojos cerrados: {eye_closure_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Emocion: {prediccion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Objetos detectados: {', '.join(detected_objects)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('Camara en vivo - Emocion y objetos detectados', frame)

    # Enviar solicitud POST cada 30 segundos
    if time.time() - last_post_time >= 30:
        last_post_time = time.time()
        try:
            response = requests.post(
                url="https://fridaplatform.com/generate",
                json={
                    "inputs": f"Escribe un resumen de 150 palabras de diagnóstico sobre un conductor en carretera según el estado de ánimo '{prediccion}', las veces que ha cerrado los ojos en 30 segundos ('{eye_closure_count}'), y los objetos detectados: {', '.join(detected_objects)}. Además, muestra qué datos guardarías en una base de datos en MongoDB con la información dada.",
                    "parameters": {"max_new_tokens": 250}
                }
            )
            print("POST realizado con éxito, respuesta:", response.json())
        except Exception as e:
            print(f"Error en el POST: {e}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
