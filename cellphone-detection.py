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

model_name = "v3-small_224_1.0_float"
model_xml_name = f"{model_name}.xml"
model_bin_name = f"{model_name}.bin"

model_xml_path = base_artifacts_dir / model_xml_name

base_url = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/mobelinet-v3-tf/FP32/"

if not model_xml_path.exists():
    download_file(base_url + model_xml_name, model_xml_name, base_artifacts_dir)
    download_file(base_url + model_bin_name, model_bin_name, base_artifacts_dir)
else:
    print(f"{model_name} already downloaded to {base_artifacts_dir}")

core = ov.Core()
model = core.read_model(model=model_xml_path)
compiled_model = core.compile_model(model=model, device_name="CPU")

output_layer = compiled_model.output(0)

# Descargar el archivo de clases de ImageNet
imagenet_filename = download_file(
    "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/datasets/imagenet/imagenet_2012.txt",
    directory="data",
)

imagenet_classes = imagenet_filename.read_text().splitlines()
imagenet_classes = ["background"] + imagenet_classes  # Agregar "background" como la clase 0

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
    if current_time - last_capture_time >= 0:
        last_capture_time = current_time
        
        # Preprocesar la imagen capturada
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_image = cv2.resize(src=image, dsize=(224, 224))
        input_image = np.expand_dims(input_image, 0)

        # Ejecutar inferencia con el modelo
        result_infer = compiled_model([input_image])[output_layer]
        result_index = np.argmax(result_infer)

        # Mostrar la clase predicha
        prediccion = imagenet_classes[result_index]
        print(f"Clase predicha: {prediccion}")
        if (result_index==488):            
            cv2.putText(frame, f"Prediccion: {prediccion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        # Mostrar la imagen y la predicción en una ventana separada
        cv2.imshow('Imagen capturada', frame)

    # Romper el loop si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
