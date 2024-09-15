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

# --- Captura de imagen desde la cámara ---
cap = cv2.VideoCapture(0)  # Captura de video desde la cámara (dispositivo 0)

if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

# Tomar una foto
ret, frame = cap.read()
if not ret:
    print("No se pudo capturar la imagen")
    cap.release()
    exit()

# Mostrar la imagen capturada
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.title("Imagen capturada")
plt.show()

# Cerrar la cámara
cap.release()

# Preprocesamiento de la imagen
# Convertir a RGB si es necesario
image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Redimensionar la imagen a las dimensiones requeridas por MobileNet (224x224)
input_image = cv2.resize(src=image, dsize=(224, 224))

# Expandir la forma de la imagen para que coincida con la entrada del modelo
input_image = np.expand_dims(input_image, 0)

# Ejecutar la inferencia con el modelo
result_infer = compiled_model([input_image])[output_layer]

# Obtener el índice de la clase con mayor probabilidad
result_index = np.argmax(result_infer)

# Descargar el archivo de clases de ImageNet
imagenet_filename = download_file(
    "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/datasets/imagenet/imagenet_2012.txt",
    directory="data",
)

# Leer las clases de ImageNet
imagenet_classes = imagenet_filename.read_text().splitlines()
imagenet_classes = ["background"] + imagenet_classes  # Agregar "background" como la clase 0

# Mostrar la clase predicha
print(f"Clase predicha: {imagenet_classes[result_index]}")
