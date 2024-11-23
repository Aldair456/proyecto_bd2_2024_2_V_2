import requests
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np
import gc
import struct
import pandas as pd
import os
from tempfile import NamedTemporaryFile

# Inicializa el modelo InceptionV3
modelo = InceptionV3(weights='imagenet', include_top=False, pooling='avg')

# Configuración
EXPECTED_LENGTH_DATA = 2048
INTEGER_BYTES = 4
FLOAT_BYTES = 4
N_FEATURES = 9

# Archivos
images_csv = r"C:\Users\semin\OneDrive\Escritorio\bd2_code\Clonación1\Proyecto_2_BD2\app\Multidimencional\images1.csv"
output_file = "output.bin"
position_data_file = "position_data.bin"
features_csv = r"C:\Users\semin\OneDrive\Escritorio\bd2_code\Clonación1\Proyecto_2_BD2\app\Multidimencional\styles1.csv"
position_feature_file = "position_feature.bin"
id_to_pos_file = "id_to_pos.bin"

id_to_pos = {}

def extract_features_url(image_link):
    try:
        response = requests.get(image_link, stream=True)
        if response.status_code != 200:
            print(f"Error al acceder a la URL {image_link}: {response.status_code}")
            return None

        with NamedTemporaryFile(delete=False) as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            temp_file_path = temp_file.name

        img = load_img(temp_file_path, target_size=(299, 299))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = modelo.predict(img_array).flatten()

        os.remove(temp_file_path)

        del img_array
        gc.collect()
        return features
    except Exception as e:
        print(f"Error al procesar la imagen {image_link}: {str(e)}")
        return None

def load_images(images_csv, output_file, position_data_file, n=None):
    ids = []
    id_to_pos = {}

    if os.path.exists(output_file) and os.path.exists(position_data_file):
        print("Los archivos binarios ya existen. No se volverán a extraer las características.")
        # Leer el archivo de posiciones para llenar id_to_pos
        with open(position_data_file, 'rb') as pos_file:
            i = 0
            while True:
                data = pos_file.read(4)
                if not data:
                    break
                position = struct.unpack('i', data)[0]
                id_to_pos[i] = position
                ids.append([i, position])
                i += 1
        return ids, id_to_pos

    with open(output_file, 'wb') as file:
        position_data = open(position_data_file, "wb")
        df_images = pd.read_csv(images_csv)
        i = 0
        for _, row in df_images.iterrows():
            if n and i >= n:
                break
            image_link = row['link']
            image_name = row['filename']
            print(f"Procesando: {image_name}, {image_link}")

            try:
                position_seek = file.tell()
                position_data.write(struct.pack('i', position_seek))

                img_encoding = extract_features_url(image_link)
                if img_encoding is None or len(img_encoding) != EXPECTED_LENGTH_DATA:
                    print(f"Error en la extracción para la imagen {image_name}")
                    continue


                id = int(image_name.split('.')[0])
                ids.append([id, position_seek])
                id_to_pos[id] = position_seek
                data = struct.pack('i' + 'f' * len(img_encoding), id, *img_encoding)
                file.write(data)
                i += 1

            except Exception as e:
                print(f"Error al procesar la imagen {image_name}: {str(e)}")
                continue

        position_data.close()

    if not ids:
        print("No se procesó ninguna imagen correctamente. Revisa los archivos de entrada.")
    else:
        print(f"Última imagen procesada: {ids[-1]}")
    print(f"Diccionario ID a Posición: {id_to_pos}")
    return ids, id_to_pos

def get_vector(output_file, position_data_file, n):
    try:
        with open(position_data_file, "rb") as pos_file:
            pos_file.seek(n * INTEGER_BYTES)
            position_in_file = int.from_bytes(pos_file.read(INTEGER_BYTES), byteorder='little')
        with open(output_file, "rb") as data_file:
            data_file.seek(position_in_file)
            data_bin = data_file.read(INTEGER_BYTES + EXPECTED_LENGTH_DATA * FLOAT_BYTES)
        return list(struct.unpack('i' + 'f'*EXPECTED_LENGTH_DATA, data_bin))[1:]
    except Exception as e:
        print(f"Error al obtener el vector de características: {str(e)}")
        return None

def get_pos_to_id(id, id_to_pos):
    pos = id_to_pos.get(id, -1)
    if pos == -1:
        print(f"ID {id} no encontrado en el diccionario.")
    return pos

def load_features():
    with open(features_csv, "r") as file:
        head = file.readline().split(',')
        head.pop()
        n_features = len(head)
        with open(position_feature_file, "wb") as pos_file:
            assert n_features == N_FEATURES, "INVALID DATA"
            while True:
                position = file.tell()
                line = file.readline()
                if not line:
                    break
                pos_file.write(struct.pack('i', position))

def get_feature(n: int, feature: int = -1):
    position_file = open(position_feature_file, 'rb')
    position_file.seek(n * INTEGER_BYTES)
    pos_in_file = int.from_bytes(position_file.read(INTEGER_BYTES), byteorder='little')
    position_file.close()
    feature_file = open(features_csv, "r")
    feature_file.seek(pos_in_file)
    data_feature = feature_file.readline().split(',')
    feature_file.close()
    if feature == -1:
        return data_feature[:N_FEATURES]
    else:
        assert feature >= 0 and feature <= N_FEATURES, "INVALID FEATURE"
        if feature == 0:
            print("FEATURE IS ID")
        return data_feature[feature]

def get_data_images(elements: list[int]):
    file = open(images_csv, "r")
    d = {}
    for i in range(len(elements)):
        d[elements[i]] = i
    data = [[]] * len(elements)
    for i in range(max(elements) + 1):
        cur = file.readline()
        if i in elements:
            cur = cur[:-1]
            cur = cur.split(',')
            cur.append(get_feature(i, N_FEATURES)[:-1])
            data[d[i]] = cur
    return data


n = None  # Cambia a un número si deseas limitar la cantidad de imágenes procesadas

# Carga de imágenes
ids, id_to_pos = load_images(images_csv, output_file, position_data_file, n)

# Carga de características
load_features()

# Ejemplo: Obtener vector y posición de un ID
if ids:
    ejemplo_id = ids[0][0]  # Toma el primer ID procesado
    pos = get_pos_to_id(ejemplo_id, id_to_pos)
    vector = get_vector(output_file, position_data_file, pos)
    print(f"Vector de características para el ID {ejemplo_id}: {vector}")

# Ejemplo: Obtener datos de imágenes
data_images = get_data_images([ejemplo_id])
print(f"Datos de la imagen para el ID {ejemplo_id}: {data_images}")