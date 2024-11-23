import struct
import numpy as np
import heapq
import pandas as pd
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input
import json
import os

VECTOR_SIZE = 2048
BINARY_FILE = 'output.bin'
POSITION_DATA_FILE = 'position_data.bin'
URL_CSV_FILE = r'C:\Users\semin\OneDrive\Escritorio\bd2_code\Clonación1\Proyecto_2_BD2\app\Multidimencional\images1.csv'

# Cargar el modelo preentrenado InceptionV3
modelo_inception = InceptionV3(weights='imagenet', include_top=False, pooling='avg')

class knnsecuencial:
    def __init__(self, vector_size=VECTOR_SIZE, binary_file=BINARY_FILE, position_data_file=POSITION_DATA_FILE, url_csv_file=URL_CSV_FILE):
        self.vector_size = vector_size
        self.binary_file = binary_file
        self.position_data_file = position_data_file
        self.url_map = pd.read_csv(url_csv_file)
        self.positions = self.load_positions()

    def load_positions(self):
        positions = []
        with open(self.position_data_file, 'rb') as f:
            while True:
                data = f.read(4)
                if not data:
                    break
                position = struct.unpack('i', data)[0]
                positions.append(position)
        return positions

    def euclidean_distance(self, x, y):
        return np.sqrt(np.sum((x - y) ** 2))

    def get_vector(self, index):
        position = self.positions[index]
        with open(self.binary_file, 'rb') as f:
            f.seek(position)
            data = f.read(4 + 4 * self.vector_size)
            if len(data) < 4 + 4 * self.vector_size:
                return None
            vector = np.array(struct.unpack(f'{self.vector_size}f', data[4:]))
        return vector

    def process_batches(self, query, process_function):
        for index in range(len(self.positions)):
            vector = self.get_vector(index)
            if vector is not None:
                process_function(index, vector, query)

    def knn_search_linear(self, query, k):
        heap = []
        unique_neighbors = set()

        def process_function(index, vector, query):
            distance = self.euclidean_distance(vector, query)
            if index not in unique_neighbors:
                unique_neighbors.add(index)
                heapq.heappush(heap, (-distance, index))
                if len(heap) > k:
                    removed = heapq.heappop(heap)
                    unique_neighbors.remove(removed[1])

        self.process_batches(query, process_function)

        neighbors = []
        for dist, index in sorted(heap, reverse=True):
            if index < len(self.url_map):
                row = self.url_map.iloc[index]
                neighbors.append({
                    "Index": index,
                    "Filename": row['filename'],
                    "Distance": -dist,
                    "Link": row['link']
                })
        return neighbors

    def save_priority_neighbors_to_json(self, query, k, filename="neighbors_priority.json"):
        neighbors = self.knn_search_linear(query, k)
        results = []
        for neighbor in neighbors:
            results.append({
                "Index": neighbor["Index"],
                "Filename": neighbor["Filename"],
                "Distance": neighbor["Distance"],
                "Link": neighbor["Link"],
            })

        # Guardar en archivo JSON
        with open(filename, 'w') as json_file:
            json.dump(results, json_file, indent=4)

        print(f"JSON guardado exitosamente en: {filename}")
        return results

# Función para convertir una imagen a un vector de características
def obtener_vector_desde_imagen(image_path):
    try:
        # Cargar la imagen desde la ruta
        img = load_img(image_path, target_size=(299, 299))

        # Convertir a un array y preprocesar
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Obtener el vector de características
        vector = modelo_inception.predict(img_array).flatten()
        return vector
    except Exception as e:
        print(f"Error al procesar la imagen: {str(e)}")
        return None

if __name__ == "__main__":
    knn = knnsecuencial()

    # Ruta de la imagen de prueba
    test_image_path = r"C:\Users\semin\OneDrive\Escritorio\bd2_code\Clonación1\Proyecto_2_BD2\app\Multidimencional\IMGEN1.jpg"

    # Generar vector de consulta
    query_vector = obtener_vector_desde_imagen(test_image_path)
    if query_vector is None:
        print("Error al generar el vector de características de la imagen.")
    else:
        print("Vector de características generado con éxito.")

        # Guardar el JSON con los vecinos más cercanos
        k = 5
        json_filename = "test_neighbors_priority.json"
        results = knn.save_priority_neighbors_to_json(query_vector, k, filename=json_filename)

        # Verificar que el archivo JSON fue guardado
        if os.path.exists(json_filename):
            print(f"El archivo JSON {json_filename} se ha guardado correctamente.")
        else:
            print(f"Error: El archivo JSON {json_filename} no fue creado.")
