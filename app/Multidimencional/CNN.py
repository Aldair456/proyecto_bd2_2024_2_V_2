import requests
from io import BytesIO
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np
import pandas as pd
import struct
import gc  


modelo = InceptionV3(weights='imagenet', include_top=False, pooling='avg')

def map_filenames_to_indices(file_path):
    data = pd.read_csv(file_path)
    filename_to_index = {row['filename']: idx for idx, row in data.iterrows()}
    for filename, index in filename_to_index.items():
        print(f"{filename} -> {index}")
    
    return filename_to_index

def extract_features_url(url, index, output_bin):
    try:
        response = requests.get(url, timeout=10)  
        if response.status_code == 200:
        
            img = image.load_img(BytesIO(response.content), target_size=(299, 299))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            features = modelo.predict(img_array).flatten()
            
            print(f"\nÍndice: {index}")
            print(f"Vector de características (primeros 10 elementos): {features[:10]}")
            print(f"Tamaño del vector: {len(features)}")

            with open(output_bin, 'ab') as f:
                f.write(struct.pack(f'i2048f', index, *features))

    
            del img_array, features
            gc.collect()  

            print(f"Características extraídas y guardadas para la imagen en el índice {index}")
        else:
            print(f"Error al descargar la imagen en el índice {index}: Status code {response.status_code}")
    except Exception as e:
        print(f"Error al procesar la imagen en el índice {index}: {str(e)}")

def process_csv(file_path, output_bin):
    data = pd.read_csv(file_path)
    
    print(f"Total de filas en el archivo CSV: {len(data)}")
    
    for idx, row in data.iterrows():
        url = row['link'] 
        try:
            extract_features_url(url, idx, output_bin) 
        except Exception as e:
            print(f"Error procesando la imagen en el índice {idx}: {str(e)}")

csv_file = 'images1.csv' 
output_bin = 'vectors.bin'  

filename_to_index = map_filenames_to_indices(csv_file)
process_csv(csv_file, output_bin)
print("\nExtracción de características completada.")
