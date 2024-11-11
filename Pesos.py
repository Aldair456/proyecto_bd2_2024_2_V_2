import pandas as pd
import numpy as np
import nltk
from nltk.stem import SnowballStemmer
from collections import defaultdict

# Asegúrate de tener los paquetes necesarios de NLTK
nltk.download('punkt')

def calcular_pesos_campos(ruta_csv, ruta_stoplist, ruta_pesos):
    stemmer = SnowballStemmer('spanish')
    stopwords = set()
    
    # Cargar stopwords
    try:
        with open(ruta_stoplist, 'r', encoding='utf-8') as archivo:
            for linea in archivo:
                palabra = linea.strip().lower()
                if palabra:
                    stopwords.add(palabra)
    except FileNotFoundError:
        print(f"Archivo de stopwords no encontrado en {ruta_stoplist}")
    
    # Añadir caracteres especiales a las stopwords
    caracteres_especiales = set("'«[]¿?$+-*'.,»:;!,º«»()@¡😆“/#|*%'&`")
    stopwords.update(caracteres_especiales)
    
    # Variables para el cálculo
    total_campos = None
    terminos_por_campo = []
    total_documentos = 0
    TAMANIO_CHUNK = 10000  # Ajusta según tus recursos
    entropias = []
    pesos_campos = []
    
    try:
        for chunk in pd.read_csv(ruta_csv, chunksize=TAMANIO_CHUNK, encoding='utf-8'):
            if total_campos is None:
                total_campos = len(chunk.columns)
                terminos_por_campo = [defaultdict(int) for _ in range(total_campos)]
            for _, fila in chunk.iterrows():
                total_documentos += 1
                for idx_campo, campo in enumerate(fila):
                    tokens = nltk.word_tokenize(str(campo).lower())
                    tokens = [token.strip() for token in tokens]
                    for token in tokens:
                        if token not in stopwords:
                            lematizado = stemmer.stem(token)
                            terminos_por_campo[idx_campo][lematizado] += 1
            if total_documentos >= 5000:  
                break
    except Exception as e:
        print(f"Error al leer el archivo CSV para calcular pesos: {e}")
        return
    
    # Calcular entropía para cada campo
    for idx_campo in range(total_campos):
        frecuencias = np.array(list(terminos_por_campo[idx_campo].values()))
        if frecuencias.sum() == 0:
            entropias.append(0)
            continue
        probabilidades = frecuencias / frecuencias.sum()
        entropia = -np.sum(probabilidades * np.log2(probabilidades + 1e-9))
        entropias.append(entropia)
    
    # Normalizar las entropías para que sumen 1 y asignar pesos
    suma_entropias = sum(entropias)
    if suma_entropias == 0:
        pesos_campos = [0 for _ in entropias]
    else:
        pesos_campos = [ent / suma_entropias for ent in entropias]
    
    print("Pesos de campos calculados:", pesos_campos)
    
    # Guardar los pesos en un archivo JSON
    import json
    with open(ruta_pesos, 'w', encoding='utf-8') as archivo:
        json.dump(pesos_campos, archivo)
    print(f"Pesos guardados en {ruta_pesos}")

# Ejemplo de uso:
if __name__ == "__main__":
    ruta_csv = r"C:\Users\semin\BD2\spotify_songs.csv"
    ruta_stoplist = r"C:\Users\semin\BD2\stopwords_personalizadas.csv"
    ruta_pesos = r"C:\Users\semin\BD2\pesos_campos.json"
    
    calcular_pesos_campos(ruta_csv, ruta_stoplist, ruta_pesos)
