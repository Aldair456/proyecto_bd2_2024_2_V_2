import pandas as pd
import nltk
from collections import Counter, defaultdict
import numpy as np
import csv

nltk.download('punkt')
nltk.download('stopwords')

def generar_stopwords_desde_csv(ruta_csv_entrada, ruta_csv_stopwords_salida):
    df = pd.read_csv(ruta_csv_entrada)
    
    # Paso 2: Combinar el texto de las columnas relevantes
    # Ajusta esta lista según las columnas de texto en tu CSV
    columnas_texto = ['track_name', 'track_artist', 'lyrics', 'playlist_name', 'playlist_genre', 'playlist_subgenre']
    df['texto_combinado'] = df[columnas_texto].astype(str).apply(' '.join, axis=1)
    
    # Paso 3: Tokenizar el texto y contar frecuencias de palabras
    word_freq = Counter()
    word_doc_freq = defaultdict(set)
    
    for idx, texto in df['texto_combinado'].items():
        tokens = nltk.word_tokenize(texto.lower())
        word_freq.update(tokens)
        for token in set(tokens):
            word_doc_freq[token].add(idx)
    
    # Paso 4: Calcular la entropía de las palabras
    total_docs = len(df)
    word_entropy = {}
    for word, doc_set in word_doc_freq.items():
        p = len(doc_set) / total_docs
        if 0 < p < 1:
            entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
            word_entropy[word] = entropy
        else:
            word_entropy[word] = 0  # Palabras que aparecen en todos o en ningún documento
    
    # Paso 5: Generar la lista de stopwords
    # Seleccionar palabras con alta frecuencia
    numero_palabras_comunes = 200  # Puedes ajustar este número
    palabras_mas_comunes = [word for word, freq in word_freq.most_common(numero_palabras_comunes)]
    
    # Seleccionar palabras con baja entropía
    umbral_entropia = 0.6  
    palabras_baja_entropia = [word for word, ent in word_entropy.items() if ent < umbral_entropia]
    
    stopwords_personalizadas = set(palabras_baja_entropia)
    
    from nltk.corpus import stopwords
    stopwords_ingles = set(stopwords.words('english'))
    stopwords_espanol = set(stopwords.words('spanish'))
    stopwords_personalizadas.update(stopwords_ingles)
    stopwords_personalizadas.update(stopwords_espanol)
    
    # Paso 6: Guardar las stopwords en un archivo CSV
    with open(ruta_csv_stopwords_salida, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for word in sorted(stopwords_personalizadas):
            writer.writerow([word])
    
    print(f"Stopwords personalizadas guardadas en {ruta_csv_stopwords_salida}")

if __name__ == '__main__':
    ruta_csv_entrada = r'C:\Users\semin\BD2\spotify_songs.csv'  
    ruta_csv_stopwords_salida = 'stopwords_personalizadas.csv'  
    
    generar_stopwords_desde_csv(ruta_csv_entrada, ruta_csv_stopwords_salida)
