
import csv
import nltk
import json
from collections import defaultdict


def contar_lineas_csv(ruta_csv):
    with open(ruta_csv, 'r', encoding='latin-1') as archivo:
        lineas = archivo.readlines()
        return len(lineas) - 1  


ruta_csv = r'C:\Users\semin\OneDrive\Escritorio\bd2\MUEST.csv'
num_lineas = contar_lineas_csv(ruta_csv)

print(f"El archivo CSV tiene {num_lineas} líneas de datos (excluyendo el encabezado).")



def tokenizar(texto):
    palabras = []
    palabra = ''
    for caracter in texto:
        if caracter.isalnum() or caracter == '-':
            palabra += caracter
        else:
            if palabra:
                palabras.append(palabra)
                palabra = ''
    if palabra:
        palabras.append(palabra)
    return palabras


def cargar_stopwords_nltk(idioma='english'):
    try:
        from nltk.corpus import stopwords
        return set(stopwords.words(idioma))
    except OSError:
        return set()


def eliminar_stopwords(lista_palabras, idioma='english'):
    stop_words = cargar_stopwords_nltk(idioma)
    return [palabra for palabra in lista_palabras if palabra.lower() not in stop_words]


def eliminar_puntuacion(lista_palabras):
    return [''.join(caracter for caracter in palabra if caracter.isalnum()) for palabra in lista_palabras if ''.join(caracter for caracter in palabra if caracter.isalnum())]


def reducir_palabras(lista_palabras):
    sufijos = ['ing', 'ed', 'ly', 'es', 's', 'ment', 'er', 'tion', 'able', 'ness', 'ful']
    palabras_reducidas = []
    for palabra in lista_palabras:
        for sufijo in sufijos:
            if palabra.endswith(sufijo):
                palabra = palabra[:-len(sufijo)]
                break
        palabras_reducidas.append(palabra)
    return palabras_reducidas


def Procesing(documento):
    tokens = tokenizar(documento)
    tokens = eliminar_stopwords(tokens)
    tokens = eliminar_puntuacion(tokens)
    tokens_reducidos = reducir_palabras(tokens)
    return tokens_reducidos


def SPIMI_Invert(token_stream):
    dictionary = defaultdict(list)

    while token_stream:
        token, doc_id = token_stream.pop(0)
        if token not in dictionary:
            dictionary[token] = []
        if doc_id not in dictionary[token]:
            dictionary[token].append(doc_id)

        if len(dictionary[token]) >= 1000:  
            dictionary[token] = dictionary[token] + [None] * len(dictionary[token])

    sorted_terms = dict(sorted(dictionary.items()))
    return sorted_terms


def MergeBlocksSimple(blocks):
    merged_index = defaultdict(list)
    for block in blocks:
        for term, postings in block.items():
            for posting in postings:
                if posting not in merged_index[term]:
                    merged_index[term].append(posting)
    return dict(merged_index)


def BSBIndexConstruction(ruta_csv, sizeBuffer, ruta_json):
    n = 0
    indice_invertido = []

    with open(ruta_csv, 'r', encoding='latin-1') as archivo:
        encabezado = archivo.readline()  
        buffer = []
        buffer_id = 1

        while True:
            linea = archivo.readline()
            if not linea:
                break
            buffer.append(linea.strip())
            if len(buffer) == sizeBuffer:
                print(f"\n--- Buffer {buffer_id} ---")
                for i, doc in enumerate(buffer, 1):
                     print(f"Documento {i} del buffer {buffer_id}: {doc}")
                n += 1
                token_stream = []
                for documento in buffer:
                    doc_id = f"buffer_{buffer_id}_doc_{buffer.index(documento) + 1}"
                    tokens = Procesing(documento)
                    token_stream.extend([(token, doc_id) for token in tokens])
                
                # Construccion del índice invertido parcial utilizando SPIMI_Invert
                fn = SPIMI_Invert(token_stream)
                indice_invertido.append(fn)
                buffer = []
                buffer_id += 1

        if buffer:
            n += 1
            token_stream = []
            for documento in buffer:
                doc_id = f"buffer_{buffer_id}_doc_{buffer.index(documento) + 1}"
                tokens = Procesing(documento)
                token_stream.extend([(token, doc_id) for token in tokens])

            fn = SPIMI_Invert(token_stream)
            indice_invertido.append(fn)

    indice_final = MergeBlocksSimple(indice_invertido)

    with open(ruta_json, 'w', encoding='utf-8') as archivo_json:
        json.dump(indice_final, archivo_json, indent=4, ensure_ascii=False)

#ruta_csv = r'C:\Users\semin\OneDrive\Escritorio\bd2\archivo_simplificado.csv'
ruta_csv = r'C:\Users\semin\OneDrive\Escritorio\bd2\MUEST.csv'
sizeBuffer = 2
ruta_json = r'C:\Users\semin\OneDrive\Escritorio\bd2\archivo_PROCESADOFinalPe13.json'

BSBIndexConstruction(ruta_csv, sizeBuffer, ruta_json)



#ruta_csv = r'C:\Users\semin\OneDrive\Escritorio\bd2\MUEST.csv'

