import json
import math
import os
# Rutas
path_local_index = r"C:\Users\semin\BD2"
ruta_archivo = r"C:\Users\semin\BD2\styles.csv"
ruta_stoplist = r"C:\Users\semin\BD2\stoplist.csv"
ruta_normas = os.path.join(path_local_index, "normas.json")
tamaño_buffer = 1024 * 10  # Tamaño del buffer en bytes (ajustable)

class IndiceInvertidoConBuffer:
    def __init__(self):
        self.stop_words = self.cargar_stoplist(ruta_stoplist)
        self.normas = {}  # Almacena las normas de cada fila
        self.num_buffers = 0  # Inicializamos la cantidad de buffers a 0

    def cargar_stoplist(self, ruta):
        with open(ruta, 'r', encoding="latin1") as archivo:
            stop_words = set(archivo.read().splitlines())
        print("Palabras vacías cargadas:", stop_words)
        return stop_words

    def tokenizar_y_stemear(self, texto):
        texto = texto.lower()
        tokens = ''.join([c if c.isalnum() else ' ' for c in texto]).split()
        tokens = [self.stem(palabra) for palabra in tokens if palabra not in self.stop_words]
        return tokens

    def stem(self, palabra):
        sufijos = ['ing', 'ed', 'ly', 'es', 's']
        for sufijo in sufijos:
            if palabra.endswith(sufijo) and len(palabra) > len(sufijo):
                return palabra[:-len(sufijo)]
        return palabra

    def construir_indice_invertido(self):
        with open(ruta_archivo, 'r', encoding="utf-8") as archivo:
            encabezado = archivo.readline().strip().split(',')
            buffer_actual = ''
            num_fila = 0
            buffer_num = 0

            while True:
                datos = archivo.read(tamaño_buffer)
                if not datos:
                    break
                buffer_actual += datos
                filas = buffer_actual.split('\n')
                buffer_actual = filas.pop()

                indice_local = {}  # Índice invertido local del buffer actual

                for linea in filas:
                    num_fila += 1
                    campos = linea.strip().split(',')
                    tokens = []
                    for campo in campos:
                        tokens.extend(self.tokenizar_y_stemear(campo))
                    self.indexar_fila(tokens, num_fila, indice_local)

                # Guardamos el índice local en un archivo JSON incluyendo ITF y Coseno
                self.calcular_itf_coseno(indice_local)
                self.guardar_indice_local(indice_local, buffer_num)
                buffer_num += 1

            # Guardamos el total de buffers procesados
            self.num_buffers = buffer_num

            # Guardamos las normas en el archivo JSON al final del proceso
            self.guardar_normas()

    def indexar_fila(self, tokens, num_fila, indice_local):
        frecuencias_terminos = {}
        for token in tokens:
            frecuencias_terminos[token] = frecuencias_terminos.get(token, 0) + 1

        for termino, freq in frecuencias_terminos.items():
            if termino not in indice_local:
                indice_local[termino] = {}
            indice_local[termino][num_fila] = freq

        self.normas[num_fila] = self.calcular_norma(frecuencias_terminos)

    def calcular_norma(self, frecuencias_terminos):
        norma = sum((1 + math.log10(freq)) ** 2 for freq in frecuencias_terminos.values())
        return round(math.sqrt(norma), 3)

    def calcular_itf_coseno(self, indice_local):
        for termino, doc_freqs in indice_local.items():
            num_docs = len(doc_freqs)
            if num_docs > 1:
                itf = 1 / math.log10(num_docs)
            else:
                itf = 1

            for doc_id, freq in doc_freqs.items():
                # Calcular peso (tf) del término en el documento
                tf = round(1 + math.log10(freq), 3)
                
                # Obtener la norma del documento
                doc_norm = self.normas.get(doc_id, 1)

                # Calcular similitud de coseno
                coseno = round((tf * itf) / doc_norm, 4) if doc_norm != 0 else 0
                
                # Almacenar ITF y Coseno en el índice local
                indice_local[termino][doc_id] = {
                    'freq': freq,
                    'tf': tf,
                    'itf': itf,
                    'coseno': coseno
                }

    def guardar_indice_local(self, indice_local, buffer_num):
        ruta_indice_local = os.path.join(path_local_index, f"indice_local_{buffer_num}.json")
        with open(ruta_indice_local, "w") as archivo_json:
            json.dump(indice_local, archivo_json, indent=4)
        print(f"Índice local guardado en: {ruta_indice_local}")

    def guardar_normas(self):
        with open(ruta_normas, "w") as archivo_json:
            json.dump(self.normas, archivo_json, indent=4)
        print(f"Normas guardadas en: {ruta_normas}")

    def procesar_consulta(self, consulta):
        tokens = self.tokenizar_y_stemear(consulta)
        frecuencias_consulta = {}
        for token in tokens:
            frecuencias_consulta[token] = frecuencias_consulta.get(token, 0) + 1
        for termino in frecuencias_consulta:
            frecuencias_consulta[termino] = round(1 + math.log10(frecuencias_consulta[termino]), 3)
        return frecuencias_consulta

    def similitud_coseno(self, terminos_consulta, top_k=0):
        norma_consulta = sum(valor ** 2 for valor in terminos_consulta.values())
        norma_consulta = round(math.sqrt(norma_consulta), 3)
        similitudes = {}

        # Cargar las normas de los documentos
        with open(ruta_normas, 'r') as archivo_json:
            normas = json.load(archivo_json)

        # Calcular similitud coseno para cada término en el índice invertido
        for termino, peso_consulta in terminos_consulta.items():
            for i in range(self.num_buffers):
                ruta_indice_local = os.path.join(path_local_index, f"indice_local_{i}.json")
                with open(ruta_indice_local, "r") as archivo_json:
                    indice_local = json.load(archivo_json)
                
                if termino in indice_local:
                    itf = 1 / math.log10(1 + len(indice_local[termino]))
                    for num_fila, datos in indice_local[termino].items():
                        tf = datos['tf']
                        peso_doc = tf * itf
                        producto = peso_doc * peso_consulta
                        if num_fila not in similitudes:
                            similitudes[num_fila] = 0
                        similitudes[num_fila] += producto

        # Normalizar los resultados con las normas
        for num_fila in similitudes:
            similitudes[num_fila] = round(similitudes[num_fila] / (norma_consulta * normas[str(num_fila)]), 4)

        # Ordenar y devolver los resultados top_k
        similitudes_ordenadas = sorted(similitudes.items(), key=lambda item: item[1], reverse=True)
        resultados_top = dict(similitudes_ordenadas[:top_k]) if top_k > 0 else dict(similitudes_ordenadas)
        print("Similitudes coseno:", resultados_top)
        return resultados_top

# Inicializar y construir el índice
indice = IndiceInvertidoConBuffer()
indice.construir_indice_invertido()

# Procesar una consulta
terminos_consulta = indice.procesar_consulta("A comfortable")
resultados_coseno = indice.similitud_coseno(terminos_consulta, top_k=10)

# Mostrar resultados finales
print("Resultados finales de similitud coseno:", resultados_coseno)