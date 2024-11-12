import os
import io
import json
import math
import pandas as pd
import nltk
from nltk.stem import SnowballStemmer
from collections import defaultdict
from typing import Dict

nltk.download('punkt')

TAMANIO_CHUNK = 467258 # Número de filas por chunk esto basandonos en el uso del 10%de la memoria disponible
RUTA_INDICE_LOCAL = r"C:\Users\semin\BD2"
RUTA_INDICE_FINAL = r"C:\Users\semin\BD2"
RUTA_ARCHIVO_CSV = r"C:\Users\semin\BD2\spotify_songs.csv"
RUTA_STOPLIST = r"C:\Users\semin\BD2\stoplist.csv"
RUTA_NORMAS = r"C:\Users\semin\BD2\normas.json"
RUTA_PESOS_CAMPO = r"C:\Users\semin\BD2\pesos_campos.json"  # Ruta para los pesos preprocesados

class IndiceInvertido:
    def __init__(self, ruta_csv: str, ruta_stoplist: str, ruta_indice: str, ruta_normas: str, ruta_pesos: str):
        self.ruta_csv = ruta_csv
        self.ruta_stoplist = ruta_stoplist
        self.ruta_indice = ruta_indice
        self.ruta_normas = ruta_normas
        self.ruta_pesos = ruta_pesos
        self.stopwords = set()
        self.stemmer = SnowballStemmer('spanish')
        self.pesos_campos = []
        self.indice_invertido = defaultdict(dict)
        self.normas_documentos = {}
        self._cargar_stopwords()
        self._cargar_pesos_campos()  # Cargar los pesos desde el archivo

    def _cargar_stopwords(self):
        """Cargar stopwords desde el archivo stoplist y añadir caracteres extendidos."""
        try:
            with open(self.ruta_stoplist, 'r', encoding='utf-8') as archivo:
                for linea in archivo:
                    palabra = linea.strip().lower()
                    if palabra:
                        self.stopwords.add(palabra)
            print("Stopwords cargadas (IndiceInvertido):", self.stopwords)
        except FileNotFoundError:
            print(f"Archivo de stopwords no encontrado en {self.ruta_stoplist}")
        
        # Añadir caracteres especiales
        caracteres_especiales = set("'«[]¿?$+-*'.,»:;!,º«»()@¡“/#|*%'&`")
        self.stopwords.update(caracteres_especiales)

    def _cargar_pesos_campos(self):
        """Carga los pesos de los campos desde un archivo JSON."""
        try:
            with open(self.ruta_pesos, 'r', encoding='utf-8') as archivo:
                self.pesos_campos = json.load(archivo)
            print(f"Pesos de campos cargados desde {self.ruta_pesos}: {self.pesos_campos}")
        except FileNotFoundError:
            print(f"Archivo de pesos no encontrado en {self.ruta_pesos}")
            self.pesos_campos = []

    def construir_indice(self):
        """Construir el índice invertido a partir del archivo CSV."""
        numero_chunk = 0
        try:
            for chunk in pd.read_csv(self.ruta_csv, chunksize=TAMANIO_CHUNK, encoding='utf-8'):
                numero_chunk += 1
                print(f"Procesando chunk {numero_chunk}")
                self._procesar_chunk(chunk)
                self._guardar_indice_parcial(numero_chunk)
            self._guardar_normas()
            print("Construcción del índice invertido completada.")
        except Exception as e:
            print(f"Error al construir el índice: {e}")

    def _procesar_chunk(self, chunk: pd.DataFrame):
        """Procesar un chunk del archivo CSV y actualizar el índice invertido y las normas."""
        for indice, fila in chunk.iterrows():
            id_documento = str(indice)  # Convertir id_documento a cadena para consistencia
            self.normas_documentos[id_documento] = 0
            frecuencia_terminos = defaultdict(float)
            
            for idx_campo, campo in enumerate(fila):
                if idx_campo >= len(self.pesos_campos):
                    continue  # Saltar campos sin pesos definidos
                peso = self.pesos_campos[idx_campo]
                if peso == 0:
                    continue  # Saltar campos con peso cero
                tokens = nltk.word_tokenize(str(campo).lower())
                tokens = [token.strip() for token in tokens]
                for token in tokens:
                    if token not in self.stopwords:
                        lematizado = self.stemmer.stem(token)
                        frecuencia_terminos[lematizado] += peso
                    else:
                        pass  # Stopwords eliminadas
                
            # actualizar indice invertido y normas
            for termino, frecuencia in frecuencia_terminos.items():
                self.indice_invertido[termino][id_documento] = math.log10(1 + frecuencia)
                self.normas_documentos[id_documento] += self.indice_invertido[termino][id_documento] ** 2

    def _guardar_indice_parcial(self, numero_chunk: int):
        """guardar el estado actual del índice invertido en un archivo JSON."""
        ruta_indice_parcial = os.path.join(self.ruta_indice, f"indice_parcial_{numero_chunk}.json")
        try:
            with open(ruta_indice_parcial, 'w', encoding='utf-8') as archivo:
                json.dump(self.indice_invertido, archivo)
            print(f"Índice parcial guardado en {ruta_indice_parcial}")
            self.indice_invertido.clear()  # limpiar indice en memoria después de guardar
        except Exception as e:
            print(f"Error al guardar el índice parcial: {e}")

    def _guardar_normas(self):
        """guardar las normas de los documentos en un archivo JSON."""
        try:
            # Convertir id_documento a cadena para consistencia
            normas_str_keys = {str(k): round(math.sqrt(v), 3) for k, v in self.normas_documentos.items()}
            with open(self.ruta_normas, 'w', encoding='utf-8') as archivo:
                json.dump(normas_str_keys, archivo)
            print(f"Normas guardadas en {self.ruta_normas}")
        except Exception as e:
            print(f"Error al guardar las normas: {e}")

class MotorConsulta:
    def __init__(self, ruta_csv: str, ruta_indice: str, ruta_normas: str, ruta_stoplist: str):
        self.ruta_csv = ruta_csv
        self.ruta_indice = ruta_indice
        self.ruta_normas = ruta_normas
        self.ruta_stoplist = ruta_stoplist
        self.stemmer = SnowballStemmer('spanish')
        self.stopwords = set()
        self._cargar_stopwords()
        self.indice_invertido = self._cargar_indice_completo()
        self.normas_documentos = self._cargar_normas()
        # Cargar el DataFrame completo una vez para evitar cargarlo múltiples veces
        self.dataframe = pd.read_csv(self.ruta_csv, index_col=None, encoding='utf-8', low_memory=False)
        self.dataframe.reset_index(drop=True, inplace=True)
        self.dataframe.index = self.dataframe.index.map(str)  # Convertir índices a str para consistencia

    def _cargar_stopwords(self):
        """cargar stopwords desde el archivo stoplist y añadir caracteres extendidos."""
        try:
            with open(self.ruta_stoplist, 'r', encoding='utf-8') as archivo:
                for linea in archivo:
                    palabra = linea.strip().lower()
                    if palabra:
                        self.stopwords.add(palabra)
            print("Stopwords cargadas (MotorConsulta):", self.stopwords)
        except FileNotFoundError:
            print(f"Archivo de stopwords no encontrado en {self.ruta_stoplist}")
        
        # Añadir caracteres especiales
        caracteres_especiales = set("'«[]¿?$+-*'.,»:;!,º«»()@¡“/#|*%'&`")
        self.stopwords.update(caracteres_especiales)

    def _cargar_indice_completo(self) -> Dict[str, Dict[str, float]]:
        """cargar todos los archivos parciales del indice invertido y fusionarlos en un indice completo."""
        indice_completo = defaultdict(dict)
        try:
            for nombre_archivo in os.listdir(self.ruta_indice):
                if nombre_archivo.startswith("indice_parcial_") and nombre_archivo.endswith(".json"):
                    ruta_archivo = os.path.join(self.ruta_indice, nombre_archivo)
                    with open(ruta_archivo, 'r', encoding='utf-8') as archivo:
                        indice_parcial = json.load(archivo)
                        for termino, postings in indice_parcial.items():
                            indice_completo[termino].update(postings)
            print("Índice invertido completo cargado.")
            return indice_completo
        except Exception as e:
            print(f"Error al cargar el índice invertido: {e}")
            return {}

    def _cargar_normas(self) -> Dict[str, float]:
        print("Estoy  aqui")
        """cargar las normas de los documentos desde el archivo normas.json."""
        try:
            with open(self.ruta_normas, 'r', encoding='utf-8') as archivo:
                normas = json.load(archivo)
            print("Normas de documentos cargadas.")
            return normas
        except Exception as e:
            print(f"Error al cargar las normas: {e}")
            return {}

    def procesar_consulta(self, consulta: str) -> Dict[str, float]:
        """procesar la cadena de consulta en un diccionario de frecuencias de terminos."""
        tokens = nltk.word_tokenize(consulta.lower())
        tokens = [token.strip() for token in tokens]
        tokens_sin_stopwords = [token for token in tokens if token not in self.stopwords]
        frecuencia_terminos = defaultdict(int)
        for token in tokens_sin_stopwords:
            lematizado = self.stemmer.stem(token)
            frecuencia_terminos[lematizado] += 1
        # Aplicar normalización logarítmica
        for termino in frecuencia_terminos:
            frecuencia_terminos[termino] = round(math.log10(1 + frecuencia_terminos[termino]), 3)
        return dict(frecuencia_terminos)

    def buscar(self, consulta: str, top_k: int = 10) -> Dict[str, Dict]:
        """Buscar en el índice invertido la consulta y devolver los Top K documentos con sus datos completos y su similitud del coseno."""
        terminos_consulta = self.procesar_consulta(consulta)
        if not terminos_consulta:
            print("No hay términos válidos en la consulta después del procesamiento.")
            return {}
        
        norma_consulta = math.sqrt(sum(freq ** 2 for freq in terminos_consulta.values()))
        puntuaciones = defaultdict(float)

        for termino, frecuencia_q in terminos_consulta.items():
            if termino in self.indice_invertido:
                postings = self.indice_invertido[termino]
                idf = math.log10(len(self.normas_documentos) / len(postings)) if len(postings) > 0 else 1
                for id_documento, frecuencia_d in postings.items():
                    puntuaciones[id_documento] += frecuencia_q * frecuencia_d * idf

        # Normalizar las puntuaciones por las normas de los documentos y la norma de la consulta
        similitud_coseno = {}
        for id_documento in puntuaciones:
            if self.normas_documentos.get(id_documento, 0) > 0 and norma_consulta > 0:
                similitud_coseno[id_documento] = puntuaciones[id_documento] / (self.normas_documentos[id_documento] * norma_consulta)
            else:
                similitud_coseno[id_documento] = 0.0

        # Ordenar y recuperar los Top K resultados
        resultados_top_ids = dict(sorted(similitud_coseno.items(), key=lambda item: item[1], reverse=True)[:top_k])

        # Cargar los datos de los documentos correspondientes y agregar la similitud del coseno
        documentos_resultados = self._cargar_documentos(resultados_top_ids.keys())
        for doc_id in documentos_resultados:
            documentos_resultados[doc_id]['similitud_coseno'] = round(resultados_top_ids[doc_id], 3)  # Redondear la similitud

        return documentos_resultados

    def _cargar_documentos(self, ids_documentos) -> Dict[str, Dict]:
        """cargar los datos de los documentos a partir de sus IDs."""
        documentos = {}
        try:
            for id_doc in ids_documentos:
                if id_doc in self.dataframe.index:
                    registro = self.dataframe.loc[id_doc].to_dict()
                    documentos[id_doc] = registro
                else:
                    print(f"ID {id_doc} no encontrado en el DataFrame.")
        except Exception as e:
            print(f"Error al cargar los documentos: {e}")
        return documentos


"""
# Ejemplo de Uso
if __name__ == "__main__":
    # Paso 1:  para calcular y guardar los pesos

    # Paso 2: construir el rndice rnvertido 
    indice = IndiceInvertido(
        ruta_csv=RUTA_ARCHIVO_CSV,
        ruta_stoplist=RUTA_STOPLIST,
        ruta_indice=RUTA_INDICE_LOCAL,
        ruta_normas=RUTA_NORMAS,
        ruta_pesos=RUTA_PESOS_CAMPO  # Ruta para los pesos preprocesados
    )
    indice.construir_indice()

    # Paso 3: inicializar el Motor de C¿consulta con la ruta_csv y ruta_stoplist
    motor_busqueda = MotorConsulta(
        ruta_csv=RUTA_ARCHIVO_CSV,
        ruta_indice=RUTA_INDICE_LOCAL,
        ruta_normas=RUTA_NORMAS,
        ruta_stoplist=RUTA_STOPLIST
    )
"""
"""
@app.route('/buscar', methods=['GET'])
def buscar():
    consulta_usuario = request.args.get('consulta', '')
    top_k = int(request.args.get('top_k', 10))
    resultados_busqueda = motor_busqueda.buscar(consulta_usuario, top_k=top_k)
    return jsonify(resultados_busqueda)
"""
"""
    # Paso 4: Procesar una Consulta
consulta_usuario = "I Feel Alive"
terminos_procesados = motor_busqueda.procesar_consulta(consulta_usuario)
print("Términos Procesados de la Consulta:", terminos_procesados)

    # Paso 5: Buscar y Recuperar los Top K Resultados
top_k = 10
resultados_busqueda = motor_busqueda.buscar(consulta_usuario, top_k=top_k)
print(f"Top {top_k} Resultados de Búsqueda:")
print(json.dumps(resultados_busqueda, indent=2, ensure_ascii=False))
"""