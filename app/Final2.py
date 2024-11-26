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

TAMANIO_CHUNK = 20000  # numero de chunks asumiendo de que tenmeos 10 % de almacenamiento 




# RUTAS  PARA PODER TRABAJAR   TODOS LOS JSON TANTO LOS QUE SE PASAN DESDE EL FRONT
RUTA_INDICE_LOCAL = r"C:\Users\semin\OneDrive\Escritorio\bd2_code\Clonación2\Proyecto_2_BD2\app\TESING"
RUTA_INDICE_FINAL = r"C:\Users\semin\OneDrive\Escritorio\bd2_code\Clonación2\Proyecto_2_BD2\app\TESING"
RUTA_ARCHIVO_CSV = r"C:\Users\semin\OneDrive\Escritorio\bd2_code\Clonación2\Proyecto_2_BD2\spotify_songs_filtrado.csv"
RUTA_STOPLIST = r"C:\Users\semin\BD2\stoplist.csv"
RUTA_NORMAS = r"C:\Users\semin\OneDrive\Escritorio\bd2_code\Clonación2\Proyecto_2_BD2\app\TESING\normas.json"
RUTA_PESOS_CAMPO = r"C:\Users\semin\OneDrive\Escritorio\bd2_code\Clonación2\Proyecto_2_BD2\app\TESING\pesos_campos.json"  # Ruta para los pesos preprocesados



class IndiceInvertido:
    def __init__(self, ruta_csv: str, ruta_stoplist: str, ruta_indice: str, ruta_normas: str, ruta_pesos: str):
        self.ruta_csv = ruta_csv
        self.ruta_stoplist = ruta_stoplist
        self.ruta_indice = RUTA_INDICE_LOCAL
        self.ruta_normas = ruta_normas
        self.ruta_pesos = ruta_pesos
        self.stopwords = set()
        self.stemmer = SnowballStemmer('spanish')
        self.pesos_campos = []
        self.indice_invertido = defaultdict(dict)
        self.normas_documentos = {}
        self._cargar_stopwords()
        self._cargar_pesos_campos()  #  cargamos los pesos previamente calculados

    def _cargar_stopwords(self):
        # CARGAMOS LAS PALABRAS QUE VAMO A FILTRAR
        try:
            with open(self.ruta_stoplist, 'r', encoding='utf-8') as archivo:
                for linea in archivo:
                    palabra = linea.strip().lower()
                    if palabra:
                        self.stopwords.add(palabra)
            print("stopwords cargadas (IndiceInvertido):", self.stopwords)
        except FileNotFoundError:
            print(f"archivo de stopwords no encontrado en {self.ruta_stoplist}")
        
        caracteres_especiales = set("'«[]¿?$+-*'.,»:;!,º«»()@¡“/#|*%'&`")
        self.stopwords.update(caracteres_especiales)

    def _cargar_pesos_campos(self):
    
        try:
            with open(self.ruta_pesos, 'r', encoding='utf-8') as archivo:
                self.pesos_campos = json.load(archivo)
            print(f"Pesos de campos cargados desde {self.ruta_pesos}: {self.pesos_campos}")
        except FileNotFoundError:
            print(f"archivo de pesos no encontrado en {self.ruta_pesos}")
            self.pesos_campos = []

    def construir_indice(self):
        numero_chunk = 0
        try:
            for chunk in pd.read_csv(self.ruta_csv, chunksize=TAMANIO_CHUNK, encoding='utf-8'):
                numero_chunk += 1
                print(f"procesando chunk {numero_chunk}")
                self._procesar_chunk(chunk)
                self._guardar_indice_parcial(numero_chunk)
            self._guardar_normas()
            print("construccion del indice invertido completa")
        except Exception as e:
            print(f"error al construir el indice: {e}")

    def _procesar_chunk(self, chunk: pd.DataFrame):
        for indice, fila in chunk.iterrows():
            id_documento = str(indice)  # Convertir id_documento a cadena para consistencia
            self.normas_documentos[id_documento] = 0
            frecuencia_terminos = defaultdict(float)
            
            for idx_campo, campo in enumerate(fila):
                if idx_campo >= len(self.pesos_campos):
                    continue  # saltar campos sin pesos definidos
                peso = self.pesos_campos[idx_campo]
                if peso == 0:
                    continue  # saltar campos con peso cero
                tokens = nltk.word_tokenize(str(campo).lower())
                tokens = [token.strip() for token in tokens]
                for token in tokens:
                    if token not in self.stopwords:
                        lematizado = self.stemmer.stem(token)
                        frecuencia_terminos[lematizado] += peso
                    else:
                        pass  # stopwords eliminadas
                
            # actualizar indice invertido y normas
            for termino, frecuencia in frecuencia_terminos.items():
                self.indice_invertido[termino][id_documento] = math.log10(1 + frecuencia)
                self.normas_documentos[id_documento] += self.indice_invertido[termino][id_documento] ** 2

    def _guardar_indice_parcial(self, numero_chunk: int):
        ruta_indice_parcial = os.path.join(self.ruta_indice, f"indice_parcial_{numero_chunk}.json")
        try:
            with open(ruta_indice_parcial, 'w', encoding='utf-8') as archivo:
                json.dump(self.indice_invertido, archivo)
            print(f"Índice parcial guardado en {ruta_indice_parcial}")
            self.indice_invertido.clear()  # limpiar indice en memoria después de guardar
        except Exception as e:
            print(f"Error al guardar el índice parcial: {e}")

    def _guardar_normas(self):
        try:
            # Convertir id_documento a cadena para consistencia
            normas_str_keys = {str(k): round(math.sqrt(v), 3) for k, v in self.normas_documentos.items()}
            with open(self.ruta_normas, 'w', encoding='utf-8') as archivo:
                json.dump(normas_str_keys, archivo)
            print(f"Normas guardadas en {self.ruta_normas}")
        except Exception as e:
            print(f"Error al guardar las normas: {e}")

class MotorConsulta:
    def __init__(self, ruta_csv: str, ruta_indice: str, ruta_normas: str, ruta_stoplist: str, tamano_bloque: int = 1000):
        self.ruta_csv = ruta_csv
        self.ruta_indice = ruta_indice
        self.ruta_normas = ruta_normas
        self.ruta_stoplist = ruta_stoplist
        self.tamano_bloque = tamano_bloque
        self.stemmer = SnowballStemmer('spanish')
        self.stopwords = set()
        self._cargar_stopwords()
        self.indice_invertido = self._cargar_indice_por_bloques()
        self.normas_documentos = self._cargar_normas()
        self.dataframe = pd.read_csv(self.ruta_csv, index_col=None, encoding='utf-8', low_memory=False)
        self.dataframe.reset_index(drop=True, inplace=True)
        self.dataframe.index = self.dataframe.index.map(str)  

    def _cargar_stopwords(self):
        try:
            with open(self.ruta_stoplist, 'r', encoding='utf-8') as archivo:
                for linea in archivo:
                    palabra = linea.strip().lower()
                    if palabra:
                        self.stopwords.add(palabra)
            print("Stopwords cargadas (motorconsulta):", self.stopwords)
        except FileNotFoundError:
            print(f"Archivo de stopwords no encontrado en {self.ruta_stoplist}")
        
        caracteres_especiales = set("'«[]¿?$+-*'.,»:;!,º«»()@¡“/#|*%'&`")
        self.stopwords.update(caracteres_especiales)
    # LECTURA MEDIANTE BLOQUES Y LUEGO LIMPIAR CUANDO SE PROCESE :D
    def _cargar_indice_por_bloques(self) -> Dict[str, Dict[str, float]]:
        indice_completo = defaultdict(dict)
        bloque_actual = defaultdict(dict)
        contador = 0
        bloque_numero = 0
        try:
            archivos_parciales = [
                archivo for archivo in os.listdir(self.ruta_indice)
                if archivo.startswith("indice_parcial_") and archivo.endswith(".json")
            ]

            for archivo in archivos_parciales:
                ruta_archivo = os.path.join(self.ruta_indice, archivo)
                with open(ruta_archivo, 'r', encoding='utf-8') as archivo_json:
                    indice_parcial = json.load(archivo_json)
                    for termino, postings in indice_parcial.items():
                        bloque_actual[termino].update(postings)
                        contador += 1

                        # si alcanzamos el tamaño del bloque, consolidamos en memoria
                        if contador >= self.tamano_bloque:
                            bloque_numero += 1
                            #print(f"Consolidando bloque {bloque_numero}")
                            self._consolidar_bloque_en_memoria(indice_completo, bloque_actual)
                            bloque_actual.clear()
                            contador = 0

            if bloque_actual:
                bloque_numero += 1
                print(f"Consolidando bloque restante {bloque_numero}")
                self._consolidar_bloque_en_memoria(indice_completo, bloque_actual)
               
            print("indice consolidado por bloques cargado en memoria")
                # Guardar el índice completo en un archivo JSON
            ruta_indice_completo = os.path.join(self.ruta_indice, "indice_completo.json")
            with open(ruta_indice_completo, 'w', encoding='utf-8') as archivo_completo:
                json.dump(indice_completo, archivo_completo, ensure_ascii=False, indent=4)
            print(f"Índice completo guardado en {ruta_indice_completo}")
            return indice_completo
        except Exception as e:
            print(f"error al cargar el índice por bloques: {e}")
            return {}
    def _consolidar_bloque_en_memoria(self, indice_completo: Dict, bloque_actual: Dict):
        for termino, postings in bloque_actual.items():
            indice_completo[termino].update(postings)



    """
    def _cargar_indice_completo(self) -> Dict[str, Dict[str, float]]:
        indice_completo = defaultdict(dict)
        try:
            for nombre_archivo in os.listdir(self.ruta_indice):
                if nombre_archivo.startswith("indice_parcial_") and nombre_archivo.endswith(".json"):
                    ruta_archivo = os.path.join(self.ruta_indice, nombre_archivo)
                    with open(ruta_archivo, 'r', encoding='utf-8') as archivo:
                        indice_parcial = json.load(archivo)
                        for termino, postings in indice_parcial.items():
                            indice_completo[termino].update(postings)
            print("Índice invertido completo cargado")
            return indice_completo
        except Exception as e:
            print(f"Error al cargar el índice invertido: {e}")
            return {}
"""
    def _cargar_normas(self) -> Dict[str, float]:
        print("Estoy  aqui")
        try:
            with open(self.ruta_normas, 'r', encoding='utf-8') as archivo:
                normas = json.load(archivo)
            print("Normas de documentos cargadas")
            return normas
        except Exception as e:
            print(f"Error al cargar las normas: {e}")
            return {}

    def procesar_consulta(self, consulta: str) -> Dict[str, float]:
        tokens = nltk.word_tokenize(consulta.lower())
        tokens = [token.strip() for token in tokens]
        tokens_sin_stopwords = [token for token in tokens if token not in self.stopwords]
        frecuencia_terminos = defaultdict(int)
        for token in tokens_sin_stopwords:
            lematizado = self.stemmer.stem(token)
            frecuencia_terminos[lematizado] += 1
        # aplicar normalizacion logaritmica
        for termino in frecuencia_terminos:
            frecuencia_terminos[termino] = round(math.log10(1 + frecuencia_terminos[termino]), 3)
        return dict(frecuencia_terminos)

    def buscar(self, consulta: str, top_k: int = 10) -> Dict[str, Dict]:
        print("ENTRO")
        terminos_consulta = self.procesar_consulta(consulta)
        if not terminos_consulta:
            print("no hay terminos validos en la consulta despues del procesamiento")
            return {}
        
        norma_consulta = math.sqrt(sum(freq ** 2 for freq in terminos_consulta.values()))
        puntuaciones = defaultdict(float)

        for termino, frecuencia_q in terminos_consulta.items():
            if termino in self.indice_invertido:
                postings = self.indice_invertido[termino]
                print("_-----------")
                print(postings)
                print("_-----------")
                idf = math.log10(len(self.normas_documentos) / len(postings)) if len(postings) > 0 else 1
                for id_documento, frecuencia_d in postings.items():
                    puntuaciones[id_documento] += frecuencia_q * frecuencia_d * idf

        # normalizar las puntuaciones por las normas de los documentos y la norma de la consulta
        similitud_coseno = {}
        for id_documento in puntuaciones:
            if self.normas_documentos.get(id_documento, 0) > 0 and norma_consulta > 0:
                similitud_coseno[id_documento] = puntuaciones[id_documento] / (self.normas_documentos[id_documento] * norma_consulta)
            else:
                similitud_coseno[id_documento] = 0.0

        # ordenar y recuperar los Top K resultados
        resultados_top_ids = dict(sorted(similitud_coseno.items(), key=lambda item: item[1], reverse=True)[:top_k])

        # cargar los datos de los documentos correspondientes y agregar la similitud del coseno
        documentos_resultados = self._cargar_documentos(resultados_top_ids.keys())
        for doc_id in documentos_resultados:
            documentos_resultados[doc_id]['similitud_coseno'] = round(resultados_top_ids[doc_id], 3)  # redondear la similitud

        return documentos_resultados

    def _cargar_documentos(self, ids_documentos) -> Dict[str, Dict]:
        """cargar los datos de los documentos a partir de sus id's"""
        documentos = {}
        try:
            for id_doc in ids_documentos:
                if id_doc in self.dataframe.index:
                    registro = self.dataframe.loc[id_doc].to_dict()
                    documentos[id_doc] = registro
                else:
                    print(f"ID {id_doc} no encontrado en el DataFrame")
        except Exception as e:
            print(f"Error al cargar los documentos: {e}")
        return documentos

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