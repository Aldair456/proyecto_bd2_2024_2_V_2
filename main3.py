


import nltk
nltk.download('punkt_tab')

import os
import io
import json
import math
import numpy as np
#from binary_search import find_word
from nltk.stem.snowball import SnowballStemmer
#nltk.download('punkt')





tamaño_maximo_buffer = int(io.DEFAULT_BUFFER_SIZE*4.15)

path_local_index = r"C:\Users\semin\BD2"


final_index = r"C:\Users\semin\BD2"


ruta_archivo = r"C:\Users\semin\BD2\styles.csv"  # Ruta del archivo CSV
ruta_stoplist = r"C:\Users\semin\BD2"

def find_word(word:str,limit = 10000):
    # Retorna el diccionario de la palabra si es que existe, si no, retorna un diccionario vacio
    ini = 0
    fin = limit
    while 1:
        mid = int((ini + fin)/2)
        ruta_indice_local=r"C:\Users\semin\BD2\index01.json"
        #ruta_indice_local = path_local_index + "\\index" + str(mid + 1).zfill(2) + ".json"
        try:
            file = open(ruta_indice_local,"r")
            data = json.load(file)
            #print(data)
            list_data = list(data)
            first_word = list_data[0]
            last_word = list_data[-1]
            if first_word <= word and word <= last_word:
                if word in data.keys():
                    print(data[word])
                    return data[word]
                else:
                    return {}
            elif word < first_word: 
                fin = mid-1
            else:
                ini = mid+1
        except:
            ini = mid-1
    return {}


"""
    Pasos:
        1. Preprocesar los documentos
        2. Contabilizar term_frecuency y doc_id (posting list)
        3. Pasar datos a memoria secundaria
"""

"""
    Cuando llega una query:
        1. Se preprocesa 
        2. Se genera una lista de terminos = indice_invertido + terminos de query
        3. Se genera un espacio vectorial con longitud fija (longitud=lista de terminos)
        4. Se genera un vector para cada fila y la query
        5. Se calcula coseno entre (vector_por_fila, vector_query)
        6. Se ordenan los resultados (orden descendente)
        7. Se retornan los k más relacionados
"""

class InvertedIndex:
    colection_header = []
    pesos = [0,0,1.5,0,1.5,1.5,1.5,0,1.5,1] # para guardar pesos de cada campo (crear una funcion que haga esto)
    stopList = []
    cont_filas_CSV = 0 #para verificar que se preprocesan todas las filas del CSV
    nro_buckets = 0

    def __init__(self):
        self.setStoplist(ruta_stoplist + r"\stoplist.csv") 

  # Línea que genera el aviso


    def do_Spimi(self):
        self.preProcessCSV(ruta_archivo) #preprocesamos cada buffer del CSV
        print("Nro de filas preprocesadas:",self.cont_filas_CSV)


    



    def setStoplist(self,nombre):
        stop_words = open(nombre, "r", encoding="latin1") 

        for i in stop_words:
            self.stopList.append(i.strip('\n')) #se agrega cada elemento quitando saltos de linea

        stop_words.close() #cierra archivo leido

        stoplist_extended = "'«[]¿?$+-*'.,»:;!,º«»()@¡😆“/#|*%'&`"
        for caracter in stoplist_extended:
            self.stopList.append(caracter)
        #print(self.stopList)
    
    def preProcessCSV(self,ruta_archivo):

    #1) Obtencion  del tamaño del archivo  y variables de posicion
        tamano_bytes = os.path.getsize(ruta_archivo) # total de bytes en el csv
        
        pos_row = 0 # pos de fila que se está leyendo
        
        
    #2) Lectura del encabezado del archivo
        # Abrir el archivo en modo lectura
        with open(ruta_archivo, "rt", encoding="utf-8") as archivo:

            """ENCABEZADO"""
            encabezado_text = archivo.readline() # Lee la primera línea del archivo para obtener el (encabezado)
            pos_row += len(encabezado_text.encode("utf-8"))  # Tamaño de la línea en bytes
            #pos_row += 1 #para contar donde inicia fila que sigue
        #divide los encabezados por coma
            self.colection_header = encabezado_text.strip().split(',')
            #print('Encabezados del CSV:', self.colection_header)
            
            #print(pos_row)

            # modificar para leer todo el csv (ahora solo lee una pagina o buffer)
            cont_buffer = 0

            normas = {} # diccionario para normas

        #para leer el archivo linea por linea hasta que se acabe el archivo
            while tamano_bytes>pos_row:#-3>=pos_row:
                pos_row = self.getBufferIndex(pos_row,archivo,cont_buffer,normas) #parametro i para nro de archivo .json
                
                #print(pos_row)
                cont_buffer += 1
            print("Indices locales creados")
            """
                for c in campos:
                    print(c,end='-')
                print("\n")
            """
        ruta_normas = ruta_stoplist+r"\normas.json" #generamos un archivo json para guardar info de las normas ([pos_row]:norma)
        print("guardando normas")
        with open(ruta_normas, "w") as archivo:
            json.dump(normas, archivo)
        #print("indice local: ",indice_local)
        #print(normas)
        #print(tamano_bytes)
    """
                # Leer líneas del archivo y agregarlas al buffer hasta que el tamaño máximo se alcance
                for linea in archivo:
                    tamaño_linea = len(linea.encode("utf-8"))  # Tamaño de la línea en bytes
                    
                    if tamaño_buffer + tamaño_linea <= tamaño_maximo_buffer: # Si la línea cabe en el buffer sin exceder el tamaño máximo
                        #buffer.append(linea.strip())  # Agrega la línea al buffer
                        self.preProcessandIndex(linea) #preProcesa la linea y retorna un indice invertido
                        tamaño_buffer += tamaño_linea  # Actualiza el tamaño del buffer
                        if cont==0:
                            print(linea.strip(), end='-;-')
                            print(tamaño_linea,end='\n')
                    else:
                        break  # Si excede el tamaño máximo, detén la lectura del buffer
                

                # Pausa el programa por 3 segundos antes de leer el siguiente buffer
                # preprocesar el buffer(linea por linea y enviar tokens a la funcion preProcesar)
                
                cont+=1
                # Si el archivo ha llegado al final, sal del bucle
                if not buffer or tamaño_buffer==0:
                    break

                # Procesar el buffer (hacer lo que necesites con las líneas leídas)
                #print("Nuevo buffer:\n\n", "\n".join(buffer))  # En este ejemplo, simplemente imprime el buffer
                #print("Tamaño buffer: ",tamaño_buffer)
            print("Listo")
            print(cont)
    """

    def getBufferIndex(self,pos_inicio,archivo,nro_buffer,normas): #funcion ´para obtener indice invertido de un buffer

        # se posiciona en el csv
        archivo.seek(pos_inicio)
        ind_result = pos_inicio #se define como la posicion de inicio
        buffer = archivo.read(tamaño_maximo_buffer) #leemos un buffer desde el csv
        ind_actual = 0 
        indice_local = {} #para indice invertido local

        """
        if nro_buffer==531:
            print(buffer)
            print(len(buffer))
        """

        #encontramos primer salto de linea y lo definimos como el límite
        i = len(buffer)-1
        while buffer[i]!='\n':
            i -= 1

        # tenemos ->  buffer[i] = '\n'  

        #print(i)
        #print(buffer)
        #pos_fila = 0
        tamaño_linea = 0 #no estamos considerando primera posicion
        #se lee una cantidad entera de lineas
        while ind_actual<i: #obtendremos cada linea posible (considerando a las que se encuentran antes del ultimo '\n')
            cont_comas = 0
            campos = [] 
            
            while cont_comas<8:
                campo = ""

                #recorremos cada campo
                while buffer[ind_actual]!=',':  
                    if buffer[ind_actual]!='\n':
                        campo += buffer[ind_actual]
                        tamaño_linea += len(buffer[ind_actual].encode('utf-8')) #aumenta el tamaño de la linea
                        
                    ind_actual += 1
                    #tamaño_linea += '\n'.encode('utf-8') #aumenta el tamaño de la linea
                
                cont_comas += 1 #aumentamos la cantidad de comas
                ind_actual += 1
                tamaño_linea += len(','.encode('utf-8')) #aumenta tamaño linea

                campos.append(campo)    #añadimos el campo

            campo = ""

            while buffer[ind_actual]!='\n':   # lee ultima columa (consideramos que hay casos en los que guarda ",", por eso se separa así) 
                campo += buffer[ind_actual]
                ind_actual += 1
                tamaño_linea += len(buffer[ind_actual].encode('utf-8')) #aumenta tamaño linea

            campos.append(campo)

            """
            for campo in campos:
                print(campo,end= ' - ')
            print('\n')
            """
            ind_actual += 1
            #ind_result += 1
            tamaño_linea += len('\n'.encode('utf-8'))

            pos_row = pos_inicio+ind_actual-tamaño_linea #se aumenta uno por siguiente posicion
            print(pos_row)

            #pos_fila += 1
            tamaño_linea = 0 #porque ya se considera posicion actual

            #preProcesa cada linea
            self.preProcessListandIndex(list_campos=campos,dicc_lexemas=indice_local, pos_row=pos_row,dicc_normas=normas) #enviamos diccionario de normas


        ind_result += ind_actual #aumenta de acuerdo al indice actual
        indice_local = dict(sorted(indice_local.items())) #ordena indice local 
    
        #enviar indice a un archivo .json
        # Escribir el conjunto de diccionarios en un archivo JSON
        ruta_indice_local = path_local_index + "\\index" + str(nro_buffer + 1).zfill(2) + ".json"
        #print("indice: ",indice_local)
        with open(ruta_indice_local, "w") as archivo:
            json.dump(indice_local, archivo)

        return ind_result

        

    # funcion para evaluar todos los campos de una fila (doc)
    def preProcessListandIndex(self,list_campos,dicc_lexemas,pos_row, dicc_normas): #recibimos también un diccionario de normas
        #los campos se encuentran separados en una lista
        #print(list_campos)
        #dicc_lexemas = {} #se imprime solo para verificar correctitud del indice invertido por linea

        if pos_row!=-1:
            self.cont_filas_CSV+=1

        dicc_normas[pos_row] = 0
        tf_local = {} #para contabilizar tf de cada palabra (en esta fila)

        for i,campo in enumerate(list_campos):

            #obtendremos la norma por cada campo (mientras preprocesamos)
            #tf_campo = self.preProcessandIndex(texto=campo,dicc_lexemas=dicc_lexemas,peso=self.pesos[i],pos_row=pos_row,tf_local=tf_local) 
            #tf_total += tf_campo # almacenamos la suma de tf's (de todos los campos)
     
            self.preProcessandIndex(texto=campo,dicc_lexemas=dicc_lexemas,peso=self.pesos[i],pos_row=pos_row,tf_local=tf_local) 
            #print("Lexemas+tf: ",dicc_lexemas) #verificacion del indice invertido por linea

        for token in tf_local:
            #tf_local[tf] = math.log10(tf_local[tf])
            #dicc_normas[pos_row] += round(math.log10(1+tf_local[token])**2,3)
            dicc_lexemas[token][pos_row] = round(math.log10(1+dicc_lexemas[token][pos_row]),3)
            dicc_normas[pos_row] += round(dicc_lexemas[token][pos_row]**2,3)
            """
            if pos_row==98:
                print(dicc_lexemas[token][pos_row],dicc_normas[pos_row])
            """
        dicc_normas[pos_row] = round(math.sqrt(dicc_normas[pos_row]),3) #agregamos la norma en el diccionario de normas (redondeada a 3 decimales)

        """
        if pos_row==98:
            print(dicc_normas[pos_row])
        """

        
    def preProcessandIndex(self,texto,dicc_lexemas,peso=1,pos_row=-1,tf_local={}): #recibe una fila y genera diccionario (pos_row, suma(tf_por_campo*peso_campo)) .
        
        #tf_local = {} #diccionario local para gestionar las normas

        """para calcular la norma:
            - manejar un diccionario local (para todas las palabras de este campo) - *se evita sobrecargar la RAM*
            - contabilizar por cada termino
                norma = log_10 (1+ tf[i]*p ) : entiendase i como la cantidad de palabras en el campo
            - guardar tf por cada termino (en disco)
            - retornar el valor de la norma (al final se saca la raiz)
        """
        
        # 1. tokenizar
        tokens = nltk.word_tokenize(texto.lower())

        #print("Tokens:",tokens)

        # 3. sacar el lexema
        stemmer = SnowballStemmer('english')
        #dicc_lexemas = {}

        # 2. obtener stoplist y eliminarlo
        """  # Primera idea de implementacion
        for i in range (len(tokens)-1,-1,-1):
            
            if tokens[i] in self.stopList:
            tokens.pop(i)
            else:
                token_lexem.append()
        """

        # 2da idea de implementacion

        for i in range (len(tokens)): #recorre tokens
            if tokens[i] not in self.stopList: #si no es stopWord
                lexema = stemmer.stem(tokens[i]) #obtenemos el lexema
                #print("termino: ",lexema,"\n","pos_row:",pos_row)
                

                if peso!=0: #solo se debe añadir si su peso realmente influye

                    # añadimos al diccionario local (para gestionar norma)
                    if lexema in tf_local:
                        tf_local[lexema] += peso
                    else:
                        tf_local[lexema] = peso


                    if pos_row==-1: 
                        if lexema in dicc_lexemas:
                            dicc_lexemas[lexema] += peso
                        else:
                            dicc_lexemas[lexema] = peso
                    else:
                        if lexema in dicc_lexemas:
                            if pos_row in dicc_lexemas[lexema]:
                                dicc_lexemas[lexema][pos_row] += peso
                            else:
                                dicc_lexemas[lexema][pos_row] = peso # antes-> dicc_lexemas[lexema]= {pos_row:peso}
                        else:
                            dicc_lexemas[lexema] = {pos_row:peso}
                 
                    #print(dicc_lexemas)      
                    """
                    # agregamos correctamente el contador en el diccionario
                    if lexema in dicc_lexemas and pos_row!=-1: #caso en el que se quiera asociar a una fila
                            if pos_row in dicc_lexemas[lexema]:
                                dicc_lexemas[lexema][pos_row] += peso #ya apareció antes
                            else:
                                dicc_lexemas[lexema] = {pos_row:peso} #primera vez que aparece

                    #caso en el que no se quiera asociar a una fila (para Query)
                    elif lexema in dicc_lexemas:  #ya existe 
                        dicc_lexemas[lexema] += peso #aparece por primera vez
                    else:
                        dicc_lexemas[lexema] = peso 
                    """
        """
        # calculamos sumatoria de tf (por este campo)
        tf_campo = 0 #para que calcule norma
        for lexema in tf_local_campo: #hasta aquí, solo tenemos el peso (en crudo) de cada palabra, para las normas necesitamos *log_10(1+peso)*
            tf_campo += math.log10(1+tf_local_campo[lexema]) #aquí ya calculamos el term_frequency  

        #print("tf_campo:",tf_campo)
        return tf_campo # term_frequency (por campo)
        """
    
        #print(dicc_lexemas)
        #return dicc_lexemas #retorna diccionario de lexemas con su tf

    def processQuery(self,query): #returns the tf of each query's word
        tokens = nltk.word_tokenize(query.lower())

        # sacar el lexema
        stemmer = SnowballStemmer('english')

        
        result = {}

        for i in range (len(tokens)): #recorre tokens
            if tokens[i] not in self.stopList: #si no es stopWord
                lexema = stemmer.stem(tokens[i]) #obtenemos el lexema

                if lexema not in result:
                    result[lexema] = 1
                else:
                    result[lexema] += 1

        # obtenemos tf de la query
        for term in result:
            result[term] = round(math.log10(1+result[term]),3)
        return result
    
    def contar_archivos_json(self): # returns the number of json files in the final merge
        # Obtener la lista de archivos en la carpeta
        archivos_en_carpeta = os.listdir(final_index)

        # Filtrar archivos con extensión .json
        archivos_json = [archivo for archivo in archivos_en_carpeta if archivo.endswith('.json')]

        # Contar el número de archivos JSON
        numero_archivos_json = len(archivos_json)

        return numero_archivos_json
    
    def cosine(self,termsQuery,topK=0): # returns the cosine between query and each row (in this case, we consider row equals to doc)
        if self.nro_buckets == 0:
            self.nro_buckets = self.contar_archivos_json()
        #print(self.nro_buckets)

        QueryNorm = 0  # we need query's norm
        cosine = {} # we need to know the rows that match with query's terms, the keys of this dict will be the pos_row's
        
        for term in termsQuery:
            QueryNorm += termsQuery[term]**2

            # we need to obtain the doc product between each term of query and row
            valuesTerm = find_word(term,self.nro_buckets)
            print(term,"aparece en",len(valuesTerm),"rows")

            if len(valuesTerm) > 1:
             itf = 1 / math.log10(len(valuesTerm))  # ITF estándar
            else:
              itf = 1
            for value in valuesTerm: #here, we itere in each pos_row
                if value not in cosine: #this means that not other term was in this row, this is the first match
                    cosine[value] = valuesTerm[value]*termsQuery[term]*itf # multiply the tf-tf of the row and of the query
                else: #this means that the row has another word that match with the term, so we increase the value of the product
                    cosine[value] += valuesTerm[value]*termsQuery[term]*itf
            
        print("cos:",cosine)

        QueryNorm = round(math.sqrt(QueryNorm),3) # we obtain the query's norm

        # until here, we have de sum of the dot product for each term of the query
        # Now, we need to divide it using the norm of the query and the norm of each row
        ruta_normas = ruta_stoplist+r"\normas.json" #read the norms
        with open(ruta_normas, 'r') as archivo_json:
            norms = json.load(archivo_json)
        #print(norms) # just in case

        for row in cosine:
            RowNorm = norms[row] # first, we obtain the norm of the row
            cosine[row] = round(cosine[row]/(QueryNorm*RowNorm),4)
        print("COSENO  : ",cosine) # just in case

        # now, we have to find the topK similarities
        if topK!=0:
            cosine = {k: v for k, v in sorted(cosine.items(), key=lambda item: item[1], reverse=True)}
            return dict(list(cosine.items())[:topK])
        print(cosine)
        return cosine
    

inverted_index = InvertedIndex()
inverted_index.do_Spimi()
""""
query_terms = inverted_index.processQuery("Classic T-Shirt")
cosine_results = inverted_index.cosine(query_terms, topK=10)
print(cosine_results)
"""
# Procesar la consulta
query_terms = inverted_index.processQuery("Classic T-Shirt")

# Verificar los términos procesados de la consulta
print("Términos de la consulta procesados:", query_terms)

# Obtener los resultados de la consulta
cosine_results = inverted_index.cosine(query_terms, topK=10)
print(cosine_results)
# Imprimir los resultados
print("Resultados de similitud de coseno:", cosine_results)