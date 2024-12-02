import psycopg2 as pg
from flask import Blueprint, render_template, request, jsonify, current_app
import os
from .Final2 import IndiceInvertido, MotorConsulta
from .Multidimencional.knn_secuencial import knnsecuencial, obtener_vector_desde_imagen
import json

import psycopg2 as pg
import pandas as pd
from psycopg2.extras import RealDictCursor
import time




class PostgresConnector:
    def __init__(self):  # Corregido a __init__
        self.connection_params = {
            "user": "postgres",
            "password": "22demarzo",
            "host": "localhost",
            "port": "5432",
            "database": "spotify_songs"
        }
        self.connect()
        
    def connect(self):
        self.conn = pg.connect(**self.connection_params)
        self.cur = self.conn.cursor(cursor_factory=RealDictCursor)
        
    def setup_database(self):
        self.cur.execute("CREATE SCHEMA IF NOT EXISTS songs;")
        # Tabla
        create_table_query = """
        CREATE TABLE IF NOT EXISTS songs.spotify_songs (
            track_id VARCHAR PRIMARY KEY,
            track_name VARCHAR,
            track_artist VARCHAR,
            lyrics TEXT,
            playlist_name,
            search_vector tsvector
        );
        """
        self.cur.execute(create_table_query)
        
        # Índice GIN
        self.cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_songs_search 
        ON songs.spotify_songs USING gin(search_vector);
        """)
        
        # Función de actualización del vector
        self.cur.execute("""
            CREATE OR REPLACE FUNCTION songs.update_search_vector()
            RETURNS trigger AS $$
            BEGIN
                NEW.search_vector = 
                    setweight(to_tsvector('english', COALESCE(NEW.track_id,'')) || 
                            to_tsvector('spanish', COALESCE(NEW.track_id,'')), 'A') ||
                    setweight(to_tsvector('english', COALESCE(NEW.track_name,'')) || 
                            to_tsvector('spanish', COALESCE(NEW.track_name,'')), 'B') ||
                    setweight(to_tsvector('english', COALESCE(NEW.track_artist,'')) || 
                            to_tsvector('spanish', COALESCE(NEW.track_artist,'')), 'C') ||
                    setweight(to_tsvector('english', COALESCE(NEW.lyrics,'')) || 
                            to_tsvector('spanish', COALESCE(NEW.lyrics,'')), 'D');
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
        """)
        
        # Trigger
        self.cur.execute("""
        DROP TRIGGER IF EXISTS trigger_search_vector ON songs.spotify_songs;
        CREATE TRIGGER trigger_search_vector
        BEFORE INSERT OR UPDATE ON songs.spotify_songs
        FOR EACH ROW
        EXECUTE FUNCTION songs.update_search_vector();
        """)
        
        self.conn.commit()

    def load_data(self, csv_path):
        self.cur.execute("SELECT COUNT(*) FROM songs.spotify_songs")
        if self.cur.fetchone()['count'] > 0:
            print("Los datos ya están cargados")
            return
            
        # Cargar datos desde CSV
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            self.cur.execute("""
            INSERT INTO songs.spotify_songs (track_id, track_name, track_artist, lyrics,playlist_name)
            VALUES (%s, %s, %s, %s,%s)
            """, (row['track_id'], row['track_name'], row['track_artist'], row['lyrics'],row['playlist_name']))
        
        self.conn.commit()
    def search2(self, consulta_usuario, top_k):
        # Realizar la consulta a la base de datos
        query = f"""
        SELECT * FROM songs.spotify_songs 
        WHERE to_tsvector('english', track_name || ' ' || track_artist) @@ plainto_tsquery('english', %s)
        LIMIT %s;
        """
        self.cur.execute(query, (consulta_usuario, top_k))
        resultados = self.cur.fetchall()  # Obtener todos los resultados

        return {"results": resultados}  # Devolver los resultados como diccionario
    def search(self, query, k=5):
        start_time = time.time()
        
        clean_query = query.lower()
        
        ts_query = ' | '.join(f"'{word}':*" for word in clean_query.split())
        
        search_query = """
                SELECT 
                    track_id,
                    track_name, 
                    track_artist,
                    lyrics,
                    playlist_name,
                    ctid::text as row_position,
                    ts_rank_cd(search_vector, to_tsquery('english', %s) || to_tsquery('spanish', %s)) as similitud
                FROM songs.spotify_songs
                WHERE search_vector @@ (to_tsquery('english', %s) || to_tsquery('spanish', %s))
                ORDER BY similitud DESC
                LIMIT %s;
                """

        self.cur.execute(search_query, (ts_query, ts_query, ts_query, ts_query, k))
        results = self.cur.fetchall()
        
        return {
            'query_time': time.time() - start_time,
            'results': results
        }

    def __del__(self):  # Corregido a __del__
        if hasattr(self, 'cur'):
            self.cur.close()
        if hasattr(self, 'conn'):
            self.conn.close()

    def eliminarchema(self):
        self.cur.execute("DROP SCHEMA IF EXISTS songs CASCADE;")
        self.conn.commit()




main = Blueprint('main', __name__)

# Configuracion de rutas de archivos
RUTA_INDICE_LOCAL = r"C:\Users\semin\BD2"
RUTA_ARCHIVO_CSV = r"C:\Users\semin\OneDrive\Escritorio\bd2_code\Clonación2\Proyecto_2_BD2\spotify_songs_filtrado.csv"
#RUTA_STOPLIST = r"C:\Users\semin\BD2\stoplist.csv"
RUTA_STOPLIST = r"C:\Users\semin\BD2\stoplist.csv"
RUTA_NORMAS = r"C:\Users\semin\OneDrive\Escritorio\bd2_code\Clonación2\Proyecto_2_BD2\app\TESING\normas.json"
RUTA_PESOS_CAMPO = r"C:\Users\semin\OneDrive\Escritorio\bd2_code\Clonación2\Proyecto_2_BD2\app\TESING\pesos_campos.json"

knn = knnsecuencial()

motor_busqueda = MotorConsulta(
    ruta_csv=RUTA_ARCHIVO_CSV,
    ruta_indice=RUTA_INDICE_LOCAL,
    ruta_normas=RUTA_NORMAS,
    ruta_stoplist=RUTA_STOPLIST
)

@main.route('/')
def home():
    return render_template('index.html')

@main.route('/about')
def about():
    return render_template('about.html')

@main.route('/consulta', methods=['POST'])
def consulta():
    data = request.get_json()
    consulta_usuario = data.get('consulta', '')
    top_k = data.get('top_k', 10)
    print("Estoy en la consulta")
    
    try:
        # Procesar la consulta
        terminos_procesados = motor_busqueda.procesar_consulta(consulta_usuario)
        print("Términos procesados:", terminos_procesados)

        # Buscar y recuperar resultados
        resultados_busqueda = motor_busqueda.buscar(consulta_usuario, top_k=top_k)
        print(resultados_busqueda)
        print(f"Top {top_k} Resultados:", json.dumps(resultados_busqueda, indent=2, ensure_ascii=False))
        return jsonify(resultados_busqueda)
    except Exception as e:
        print(f"Error en la consulta: {str(e)}")
        return jsonify({"error": "Error interno en el servidor"}), 500

@main.route('/knn/priority', methods=['POST'])
def knn_priority():
    try:
        # Verificar que se haya enviado una imagen
        if 'image' not in request.files:
            return jsonify({"error": "No se encontró un archivo de imagen."}), 400

        image_file = request.files['image']

        # Validar que la imagen tenga un nombre válido
        if image_file.filename == '':
            return jsonify({"error": "El archivo de imagen no tiene un nombre válido."}), 400

        # Guardar el archivo en la carpeta 'uploads'
        upload_folder = os.path.join(current_app.root_path, 'static/uploads')
        os.makedirs(upload_folder, exist_ok=True)
        image_path = os.path.join(upload_folder, image_file.filename)
        image_file.save(image_path)

        # Procesar la imagen y obtener el vector
        query_vector = obtener_vector_desde_imagen(image_path)
        if query_vector is None:
            os.remove(image_path)
            return jsonify({"error": "No se pudo procesar la imagen. Verifica que sea válida."}), 400

        # Realizar la búsqueda KNN
        k = int(request.form.get('k', 8))  # Número de vecinos por defecto: 8
        results = knn.save_priority_neighbors_to_json(query_vector, k)

        # Eliminar la imagen después de procesarla
        os.remove(image_path)

        return jsonify(results)
    except Exception as e:
        print(f"Error procesando la solicitud: {str(e)}")
        return jsonify({"error": "Error interno en el servidor. Revisa los logs para más detalles."}), 500





def get_db_connection():
    conn = pg.connect(
        host=current_app.config['DB_HOST'],
        database=current_app.config['DB_NAME'],
        user=current_app.config['DB_USER'],
        password=current_app.config['DB_PASSWORD']
    )
    return conn





@main.route('/consulta/postgres', methods=['POST'])
def consulta_postgres():
    data = request.get_json()
    consulta_usuario = data.get('consulta', '')
    top_k = data.get('top_k', 10)

    try:
        db = PostgresConnector()  # Crear una instancia de PostgresConnector
        resultados = db.search2(consulta_usuario, top_k)  # Usar el método de búsqueda
        print(resultados)
        print("..................")
        return jsonify(resultados['results'])  # Devolver resultados
    except Exception as e:
        print(f"Error en la consulta a PostgreSQL: {str(e)}")
        return jsonify({"error": "Error interno en la consulta a la base de datos."}), 500


