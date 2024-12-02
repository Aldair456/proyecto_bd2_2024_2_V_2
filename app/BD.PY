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
            playlist_name VARCHAR,
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
            INSERT INTO songs.spotify_songs (track_id, track_name, track_artist, lyrics, playlist_name)
            VALUES (%s, %s, %s, %s,%s)
            """, (row['track_id'], row['track_name'], row['track_artist'], row['lyrics'],row['playlist_name']))
        
        self.conn.commit()

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

# # Uso

db = PostgresConnector()
# db.eliminarchema()

db.setup_database()
db.load_data('spotify_songs_filtrado.csv')

# Búsqueda
results = db.search("amor no es amor sin amor", 5)
for result in results['results']:
    print(f"ID de la canción: {result['track_id']}")
    print(f"Nombre de la canción: {result['track_name']}")
    print(f"Artista: {result['track_artist']}")
    print(f"Letras: {result['lyrics']}")
    print(f"Posición de la fila: {result['playlist_name']}")
    print(f"Similitud: {result['similitud']}")
