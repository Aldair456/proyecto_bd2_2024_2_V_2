import pandas as pd

def filtrar_columnas(input_csv, output_csv, columnas_a_mantener):
    df = pd.read_csv(input_csv)
    df_filtrado = df[columnas_a_mantener]
    df_filtrado.to_csv(output_csv, index=False)

input_csv = r"C:\Users\semin\OneDrive\Escritorio\bd2_code\Clonación1\Proyecto_2_BD2\spotify_songs.csv"
output_csv = "spotify_songs_filtrado.csv"
columnas_a_mantener = ["track_id", "track_name", "track_artist", "lyrics", "playlist_name"]

filtrar_columnas(input_csv, output_csv, columnas_a_mantener)
