from flask import Blueprint, render_template, request, jsonify, current_app
import os
from .Final2 import IndiceInvertido, MotorConsulta
from .Multidimencional.knn_secuencial import knnsecuencial, obtener_vector_desde_imagen
import json

main = Blueprint('main', __name__)

# Configuración de rutas de archivos
RUTA_INDICE_LOCAL = r"C:\Users\semin\BD2"
RUTA_ARCHIVO_CSV = r"C:\Users\semin\BD2\spotify_songs.csv"
RUTA_STOPLIST = r"C:\Users\semin\BD2\stoplist.csv"
RUTA_NORMAS = r"C:\Users\semin\BD2\normas.json"
RUTA_PESOS_CAMPO = r"C:\Users\semin\BD2\pesos_campos.json"

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
