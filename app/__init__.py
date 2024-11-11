from flask import Flask
from .routes import main

def create_app():
    app = Flask(__name__)

    # Configuración de la aplicación (opcional)
    app.config['SECRET_KEY'] = 'random'

    app.register_blueprint(main)

    return app
