import atexit
from django.apps import AppConfig
from .database import init_connection_pool, close_all_connections
import logging
from django.apps import AppConfig
from .utils import load_facenet_model

class ApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'

    def ready(self):
        # Initialize the connection pool when the app starts
        init_connection_pool()
        logging.info("Database connection pool initialized.")
        from api.utils import load_facenet_model
        load_facenet_model()

        # Register the shutdown hook to close the connection pool
        atexit.register(close_all_connections)
