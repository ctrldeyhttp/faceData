import logging
from celery.signals import worker_process_init
from celery import shared_task
import tensorflow as tf
from api.utils import load_facenet_model, get_embedding

# Global variable for the Facenet model
facenet_model = None

@worker_process_init.connect
def initialize_worker(**kwargs):
    """
    Configure TensorFlow threading and initialize the Facenet model for each worker process.
    """
    global facenet_model

    # Configure TensorFlow threading settings
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    # Load the Facenet model
    if facenet_model is None:
        logging.info("Loading Facenet model for worker...")
        facenet_model = load_facenet_model()
        logging.info("Facenet model loaded successfully.")

@shared_task(bind=True, max_retries=3)
def process_face_embeddings(self, image_path):
    """
    Celery task to process an image and generate its face embeddings.
    """
    global facenet_model

    # Ensure the Facenet model is initialized
    if facenet_model is None:
        raise RuntimeError("Facenet model is not loaded. Initialization failed.")

    try:
        # Generate embeddings
        embedding = get_embedding(image_path)
        return embedding.tolist()
    except Exception as e:
        logging.error(f"Error processing embeddings for {image_path}: {e}")
        raise self.retry(exc=e, countdown=5)
