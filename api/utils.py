from deepface import DeepFace
import numpy as np

# Global variable to hold the preloaded model
facenet_model = None

def load_facenet_model():
    """
    Load and cache the Facenet512 model.
    
    Returns:
        facenet_model: The loaded Facenet512 model.
    """
    global facenet_model
    if facenet_model is None:
        print("Loading Facenet512 model...")
        facenet_model = DeepFace.build_model("Facenet512")
    return facenet_model

def get_embedding(image_path):
    """
    Generate a 512-dimensional embedding from the given image using DeepFace.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: A 512-dimensional embedding vector.
    """
    try:
        # Ensure the model is loaded
        load_facenet_model()

        # Use DeepFace to directly generate the embedding
        embeddings = DeepFace.represent(
            img_path=image_path,
            model_name="Facenet512",
            align=True,
            enforce_detection=False,  # Assumes face detection is done earlier
            normalization="ArcFace"   # Normalization expected by Facenet
        )

        # DeepFace.represent returns a list; extract the first embedding
        if embeddings and isinstance(embeddings, list) and 'embedding' in embeddings[0]:
            return np.array(embeddings[0]['embedding'])  # Convert to numpy array
        else:
            raise ValueError("Embedding data not found in the response.")

    except Exception as e:
        print(f"Error in generating embedding: {e}")
        raise
