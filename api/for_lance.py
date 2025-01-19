from django.shortcuts import render, redirect
from .models import UploadedImage
from django.core.files.storage import default_storage
import cv2
from deepface import DeepFace
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from django.db import connection  # For executing raw SQL queries
from django.utils import timezone

def upload_image(request):
    if request.method == 'POST':
        name = request.POST.get('name')  # Get name from the form
        image_file = request.FILES['image']  # Get image file from the form

        # Save the uploaded file to the media folder using default storage
        image_path = default_storage.save(f"uploads/{image_file.name}", image_file)
        absolute_image_path = default_storage.path(image_path)  # Get the absolute path

        try:
            # Process the uploaded image with DeepFace
            result = DeepFace.represent(
                img_path=absolute_image_path, 
                model_name="Facenet512", 
                detector_backend="retinaface",
                align=True,
                enforce_detection=True,
                normalization="ArcFace"
            )

            # Extract the embedding
            embedding = result[0]["embedding"]

            print(f"Type of embedding: {type(embedding)}")
            print(f"Embedding length: {len(embedding)}")

            # Convert the embedding to a Python list (if not already a list)
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()

            # Ensure the embedding is of size 512 (this is a sanity check)
            if len(embedding) != 512:
                raise ValueError(f"Embedding is not of size 512. Got size {len(embedding)}")

            # Create the HNSW index in PostgreSQL if it doesn't exist
            with connection.cursor() as cursor:
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS api_uploadedimage_embedding_idx ON api_uploadedimage USING hnsw (embedding vector_cosine_ops);"
                )
                print("Cosine similarity HNSW index created on embeddings.")

            # Insert the record into the PostgreSQL api_uploadedimage table
            with connection.cursor() as cursor:
                sql = """
                INSERT INTO api_uploadedimage (name, image, date_uploaded, embedding)
                VALUES (%s, %s, %s, %s)
                """
                cursor.execute(sql, [
                    name,  # Name from the form
                    image_path,  # Relative path to the image (as stored in MEDIA_ROOT)
                    timezone.now(),  # Current timestamp for date_uploaded
                    embedding  # The embedding as a list (PostgreSQL will convert it to vector(512))
                ])
        except Exception as e:
            print(f"Error during image processing or database insertion: {e}")
            # Optionally, show an error page or flash a message

        # Redirect to a success page or render a response
        return redirect('success')  # Replace with the actual success URL

    return render(request, 'upload_image.html')


def success(request):
    return render(request, 'success.html')