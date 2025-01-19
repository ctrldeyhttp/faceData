import cv2
import psycopg2
from deepface import DeepFace
import numpy as np
import pickle
from django.shortcuts import render, redirect
from .models import UploadedImage
from django.core.files.storage import default_storage
from datetime import datetime
import tensorflow as tf
import os
import base64
import time
from sklearn.metrics.pairwise import cosine_similarity
from django.db import connection  # For executing raw SQL queries
from retinaface import RetinaFace




model_path = os.path.join(os.path.dirname(__file__), 'liveness.model')
model = tf.keras.models.load_model(model_path)

def capture_face_from_webcam():
    """Capture a frame from the webcam and return it."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return None
    
    time.sleep(2)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return frame
    else:
        print("Error: Unable to capture a frame.")
        return None

def draw_bounding_boxes(frame):
    """
    Detect all faces in the frame and draw bounding boxes around them.
    Highlight the largest face with a distinct color.
    """
    detections = RetinaFace.detect_faces(frame)

    if not detections:
        print("No faces detected.")
        return frame, None

    largest_face = None
    largest_area = 0

    for key, face_info in detections.items():
        # Extract the facial area
        facial_area = face_info["facial_area"]
        x1, y1, x2, y2 = facial_area

        # Calculate the area of the bounding box
        area = (x2 - x1) * (y2 - y1)

        # Draw bounding box (default color: blue)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Identify the largest face
        if area > largest_area:
            largest_area = area
            largest_face = facial_area

    # Highlight the largest face with a different color (e.g., green)
    if largest_face:
        x1, y1, x2, y2 = largest_face
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    return frame, largest_face

def upload_image(request):
    # Capture the webcam frame
    frame = capture_face_from_webcam()
    if frame is None:
        return render(request, 'image_recognition_result.html', {'error': 'Unable to capture face from webcam.'})

    # Draw bounding boxes around all detected faces
    processed_frame, largest_face = draw_bounding_boxes(frame)

    if largest_face is None:
        return render(request, 'image_recognition_result.html', {'error': 'No faces detected in the webcam frame.'})

    # Convert the frame with bounding boxes into Base64 for display in the webpage
    captured_image = frame_to_base64(processed_frame)

    # Extract the largest face for processing
    x1, y1, x2, y2 = largest_face
    cropped_face = frame[y1:y2, x1:x2]

    # Process the cropped face with DeepFace
    try:
        result = DeepFace.represent(
            img_path=cropped_face,         # Pass the cropped face
            model_name="Facenet512",       # Use Facenet512 for embedding
            detector_backend="retinaface", # Use RetinaFace for detection
            align=True,
            enforce_detection=False,       # Disable detection since we've already done it
            normalization="ArcFace"
        )
    except Exception as e:
        return render(request, 'image_recognition_result.html', {'error': f"DeepFace processing error: {str(e)}"})

    if not result:
        return render(request, 'image_recognition_result.html', {'error': 'Face not detected in the cropped region.'})

    # Extract the embedding
    target_embedding = result[0]["embedding"]

    # Compare the extracted target embedding with the database images
    recognized_name = compare_face_to_uploaded_images(target_embedding)

    if recognized_name:
        return render(request, 'image_recognition_result.html', {
            'name': recognized_name,
            'captured_image': captured_image
        })    
    else:
        return render(request, 'image_recognition_result.html', {
            'name': "No match found.",
            'captured_image': captured_image
        })


def success(request):
    return render(request, 'success.html')

def compare_face_to_uploaded_images(target_embedding):
    """Compare the given embedding with stored embeddings in the UploadedImage model."""

    
    # Connect to PostgreSQL database to query pgvector embeddings
    conn = psycopg2.connect(
        dbname="railway", 
        user="postgres", 
        password="MbNDt6E2SeF26YMjH3kiE27NC-lmvBnl", 
        host="autorack.proxy.rlwy.net", 
        port="17564"
    )

    cursor = conn.cursor()
    
    start_time = time.time()
    
    threshold = 0.3
     
    
    print(f"Type of target_embedding before querying: {type(target_embedding)}")

    # Query the database to find the most similar embeddings
    cursor.execute(
        """
        SELECT id, name, embedding <=> %s::vector as distance
        FROM api_uploadedimage
        WHERE embedding IS NOT NULL
        AND embedding <=> %s::vector < %s  -- Add threshold condition here
        ORDER BY distance ASC
        LIMIT 5
        """,
        (target_embedding, target_embedding, threshold)
    )

    results = cursor.fetchall()
    
    end_time = time.time()

    best_match = None
    best_similarity = float('inf')  # To store the highest similarity score

    for result in results:
        image_id, image_name, distance, *extra_values = result
        distance = float(distance)
        print(f"Name: {image_name}, Distance: {distance}")
        
        # Check if the distance is below the threshold

        if distance < best_similarity:
            best_similarity = distance
            best_match = image_name
                
                
    query_time = end_time - start_time
    print(f"Query completed in {query_time:.4f} seconds.")
    cursor.close()
    conn.close()

    return best_match if best_match else "No match found"

def show_uploaded_images(request):
    images = UploadedImage.objects.all()

    return render(request, 'show_images.html', {'images': images})

def frame_to_base64(frame):
    """Convert a webcam frame (NumPy array) to Base64 string."""
    _, buffer = cv2.imencode('.jpg', frame)
    base64_image = base64.b64encode(buffer).decode('utf-8')
    return base64_image