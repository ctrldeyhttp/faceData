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
from mtcnn import MTCNN


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

def upload_image(request):

    frame = capture_face_from_webcam()
    if frame is None:
        return render(request, 'image_recognition_result.html', {'error': 'Unable to capture face from webcam.'})
    
    # Convert the frame to RGB (DeepFace expects RGB format)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    detector = MTCNN()
    detections = detector.detect_faces(frame_rgb)
    
    if detections is None or len(detections) == 0:
        return render(request, 'image_recognition_result.html', {'error': 'No face detected in the image.'})

    # Determine the largest face in the image by calculating the area of the bounding box
    largest_face = None
    largest_area = 0
    largest_face_box = None
    
    for detection in detections:
        x, y, w, h = detection['box']  # Use the 'box' key from the MTCNN detection
        area = w * h
        if area > largest_area:
            largest_area = area
            largest_face = frame_rgb[y:y + h, x:x + w]  # Crop out the largest face
            largest_face_box = (x, y, w, h)

    if largest_face is None:
        return render(request, 'image_recognition_result.html', {'error': 'No face detected in the image.'})

    # Draw a bounding box around the largest face on the original frame
    x, y, w, h = largest_face_box
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    captured_image = frame_to_base64(frame)


    # Process the captured frame with DeepFace to extract embeddings
    result = DeepFace.represent(
            img_path=largest_face,               # Pass the frame directly
            model_name="Facenet512",      # Use Facenet512 for embedding
            #detector_backend="retinaface", # Use RetinaFace for detection
            align=True,
            enforce_detection=True,         # Ensure a face is detected
            normalization="ArcFace"
        )
    if not result:
        return render(request, 'image_recognition_result.html', {'error': 'Face not detected in the webcam frame.'})

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
            'name': "No match found.122",
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