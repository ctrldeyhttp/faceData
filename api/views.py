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
from django.db import connection
from retinaface import RetinaFace
from .database import get_connection, release_connection
from .utils import load_facenet_model, get_embedding



def capture_face_from_webcam():
    """Capture a frame from the webcam and return it."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return None
    
    time.sleep(0.5)
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

    # Save cropped face temporarily for embedding generation
    cropped_face_path = 'temp_cropped_face.jpg'
    cv2.imwrite(cropped_face_path, cropped_face)

    # Process the cropped face with the preloaded Facenet512 model
    try:
        target_embedding = get_embedding(cropped_face_path)
        if isinstance(target_embedding, np.ndarray):
            target_embedding = target_embedding.tolist()
    except Exception as e:
        return render(request, 'image_recognition_result.html', {'error': f"Error generating embedding: {str(e)}"})
    finally:
        if os.path.exists(cropped_face_path):
            os.remove(cropped_face_path)

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
    connection = None
    try:
            connection = get_connection()
            cursor = connection.cursor()

            start_time = time.time()
            threshold = 0.3

            cursor.execute(
                """
                SELECT id, name, embedding <=> %s::vector as distance
                FROM api_uploadedimage
                WHERE embedding IS NOT NULL
                AND embedding <=> %s::vector < %s
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

                if distance < best_similarity:
                    best_similarity = distance
                    best_match = image_name

            query_time = end_time - start_time
            print(f"Query completed in {query_time:.4f} seconds.")

            return best_match if best_match else None

    except Exception as e:
            print(f"Database query failed: {str(e)}")
            return None

    finally:
            if connection:
                release_connection(connection)
                
def show_uploaded_images(request):
    images = UploadedImage.objects.all()

    return render(request, 'show_images.html', {'images': images})

def frame_to_base64(frame):
    """Convert a webcam frame (NumPy array) to Base64 string."""
    _, buffer = cv2.imencode('.jpg', frame)
    base64_image = base64.b64encode(buffer).decode('utf-8')
    return base64_image