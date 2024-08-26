import cv2
from deepface import DeepFace
from datetime import datetime, timedelta
import numpy as np
from collections import Counter
import time
import csv
import os
import json

def create_tracker():
    """Create a new tracker."""
    if cv2.__version__.startswith('4.'):
        return cv2.TrackerKCF_create()
    else:
        return cv2.legacy.TrackerKCF_create()

def cosine_similarity(embedding1, embedding2):
    """Calculate the cosine similarity between two embeddings."""
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)
    if vec1.shape != vec2.shape:
        print(f"Skipping comparison due to dimension mismatch: {vec1.shape} vs {vec2.shape}")
        return 0  # Skip this comparison or return a low similarity score
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

def find_matching_face_id(new_embedding, stored_embeddings):
    """Find the matching face ID based on face embedding similarity."""
    for face_id, embedding in stored_embeddings.items():
        similarity = cosine_similarity(embedding, new_embedding)
        if similarity > 0.6:  # Adjust threshold as needed for matching
            return face_id
    return None

def get_current_time():
    """Return the current time."""
    return datetime.now()

def save_to_csv(data, filename):
    """Save collected data to a CSV file."""
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

def save_embeddings_to_file(stored_embeddings, filename):
    """Save the stored face embeddings to a file."""
    with open(filename, 'w') as file:
        json.dump(stored_embeddings, file)

def load_embeddings_from_file(filename):
    """Load face embeddings from a file."""
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            return json.load(file)
    return {}

# File paths for storing embeddings and logs
embedding_file = 'stored_embeddings.json'
log_file = 'face_tracking_log.csv'

# Ensure the directory for the CSV file exists
log_dir = os.path.dirname(log_file)
if log_dir and not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Load stored embeddings from the file
stored_embeddings = load_embeddings_from_file(embedding_file)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

face_trackers = {}
face_times = {}
face_emotions = {}
face_last_update = {}
face_id_count = len(stored_embeddings)

last_detection_time = time.time()
detection_interval = 5  # Reduced to check more frequently

# Write CSV header if the file is empty
if not os.path.exists(log_file):
    save_to_csv(['Face ID', 'Event', 'Time', 'Duration', 'Cumulative Emotion'], log_file)

try:
    while True:
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (320, 240))  # Reduce resolution
        rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # Update all trackers
        face_ids_to_remove = []
        for face_id, tracker in face_trackers.items():
            success, bbox = tracker.update(frame_resized)
            if success:
                x, y, w, h = [int(v) for v in bbox]
                face_roi = frame_resized[y:y + h, x:x + w]
                try:
                    result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                    emotion = result[0]['dominant_emotion']
                    face_emotions[face_id].append(emotion)
                    cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame_resized, f"ID {face_id} {emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                except Exception as e:
                    print(f"Error analyzing face {face_id}: {e}")
                face_last_update[face_id] = get_current_time()
            else:
                # Check if face has timed out
                if face_times[face_id]['time_out'] is None:
                    face_times[face_id]['time_out'] = get_current_time()
                    duration = face_times[face_id]['time_out'] - face_times[face_id]['time_in']
                    cumulative_emotion = Counter(face_emotions[face_id]).most_common(1)[0][0]
                    save_to_csv([face_id, 'time-out', face_times[face_id]['time_out'], str(duration), cumulative_emotion], log_file)
                    print(f"Face ID {face_id} time-out at: {face_times[face_id]['time_out']}")
                    print(f"Face ID {face_id} was present for: {duration}")
                    print(f"Face ID {face_id} cumulative emotion: {cumulative_emotion}")
                face_ids_to_remove.append(face_id)
        
        for face_id in face_ids_to_remove:
            face_trackers.pop(face_id, None)
            face_times.pop(face_id, None)
            face_emotions.pop(face_id, None)
            face_last_update.pop(face_id, None)
            stored_embeddings.pop(face_id, None)

        current_time = time.time()
        if current_time - last_detection_time > detection_interval:
            faces = face_cascade.detectMultiScale(rgb_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            last_detection_time = current_time
            
            # Use DeepFace to get face embeddings with a consistent model
            for (x, y, w, h) in faces:
                face_roi = frame_resized[y:y + h, x:x + w]
                result = DeepFace.represent(face_roi, model_name='Facenet', enforce_detection=False)
                if result:
                    face_encoding = result[0]['embedding']
                    matched_face_id = find_matching_face_id(face_encoding, stored_embeddings)

                    if matched_face_id is None:
                        # New face detected
                        face_id_count += 1
                        tracker = create_tracker()
                        tracker.init(frame_resized, (x, y, w, h))
                        face_trackers[face_id_count] = tracker
                        face_times[face_id_count] = {'time_in': get_current_time(), 'time_out': None}
                        face_emotions[face_id_count] = []
                        face_last_update[face_id_count] = get_current_time()
                        stored_embeddings[face_id_count] = face_encoding  # Store the encoding with the new ID
                        save_to_csv([face_id_count, 'time-in', face_times[face_id_count]['time_in'], '', ''], log_file)
                        print(f"Face ID {face_id_count} time-in at: {face_times[face_id_count]['time_in']}")
                    else:
                        # Existing face detected; update tracker if necessary
                        if matched_face_id in face_last_update and get_current_time() - face_last_update[matched_face_id] > timedelta(seconds=1):
                            tracker = create_tracker()
                            tracker.init(frame_resized, (x, y, w, h))
                            face_trackers[matched_face_id] = tracker
                            face_last_update[matched_face_id] = get_current_time()
                            stored_embeddings[matched_face_id] = face_encoding  # Update stored encoding
                            print(f"Face ID {matched_face_id} re-entered at: {datetime.now()}")
                else:
                    print("No embedding found for detected face.")

        fps = 1 / (time.time() - start_time)
        cv2.putText(frame_resized, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Real-time Emotion Detection', frame_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user.")

finally:
    # Save the embeddings to the file before exiting
    save_embeddings_to_file(stored_embeddings, embedding_file)
    cap.release()
    cv2.destroyAllWindows()
