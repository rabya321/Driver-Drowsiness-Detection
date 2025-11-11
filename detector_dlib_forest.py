print("[DEBUG] SCRIPT STARTED (DLib + Isolation Forest Version)")
"""
TRAIN ANOMALY DETECTOR (DLib + Isolation Forest)
----------------------
This script processes 'test_video.mp4', extracts
DLib (face_recognition) features, and trains an Isolation Forest
to learn "normal" driving patterns.
"""

import cv2
import numpy as np
import face_recognition  # Using DLib!
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import warnings
import time
import pickle

warnings.filterwarnings('ignore')

# --- DLib Feature Extractor ---
# These functions are from your original project code

def eye_aspect_ratio(eye):
    """Calculate Eye Aspect Ratio (EAR) for DLib landmarks."""
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    if C == 0: return 0.0
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(top_lip, bottom_lip):
    """Calculate Mouth Aspect Ratio (MAR) for DLib landmarks."""
    A = distance.euclidean(top_lip[2], bottom_lip[2])
    B = distance.euclidean(top_lip[3], bottom_lip[3])
    C = distance.euclidean(top_lip[4], bottom_lip[4])
    D = distance.euclidean(top_lip[0], top_lip[6]) # This is width
    if D == 0: return 0.0
    mar = (A + B + C) / (3 * D)
    return mar

def get_features(frame):
    """Extracts EAR and MAR using DLib (face_recognition)"""
    
    # Resize for faster processing, same as your optimized script
    height, width = frame.shape[:2]
    scale = 1.0
    if width > 640:
        scale = 640 / width
        frame_small = cv2.resize(frame, (int(width * scale), int(height * scale)))
    else:
        frame_small = frame
        
    # DLib needs RGB
    frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

    # Find faces (using fast 'hog' model)
    face_locations = face_recognition.face_locations(frame_rgb, model='hog')
    if not face_locations:
        return None # No face found

    # Get landmarks for the first face
    landmarks_list = face_recognition.face_landmarks(frame_rgb, face_locations)
    if not landmarks_list:
        return None # No landmarks found

    landmarks = landmarks_list[0]

    # Extract landmark points
    left_eye = np.array(landmarks['left_eye'])
    right_eye = np.array(landmarks['right_eye'])
    top_lip = np.array(landmarks['top_lip'])
    bottom_lip = np.array(landmarks['bottom_lip'])

    # Calculate EAR and MAR
    ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
    mar = mouth_aspect_ratio(top_lip, bottom_lip)
    
    feature_vector = [ear, mar]
    return feature_vector

# --- Main Training ---

def train():
    print("[INFO] Starting training data extraction...")
    video_cap = cv2.VideoCapture('normal_driving.mp4') 

    if not video_cap.isOpened():
        print("[DEBUG ERROR] OpenCV could not open the video file.")
        return
    else:
        print("[DEBUG INFO] Video file opened successfully.")

    all_features = []
    frame_count = 0
    found_face_count = 0

    while video_cap.isOpened():
        success, frame = video_cap.read()
        if not success:
            break 
        
        frame_count += 1
        features = get_features(frame)

        if features is None:
            continue # No face found in this frame

        found_face_count += 1
        all_features.append(features)

        if frame_count % 30 == 0: # Print status
            print(f"[DEBUG STATUS] Processing frame {frame_count}... Faces found: {found_face_count}")

    video_cap.release()
    print(f"[INFO] Total frames read: {frame_count}")
    print(f"[INFO] Frames with a face found: {found_face_count}")

    if not all_features:
        print("[ERROR] No features were extracted. Cannot train model.")
        return

    # 1. Scale the data
    scaler = StandardScaler()
    all_features_scaled = scaler.fit_transform(all_features)
    
    # 2. Build and train Isolation Forest
    print("[INFO] Training Isolation Forest...")
    model = IsolationForest(contamination=0.01, random_state=42)
    model.fit(all_features_scaled)
    print("[INFO] Model training complete.")

    # 3. Save the model and the scaler
    with open('dlib_anomaly_model.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    with open('dlib_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print("[INFO] Training complete. Model 'dlib_anomaly_model.pkl' and 'dlib_scaler.pkl' saved.")


if __name__ == "__main__":
    train()