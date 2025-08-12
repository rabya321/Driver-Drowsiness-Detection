"""
Driver Drowsiness Detection
---------------------------
Detects signs of driver fatigue using Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR).
Works with both images and real-time webcam feed.

Usage:
    python main.py                # Runs real-time detection from webcam
    python main.py image.jpg       # Runs detection on a single image
"""

import sys
import cv2
import numpy as np
from PIL import Image
import face_recognition
from scipy.spatial import distance
import warnings

warnings.filterwarnings('ignore')

# Threshold values (tweak if needed)
EAR_THRESH = 0.25   # Eye Aspect Ratio threshold
MAR_THRESH = 0.6    # Mouth Aspect Ratio threshold
SCORE_LIMIT = 5     # Score threshold to trigger drowsiness alert

# -------------------- Utility Functions --------------------

def eye_aspect_ratio(eye):
    """Calculate Eye Aspect Ratio (EAR) for one eye."""
    A = distance.euclidean(eye[2], eye[4])
    B = distance.euclidean(eye[1], eye[5])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(top_lip, bottom_lip):
    """Calculate Mouth Aspect Ratio (MAR)."""
    A = distance.euclidean(top_lip[2], bottom_lip[2])
    B = distance.euclidean(top_lip[3], bottom_lip[3])
    C = distance.euclidean(top_lip[4], bottom_lip[4])
    D = distance.euclidean(top_lip[0], top_lip[6])
    mar = (A + B + C) / (3 * D)
    return mar

def process_frame(frame):
    """
    Process a single frame/image.
    Returns:
        eye_flag  (bool): True if eyes are closed
        mouth_flag (bool): True if yawning
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(frame_rgb)

    eye_flag = mouth_flag = False

    for face_location in face_locations:
        landmarks = face_recognition.face_landmarks(frame_rgb, [face_location])[0]
        left_eye = np.array(landmarks['left_eye'])
        right_eye = np.array(landmarks['right_eye'])
        top_lip = np.array(landmarks['top_lip'])
        bottom_lip = np.array(landmarks['bottom_lip'])

        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
        mar = mouth_aspect_ratio(top_lip, bottom_lip)

        if ear < EAR_THRESH:
            eye_flag = True
        if mar > MAR_THRESH:
            mouth_flag = True

    return eye_flag, mouth_flag

# -------------------- Modes --------------------

def run_image_mode(image_path):
    """Run detection on a single image."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Unable to load image: {image_path}")
        return
    eye_flag, mouth_flag = process_frame(img)
    print(f"Eyes closed: {eye_flag}, Yawning: {mouth_flag}")
    cv2.imshow("Drowsiness Detection - Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_webcam_mode():
    """Run real-time detection from webcam."""
    video_cap = cv2.VideoCapture(0)
    score = 0
    count = 0

    if not video_cap.isOpened():
        print("[ERROR] Unable to access webcam.")
        return

    print("[INFO] Press 'q' to quit.")

    while True:
        success, frame = video_cap.read()
        if not success:
            break

        frame = cv2.resize(frame, (800, 500))
        count += 1

        if count % 5 == 0:
            eye_flag, mouth_flag = process_frame(frame)
            if eye_flag or mouth_flag:
                score += 1
            else:
                score = max(score - 1, 0)

        # Display score
        cv2.putText(frame, f"Score: {score}", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Drowsiness alert
        if score >= SCORE_LIMIT:
            cv2.putText(frame, "DROWSY!", (frame.shape[1] - 150, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        cv2.imshow("Drowsiness Detection - Webcam", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_cap.release()
    cv2.destroyAllWindows()

# -------------------- Main Entry --------------------

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_image_mode(sys.argv[1])
    else:
        run_webcam_mode()
# -------------------- End of Main.py --------------------
