"""
Optimized Driver Drowsiness Detection - Industry Standards
----------------------------------------------------------
Target: 30 FPS, <100ms latency, <33ms processing per frame

Features:
- DLib ('face_recognition') for fast feature extraction
- Unsupervised Anomaly Detection (Isolation Forest)
- Drowsiness scoring (using EAR + MAR)
- Occlusion checking
- Ambiguity-aware logic
"""

import cv2
import numpy as np
import face_recognition 
from scipy.spatial import distance
import time
from collections import deque
import warnings
import pickle 
from sklearn.ensemble import IsolationForest 

warnings.filterwarnings('ignore')

# --- Constants ---
EAR_THRESH = 0.25      # Drowsiness
MAR_THRESH = 0.5       # Yawn threshold (NEW)
SCORE_LIMIT = 5        # Drowsiness
TARGET_FPS = 30
MAX_PROCESSING_TIME_MS = 33

# --- Performance Monitor Class (Same as before) ---
class PerformanceMonitor:
    def __init__(self, window_size=30):
        self.frame_times = deque(maxlen=window_size)
        self.processing_times = deque(maxlen=window_size)
        self.last_time = time.time()
    
    def update_frame(self):
        current_time = time.time()
        self.frame_times.append(current_time - self.last_time)
        self.last_time = current_time
    
    def update_processing(self, processing_time):
        self.processing_times.append(processing_time * 1000) # ms
    
    def get_fps(self):
        if len(self.frame_times) < 2: return 0
        return 1.0 / (sum(self.frame_times) / len(self.frame_times))
    
    def get_avg_processing_time_ms(self):
        if not self.processing_times: return 0
        return sum(self.processing_times) / len(self.processing_times)

# --- DLib Feature Extractor Functions ---

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    if C == 0: return 0.0
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(top_lip, bottom_lip):
    A = distance.euclidean(top_lip[2], bottom_lip[2])
    B = distance.euclidean(top_lip[3], bottom_lip[3])
    C = distance.euclidean(top_lip[4], bottom_lip[4])
    D = distance.euclidean(top_lip[0], top_lip[6]) # Width
    if D == 0: return 0.0
    mar = (A + B + C) / (3 * D)
    return mar

def get_features(frame):
    """Extracts EAR and MAR using DLib (face_recognition)"""
    height, width = frame.shape[:2]
    scale = 1.0
    if width > 640:
        scale = 640 / width
        frame_small = cv2.resize(frame, (int(width * scale), int(height * scale)))
    else:
        frame_small = frame
        
    frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(frame_rgb, model='hog')
    if not face_locations:
        return None 

    landmarks_list = face_recognition.face_landmarks(frame_rgb, face_locations)
    if not landmarks_list:
        return None 

    landmarks = landmarks_list[0]

    try:
        left_eye = np.array(landmarks['left_eye'])
        right_eye = np.array(landmarks['right_eye'])
        top_lip = np.array(landmarks['top_lip'])
        bottom_lip = np.array(landmarks['bottom_lip'])
    except:
        return None 

    ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
    mar = mouth_aspect_ratio(top_lip, bottom_lip)
    
    feature_vector = [ear, mar]
    return feature_vector
        
# --- Main Drowsiness Detector Class ---

class DrowsinessDetector:
    def __init__(self):
        self.score = 0
        self.performance_monitor = PerformanceMonitor()
        
        print("[INFO] Loading anomaly detection model...")
        try:
            with open('dlib_anomaly_model.pkl', 'rb') as f:
                self.anomaly_model = pickle.load(f)
            with open('dlib_scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            print("[INFO] Anomaly model and scaler loaded.")
        except Exception as e:
            print(f"[ERROR] Could not load model/scaler: {e}")
            print("[ERROR] Please run the training script first.")
            exit()
        
        self.alert_active = False
        self.occlusion_active = False
        self.anomaly_active = False

    def draw_interface(self, frame, processing_time):
        fps = self.performance_monitor.get_fps()
        avg_latency = self.performance_monitor.get_avg_processing_time_ms()
        
        # Performance
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Latency: {avg_latency:.1f}ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        color = (0, 255, 0) if processing_time * 1000 < MAX_PROCESSING_TIME_MS else (0, 0, 255)
        cv2.putText(frame, f"Frame: {processing_time*1000:.1f}ms", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Status
        status_text = "STATUS: NOMINAL"
        status_color = (0, 255, 0)
        if self.occlusion_active:
            status_text = "INVALID: SENSOR OCCLUDED"
            status_color = (0, 255, 255)
        elif self.anomaly_active:
            status_text = "STATUS: ANOMALY DETECTED"
            status_color = (0, 165, 255)
            
        cv2.putText(frame, status_text, (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame, f"Score: {self.score}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Main Drowsiness Alert
        if self.alert_active:
            alert_text = "DROWSY ALERT!"
            cv2.putText(frame, alert_text, (frame.shape[1] - 300, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), (0, 0, 255), 5)


    def run_webcam_mode(self):
        video_cap = cv2.VideoCapture(0)
        video_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        video_cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        
        if not video_cap.isOpened():
            print("[ERROR] Unable to access webcam.")
            return

        print(f"[INFO] Target: {TARGET_FPS} FPS, <{MAX_PROCESSING_TIME_MS}ms processing")
        print("[INFO] Press 'q' to quit.")
        
        while True:
            self.performance_monitor.update_frame()
            success, frame = video_cap.read()
            if not success: break
            
            start_time = time.time()
            features = get_features(frame) 
            processing_time = time.time() - start_time
            self.performance_monitor.update_processing(processing_time)
            
            # 1. Handle Occlusion (Hard Filter)
            if features is None:
                self.occlusion_active = True
                self.anomaly_active = False
                self.score = max(self.score - 1, 0) 
            
            # 2. Process Valid Data
            else:
                self.occlusion_active = False
                ear, mar = features
                
                # --- Anomaly Logic (NEW) ---
                scaled_features = self.scaler.transform([features])
                prediction = self.anomaly_model.predict(scaled_features)
                self.anomaly_active = (prediction == -1)
                
                # --- Drowsiness Logic (UPDATED) ---
                if ear < EAR_THRESH or mar > MAR_THRESH: # <-- CHANGE IS HERE
                    self.score = min(self.score + 1, SCORE_LIMIT + 2)
                else:
                    self.score = max(self.score - 1, 0)

            # Final Alert Logic
            self.alert_active = (self.score >= SCORE_LIMIT)
            
            self.draw_interface(frame, processing_time)
            cv2.imshow("Industry-Standard Drowsiness Detection (DLib)", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = DrowsinessDetector()
    detector.run_webcam_mode()