"""
Optimized Driver Drowsiness Detection - Industry Standards
----------------------------------------------------------
Target: 30 FPS, <100ms latency, <33ms processing per frame
"""

import sys
import cv2
import numpy as np
import face_recognition
from scipy.spatial import distance
import time
import threading
from collections import deque
import warnings

warnings.filterwarnings('ignore')

# Industry-standard thresholds
EAR_THRESH = 0.25
MAR_THRESH = 0.6
SCORE_LIMIT = 5
TARGET_FPS = 30
MAX_PROCESSING_TIME_MS = 33  # For 30 FPS capability

class PerformanceMonitor:
    """Monitor FPS and latency metrics"""
    def __init__(self, window_size=30):
        self.frame_times = deque(maxlen=window_size)
        self.processing_times = deque(maxlen=window_size)
        self.last_time = time.time()
    
    def update_frame(self):
        current_time = time.time()
        self.frame_times.append(current_time - self.last_time)
        self.last_time = current_time
    
    def update_processing(self, processing_time):
        self.processing_times.append(processing_time * 1000)  # Convert to ms
    
    def get_fps(self):
        if len(self.frame_times) < 2:
            return 0
        return 1.0 / (sum(self.frame_times) / len(self.frame_times))
    
    def get_avg_processing_time_ms(self):
        if not self.processing_times:
            return 0
        return sum(self.processing_times) / len(self.processing_times)
    
    def get_latency_ms(self):
        return self.get_avg_processing_time_ms()

def eye_aspect_ratio(eye):
    """Calculate Eye Aspect Ratio (EAR) - optimized"""
    A = distance.euclidean(eye[2], eye[4])
    B = distance.euclidean(eye[1], eye[5])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(top_lip, bottom_lip):
    """Calculate Mouth Aspect Ratio (MAR) - optimized"""
    A = distance.euclidean(top_lip[2], bottom_lip[2])
    B = distance.euclidean(top_lip[3], bottom_lip[3])
    C = distance.euclidean(top_lip[4], bottom_lip[4])
    D = distance.euclidean(top_lip[0], top_lip[6])
    return (A + B + C) / (3 * D)

def process_frame_optimized(frame):
    """
    Optimized frame processing for industry standards
    Target: <33ms processing time
    """
    start_time = time.time()
    
    # Resize for faster processing while maintaining accuracy
    height, width = frame.shape[:2]
    if width > 640:
        scale = 640 / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        frame_small = cv2.resize(frame, (new_width, new_height))
    else:
        frame_small = frame
    
    frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
    
    # Use faster face detection model
    face_locations = face_recognition.face_locations(frame_rgb, model='hog') #HOG(Histogram of Oriented Gradients) is faster than CNN
    
    eye_flag = mouth_flag = False
    
    if face_locations:  # Only process if face detected
        # Process only the first (largest) face for speed
        landmarks = face_recognition.face_landmarks(frame_rgb, [face_locations[0]])[0]
        
        left_eye = np.array(landmarks['left_eye'])
        right_eye = np.array(landmarks['right_eye'])
        top_lip = np.array(landmarks['top_lip'])
        bottom_lip = np.array(landmarks['bottom_lip'])
        
        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
        mar = mouth_aspect_ratio(top_lip, bottom_lip)
        
        eye_flag = ear < EAR_THRESH
        mouth_flag = mar > MAR_THRESH
    
    processing_time = time.time() - start_time
    return eye_flag, mouth_flag, processing_time

class DrowsinessDetector:
    """Industry-standard drowsiness detection system"""
    
    def __init__(self):
        self.score = 0
        self.performance_monitor = PerformanceMonitor()
        self.alert_active = False
        self.last_alert_time = 0
        
    def run_webcam_mode(self):
        """Optimized webcam processing for 30 FPS target"""
        video_cap = cv2.VideoCapture(0)
        
        # Set camera properties for optimal performance
        video_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        video_cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        
        if not video_cap.isOpened():
            print("[ERROR] Unable to access webcam.")
            return
        
        print(f"[INFO] Target: {TARGET_FPS} FPS, <{MAX_PROCESSING_TIME_MS}ms processing")
        print("[INFO] Press 'q' to quit.")
        
        frame_count = 0
        
        while True:
            self.performance_monitor.update_frame()
            success, frame = video_cap.read()
            
            if not success:
                break
            
            frame_count += 1
            
            # Process every frame for industry standard (vs every 5th)
            eye_flag, mouth_flag, processing_time = process_frame_optimized(frame)
            self.performance_monitor.update_processing(processing_time)
            
            # Update drowsiness score
            if eye_flag or mouth_flag:
                self.score = min(self.score + 1, SCORE_LIMIT + 2)
            else:
                self.score = max(self.score - 1, 0)
            
            # Alert logic with timing
            current_time = time.time()
            if self.score >= SCORE_LIMIT:
                if not self.alert_active or (current_time - self.last_alert_time) > 0.5:
                    self.alert_active = True
                    self.last_alert_time = current_time
            else:
                self.alert_active = False
            
            # Display metrics and alerts
            self.draw_interface(frame, processing_time)
            
            cv2.imshow("Industry-Standard Drowsiness Detection", frame)
            
            # Maintain target FPS
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Print final performance metrics
        self.print_performance_summary()
        video_cap.release()
        cv2.destroyAllWindows()
    
    def draw_interface(self, frame, processing_time):
        """Draw performance metrics and alerts"""
        fps = self.performance_monitor.get_fps()
        avg_latency = self.performance_monitor.get_avg_processing_time_ms()
        
        # Performance metrics
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Latency: {avg_latency:.1f}ms", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Score: {self.score}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Current frame processing time
        color = (0, 255, 0) if processing_time * 1000 < MAX_PROCESSING_TIME_MS else (0, 0, 255)
        cv2.putText(frame, f"Frame: {processing_time*1000:.1f}ms", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Drowsiness alert
        if self.alert_active:
            cv2.putText(frame, "DROWSY ALERT!", (frame.shape[1] - 250, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            # Add red border for alert
            cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), 
                         (0, 0, 255), 5)
    
    def print_performance_summary(self):
        """Print final performance metrics"""
        fps = self.performance_monitor.get_fps()
        avg_latency = self.performance_monitor.get_avg_processing_time_ms()
        
        print("\n" + "="*50)
        print("PERFORMANCE SUMMARY")
        print("="*50)
        print(f"Average FPS: {fps:.2f}")
        print(f"Average Processing Latency: {avg_latency:.2f}ms")
        print(f"Target FPS: {TARGET_FPS} ({'✓' if fps >= TARGET_FPS else '✗'})")
        print(f"Target Latency: <{MAX_PROCESSING_TIME_MS}ms ({'✓' if avg_latency < MAX_PROCESSING_TIME_MS else '✗'})")
        print("="*50)

def run_image_mode(image_path):
    """Test processing time on single image"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Unable to load image: {image_path}")
        return
    
    eye_flag, mouth_flag, processing_time = process_frame_optimized(img)
    print(f"Eyes closed: {eye_flag}, Yawning: {mouth_flag}")
    print(f"Processing time: {processing_time*1000:.2f}ms")
    
    cv2.imshow("Drowsiness Detection - Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = DrowsinessDetector()
    
    if len(sys.argv) > 1:
        run_image_mode(sys.argv[1])
    else:
        detector.run_webcam_mode()