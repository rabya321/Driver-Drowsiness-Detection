# Driver Drowsiness Detection System

This project detects signs of driver fatigue in real-time using a webcam feed.  
It monitors **eye closure** and **yawning** by tracking facial landmarks, and displays alerts when drowsiness is detected.

## Features
- Real-time face, eye, and mouth detection using `face_recognition` & `OpenCV`
- Eye Aspect Ratio (EAR) for detecting prolonged eye closure
- Mouth Aspect Ratio (MAR) for detecting yawning
- Adjustable thresholds for sensitivity
- Score-based drowsiness detection (avoids false alarms)
- Works with any USB webcam

## Tech Stack
- **Python 3.13**
- OpenCV
- face_recognition (dlib-based)
- NumPy, Pillow, Matplotlib, SciPy

## 📂 Project Structure
Driver-Drowsiness-Detection
- drowsiness_detection.py — Main Script for detection
- drowsiness_detection.ipynb — development & explanation
- person.jpeg — Sample input image
- dlib-20.0.0-cp313-cp313-win_amd64.whl
- requirements.txt — Python dependencies
- README.md — Project documentation
- assets/ — Results folder

## Installation
1. Clone the repo:
- git clone https://github.com/rabya321/Driver-Drowsiness-Detection.git
- cd Driver-Drowsiness-Detection
2. Create and activate a virtual environment
- python -m venv venv
- source venv/bin/activate   # Mac/Linux
- venv\Scripts\activate      # Windows
3. Install dlib
- pip install dlib-20.0.0-cp313-cp313-win_amd64.whl
4. Install dependencies
- pip install -r requirements.txt
- pip install cmake

## Usage
- **Run with sample image**
python drowsiness_detection.py person.jpeg

- **Run with webcam**
python drowsiness_detection.py


##  Example Output
![Demo Output](assets/demo.png)
![Webcam Output] (assests/demo.png)

