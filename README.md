# AI-Powered Online Exam Proctoring System

## Overview
An automated proctoring solution using YOLOv8 and OpenCV to detect unauthorized devices (phones), suspicious head movements, and eye presence during online exams.

## Features
- **Object Detection**: Real-time identification of cell phones and extra people.
- **Centroid Tracking**: Monitors head position to detect lateral gaze deviation.
- **Smart Alerts**: Visual warnings triggered only after sustained suspicious behavior.

## 🛠️ Installation
1. Clone the repo: `git clone https://github.com/your-username/ai-proctoring-system.git`
2. Install dependencies: `pip install opencv-python ultralytics numpy`
3. Run the proctor: `python proctor.py`
