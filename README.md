# 🖥️ Backend: CV-Based Crowd Analysis API
*FastAPI backend for real-time crowd behavior analysis using YOLOv8 and DeepSort*

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![Demo](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExcDd0d3VkYjV1dW1vN2R6Z2R4eGZ4Z3JtYzNybjBqY2R5dGJmYyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/your-demo-gif-url.gif)

## 🔍 Features
- **YOLOv8** for person detection
- **DeepSort** for object tracking
- **Behavior Analysis**:
  - Handshake/Pushing/Wrestling detection
  - Velocity and trajectory analysis
- **REST API** endpoints:
  - Video processing
  - Report generation
- **FFmpeg integration** for web-optimized video output

## 🛠️ Installation
```bash
# Clone repository
git clone https://github.com/atharvaishere/Backend_CV_Analysis.git
cd Backend_CV_Analysis

# Set up environment
python -m venv venv
source venv/bin/activate  # Linux/MacOS
venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Download models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt
wget https://drive.google.com/uc?id=1hD8I0L8sZQmW3JQSGuOQxkg7K2wTfWQN -O osnet_x1_0_imagenet.pth

# Run server
uvicorn main:app --reload --host 0.0.0.0 --port 8000




Last updated: 2025-04-15 **