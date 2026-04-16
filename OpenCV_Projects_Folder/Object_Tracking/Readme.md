# Real Time Object Tracking with Optical Flow with OpenCV

## Overview

In this project Real-Time Feature Tracking System is being used with Optical Flow through Webcam with algorithms
- **Shi-Tomasi Corner Detection**
- **Lucas-Kanade Optical Flow (Pyramidal)**

The system detects strong feature points in a video stream and tracks their motion across consecutive frames, visualizing trajectories and movement patterns in real time.

## System Architecture

Webcam Input
     ↓
Frame Preprocessing (Grayscale Conversion)
     ↓
Feature Detection (Initial Frame)
     ↓
Optical Flow Tracking (Frame-to-Frame)
     ↓
Filtering Valid Points
     ↓
Visualization (Tracks + Points)
     ↓
Reinitialization (if needed)



