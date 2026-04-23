# PyQt5 Face Tracking Application

This project is a realtime face tracking desktop application developed using Python, OpenCV, and PyQt5. The application captures webcam video, detects a human face, and tracks the detected face dynamically inside a graphical user interface (GUI).

The project combines classical Computer Vision techniques with a modern PyQt5-based interface. Initially, Haar Cascade face detection is used to locate the largest visible face in the webcam frame. After the face is detected, a histogram-based tracking approach using MeanShift is applied to follow the face movement in realtime.

The graphical interface was developed with PyQt5 and includes:
- Live webcam display
- Start and stop camera controls
- Face re-detection button
- Realtime status information

Several improvements were implemented to increase tracking stability:
- MeanShift was used instead of CamShift to reduce unstable rotation behavior.
- Tracking window smoothing was added to reduce sudden rectangle jumps.
- Size clamping was implemented to prevent the tracking rectangle from growing excessively.
- Periodic face re-detection helps recover from tracking drift.

<img width="1920" height="1080" alt="Screenshot from 2026-04-23 17-43-51" src="https://github.com/user-attachments/assets/8d62c115-2cb0-4561-a3be-0f83349e159f" />


## Technologies Used

- Python
- OpenCV
- PyQt5
- NumPy

## Features

- Realtime webcam face tracking
- GUI-based application
- Stable rectangle tracking
- Automatic tracking recovery
- Lightweight desktop application

## Limitations

The project currently uses Haar Cascade XML-based face detection, which is a classical Computer Vision method. While lightweight and fast, Haar cascades can struggle under:
- lighting changes
- shadows
- head rotation
- occlusion
- complex backgrounds

For more robust and modern face detection, future versions may integrate:
- MediaPipe Face Detection
- OpenCV DNN face detectors
- YOLO-based face detection systems

## Future Improvements

- Multi-face tracking
- Deep Learning face detectors
- Kalman filter integration
- FPS monitoring
- Face recognition support
- Video recording functionality

This project is useful for learning realtime Computer Vision, GUI programming, and object tracking concepts in Python.
