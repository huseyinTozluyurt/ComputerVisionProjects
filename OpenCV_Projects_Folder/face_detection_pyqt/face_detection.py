import sys
import cv2
import numpy as np

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QPushButton,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QMessageBox,
)


class FaceTrackingApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("PyQt5 Face Tracking - Stable MeanShift Version")
        self.resize(900, 700)

        # UI
        self.video_label = QLabel("Camera is stopped.")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setStyleSheet("background-color: black; color: white;")

        self.status_label = QLabel("Status: Ready")
        self.status_label.setAlignment(Qt.AlignCenter)

        self.start_button = QPushButton("Start Camera")
        self.stop_button = QPushButton("Stop Camera")
        self.redetect_button = QPushButton("Re-detect Face")

        self.stop_button.setEnabled(False)
        self.redetect_button.setEnabled(False)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.redetect_button)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.status_label)
        layout.addLayout(button_layout)
        self.setLayout(layout)

        # Connections
        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)
        self.redetect_button.clicked.connect(self.force_redetect)

        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Camera / detection / tracking state
        self.cap = None
        self.face_cascade = None

        self.tracking_initialized = False
        self.track_window = None
        self.roi_hist = None

        self.initial_face_size = None
        self.prev_window = None

        self.frame_count = 0
        self.redetect_interval = 15  # every 15 frames try correction

        self.term_crit = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            10,
            1
        )

        self.load_cascade()

    def load_cascade(self):
        cascade_path = "DATA/haarcascades/haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        if self.face_cascade.empty():
            QMessageBox.critical(
                self,
                "Cascade Error",
                f"Could not load cascade file:\n{cascade_path}"
            )
            self.start_button.setEnabled(False)
            self.status_label.setText("Status: Cascade could not be loaded")

    def start_camera(self):
        if self.face_cascade is None or self.face_cascade.empty():
            QMessageBox.critical(self, "Error", "Face cascade is not loaded.")
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Camera Error", "Could not open webcam.")
            self.cap = None
            return

        self.reset_tracking_state()

        self.timer.start(30)

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.redetect_button.setEnabled(True)
        self.status_label.setText("Status: Camera started")

    def stop_camera(self):
        self.timer.stop()

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.video_label.setText("Camera is stopped.")
        self.video_label.setPixmap(QPixmap())
        self.status_label.setText("Status: Camera stopped")

        self.reset_tracking_state()

        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.redetect_button.setEnabled(False)

    def reset_tracking_state(self):
        self.tracking_initialized = False
        self.track_window = None
        self.roi_hist = None
        self.initial_face_size = None
        self.prev_window = None
        self.frame_count = 0

    def force_redetect(self):
        self.reset_tracking_state()
        self.status_label.setText("Status: Re-detection requested")

    def initialize_tracking(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_rects = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(60, 60)
        )

        if len(face_rects) == 0:
            self.status_label.setText("Status: No face detected")
            return False

        # Choose the largest face
        face_rects = sorted(face_rects, key=lambda r: r[2] * r[3], reverse=True)
        face_x, face_y, w, h = face_rects[0]

        self.track_window = (face_x, face_y, w, h)
        self.initial_face_size = (w, h)
        self.prev_window = self.track_window

        roi = frame[face_y:face_y + h, face_x:face_x + w]
        if roi.size == 0:
            self.status_label.setText("Status: Invalid ROI")
            return False

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(
            hsv_roi,
            np.array((0, 30, 32)),
            np.array((180, 255, 255))
        )

        self.roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv2.normalize(self.roi_hist, self.roi_hist, 0, 255, cv2.NORM_MINMAX)

        self.tracking_initialized = True
        self.status_label.setText("Status: Face detected and tracking initialized")
        return True

    def try_redetect_nearby(self, frame):
        if self.track_window is None:
            return False

        x, y, w, h = self.track_window
        frame_h, frame_w = frame.shape[:2]

        margin_x = int(0.5 * w)
        margin_y = int(0.5 * h)

        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(frame_w, x + w + margin_x)
        y2 = min(frame_h, y + h + margin_y)

        search_roi = frame[y1:y2, x1:x2]
        if search_roi.size == 0:
            return False

        gray_roi = cv2.cvtColor(search_roi, cv2.COLOR_BGR2GRAY)

        face_rects = self.face_cascade.detectMultiScale(
            gray_roi,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(60, 60)
        )

        if len(face_rects) == 0:
            return False

        face_rects = sorted(face_rects, key=lambda r: r[2] * r[3], reverse=True)
        fx, fy, fw, fh = face_rects[0]

        # Convert local ROI coordinates back to frame coordinates
        new_x = x1 + fx
        new_y = y1 + fy

        self.track_window = (new_x, new_y, fw, fh)
        self.prev_window = self.track_window

        # Refresh histogram with new face ROI
        roi = frame[new_y:new_y + fh, new_x:new_x + fw]
        if roi.size != 0:
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(
                hsv_roi,
                np.array((0, 30, 32)),
                np.array((180, 255, 255))
            )
            self.roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
            cv2.normalize(self.roi_hist, self.roi_hist, 0, 255, cv2.NORM_MINMAX)

        self.status_label.setText("Status: Face re-detected")
        return True

    def smooth_window(self, new_window):
        x, y, w, h = new_window

        if self.prev_window is not None:
            px, py, pw, ph = self.prev_window
            alpha = 0.8
            x = int(alpha * px + (1 - alpha) * x)
            y = int(alpha * py + (1 - alpha) * y)
            w = int(alpha * pw + (1 - alpha) * w)
            h = int(alpha * ph + (1 - alpha) * h)

        smoothed = (x, y, w, h)
        self.prev_window = smoothed
        return smoothed

    def clamp_window(self, frame, window):
        x, y, w, h = window
        frame_h, frame_w = frame.shape[:2]

        if self.initial_face_size is not None:
            init_w, init_h = self.initial_face_size
            min_w = int(init_w * 0.75)
            max_w = int(init_w * 1.35)
            min_h = int(init_h * 0.75)
            max_h = int(init_h * 1.35)

            w = max(min_w, min(w, max_w))
            h = max(min_h, min(h, max_h))

        x = max(0, min(x, frame_w - w))
        y = max(0, min(y, frame_h - h))

        return (x, y, w, h)

    def update_frame(self):
        if self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.status_label.setText("Status: Failed to read frame")
            return

        frame = cv2.flip(frame, 1)
        self.frame_count += 1

        # First detect face if tracking not ready
        if not self.tracking_initialized:
            found = self.initialize_tracking(frame)
            if found and self.track_window is not None:
                x, y, w, h = self.track_window
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            self.display_frame(frame)
            return

        # Periodic correction with Haar face detector
        if self.frame_count % self.redetect_interval == 0:
            self.try_redetect_nearby(frame)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)

        try:
            _, new_window = cv2.meanShift(
                dst,
                self.track_window,
                self.term_crit
            )

            new_window = self.clamp_window(frame, new_window)
            new_window = self.smooth_window(new_window)
            new_window = self.clamp_window(frame, new_window)

            self.track_window = new_window

            x, y, w, h = self.track_window

            # Optional confidence check: if window gets too small/invalid, reset
            if w <= 0 or h <= 0:
                self.reset_tracking_state()
                self.status_label.setText("Status: Tracking lost, re-detecting")
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                self.status_label.setText("Status: Tracking face")

        except cv2.error:
            self.reset_tracking_state()
            self.status_label.setText("Status: Tracking lost, trying detection again")

        self.display_frame(frame)

    def display_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w

        qt_image = QImage(
            rgb_frame.data,
            w,
            h,
            bytes_per_line,
            QImage.Format_RGB888
        )

        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = FaceTrackingApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
