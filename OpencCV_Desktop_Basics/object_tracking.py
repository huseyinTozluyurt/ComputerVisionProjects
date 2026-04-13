import cv2
import numpy as np


def main():
    # Shi-Tomasi corner detection parameters
    corner_track_params = dict(
        maxCorners=100,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=7
    )

    # Lucas-Kanade optical flow parameters
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Hata: Webcam açılamadı.")
        return

    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Hata: İlk kare okunamadı.")
        cap.release()
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Detect initial points
    prev_pts = cv2.goodFeaturesToTrack(
        prev_gray,
        mask=None,
        **corner_track_params
    )

    if prev_pts is None:
        print("Hata: Takip edilecek köşe bulunamadı.")
        cap.release()
        return

    # Mask for drawing tracks
    mask = np.zeros_like(prev_frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Hata: Kare okunamadı.")
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            frame_gray,
            prev_pts,
            None,
            **lk_params
        )

        if next_pts is None or status is None:
            # Yeni noktalar bulunamazsa yeniden köşe seç
            prev_pts = cv2.goodFeaturesToTrack(
                frame_gray,
                mask=None,
                **corner_track_params
            )
            prev_gray = frame_gray.copy()
            mask = np.zeros_like(frame)
            cv2.imshow("tracking", frame)

            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                break
            continue

        # Select good points
        good_new = next_pts[status == 1]
        good_prev = prev_pts[status == 1]

        # Draw tracks
        for new, prev in zip(good_new, good_prev):
            x_new, y_new = new.ravel()
            x_prev, y_prev = prev.ravel()

            x_new, y_new = int(x_new), int(y_new)
            x_prev, y_prev = int(x_prev), int(y_prev)

            mask = cv2.line(mask, (x_new, y_new), (x_prev, y_prev), (0, 255, 0), 2)
            frame = cv2.circle(frame, (x_new, y_new), 5, (0, 0, 255), -1)

        img = cv2.add(frame, mask)
        cv2.imshow("tracking", img)

        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('r'):
            # reset tracking
            mask = np.zeros_like(frame)
            prev_pts = cv2.goodFeaturesToTrack(
                frame_gray,
                mask=None,
                **corner_track_params
            )
            prev_gray = frame_gray.copy()
            continue

        # Update previous frame and points
        prev_gray = frame_gray.copy()
        prev_pts = good_new.reshape(-1, 1, 2)

        # Eğer nokta kalmazsa yeniden algıla
        if len(prev_pts) < 5:
            prev_pts = cv2.goodFeaturesToTrack(
                prev_gray,
                mask=None,
                **corner_track_params
            )
            mask = np.zeros_like(frame)
            if prev_pts is None:
                print("Uyarı: Yeni takip noktası bulunamadı.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()