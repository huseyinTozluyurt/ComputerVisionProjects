import cv2
import numpy as np


def main():
    # Shi-Tomasi parameters
    corner_track_params = {
        "maxCorners": 150,
        "qualityLevel": 0.2,
        "minDistance": 10,
        "blockSize": 7
    }

    # Lucas-Kanade parameters
    lk_params = {
        "winSize": (21, 21),
        "maxLevel": 3,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
    }

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Webcam could not be opened.")
        return

    # Optional: set webcam resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    ret, prev_frame = cap.read()
    if not ret:
        print("Error: First frame could not be read.")
        cap.release()
        return

    prev_frame = cv2.flip(prev_frame, 1)  # mirror effect
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **corner_track_params)
    if prev_pts is None:
        print("Error: No good feature points found.")
        cap.release()
        return

    mask = np.zeros_like(prev_frame)
    frame_count = 0

    print("Press ESC to quit, R to reset tracking, C to clear trails.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Frame could not be read.")
            break

        frame = cv2.flip(frame, 1)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        next_pts, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            frame_gray,
            prev_pts,
            None,
            **lk_params
        )

        if next_pts is None or status is None:
            prev_pts = cv2.goodFeaturesToTrack(frame_gray, mask=None, **corner_track_params)
            prev_gray = frame_gray.copy()
            mask = np.zeros_like(frame)

            cv2.imshow("Tracking", frame)
            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                break
            continue

        good_new = next_pts[status.flatten() == 1]
        good_prev = prev_pts[status.flatten() == 1]

        for new, prev in zip(good_new, good_prev):
            x_new, y_new = new.ravel()
            x_prev, y_prev = prev.ravel()

            x_new, y_new = int(x_new), int(y_new)
            x_prev, y_prev = int(x_prev), int(y_prev)

            cv2.line(mask, (x_prev, y_prev), (x_new, y_new), (0, 255, 0), 2)
            cv2.circle(frame, (x_new, y_new), 4, (0, 0, 255), -1)

        img = cv2.add(frame, mask)

        # Show number of tracked points
        cv2.putText(
            img,
            f"Tracked Points: {len(good_new)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.imshow("Tracking", img)

        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('r'):
            prev_pts = cv2.goodFeaturesToTrack(frame_gray, mask=None, **corner_track_params)
            mask = np.zeros_like(frame)
            prev_gray = frame_gray.copy()
            continue
        elif key == ord('c'):
            mask = np.zeros_like(frame)

        prev_gray = frame_gray.copy()

        if len(good_new) > 0:
            prev_pts = good_new.reshape(-1, 1, 2)
        else:
            prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **corner_track_params)
            mask = np.zeros_like(frame)

        # Periodically refresh feature points
        frame_count += 1
        if frame_count % 50 == 0 or prev_pts is None or len(prev_pts) < 10:
            new_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **corner_track_params)
            if new_pts is not None:
                prev_pts = new_pts
                mask = np.zeros_like(frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
