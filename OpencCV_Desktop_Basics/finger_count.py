import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Fingertip landmark indices
tip_ids = [4, 8, 12, 16, 20]

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)  # mirror view
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    finger_count = 0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = hand_landmarks.landmark
            h, w, _ = frame.shape

            # Convert normalized landmarks to pixel coordinates
            points = []
            for lm in landmarks:
                points.append((int(lm.x * w), int(lm.y * h)))

            # Thumb: compare x positions
            if points[tip_ids[0]][0] > points[tip_ids[0] - 1][0]:
                finger_count += 1

            # Other 4 fingers: fingertip y should be above lower joint y
            for i in range(1, 5):
                if points[tip_ids[i]][1] < points[tip_ids[i] - 2][1]:
                    finger_count += 1

    cv2.putText(frame, f'Fingers: {finger_count}', (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Finger Counter", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
