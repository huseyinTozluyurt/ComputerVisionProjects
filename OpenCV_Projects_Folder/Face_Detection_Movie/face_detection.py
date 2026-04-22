import cv2

video_path = "DATA/youtube_video_dataset/movie.mp4"
cascade_path = "DATA/haarcascades/haarcascade_frontalface_default.xml"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video: {video_path}")
    exit()

face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    print(f"Error: Could not load cascade: {cascade_path}")
    cap.release()
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break

    # Optional (remove if dataset video)
    # frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(60, 60)
    )

    # Draw only if faces are detected
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Face Detection", frame)

    key = cv2.waitKey(30) & 0xFF
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
