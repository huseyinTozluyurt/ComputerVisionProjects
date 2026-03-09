import cv2
import numpy as np

# Use Konftel camera device
cap = cv2.VideoCapture(2, cv2.CAP_V4L2)   # /dev/video2

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Optional: set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame")
        break

    # Convert BGR image to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Green range in HSV
    # You may tune these values depending on lighting/environment
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    # Create mask for green pixels
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Morphological cleaning
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours in the green mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = frame.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Ignore very small green regions
        if area > 2000:
            x, y, w, h = cv2.boundingRect(cnt)

            # Draw rectangle around green object
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(output, "Green Region", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show original / processed views
    cv2.imshow("Konftel Camera", frame)
    cv2.imshow("Green Mask", mask)
    cv2.imshow("Detected Green Trees/Regions", output)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()