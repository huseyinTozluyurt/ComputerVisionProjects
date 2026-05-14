from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

image_path = "/home/huseyin/opencv_env/goruntu_open-main/DATA/solvay_conference.jpg"

image = cv2.imread(image_path)
# image = cv2.resize(image, (1280, 1020))

if image is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")

results = model.predict(image, conf=0.05, verbose=False)

result = results[0]

for box in result.boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    class_name = result.names[class_id]

    x1, y1, x2, y2 = box.xyxy[0]

    print("Class ID:", class_id)
    print("Class name:", class_name)
    print("Confidence:", confidence)
    print("Box coordinates:", x1.item(), y1.item(), x2.item(), y2.item())
    print("-------------------------")

    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.putText(image, f"{class_name} {confidence:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


cv2.imshow("YOLO Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
