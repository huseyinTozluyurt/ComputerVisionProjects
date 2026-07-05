import cv2

img = cv2.imread("DATA/elevator_raw.jpg")

if img is None:
    print("Image could not be loaded. Check the file path.")
else:
    img = cv2.resize(img, (1024, 800))

    height, width, channels = img.shape

    center_x = width // 2
    center_y = height // 2

    print("Image width:", width)
    print("Image height:", height)
    print("Center x:", center_x)
    print("Center y:", center_y)

    # Horizontal margin from center
    center_margin = 100

    # Vertical padding as proportion of image height
    center_padding = 37/100

    # Box coordinates
    left_x = center_x - center_margin
    right_x = center_x + center_margin

    top_y = int(height * center_padding)
    bottom_y = int(height * (1 - center_padding))

    # Draw center box
    cv2.line(img, (left_x, top_y), (left_x, bottom_y), (0, 0, 255), 3)
    cv2.line(img, (right_x, top_y), (right_x, bottom_y), (0, 0, 255), 3)
    cv2.line(img, (left_x, top_y), (right_x, top_y), (0, 0, 255), 3)
    cv2.line(img, (left_x, bottom_y), (right_x, bottom_y), (0, 0, 255), 3)

    cv2.imshow("Centered Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()