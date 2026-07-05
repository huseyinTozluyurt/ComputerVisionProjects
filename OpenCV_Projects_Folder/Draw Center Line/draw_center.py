import cv2

img = cv2.imread("DATA/elevator.jpg")

img = cv2.resize(img,(1024,800))

height, width, channels = img.shape

center_x = width / 2
center_x_ = width // 2

center_y = height / 2
center_y_ = height // 2

print("Center_x / : " + str(center_x))
print("Center_y / : " + str(center_y))

print("Center_x //: " + str(center_x_))
print("Center_y //: " + str(center_y_))


# Draw lines to visualize the center of the image with a margin of 100 pixels on either side
# cv2.line(img, (center_x_ - 100, 0), (center_x_ - 100, height), (255,0,0), 2)
# cv2.line(img,(center_y_ + 100, 0), (center_y_ + 100, width), (255,0,0), 2)


# Center of the image
# cv2.line(img, (center_x_, 0), (center_x_, height), (255, 0, 0), 2)



# Draw a box at the center of the image with a margin parameter
center_margin = 200
# center_padding = 40/100
cv2.line(img, (center_x_ - center_margin, int(height * (1/4))), (center_x_ - center_margin, int(height * (3/4))), (0, 0, 255), 3)

cv2.line(img, (center_x_ - center_margin, int(height * (1/4))), (center_x_ + center_margin, int(height * (1/4))), (0,0,255), 3)
cv2.line(img, (center_x_ - center_margin, int(height * (3/4))), (center_x_ + center_margin, int(height * (3/4))), (0,0,255), 3)

cv2.line(img, (center_x_ + center_margin, int(height * (1/4))), (center_x_ + center_margin, int(height * (3/4))), (0, 0, 255), 3)


cv2.imshow("Centered Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
