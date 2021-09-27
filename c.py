import cv2
import imutils

img = cv2.imread('images/object1.jpg' , 0)
img = imutils.resize(img, height=720)
# cv2.imshow('image' , img)
# cv2.waitKey(0)
cv2.imwrite("result.png", img)