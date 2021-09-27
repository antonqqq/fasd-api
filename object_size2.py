import cv2
import argparse
from imutils import contours

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())

src = cv2.imread(args["image"], cv2.IMREAD_COLOR)

#Transform source image to gray if it is not already
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

bilateral_filtered_image = cv2.bilateralFilter(src, 5, 175, 175)
edge_detected_image = cv2.Canny(bilateral_filtered_image, 75, 200)
contours1, hierarchy = cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# (contours1, _) = contours.sort_contours(contours1, method="top-to-bottom")

# ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
hierarchy = hierarchy[0]

for i, c in enumerate(contours1):
    if hierarchy[i][2] < 0 and hierarchy[i][3] < 0:
        cv2.drawContours(src, contours1, i, (0, 0, 255), 2)
    else:
        cv2.drawContours(src, contours1, i, (0, 255, 0), 2)

#write to the same directory
cv2.imwrite("result.png", src)