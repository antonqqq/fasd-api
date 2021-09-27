from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import csv
import pandas as pd

# define the automatic canny edge detection method
def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)

	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)

	# return the edged image
	return edged

# define midpoint method
def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-w", "--width", type=float, required=True,
	help="width of the left-most object in the image (in inches)")
args = vars(ap.parse_args())

# load the image, convert it to grayscale, and blur it slightly, twice
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (9, 9), 0)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged = auto_canny(gray)
edged = cv2.dilate(edged, None, iterations=3)
edged = cv2.erode(edged, None, iterations=2)

# find contours in the edge map
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# sort the contours from left-to-right and initialize the
# 'pixels per metric' calibration variable
(cnts, _) = contours.sort_contours(cnts)
pixelsPerMetric = None

# create list object
measure = []

# loop over the contours individually
for c in cnts:
	# if the contour is not sufficiently large, ignore it
    if cv2.contourArea(c) < 100:
        continue

    orig = image.copy()
    box = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    # order the points in the contour such that they appear
    # in top-left, top-right, bottom-right, and bottom-left
    # order.
    box = perspective.order_points(box)

    # unpack the ordered bounding box, then compute the midpoint
    # between the top-left and top-right coordinates, followed by
    # the midpoint between bottom-left and bottom-right coordinates
    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)

    # compute the midpoint between the top-left and top-right points,
    # followed by the midpoint between the top-right and bottom-right
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    # compute the Euclidean distance between the midpoints
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    # if the pixels per metric has not been initialized, then
    # compute it as the ratio of pixels to supplied metric
    # (in this case, inches)
    if pixelsPerMetric is None:
        pixelsPerMetric = dB / args["width"]
    
    # compute the size and surface area of the object
    dimA = (dA / pixelsPerMetric)
    dimB = (dB / pixelsPerMetric)
    SA = ((cv2.contourArea(c) / pixelsPerMetric) / pixelsPerMetric)

    # add measurements to list
    measure.append((dimA, dimB, SA))

    # compute centre of the contour
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    # draw contours in red
    cv2.drawContours(orig, [c.astype("int")], -1, (0, 0, 255), 2)

    # draw the object area on the image
    cv2.putText(orig, "{:.2f}sqcm".format(SA),
        (int (cX), int (cY)), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (0, 0, 0), 3)
    cv2.putText(orig, "{:.2f}sqcm".format(SA),
        (int (cX), int (cY)), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (255, 255, 255), 2)

    # show the output image
    # origS = cv2.resize(orig, (1536, 1024))
    cv2.imshow("Image", orig)
    cv2.waitKey(0)