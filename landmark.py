# USAGE
# python facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg 

# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
from scipy.spatial import distance as dist
import base64
from urllib.parse import quote
from imutils import perspective
from imutils import contours

def get_image_with_landmarks(file_path: str):
    width = 23.00

    def midpoint(ptA, ptB):
        return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

    # construct the argument parser and parse the arguments
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-p", "--shape-predictor", required=True,
    # 	help="path to facial landmark predictor")
    # ap.add_argument("-i", "--image", required=True,
    # 	help="path to input image")
    # args = vars(ap.parse_args())

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    # predictor = dlib.shape_predictor(args["shape_predictor"])
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    # load the input image, resize it, and convert it to grayscale
    # image = cv2.imread(args["image"])
    # image = cv2.imread('images/face7.jpg')
    image = cv2.imread(file_path, 1)
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        # (x, y, w, h) = face_utils.rect_to_bb(rect)
        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # # show the face number
        # cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
        # 	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        # count = 0
        for (x, y) in shape:
            # cv2.putText(image, str(count), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
            # count += 1

    bilateral_filtered_image = cv2.bilateralFilter(image, 5, 175, 175)
    edge_detected_image = cv2.Canny(bilateral_filtered_image, 75, 200)
    contours1, _= cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    (contours1, _) = contours.sort_contours(contours1, method="top-to-bottom")
    pixelsPerMetric = None

    contour_list = []
    for contour in contours1:
        approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
        area = cv2.contourArea(contour)
        if ((len(approx) > 8) & (area > 30) ):
            contour_list.append(contour)
        
    cv2.drawContours(image, contour_list, -1, (255, 255, 255), 1)

    for c in contour_list:
        image = image.copy()
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        box = perspective.order_points(box)
        cv2.drawContours(image, [box.astype("int")], -1, (0, 255, 0), 2)

        # loop over the imageinal points and draw them
        for (x, y) in box:
            cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)

        # unpack the ordered bounding box, then compute the midpoint
        # between the top-left and top-right coordinates, followed by
        # the midpoint between bottom-left and bottom-right coordinates
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        # draw the midpoints on the image
        cv2.circle(image, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(image, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(image, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(image, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

        # draw lines between the midpoints
        cv2.line(image, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
            (255, 0, 255), 2)
        cv2.line(image, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
            (255, 0, 255), 2)

        # compute the Euclidean distance between the midpoints
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        # if the pixels per metric has not been initialized, then
        # compute it as the ratio of pixels to supplied metric
        # (in this case, inches)
        # if pixelsPerMetric is None:
        pixelsPerMetric = dB / width

        lxy = (shape[36][0], shape[36][1]), (shape[39][0], shape[39][1])
        l_eye = dist.euclidean(lxy[0], lxy[1])
        l_size = l_eye / pixelsPerMetric
        cv2.putText(image, "{:.2f}mm".format(l_size), lxy[1], cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.line(image, lxy[0], lxy[1], (255, 0, 255), 2)

        rxy = (shape[42][0], shape[42][1]), (shape[45][0], shape[45][1])
        r_eye = dist.euclidean(rxy[0], rxy[1])
        r_size = r_eye / pixelsPerMetric
        cv2.putText(image, "{:.2f}mm".format(r_size), rxy[1], cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.line(image, rxy[0], rxy[1], (255, 0, 255), 2)

        print(l_size)
        print(r_size)

        ul = []
        upp = [48, 49, 50, 51, 52, 53, 54, 64, 63, 62, 61, 60]
        for x in upp:
            ul.append([shape[x][0], shape[x][1]])
        ulnp = np.array(ul)
        ulctr = ulnp.reshape((-1,1,2)).astype(np.int32)
        sa = ((cv2.contourArea(ulctr) / pixelsPerMetric) / pixelsPerMetric)

        cv2.drawContours(image, ulctr, -1, (255,255,255), 1)

        interval = 1
        perimeter = 0
        for x in upp[:-1]:
            if x == 54:
                interval = 10
            if x == 64:
                interval = -1
            xy = (shape[x][0], shape[x][1]), (shape[x+interval][0], shape[x+interval][1])
            size = dist.euclidean(xy[0], xy[1]) / pixelsPerMetric
            perimeter += size

        circularity = (perimeter * perimeter) / sa
        print(circularity)
        cv2.putText(image, "{:.2f}".format(circularity), (shape[54][0], shape[54][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)


        # compute the size of the object
        dimA = dA / pixelsPerMetric
        dimB = dB / pixelsPerMetric

        # draw the object sizes on the image
        cv2.putText(image, "{:.2f}mm".format(dimA),
            (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (255, 255, 255), 2)
        cv2.putText(image, "{:.2f}mm".format(dimB),
            (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (255, 255, 255), 2)

        # show the output image with the face detections + facial landmarks
        # cv2.imshow("Output", image)
        # cv2.waitKey(0)
        retval, buffer = cv2.imencode('.jpg', image)
        image_as_text = base64.b64encode(buffer)

        return {'image_with_landmarks': 'data:image/png;base64,{}'.format(quote(image_as_text))}