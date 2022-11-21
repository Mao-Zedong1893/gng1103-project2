import cvzone
from cvzone.ColorModule import ColorFinder
import cv2
import numpy as np

cv2.bootstrap()
prevCircle = None
dist = lambda x1,y1,x2,y2: (x1-x2)*2+(y1-y2)*2
cap = cv2.VideoCapture(1)
cap.set(2, 1920)
cap.set(3, 1080)

myColourFinder = ColorFinder(False)
hsvVals = {'hmin': 43, 'smin': 60, 'vmin': 111, 'hmax': 98, 'smax': 255, 'vmax': 255}

while True:
    success, img = cap.read()
    imgColour, mask = myColourFinder.update(img, hsvVals)
    imgContour, contours = cvzone.findContours(img, mask, minArea=5000)

    img_grey_blurred = cv2.blur(mask, (6, 6))

    circles = cv2.HoughCircles(img_grey_blurred, cv2.HOUGH_GRADIENT, 
                                        1, 99999, param1 = 100, param2 = 30, 
                                        minRadius = 10, maxRadius = 800)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(img, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(img, center, radius, (255, 0, 255), 3)
            print(center)

    imgStack = cvzone.stackImages([img, imgColour, mask, imgContour], 2, 0.5)
    cv2.imshow("Image", imgStack)
    cv2.imshow("circles", img)
    cv2.imshow("blur", img_grey_blurred)
    cv2.waitKey(1)