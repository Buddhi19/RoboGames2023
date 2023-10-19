import cv2
import numpy as np
bgr = cv2.imread("new.jpg")
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

while True:
    ret, thresh = cv2.threshold(gray, 150, 255, 0)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    approx = [cv2.approxPolyDP(
        cnt, 0.015*cv2.arcLength(cnt, True), True) for cnt in contours]
    quadrilaterals = list(filter(lambda x: 4 <= len(x) <= 5, approx))
    large_quads = list(
        filter(lambda quad: cv2.contourArea(quad) > 50, quadrilaterals))
    drawn = cv2.drawContours(bgr, large_quads, -1, (0, 255, 0), 3)
    cv2.imshow("debug", drawn)
    cv2.waitKey(1)

cv2.destroyAllWindows()
