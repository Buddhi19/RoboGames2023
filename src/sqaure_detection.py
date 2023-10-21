import cv2
import numpy as np
from circle_workflows import Circle_Workflow


class Square(Circle_Workflow):
    def get_square_center(self,contour):
        M = cv2.moments(contour)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        area = cv2.contourArea(contour)
        return (cx, cy, area)


    def find_circles(self,bgr):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 150, 255, 0)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        approx = [cv2.approxPolyDP(
            cnt, 0.015*cv2.arcLength(cnt, True), True) for cnt in contours]
        quadrilaterals = list(filter(lambda x: 4 <= len(x) <= 5, approx))
        large_quads = list(
            filter(lambda quad: cv2.contourArea(quad) > 50, quadrilaterals))
        centers = [self.get_square_center(quad) for quad in large_quads]

        drawn = cv2.drawContours(bgr, large_quads, -1, (0, 255, 0), 3)
        for cx, cy, a in centers:
            drawn = cv2.circle(drawn, (cx, cy), 5, (255, 0, 0), -1)
        cv2.imshow("debug", drawn)
        cv2.waitKey(1)
        return centers
    

# cv2.destroyAllWindows()
