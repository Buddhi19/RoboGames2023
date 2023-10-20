import cv2
import numpy as np
import computer_vision as com_v
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class Circle_Workflow:
    @abstractmethod
    def find_circles(self, image_np_bgra: np.ndarray) -> Tuple[np.ndarray]:
        ...

    def draw_circles(self, circles, image):
        if(circles is None):
            return image

        for circle in circles:
            image = com_v.draw_circle_on_image(
                image, (int(circle[0]), int(circle[1])), (int(circle[2])))
        return image


class Hugh_circle_Workflow(Circle_Workflow):
    def __init__(self, color_filter: com_v.Color_Filter_Generic):
        self._erosion_level1 = 5
        self._dilation_level1 = 5
        self._dilation_after_canny = 3
        self._gaussian_blur_before_hugh = 5
        self._min_distance_scale = 8
        self._param1 = 1
        self._param2 = 23  # 27
        self._color_filter = color_filter

    def set_color_filter(self, color_filter):
        self._color_filter = color_filter

    def find_circles(self, image_np_bgra: np.ndarray):
        float_image = image_np_bgra.astype(np.float32)
        float_image = image_np_bgra
        mask = self._color_filter.get_mask_for_bgra(float_image)
        # print("hhhh")
        # return 0, 0, mask
        # print("adsads")
        masked_image = self._color_filter.mask_image(image_np_bgra, mask)
        # return 0, 0, masked_image
        gray_masked_image = com_v.bgra_to_gray_scale(masked_image)
        eroded_mask = cv2.erode(mask, np.ones(
            (self._erosion_level1, self._erosion_level1)), iterations=1)
        dilated_eroded_mask = cv2.dilate(eroded_mask, np.ones(
            (self._dilation_level1, self._dilation_level1)), iterations=1)
        canny_image = com_v.canny(dilated_eroded_mask)
        dilated_canny_image = cv2.dilate(canny_image, np.ones(
            (self._dilation_after_canny, self._dilation_after_canny)), iterations=1)

        blurred_image = cv2.GaussianBlur(
            dilated_canny_image, (self._gaussian_blur_before_hugh, self._gaussian_blur_before_hugh), 0)

        min_distance_between_centers = blurred_image.shape[0]//self._min_distance_scale

        circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT,
                                   1, min_distance_between_centers, param1=self._param1, param2=self._param2)

        if(circles is not None):
            circles = circles[0, :]
        # print(circles)
        return circles
