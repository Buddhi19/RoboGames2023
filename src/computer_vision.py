import cv2
import numpy as np
from typing import Collection, Tuple
from abc import ABC, abstractmethod


def bgra_to_rgb(bgra_image: np.ndarray):
    return cv2.cvtColor(bgra_image, cv2.COLOR_BGRA2RGB)


def rgb_to_hsv(rgb_image: np.ndarray):
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)


def rgb_to_bgra(rgb_image: np.ndarray):
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)


def bgra_to_hsv(bgra_image: np.ndarray):
    return rgb_to_hsv(bgra_to_rgb(bgra_image))


def bgra_to_gray_scale(bgra_image: np.ndarray):
    return cv2.cvtColor(bgra_image, cv2.COLOR_BGR2GRAY)


def gray_to_bgra(gray_image: np.ndarray):
    return cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)


class Color_Filter_Generic:

    @abstractmethod
    def get_mask_for_hsv(self, image_hsv: np.ndarray) -> np.ndarray:
        ...

    def get_mask_for_bgra(self, image_bgra: np.ndarray):
        hsv_image = bgra_to_hsv(image_bgra)
        return self.get_mask_for_hsv(hsv_image)

    @staticmethod
    def mask_image(image: np.ndarray, mask: np.ndarray):
        return cv2.bitwise_and(image, image, mask=mask)


class Color_Filter_HSV(Color_Filter_Generic):
    def __init__(self, hl, hh, sl, sh):
        self.high = np.array([hh, sh, 255])
        self.low = np.array([hl, sl, 0])

    def get_mask_for_hsv(self, image_hsv: np.ndarray):
        return cv2.inRange(image_hsv, self.low, self.high)


class Color_Filter(Color_Filter_Generic):
    def __init__(self, hex_color: int,  delta_h: int, delta_s: float, brightness_range=(0, 255)):
        self.color = (hex_color >> 16, hex_color >> 8 & 0xff, hex_color & 0xff)
        hsv_color = cv2.cvtColor(
            np.array(self.color, dtype='float32').reshape((1, 1, 3)), cv2.COLOR_RGB2HSV).flatten()
        delta = np.array([delta_h, delta_s, 0])
        low_color, high_color = hsv_color.copy(), hsv_color.copy()
        low_color[2] = brightness_range[0]
        high_color[2] = brightness_range[1]
        self.low_color = low_color-delta
        self.high_color = high_color+delta

    def get_mask_for_hsv(self, image_hsv: np.ndarray):
        return cv2.inRange(image_hsv, self.low_color, self.high_color)


class Combined_Filter(Color_Filter_Generic):
    def __init__(self, filters: Collection[Color_Filter_Generic]):
        self.filter_collection = filters

    def get_mask_for_hsv(self, image_hsv: np.ndarray):
        mask = np.zeros(image_hsv.shape[:2], dtype=np.uint8)
        for filter in self.filter_collection:
            mask = mask | filter.get_mask_for_hsv(image_hsv)  # type:ignore
        return mask


def get_circles(image_binary: np.ndarray):
    # print(image_binary.dtype)
    # print(image_binary.shape)
    image_binary = cv2.GaussianBlur(image_binary, (5, 5), 0)
    min_distance_between_centers = image_binary.shape[0]//16

    circles = cv2.HoughCircles(image_binary, cv2.HOUGH_GRADIENT,
                               1, min_distance_between_centers, param1=1, param2=27)
    if(circles is None):
        return None
    return circles[0, :]


def get_circles_blob(gray_image: np.ndarray):
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()  # type:ignore

    params.filterByArea = True
    params.minArea = 50

    # Filter by Circularity
    params.filterByCircularity = False
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.1

    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0.1
    detector = cv2.SimpleBlobDetector_create(params)  # type:ignore
    key_points = detector.detect(gray_image)
    return tuple((key_point.pt[0], key_point.pt[1], key_point.size/2) for key_point in key_points)


def draw_circle_on_image(image: np.ndarray, center: Tuple[int, int], radius: int, color=(255, 0, 0)):
    image = bgra_to_rgb(image)
    image = cv2.circle(image, center, radius, color, 3)
    return rgb_to_bgra(image)


def canny(gray_image: np.ndarray):
    return cv2.Canny(gray_image, 100, 200, L2gradient=True)


class Debug_Window:
    def __init__(self, window_name):
        cv2.namedWindow(window_name)
        self.window_name = window_name
        self.Kp = 0.00001
        self.Ki = 0
        self.Kd = 0
        for name in {'Kp', 'Ki', 'Kd'}:
            cv2.createTrackbar(name, self.window_name, 0,  # type: ignore
                               100, self.get_function(name))

    def get_function(self, name):
        def func(x):
            setattr(self, name, x/100)
        return func

    def show_image(self, image: np.ndarray):
        cv2.imshow(self.window_name, image)
        cv2.waitKey(1)

    def __del__(self):
        cv2.destroyWindow(self.window_name)


class Computer_Vision_Helper:
    def __init__(self, window_name):
        cv2.namedWindow(window_name)
        self.window_name = window_name
        self.param1 = 1
        self.param2 = 50
        self.blur = 1
        cv2.createTrackbar("param1", self.window_name, 1,  # type: ignore
                           100, self.set_param1)
        cv2.createTrackbar("param2", self.window_name, 1,  # type:ignore
                           100, self.set_param2)
        cv2.createTrackbar("blur", self.window_name, 1, 100,  # type:ignore
                           self.set_blur)

    def set_param1(self, x):
        self.param1 = x

    def set_param2(self, x):
        self.param2 = x

    def set_blur(self, x):
        self.blur = 2*x-1
        print(self.blur)

    def show_image(self, image: np.ndarray):
        cv2.imshow(self.window_name, image)
        cv2.waitKey(1)

    def __del__(self):
        cv2.destroyWindow(self.window_name)

    def get_circles(self, image: np.ndarray):

        image_binary = cv2.GaussianBlur(image, (self.blur, self.blur), 0)
        min_distance_between_centers = image_binary.shape[0]//4

        circles = cv2.HoughCircles(image_binary, cv2.HOUGH_GRADIENT,
                                   1, min_distance_between_centers, param1=self.param1, param2=self.param2)
        if(circles is None):
            return None
        return circles[0, :]
