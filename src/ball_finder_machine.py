
from circle_workflows import Circle_Workflow
import numpy as np
from typing import Any, Tuple
import computer_vision as com_v


class Ball_Finder_Machine:
    def __init__(self, circle_workflow: Circle_Workflow, image_shape: Tuple[int, int]) -> None:
        self.circle_workflow = circle_workflow
        self._current_circle = None
        self._circle_seen = None
        self._image_shape = image_shape
        self._screen_center_x = image_shape[0]/2

    def reset(self):
        self._current_circle = None
        self._circle_seen = None

    @staticmethod
    def sort_circles_by_radius(circles):
        return sorted(circles, key=lambda x: x[2])

    @property
    def current_circle(self):
        return self._current_circle

    @property
    def current_center(self):
        if(self._current_circle is None):
            return None
        return (self._current_circle[0], self._current_circle[1])

    @property
    def current_radius(self):
        if(self._current_circle is None):
            return None
        return self._current_circle[2]

    @property
    def circle_present(self):
        return self._circle_seen

    @staticmethod
    def circle_distance_squared(circle1, circle2):
        return (circle1[0]-circle2[0])**2+(circle2[1]-circle1[1])**2

    def run(self, image: np.ndarray):
        circles = self.circle_workflow.find_circles(image)
        print(circles)
        if(circles is None):
            self._circle_seen = False
            return False
        self._circle_seen = True
        circles_sorted_by_radius = self.sort_circles_by_radius(circles)

        if(self._current_circle is None):
            self._current_circle = circles_sorted_by_radius[0]
            return True

        self._current_circle = min(circles_sorted_by_radius, key=lambda circle: self.circle_distance_squared(
            circle, self._current_circle))
        return True

    def deviation_from_center(self):
        if(self.current_center is None):
            return None

        center_x = self.current_center[0]
        return (self._screen_center_x - center_x)/self._image_shape[0]

    def debug_draw_current_center(self, image: np.ndarray, center=None):
        if(self.current_center is None or not self.circle_present):
            return image

        if(center is None):
            center = self.current_center
        return com_v.draw_circle_on_image(
            image, tuple(map(int, center)), 3, color=(0, 255, 0))  # type:ignore


class Filtered_Value:
    def __init__(self, beta: float):
        self._value = None
        self._beta = beta
        assert(0 < self._beta <= 1)

    def __call__(self, val: float) -> float:
        if(self._value is None):
            self._value = val
        self._value = self._value * (1-self._beta)+val*self._beta
        return self._value


class Double_Filter:
    def __init__(self, beta: float):
        self.f1 = Filtered_Value(beta)
        self.f2 = Filtered_Value(beta)

    def __call__(self, two_floats: Tuple[float, float]) -> Tuple[float, float]:
        return self.f1(two_floats[0]), self.f2(two_floats[1])
