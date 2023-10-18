import numpy as np

class RobotVision:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.camera_width = 256
        self.camera_height = 256

    def get_numpy_image_bgra(self,image):

        arr = np.frombuffer(image, dtype=np.uint8)
        return arr#.reshape((self.camera_height, self.camera_width, -1))

    # def display_numpy_image_bgra(self, image: np.ndarray):
    #     data = bytes(image.flatten())
    #     display_image = self.display.imageNew(
    #         data, Display.BGRA, self.camera_width, self.camera_height)
    #     self.display.imagePaste(display_image, 0, 0, False)
    #     self.display.imageDelete(display_image)

    def get_numpy_image_cv2_bgra(self,image):
        return self.get_numpy_image_bgra(image).astype('float32')

    # def display_numpy_image_cv2_bgra(self, image: np.ndarray):
    #     self.display_numpy_image_bgra(image.astype(int))
