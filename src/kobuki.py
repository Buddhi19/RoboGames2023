import freenect
import cv2
from find_and_pickup import Super_Machine

def get_depth_and_bgr():
    depth, timestamp = freenect.sync_get_depth()
    rgb, timestamp = freenect.sync_get_video()
    depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return depth, bgr

def main():
    Sup = Super_Machine()
    while True:
        bgr = get_depth_and_bgr()[1]
        Sup.run(bgr)


if __name__ == "__main__":
    main()
    