from camera import Camera
from drowsy_detector import *

if __name__ == "__main__":
    # Example usage of Camera class
    cam = Camera()
    cam.initialize()
    start_detecting(cam)

