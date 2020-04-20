from contextlib import contextmanager

import cv2


@contextmanager
def opencv_video_capture(webcam_index):
    camera = cv2.VideoCapture(webcam_index)
    yield camera
    camera.release()
    cv2.destroyAllWindows()
