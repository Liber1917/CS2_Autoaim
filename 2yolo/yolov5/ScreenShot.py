import cv2
from mss import mss
import numpy

ScrX = 2560
ScrY = 1600
ScrSht_size_h = 520
window_size=(
    int(ScrX/2-ScrSht_size_h),
    int(ScrY/2-ScrSht_size_h),
    int(ScrX/2+ScrSht_size_h),
    int(ScrY/2+ScrSht_size_h),
)
Screenshot_value = mss()
def screenshot():
    img =Screenshot_value.grab(window_size)
    img = numpy.array(img)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    return img

# while True:
#     cv2.imshow('a',numpy.array(screenshot()))
#     cv2.waitKey(1)
