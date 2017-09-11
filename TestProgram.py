import cv2
import numpy as np

cap = cv2.VideoCapture(0)
frame = cap.read()

x, y, w, h = -1, -1, -1, -1

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
COLOR_MIN = np.array([0, 137, 77], np.uint8)
COLOR_MAX = np.array([255, 173, 127], np.uint8)

mask = cv2.inRange(hsv, COLOR_MIN, COLOR_MAX)

hsv_roi = hsv[y:y+h, x:x+w]
mask_roi = mask[y:y+h, x:x+w]

hist = cv2.calcHist([hsv_roi], [0], mask_roi, [16], [0, 180])