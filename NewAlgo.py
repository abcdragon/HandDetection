import cv2
import numpy as np
import math

startTrack = False

cam = cv2.VideoCapture('TestingMovie2.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2() # 제거할 배경 생성

#cv2.waitKey(0)

while(cam.isOpened()):
    ret, frame = cam.read()

    fgmask = fgbg.apply(frame) # 배경제거

    YCrCb = cv2.cvtColor(fgmask, cv2.COLOR_BGR2YCrCb)
    skinColorMin = np.array([0, 133, 77])
    skinColorMax = np.array([255, 173, 127])

    Range = cv2.inRange(YCrCb, skinColorMin, skinColorMax)
    cv2.imshow('Pick', Range)

    _, thresholdImage = cv2.threshold(Range, 127, 255, cv2.THRESH_BINARY)
    #_, threshold_image = cv2.adaptiveThreshold(Range, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 2)
    cv2.imshow('threshold', thresholdImage)

    kernel = np.ones( (5, 5), np.uint8 )
    c1 = cv2.morphologyEx( thresholdImage, cv2.MORPH_CLOSE, kernel)
    c2 = cv2.morphologyEx( c1, cv2.MORPH_OPEN, kernel)
    thresholdImage = c2

    _, contours, hierarchy = cv2.findContours( thresholdImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE ) # 3번째 인자 대체 : RETR_TREE

    max = cv2.contourArea(contours[0])
    max_idx = 0

    for idx, cnt in enumerate(contours[1:]):
        tmp = cv2.contourArea(cnt)

        if tmp > max :
            max = tmp
            max_idx = idx

    cnt = contours[max_idx]
    cv2.drawContours(frame, [cnt], -1, [255, 0, 0], 3)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cam.release()
quit()