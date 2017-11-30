import os

import ExistFile
import cv2
import numpy as np

from NewVer import NoExistFile

startTrack = False

cam = cv2.VideoCapture('TestingMovie.mp4')
# cv2.waitKey(0)

FileChk = os.path.exists('Result.jpg')
centerArray = list()
SavingImage = None

start = False

try:
    # 검사
    while True:
        if cv2.waitKey(10) & 0xFF == ord('c'): # 만약 C 를 입력받으면 손 체크 시작
            start = True

        while start:
            ret, frame = cam.read()  # (848, 480), (width, height)

            HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            skinColorMin = np.array([0, 48, 80])
            skinColorMax = np.array([20, 255, 255])

            Range = cv2.inRange(HSV, skinColorMin, skinColorMax)
            _, thresholdImage = cv2.threshold(Range, 200, 255, cv2.THRESH_BINARY)
            # _, threshold_image = cv2.adaptiveThreshold(Range, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 2)
            cv2.imshow('threshold', thresholdImage)

            kernel = np.ones((5, 5), np.uint8)
            c1 = cv2.morphologyEx(thresholdImage, cv2.MORPH_CLOSE, kernel)
            thresholdImage = cv2.morphologyEx(c1, cv2.MORPH_OPEN, kernel)

            _, contours, hierarchy = cv2.findContours(thresholdImage, cv2.RETR_EXTERNAL,
                                                      cv2.CHAIN_APPROX_SIMPLE)  # 3번째 인자 대체 : RETR_TREE

            if contours:
                max = cv2.contourArea(contours[0])
                maxIdx = 0

                for i in range(len(contours)):
                    if cv2.contourArea(contours[i]) > max:
                        maxIdx = i
                        max = cv2.contourArea(contours[i])

                cnt = contours[maxIdx]

                moments = cv2.moments(cnt)

                cx = 0
                cy = 0

                if moments['m00'] != 0:
                    cx = int(moments['m10'] / moments['m00'])
                    cx = int(moments['m01'] / moments['m00'])

                center = (cx, cy)

                # find the circle which completely covers the object with minimum area
                (x, y), radius = cv2.minEnclosingCircle(cnt)  # rasdius와 (x, y) 에 해당 범위를 감싸는 원을 그린다.
                center = (int(x), int(y))

                centerArray.append(center)

                radius = int(radius)
                cv2.circle(frame, center, radius, (255, 255, 255), 3)

                for centerDot in centerArray:
                    cv2.circle(frame, centerDot, 10, [0, 0, 255], -1)

                SavingImage = frame
                # cv2.drawContours(frame, [cnt], -1, [255, 0, 0], 3)

            cv2.imshow('frame', frame)

            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                quit()

            elif key == ord('s'): # S 를 누르면
                if os.path.exists('Result.jpg') : # 사진 유무 체크
                    nowChkImage = frame
                    if ExistFile.existFile( nowChkImage ) : # 체크할 사진 보내고 원본이랑 비교
                        print("암호가 맞습니다.")

                        cv2.waitKey(0)
                        quit()
                    else :
                        print("틀렸습니다.") # 틀릴 경우 틀렸다고 출력
                        start = False
                else:
                    NoExistFile.NoFile( frame ) # 아니면 저장 하고 처음으로
                    start = False # 이 while 문 종료하고, 다시 C를 누를 때까지 체크
            # 검사

            # Q를 누를 때까지 암호 만들기
except:
    pass

cv2.destroyAllWindows()
quit()