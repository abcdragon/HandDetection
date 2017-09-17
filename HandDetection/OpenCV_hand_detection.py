import cv2
import numpy as np
import math
import MiddlePoint

cap = cv2.VideoCapture(0)

roi = None
trackWindow = None
termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret, frame = cap.read()
    '''if trackWindow is not None:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        ret, trackWindow = cv2.CamShift(dst, trackWindow, termination)
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)'''

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    COLOR_MIN = np.array([61, 132, 109], np.uint8)
    COLOR_MAX = np.array([166, 166, 166], np.uint8)
    #COLOR_MIN = np.array([0, 137, 77], np.uint8)
    #COLOR_MAX = np.array([255, 173, 127], np.uint8)

    frame_threshed = cv2.inRange(hsv, COLOR_MIN, COLOR_MAX)
    ret, threshold_image = cv2.threshold(frame_threshed, 100, 255,  cv2.THRESH_BINARY)
    _, contours, hierarchy = cv2.findContours(threshold_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) != 0:
    
        # 윤곽선을 갖는 것들 중 가장 큰것을 고른다.
        max_area = 0
        ci = 0
        for i in range(len(contours)):
            cnt = contours[i]
            area = cv2.contourArea(cnt)
            if (area > max_area):
                max_area = area
                ci = i  # contour에서의 배열 번호를 차례대로 저장한다.

        cnt = contours[ci]  # 위에서 가장 큰 영역으로 뽑힌 부분을 cnt에 저장한다.
        
        mmt = cv2.moments(cnt)
        cx = int(mmt['m10']/mmt['m00'])
        cy = int(mmt['m01'] /mmt['m00'])

        chk = cv2.isContourConvex(cnt)

        tempPoint = []

        if not chk:
            hull = cv2.convexHull(cnt)
            for i in range(len(hull)):
                tempHull = (hull[i][0][0], hull[i][0][1])
                if(cy > tempHull[1]):
                    tempPoint.append(tempHull)
                    #cv2.circle(img1, tempHull, 10, (255, 0, 0), -1)

            cv2.drawContours(frame, [hull], 0, (0, 255, 0), 3)

        #middlePoint = MiddlePoint.FindMiddlePoint(tempPoint)
        #print(middlePoint)
            
        if len(tempPoint) != 0:
            minX, maxX = tempPoint[0][0], tempPoint[0][0]
            for i in range(len(tempPoint)):
                print(tempPoint[i])
                #cv2.putText(img1, str(middlePoint[i]), (middlePoint[i][0] + 20, middlePoint[i][1] + 20), 2, 1.2, (255, 255, 255))
                cv2.circle(frame, tempPoint[i], 10, (255, 255, 0), -1)
                if minX > tempPoint[i][0] : minX = tempPoint[i][0]
                if maxX < tempPoint[i][0] : maxX = tempPoint[i][0]

            x1, y1, x2, y2 = cv2.boundingRect(cnt)
            x1, x2 = minX, maxX
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if x2 - x1 < y2 - y1:
                trackWindow = (x1, y1, x2, y2)
                roi = frame[y1:y2, x1:x2]
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                roi_hist = cv2.calcHist([roi], [0], None, [180], [0, 180])
                cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
                

            cv2.drawContours(frame, [cnt], 0, (255, 0, 255), 2)
            #cv2.imshow('contour image', img)
            
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()
