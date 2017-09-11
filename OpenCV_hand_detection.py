import cv2
import numpy as np
import math
import MiddlePoint

img = cv2.imread('C:\\Users\\HelloGuys\\Desktop\\HandDetectionTestImages\\TestImageVertical.jpg', cv2.IMREAD_COLOR)
img1 = img.copy()

def YCbCr_Skin_Rectangle():
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    COLOR_MIN = np.array([0, 137, 77], np.uint8)
    COLOR_MAX = np.array([255, 173, 127], np.uint8)

    frame_threshed = cv2.inRange(hsv, COLOR_MIN, COLOR_MAX)
    ret, threshold_image = cv2.threshold(frame_threshed, 100, 255,  cv2.THRESH_BINARY)
    _, contours, hierarchy = cv2.findContours(threshold_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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

        cv2.drawContours(img1, [hull], 0, (0, 255, 0), 3)

    middlePoint = MiddlePoint.FindMiddlePoint(tempPoint)
    #print(middlePoint)

    for i in range(len(middlePoint)):
        #print(middlePoint[i])
        #cv2.putText(img1, str(middlePoint[i]), (middlePoint[i][0] + 20, middlePoint[i][1] + 20), 2, 1.2, (255, 255, 255))
        cv2.circle(img1, middlePoint[i], 10, (255, 255, 0), -1)

    cv2.imshow('ConvexHull', img1)

    x, y, w, h = cv2.boundingRect(cnt)
    area = cv2.contourArea(cnt)
    ellipse = cv2.fitEllipse(cnt)

    equivalent_diameter = np.sqrt(4*area/np.pi)
    Temp_radius = int(equivalent_diameter/2)

    cv2.circle(img, (cx, cy), 3, (0, 0, 255), -1)
    cv2.circle(img, (cx, cy), Temp_radius, (0, 0, 255), 2)
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.ellipse(img, ellipse, (50, 50, 50), 2)

    topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
    cv2.circle(img1, (topmost[0], topmost[1]), 44, (255, 255, 255), -1)

    cv2.drawContours(img, [cnt], 0, (255, 0, 255), 2)
    #cv2.imshow('contour image', img)
    cv2.waitKey(0)

YCbCr_Skin_Rectangle()
