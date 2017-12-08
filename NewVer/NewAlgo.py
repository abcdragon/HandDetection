import os

import cv2
import numpy as np

startTrack = False

cam = cv2.VideoCapture( 0 )
# cv2.waitKey(0)

FileChk = os.path.exists( 'Result.jpg' )
centerArray = list()
trackWindow = None
HSV = None
roi_hist = None
start = False

try:
    # 검사
    while True:
        if not start:
            centerArray = list()
            _, preStart = cam.read()
            cv2.imshow( 'Test', preStart )

            key = cv2.waitKey( 10 ) & 0xFF
            if key == ord( 'c' ):
                cv2.destroyAllWindows()
                start = not start

            if key == ord( 's' ):
                cv2.destroyAllWindows()
                break

        while start and (trackWindow is None):
            ret, frame = cam.read()  # (848, 480), (width, height)

            HSV = cv2.cvtColor( frame, cv2.COLOR_BGR2HSV )
            skinColorMin = np.array( [0, 48, 80] )
            skinColorMax = np.array( [20, 255, 255] )

            Range = cv2.inRange( HSV, skinColorMin, skinColorMax )
            _, thresholdImage = cv2.threshold( Range, 200, 255, cv2.THRESH_BINARY )
            # _, threshold_image = cv2.adaptiveThreshold(Range, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 2)

            kernel = np.ones( (5, 5), np.uint8 )
            c1 = cv2.morphologyEx( thresholdImage, cv2.MORPH_CLOSE, kernel )
            thresholdImage = cv2.morphologyEx( c1, cv2.MORPH_OPEN, kernel )

            _, contours, hierarchy = cv2.findContours( thresholdImage, cv2.RETR_EXTERNAL,
                                                       cv2.CHAIN_APPROX_SIMPLE )  # 3번째 인자 대체 : RETR_TREE
            if contours:
                max = cv2.contourArea( contours[0] )
                maxIdx = 0

                for i in range( len( contours ) ):
                    if cv2.contourArea( contours[i] ) > max:
                        maxIdx = i
                        max = cv2.contourArea( contours[i] )

                cnt = contours[maxIdx]
                (x, y), radius = cv2.minEnclosingCircle( cnt )  # rasdius와 (x, y) 에 해당 범위를 감싸는 원을 그린다.

                center = (int( x ), int( y ))  # 비교 해보기

                # find the circle which completely covers the object with minimum area
                col, row, x, y = cv2.boundingRect( cnt )  # re : rect
                height, width = abs( row - y ), abs( col - x )
                if height * width < 100: break # 화면 크기 체크로 너무 작으면 break
                
                trackWindow = (col, row, width, height)
                roi = frame[row:row + height, col:col + width]
                hsv_roi = cv2.cvtColor( roi, cv2.COLOR_BGR2HSV )
                mask = cv2.inRange(hsv_roi, skinColorMin, skinColorMax)
                roi_hist = cv2.calcHist( [hsv_roi], [0], mask, [180], [0, 180] )
                
                cv2.normalize( roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX )
                cv2.imshow( "this is roi", roi )
                
                # center = (int(x), int(y))
                centerArray.append( center )

                radius = int( radius )
                cv2.circle( frame, center, radius, (255, 255, 255), 3 )
                # cv2.drawContours(frame, [cnt], -1, [255, 0, 0], 3)

                for center in centerArray:
                    cv2.circle( frame, center, 2, (255, 0, 0), -1 )

                cv2.imshow( 'frame', frame )

            if trackWindow is not None : break
                
            key = cv2.waitKey( 30 ) & 0xFF

            if key == ord( 'q' ):
                quit()

            elif key == ord( 's' ):  # S 를 누르면
                if os.path.exists( 'Result.jpg' ):  # 사진 유무 체크
                    nowChkImage = frame
                    # if NewVer.ExistFile.existFile( nowChkImage ) : # 체크할 사진 보내고 원본이랑 비교
                    print( "암호가 맞습니다." )

                    cv2.waitKey( 0 )
                    quit()
                else:
                    cv2.imwrite( 'Result.jpg', frame )

                    print( '암호가 새로 저장되었습니다. 다시 암호를 풀어주세요' )
                    start = False  # 이 while 문 종료하고, 다시 C를 누를 때까지 체크
                    # 검사
                
                    # Q를 누를 때까지 암호 만들기
    while trackWindow is not None:
        _, frame = cam.read()

        HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject( [HSV], [0], roi_hist, [0, 180], 1 )

        termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        getData, trackWindow = cv2.CamShift( dst, trackWindow, termination )

        pts = cv2.boxPoints( getData )
        pts = np.int0( pts )
        Show_frame = cv2.polylines( frame, [pts], True, (0, 255, 0), 2 )
        cv2.imshow('Showing', Show_frame)
except:
    pass

cv2.destroyAllWindows()
quit()
