import cv2
import numpy as np
import math

startTrack = False

cam = cv2.VideoCapture( 0 )
# cv2.waitKey(0)

while (cam.isOpened()):
    ret, frame = cam.read()

    HSV = cv2.cvtColor( frame, cv2.COLOR_BGR2HSV )
    skinColorMin = np.array( [0, 48, 80] )
    skinColorMax = np.array( [20, 255, 255] )

    Range = cv2.inRange( HSV, skinColorMin, skinColorMax )
    _, thresholdImage = cv2.threshold( Range, 200, 255, cv2.THRESH_BINARY )
    # _, threshold_image = cv2.adaptiveThreshold(Range, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 2)
    cv2.imshow( 'threshold', thresholdImage )

    kernel = np.ones( (5, 5), np.uint8 )
    c1 = cv2.morphologyEx( thresholdImage, cv2.MORPH_CLOSE, kernel )
    c2 = cv2.morphologyEx( c1, cv2.MORPH_OPEN, kernel )
    thresholdImage = cv2.morphologyEx( c2, cv2.MORPH_OPEN, kernel )

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

        moments = cv2.moments( cnt )

        cx = 0
        cy = 0

        if moments['m00'] != 0:
            cx = int( moments['m10'] / moments['m00'] )
            cx = int( moments['m01'] / moments['m00'] )

        center = (cx, cy)

        # find the circle which completely covers the object with minimum area
        (x, y), radius = cv2.minEnclosingCircle( cnt )  # rasdius와 (x, y) 에 해당 범위를 감싸는 원을 그린다.
        center = (int( x ), int( y ))
        radius = int( radius )
        cv2.circle( frame, center, radius, (255, 255, 255), 3 )
        cv2.circle( frame, center, 10, [0, 0, 255], -1 )

        # cv2.drawContours(frame, [cnt], -1, [255, 0, 0], 3)

    cv2.imshow( 'frame', frame )

    if cv2.waitKey( 30 ) & 0xFF == ord( 'q' ):
        break

cv2.destroyAllWindows()
cam.release()
quit()

'''import cv2
import numpy as np
import math

startTrack = False

cam = cv2.VideoCapture( 'TestingMovie2.mp4' )
# cv2.waitKey(0)

centerArray = list()
SavingImage = None

start = False

try:
    while not start:
        ret, frame = cam.read()
        
        cv2.imshow('HIHI', frame)

        if cv2.waitKey(30) & 0xFF == ord('s'):
            start = not start

    # TODO 파일 검색 후 있으면 가져와서 비교하고, 아니면, 새로 암호 만들기
    # S를 입력할 경우 시작
    while True:
        ret, frame = cam.read() # (848, 480), (width, height)

        HSV = cv2.cvtColor( frame, cv2.COLOR_BGR2HSV )
        skinColorMin = np.array( [0, 48, 80] )
        skinColorMax = np.array( [20, 255, 255] )

        Range = cv2.inRange( HSV, skinColorMin, skinColorMax )
        _, thresholdImage = cv2.threshold( Range, 200, 255, cv2.THRESH_BINARY )
        # _, threshold_image = cv2.adaptiveThreshold(Range, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 2)
        cv2.imshow( 'threshold', thresholdImage )

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

            moments = cv2.moments( cnt )

            cx = 0
            cy = 0

            if moments['m00'] != 0:
                cx = int( moments['m10'] / moments['m00'] )
                cx = int( moments['m01'] / moments['m00'] )

            center = (cx, cy)

            # find the circle which completely covers the object with minimum area
            (x, y), radius = cv2.minEnclosingCircle( cnt )  # rasdius와 (x, y) 에 해당 범위를 감싸는 원을 그린다.
            center = (int( x ), int( y ))

            centerArray.append(center)

            radius = int( radius )
            cv2.circle( frame, center, radius, (255, 255, 255), 3 )

            for centerDot in centerArray:
                cv2.circle( frame, centerDot, 10, [0, 0, 255], -1 )

            SavingImage = frame
            # cv2.drawContours(frame, [cnt], -1, [255, 0, 0], 3)

        cv2.imshow( 'frame', frame )

        if cv2.waitKey( 30 ) & 0xFF == ord( 'q' ):
            break
        
        # Q를 누를 때까지 암호 만들기
except:
    pass

cv2.imwrite('Result.jpg', SavingImage)

cv2.destroyAllWindows()

cam.release()
quit()
'''