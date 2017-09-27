import cv2
import numpy as np
import math

# Define put text font
font = cv2.FONT_HERSHEY_SIMPLEX
# Define the codec and create VideoWriter object
#fourcc = cv2.VideoWriter_fourcc(*'XVID')                                                       # - 동영상 저장 - 1
#record = cv2.VideoWriter('output/' + str(datetime.now()) + '.avi', fourcc, 20.0, (640, 480))  # - 동영상 저장 - 2

def main():
    # Capture from webcam
    cam = cv2.VideoCapture('TestingMovie.mp4') # 카메라에서 읽어오기
    fgbg = cv2.createBackgroundSubtractorMOG2() # 영상 속 배경 추출, http://eehoeskrap.tistory.com/117 - 참고 주소

    while True:
        ret, frame = cam.read()

        # print(frame.shape)

        if ret is False:
            return

        # Show ROI rectangle
        cv2.rectangle(frame, (50, 50), (400, 400), (255, 255, 2), 4)  # outer most rectangle
        ROI = frame[50:400, 50:400] # ROI 저장

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        # COLOR_MIN = np.array([61, 132, 109], np.uint8)
        # COLOR_MAX = np.array([166, 166, 166], np.uint8)
        COLOR_MIN = np.array([0, 137, 77], np.uint8)
        COLOR_MAX = np.array([255, 173, 127], np.uint8)

        frame_threshed = cv2.inRange(hsv, COLOR_MIN, COLOR_MAX)
        ret, threshold_image = cv2.threshold(frame_threshed, 100, 255, cv2.THRESH_BINARY)

        # MOG2 Background Subtraction
        #fgmask = ROI
        #fgbg.setBackgroundRatio(0.005)
        #fgmask = fgbg.apply(ROI, fgmask)

        # Noise remove
        kernel = np.ones((5, 5), np.uint8)
        c1 = cv2.morphologyEx(threshold_image, cv2.MORPH_OPEN, kernel) # 잡음 제거 기술로 열림필터와 닫힘필터로 제거한다. 아래 주소에서 참고
        c2 = cv2.morphologyEx(c1, cv2.MORPH_CLOSE, kernel)
        closing = cv2.morphologyEx(c2, cv2.MORPH_CLOSE, kernel) # http://hongkwan.blogspot.kr/2013/01/opencv-5-2-example.html - 참고

        # Blur the image
        #blur = cv2.blur(ROI, (3, 3))

        # Convert to HSV color space
        #hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        # Create a binary image with where white will be skin colors and rest is black
        #thresh = cv2.inRange(hsv, np.array([2, 50, 50]), np.array([20, 255, 255]))

        #_, thresh = cv2.threshold(fgmask, 75, 255, cv2.THRESH_BINARY);
        # Find contours of the filtered frame
        _, contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # print(contours)

        # Draw Contours
        for cnt in contours:
            color = [222, 222, 222]  # contours color
            cv2.drawContours(ROI, [cnt], -1, color, 3)

        if contours:

            cnt = contours[0]

            # Find moments of the contour
            moments = cv2.moments(cnt) # 무게 중심

            cx = 0
            cy = 0
            # Central mass of first order moments
            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])  # cx = M10/M00
                cy = int(moments['m01'] / moments['m00'])  # cy = M01/M00

            center = (cx, cy)

            # Draw center mass
            cv2.circle(ROI, center, 15, [0, 0, 255], 2)

            # find the circle which completely covers the object with minimum area
            (x, y), radius = cv2.minEnclosingCircle(cnt) # rasdius와 (x, y) 에 해당 범위를 감싸는 원을 그린다.
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(ROI, center, radius, (0, 0, 0), 3)
            area_of_circle = math.pi * radius * radius

            # drawn bounding rectangle with minimum area, also considers the rotation
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(ROI, [box], 0, (0, 0, 255), 2)

            # approximate the shape
            cnt = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True) # contour를 찾고, contour를 다각형으로 근사화 시키는 함수

            # Find Convex Defects
            hull = cv2.convexHull(cnt, returnPoints=False) # <- 외각선 그리기
            defects = cv2.convexityDefects(cnt, hull) # <- 손가락 안쪽 점을 그림
            # https://m.blog.naver.com/PostView.nhn?blogId=samsjang&logNo=220524551089&proxyReferer=https%3A%2F%2Fwww.google.co.kr%2F - 참고

            fingers = 0 # 손가락 갯수를 담는 변수

            # Get defect points and draw them in the original image
            if defects is not None:
                # print('defects shape = ', defects.shape[0])
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(cnt[s][0])
                    end = tuple(cnt[e][0])
                    far = tuple(cnt[f][0])
                    cv2.line(ROI, start, end, [0, 255, 0], 3)
                    cv2.circle(ROI, far, 8, [211, 84, 0], -1)
                    #  finger count
                    a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                    b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                    c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                    angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # arc cos - > cos의 역함수 (밑변/빗변)을 대입하여 각도를 구한다.
                    #       math.acos((          빗변          ) / (   높이  ))
                    # 또한 아크 코사인은 cos의 역함수 이기 때문에 대입 될 수 있는 값이 1을 넘지 못한다.
                    # 아크 코사인의 치역은 -(파이 값) <= 치역 <= 파이(각도로 변환하면 180) 이다. (호도법으로 나타내서 파이로 표현)
                    # http://tip.daum.net/question/56035762 - 표와 그래프 참고
                    # 대입 되는 값이 1에 가까우면 0, -1에 가까우면 180, 0에 가까우면 90도이다.
                    area = cv2.contourArea(cnt)

                    if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers -> 90도 아래면 손가락
                        fingers += 1
                        cv2.circle(ROI, far, 1, [255, 0, 0], -1)

                    if len(cnt) >= 5:
                        (x_centre, y_centre), (minor_axis, major_axis), angle_t = cv2.fitEllipse(cnt)
                        # 첫번째 튜플은 중심 좌표, 두번째 튜플은 중심선(?)

                    letter = ''
                    if area_of_circle - area < 5000:
                        #  print('A')
                        letter = 'A'
                    elif angle_t > 120:
                        letter = 'U'
                    elif area > 120000:
                        letter = 'B'
                    elif fingers == 1:
                        if 40 < angle_t < 66:
                            # print('C')
                            letter = 'C'
                        elif 20 < angle_t < 35:
                            letter = 'L'
                        else:
                            letter = 'V'
                    elif fingers == 2:
                        if angle_t > 100:
                            letter = 'F'
                        # print('W')
                        else:
                            letter = 'W'
                    elif fingers == 3:
                        # print('4')
                        letter = '4'
                    elif fingers == 4:
                        # print('Ola!')
                        letter = 'Ola!'
                    else:
                        if 169 < angle_t < 180:
                            # print('I')
                            letter = 'I'
                        elif angle_t < 168:
                            # print('J')
                            letter = 'J'

                    # Prints the letter and the number of pointed fingers and
                    print('Fingers = '+str(fingers)+' | Letter = '+str(letter))

        else:
            # prints msg: no hand detected
            cv2.putText(frame, "No hand detected", (45, 450), font, 2, np.random.randint(0, 255, 3).tolist(), 2)

        # Show outputs images
        cv2.imshow('frame', frame)
        #cv2.imshow('blur', blur)
        #cv2.imshow('hsv', hsv)
        #cv2.imshow('thresh', thresh)
        #cv2.imshow('mog2', fgmask)
        cv2.imshow('ROI', ROI)

        #record.write(frame)

        # Check key pressed
        if cv2.waitKey(100) == 27:
            break  # ESC to quit

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()