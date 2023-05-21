from cv2 import *
import numpy as np, os

camera = VideoCapture(0)
camera.set(CAP_PROP_FRAME_WIDTH, 1280)
size = (640 - 300, 360 - 300, 640 + 300, 360 + 300)
lower_orange, upper_orange = (0, 45, 45), (30, 255, 255)
contours, roi_hist, bR = [], None, None

while True:
        ret, frame = camera.read()
        if ret:
            ROI = frame[size[1]:size[3], size[0]:size[2]] # ROI 추출
            hsv = cvtColor(ROI, COLOR_BGR2HSV)
            ma = inRange(hsv, lower_orange, upper_orange)
            gyoziphap = bitwise_and(ROI, ROI, mask=ma) # 정한 범위의 색상들만 출력
            _, _, v = split(gyoziphap)
            blur = GaussianBlur(v, (5, 5), 0)
            hist = [0 for i in range(256)]

            for pixel in blur:
                for pix in pixel:
                    hist[pix] += 1

            n, prob, uT = blur.shape[0] * blur.shape[1], [], 0
            for i in range(256):
                prob.append(hist[i] / n)
                uT += i * prob[i]

            w0, u0, maxB, T = prob[0], 0, 0, 0
            for i in range(1, 256):
                w1 = 1 - w0
                if w0 > 0 and w1 > 0:
                    if maxB < (w0 / w1) * (uT - u0 / w0) ** 2:
                        maxB = (w0 / w1) * ((uT - u0 / w0) ** 2)
                        T = i
                
                u0 += i * prob[i]
                w0 += prob[i]
            _, thresholdImg = threshold(blur, T, 255, THRESH_BINARY)
            imshow('threshold', thresholdImg)
            
            kernel = np.ones((5, 5), np.uint8)
            c1 = morphologyEx(thresholdImg, MORPH_CLOSE, kernel) # 어두운 색 감소
            thresholdImg = morphologyEx(c1, MORPH_OPEN, kernel) # 밝은 색 감소

            contours, _ = findContours(thresholdImg, RETR_LIST, CHAIN_APPROX_SIMPLE)
            if len(contours):
                maxIDX, contours = 0, sorted(contours, key=contourArea)

                drawContours(thresholdImg, contours, -1, 255, 3)
                bR = boundingRect(contours[-1])
                imshow('trackWindow', ROI[bR[1]:bR[1]+bR[3], bR[0]:bR[0]+bR[2]])
            else: destroyWindow('trackWindow')

        if len(contours) and waitKey(1) == ord('q'):
            hsv_roi = cvtColor(ROI[bR[1]:bR[1]+bR[3], bR[0]:bR[0]+bR[2]], COLOR_BGR2HSV)
            ma2 = inRange(hsv_roi, lower_orange, upper_orange)
            roi_hist = calcHist([hsv_roi], [0], ma2, [180], [0, 180])
            destroyAllWindows()
            break

# CamShift 를 활용하여 손의 경로 추적 및 임시 암호 파일 취득(timg)
ROI, dotList = None, []
while bR:
    ret, frame = camera.read()
    if not ret: break
    ROI = frame[size[1]:size[3], size[0]:size[2]]
    hsv = cvtColor(ROI, COLOR_BGR2HSV)
    dst = calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
    ret, trackWindow = CamShift(dst, bR, (TERM_CRITERIA_EPS | TERM_CRITERIA_COUNT, 10, 1))
    pts = np.int0(boxPoints(ret))
    dotList.append(np.int0(sum(pts) / 4))
    polylines(ROI, [np.array(dotList)], False, (0, 255, 0), 5) # 경로 그리기
    imshow('drawPATH', ROI)
    if waitKey(1) == ord('s'): # 경로 저장
        destroyAllWindows()
        break

tRes = cvtColor(ROI, COLOR_BGR2HSV)
mask = inRange(tRes, (60, 250, 250), (60, 255, 255))
tRes = bitwise_and(tRes, tRes, mask=mask)
_, _, res = split(tRes)

resCons, _ = findContours(res, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
x, y, w, h = boundingRect(sorted(resCons, key=contourArea)[-1])
res = res[y:y+h, x:x+w] # 암호 영역 추출
destroyAllWindows()

if not os.path.isfile('crypto.jpg'):
    imwrite('crypto.jpg', res)

else:
    crypto = imread('crypto.jpg', IMREAD_GRAYSCALE)
    crypto, tcrypto = np.array(crypto), np.array(res)
    if crypto.shape != tcrypto.shape:
        height, width = tcrypto.shape
        crypto = resize(crypto, (width, height), interpolation=INTER_AREA)
    imshow('crypto', crypto)
    imshow('timg', tcrypto)

    while True:
        if waitKey(1) == ord('q'):
            destroyAllWindows()
            break
    
    err = np.sum((crypto.astype('float') - tcrypto.astype('float')) ** 2)
    err /= float(crypto.shape[0] * crypto.shape[1])
    if err < 12000: print('문이 열립니다!')
    else: print('실패하였습니다.')
    print(err)
exit()
