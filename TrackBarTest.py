import numpy as np
import cv2

# dummy function
def nothing(x):
    pass

# Track bar window
cv2.namedWindow('thresh')

# Track bars
cv2.createTrackbar('lH','thresh', 0, 255, nothing)
cv2.createTrackbar('lS', 'thresh', 0, 255, nothing)
cv2.createTrackbar('lV', 'thresh', 0, 255, nothing)
cv2.createTrackbar('uH', 'thresh', 0, 255, nothing)
cv2.createTrackbar('uS', 'thresh', 0, 255, nothing)
cv2.createTrackbar('uV', 'thresh', 0, 255, nothing)

# capture video
cap = cv2.VideoCapture(0)

while True:
    # Read from camera
    source, frame = cap.read()
    if not source:
        break

    # converting image color to HSV color space
    cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb, frame, 0)

    # Getting values from track bars
    lH = cv2.getTrackbarPos('lH', 'thresh')
    uH = cv2.getTrackbarPos('uH', 'thresh')
    lS = cv2.getTrackbarPos('lS', 'thresh')
    uS = cv2.getTrackbarPos('uS', 'thresh')
    lV = cv2.getTrackbarPos('lV', 'thresh')
    uV = cv2.getTrackbarPos('uV', 'thresh')

    lowerb = np.array([lH, lS, lV], np.uint8)
    upperb = np.array([uH, uS, uV], np.uint8)

    cv2.imshow('image', frame)
    frame = cv2.inRange(frame, lowerb, upperb)
    cv2.flip(frame, 1, frame)
    cv2.imshow('thresh', frame)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()