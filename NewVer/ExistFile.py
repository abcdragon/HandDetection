import cv2
import numpy as np

def existFile(nowChkImage):
    try:
        CompareImage = cv2.imread('Result.jpg')

        sift = cv2.SIFT()

        kp1, des1 = sift.detectAndCompute(nowChkImage, None)
        kp2, des2 = sift.detectAndCompute(CompareImage, None)

        bf = cv2.BFMatcher()
        matches = bf.match(des1, des2)

        matches = sorted(matches, key=lambda val: val.distance)

        return False
    except:
        pass