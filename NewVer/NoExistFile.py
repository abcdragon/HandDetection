import cv2
import numpy as np

SavingImage = None

def NoFile(Image) :
    cv2.imwrite('Result.jpg', SavingImage)
    print('암호가 새로 저장되었습니다. 다시 암호를 풀어주세요')