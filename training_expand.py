
import numpy as np
import cv2

# translation with horizontally ±1 and vertically ±1
def ImageTranslation(img,step1,step2):
    rows, cols = img.shape
    T1 = np.random.uniform(-1,1)
    T1 = (T1>=0)*step1+(T1<0)*(-step1)
    T2 = np.random.uniform(-1,1)
    T2 = (T2>=0)*step2+(T2<0)*(-step2)
    M = np.float32([[1, 0, T1], [0, 1, T2]])
    return cv2.warpAffine(img, M, (cols, rows))

# counter-clockwise rotation with degree U(1,6)
def ImageRotation1(img,thetaL,thetaU):
    rows, cols = img.shape
    theta = np.random.uniform(thetaL,thetaU)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), theta, 1)
    return cv2.warpAffine(img, M, (cols, rows))

# clockwise rotation withe degree
def ImageRotation2(img,thetaL,thetaU):
    rows, cols = img.shape
    theta = np.random.uniform(thetaL,thetaU)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -theta, 1)
    return cv2.warpAffine(img, M, (cols, rows))