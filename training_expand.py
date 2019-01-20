
import numpy as np
import cv2

def ImageTranslation(img,step1,step2):
    """image translation with horizontal step ±step1
    and vertical step ±step2"""
    rows, cols = img.shape
    T1 = np.random.uniform(-1,1)
    T1 = (T1>=0)*step1+(T1<0)*(-step1)
    T2 = np.random.uniform(-1,1)
    T2 = (T2>=0)*step2+(T2<0)*(-step2)
    M = np.float32([[1, 0, T1], [0, 1, T2]])
    return cv2.warpAffine(img, M, (cols, rows))

def ImageRotation1(img,thetaL,thetaU):
    """counter-clockwise rotation with degree U(thetaL,thetaU)"""
    rows, cols = img.shape
    theta = np.random.uniform(thetaL,thetaU)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), theta, 1)
    return cv2.warpAffine(img, M, (cols, rows))

def ImageRotation2(img,thetaL,thetaU):
    """clockwise rotation with degree U(thetaL,thetaU)"""
    rows, cols = img.shape
    theta = np.random.uniform(thetaL,thetaU)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -theta, 1)
    return cv2.warpAffine(img, M, (cols, rows))