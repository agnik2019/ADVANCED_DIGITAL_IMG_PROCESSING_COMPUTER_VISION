import copy as cp
import cv2
import numpy as np
import sys
import os

def harris_corners(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_img = np.float32(gray_img)

    corners_img = cv2.cornerHarris(gray_img, 3, 3, 0.04)

    image[corners_img > 0.001 * corners_img.max()] = [255, 255, 0]

    return image

def shi_tomasi(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    corners_img = cv2.goodFeaturesToTrack(gray_img, 1200, 0.01, 10)
    # corners_img = np.int0(corners_img)

    blank_img = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)

    for corners in corners_img:
        x, y = corners.ravel()
        cv2.circle(image, (int(x), int(y)), 3, [255, 255, 0], -1)
        cv2.circle(blank_img, (int(x), int(y)), 2, [255, 255, 0], -1)

    return image, blank_img


def find_corner(img):
    img_dup = cp.copy(img)
    img_dup1 = cp.copy(img)

    harris = harris_corners(img)
    shitomasi, silhouette = shi_tomasi(img_dup)

    # Display different corner detection methods side by side

    out1 = np.concatenate((harris, shitomasi), axis=1)
    out2 = np.concatenate((img_dup1, silhouette), axis=1)

    out3 = np.concatenate((out1, out2), axis=0)
    cv2.imshow('Corners', out3)
    return harris, shitomasi, silhouette, out3



    # loading image
imgname = "PataChitraPuri_1.jpg"
img = cv2.imread(imgname)

harris, shitomasi, silhoutte, out3 = find_corner(img)
cv2.imwrite('qn2_corner_detect.jpg', out3)
cv2.waitKey(0)
cv2.destroyAllWindows()