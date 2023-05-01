import numpy as np
import os
import cv2

corner = 8
block = 14
resolution = [1440, 1080]

rootdir = os.getcwd()
imgPath = os.path.join(rootdir, 'intrinsic')

objp = np.zeros((corner*corner,3), np.float32)
objp[:,:2] = np.mgrid[0:corner, 0:corner].T.reshape(-1,2)
objp = objp * block

# get the chessboard image
images = []
for filename in os.listdir(imgPath):
    fullpath = os.path.join(imgPath, filename)
    if filename.startswith('chessboard_') and filename.endswith('.jpg'):
        images.append(fullpath)

print('Start finding chessboard corners...')
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    img = cv2.resize(img, (resolution[0], resolution[1]), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (corner, corner), None)
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)
img_size = (img.shape[1], img.shape[0])

# calculate camera intrinsic
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
