# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 13:19:20 2020

@author: Deepak
"""

import cv2
import numpy as np
import imp
import GravityMachineTrack 
imp.reload(GravityMachineTrack)
import pandas as pd
import os
from scipy.ndimage.filters import uniform_filter1d

def threshold_image_gray(image_gray, LOWER, UPPER):
    imgMask = np.array((image_gray >= LOWER) & (image_gray <= UPPER), dtype='uint8')
    
    # imgMask = cv2.inRange(cv2.UMat(image_gray), LOWER, UPPER)  #The tracked object will be in white
    imgMask = cv2.erode(imgMask, None, iterations=2) # Do a series of erosions and dilations on the thresholded image to reduce smaller blobs
    imgMask = cv2.dilate(imgMask, None, iterations=2)
    
    return imgMask

def find_centroid_basic(image):
    #find contour takes image with 8 bit int and only one channel
    #find contour looks for white object on a black back ground
    # This finds the centroid with the maximum area in the current frame
    contours = cv2.findContours(image, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2]
    centroid=False
    isCentroidFound=False
    if len(contours)>0:
        cnt = max(contours, key=cv2.contourArea)
        M = cv2.moments(cnt)
        if M['m00']!=0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            centroid=np.array([cx,cy])
            isCentroidFound=True
    return isCentroidFound,centroid


 # Contrast factor for image
clahe_cliplimit = 3.0

# Create a CLAHE object 
clahe = cv2.createCLAHE(clipLimit = clahe_cliplimit, tileGridSize = (6,6))

# 3 vorticella
track_file = 'H:/2019 Monterey Trip/Vorticella_GM/2019_08_22/track9/track000.csv'
sphere = 'Sphere006'
T_start = 60
T_end = 90

track = GravityMachineTrack.gravMachineTrack(trackFile = track_file, Tmin = T_start, Tmax = T_end, findDims = True, flip_z = False, scaleFactor = 5)
save_file_name = sphere + 'RotationTracks_Tmin_{}_Tmax_{}'.format(T_start, T_end)

image_save_path = os.path.join('D:/Vorticella_GravityMachine/SphereRotation_analysis', save_file_name)

if(not os.path.exists(image_save_path)):
    os.makedirs(image_save_path)

stride = 10     # No:of image frames to skip

image_index_subsampled = track.imageIndex[::stride]

TimeStamp_start = track.df['Time'][image_index_subsampled[0]]

nFrames = len(image_index_subsampled)

print("No:of frames in track: {}".format(nFrames))


counter = 0
 # Display initial image and allow users to choose the n_points that are tracked
imageindex = image_index_subsampled[0] 
                
image_name = track.df['Image name'][imageindex]

image = cv2.imread(os.path.join(track.path, track.image_dict[image_name], image_name),0)
image = clahe.apply(image)

bbox = cv2.selectROI(image, False)

image_cropped = image[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])]

cv2.imshow("Cropped image", image_cropped)
cv2.waitKey(0)

kernel = np.ones((5,5),np.uint8)

while True and counter < nFrames-1:
    
    imageindex = image_index_subsampled[counter] 
    image_name = track.df['Image name'][imageindex]
    TimeStamp = track.df['Time'][imageindex] - TimeStamp_start
    
    
    image = cv2.imread(os.path.join(track.path, track.image_dict[image_name], image_name),0)
    image = clahe.apply(image)
    
    image_raw = np.copy(image)
    
    image_cropped = image[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])]

    ret2, imgMask = cv2.threshold(image_cropped,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#    imgMask = cv2.adaptiveThreshold(image_cropped, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,15,2)
    
#    imgMask = cv2.erode(imgMask, None, iterations = 2) # Do a series of erosions and dilations on the thresholded image to reduce smaller blobs
#    imgMask = cv2.dilate(imgMask, None, iterations = 2)
    
    # Morphological closing
    imgMask = cv2.morphologyEx(imgMask, cv2.MORPH_CLOSE, kernel)
    
#    edges = cv2.Canny(image_cropped,100, 200)

#    cv2.imshow("Raw image", image)
    
    cv2.imshow("Thres imageholded", imgMask)
#    cv2.imshow("Edge detection", edges)
    key = cv2.waitKey(1) & 0xff
    counter += 1    
    if(key == 27):
            # ESC. If the organism is detected correctly, then store the value and break from the loop
        break
    elif(key == 32):
            # SPACEBAR
        continue
    
    
cv2.destroyAllWindows()

    