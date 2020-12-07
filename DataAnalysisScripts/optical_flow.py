# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 09:14:53 2020
Optical flow from images
@author: Deepak
"""

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

video_folder = 'D:/Vorticella_GravityMachine/SphereRotation_analysis/2019_08_21_Afternoon_Track7_enhanced' 





files = os.listdir(video_folder)

first_frame = np.array(cv2.imread(os.path.join(video_folder, files[0]), 0), dtype = 'uint8')

print(np.shape(first_frame))
cv2.imshow("First frame", first_frame)
cv2.waitKey(1)

prev_frame = first_frame

imW, imH = np.shape(first_frame)[0], np.shape(first_frame)[1]

print(imW)
print(imH) 
# Creates an image filled with zero 
# intensities with the same dimensions  
# as the frame 
mask = np.zeros((imW, imH, 3))

print(np.shape(mask)) 
  

# Sets image saturation to maximum 
mask[:,:, 1] = 255
  
stride = 50

flow_array = []

x, y = np.meshgrid(range(imH), range(imW))

print('x array shape: {}'.format(np.shape(x)))


plt.figure(1)

for ii, file in enumerate(files[1:-1:stride]): 
      
    # ret = a boolean return value from getting 
    # the frame, frame = the current frame being 
    # projected in the video 
    frame = np.array(cv2.imread(os.path.join(video_folder, file), 0), dtype = 'uint8')
      
    # Opens a new window and displays the input 
    # frame 
    cv2.imshow("input", frame) 
      
    # Converts each frame to grayscale - we previously  
    # only converted the first frame to grayscale 
#    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) 
      
    # Calculates dense optical flow by Farneback method 
    flow = cv2.calcOpticalFlowFarneback(prev_frame, frame, None, pyr_scale = 0.5, levels = 5, winsize = 11, iterations = 5, poly_n = 5, poly_sigma = 1.1, flags = 0)
      
    flow_array.append(flow)
    # Computes the magnitude and angle of the 2D vectors 
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1]) 
    
    plt.clf()
    ax1 = plt.imshow(frame,cmap=plt.cm.gray, alpha=1.0)
    ax2 = plt.quiver(x[::100,::100], y[::100,::100], flow[::100,::100,0], flow[::100,::100, 1])
    plt.axis('image')
    plt.show(block=False)
    plt.pause(0.001)
      
    # Sets image hue according to the optical flow  
    # direction 
    mask[..., 0] = angle * 180 / np.pi / 2
      
    # Sets image value according to the optical flow 
    # magnitude (normalized) 
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX) 
      
    # Converts HSV to RGB (BGR) color representation 
    rgb = cv2.cvtColor(np.float32(mask), cv2.COLOR_HSV2BGR) 
      
    # Opens a new window and displays the output frame 
    cv2.imshow("dense optical flow", rgb) 
      
    # Updates previous frame 
    prev_frame = frame 
      
    # Frames are read by intervals of 1 millisecond. The 
    # programs breaks out of the while loop when the 
    # user presses the 'q' key 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
# The following frees up resources and 
# closes all windows 

cv2.destroyAllWindows() 


plt.figure(1)
plt.quiver(flow_array[0][::100,::100,0], flow_array[0][::100,::100,1])

#for ii in range(len(flow_array)):
#    
#    flow = flow_array[ii]
#    
#    plt.cla()
#    plt.quiver(flow[:,:,0], flow[:,:,1])
#    plt.pause(0.001)
#    plt.show()
    

    